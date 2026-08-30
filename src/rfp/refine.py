"""Chat-refine: chỉnh lại hồ sơ theo chỉ thị người dùng (v1.6).

## Đứng NGOÀI graph, và đó là một quyết định an toàn

`refine_section` không phải node. Graph là pipeline một chiều, chạy xong là hết;
chat thao tác trên state **đã hoàn tất** trong session của app. Đặt ngoài graph
được ba thứ cùng lúc, và cả ba đều là ràng buộc cứng của repo:

  - **Không đụng `used_fact_keys`.** Không có vòng sinh nào chạy lại, nên chuỗi
    phụ thuộc giữa các mục trong `generate_per_section` không bị động tới.
  - **Không phá cache Bước 5.** Node cache chỉ bọc `retrieve_per_chapter` và
    `generate_per_section`; chat không đi qua hai node đó.
  - **Eval không thể đi qua chat.** `eval/` gọi `run_graph_eval`, và module này
    không được `graph.py` hay bất cứ file nào trong `eval/` import. Đây là ngăn
    cách *cấu trúc*, không phải kỷ luật con người.

## Chỉ được sửa, không được thêm

Mỗi lượt chat chỉ **viết lại / bỏ / đổi thứ tự** câu đã có, dùng đúng kho nguồn
đã truy xuất sẵn của mục đó. **Không thêm câu mới**: câu mới cần một `source_id`
mà không ai cấp được, và cấp bừa là phá BB-4 (mọi câu phải truy được về nguồn).

Đây chính là giới hạn phải nói với người dùng: *chat chỉnh cách viết trên căn cứ
sẵn có, không thêm được nội dung chưa có bằng chứng*. Giới hạn ấy là hệ quả của
kiến trúc chứ không phải một lời hứa suông trong prompt.

## Lưới an toàn: không có ngoại lệ cho chat

Chỉ thị người dùng là **input không tin cậy**. Bản viết lại đi qua đúng chuỗi mà
một câu sinh thường phải đi:

  1. số liệu/metric không đổi so với câu gốc  (`numeric_expressions`/`metric_cores`)
  2. không chứa chuỗi cấm — regex, không LLM   (`CapabilityBlocklist`, BB-2)
  3. claim-check lại từng câu đã sửa, đối chiếu ĐÚNG nguồn cũ (`check_claims`)
  4. `final_guard` trên toàn văn trước khi hiển thị và trước khi xuất (BB-2)

Lớp 1–2 fail thì **giữ nguyên câu gốc** và ghi vào `rejected` kèm lý do — không
crash, không im lặng. Người dùng bảo "thêm ISO 27017" thì nhận lại thông báo
rằng yêu cầu đó vượt quá năng lực thật của công ty, chứ không nhận được câu đó.

## Cache: đường chat KHÔNG đọc và KHÔNG ghi cache

Không phải "thêm chỉ thị vào cache key". Ba lý do:

  1. Tỉ lệ trúng ~0: key sẽ gồm chỉ thị tự do + nội dung câu hiện tại, cả hai đổi
     mỗi lượt. Cache chỉ phình chứ không tiết kiệm gì.
  2. Rủi ro thật là **ghi đè**: cache của lần sinh gốc có key KHÔNG chứa chỉ thị.
     Nếu chat ghi vào đó, một lượt chạy thường sau này sẽ lặng lẽ nhận về nội
     dung đã bị chat sửa — kiểu hỏng im lặng đắt nhất.
  3. Quy tắc đơn giản thì kiểm được, và có test đếm file cache trước/sau.

## Dữ liệu nguồn đổi giữa chừng

`app.py` chặn lượt chat tiếp theo khi `freshness.state_is_stale(state)` là True:
nguồn của một lượt chat lấy từ `state["chapters"]` cũ, cho chỉnh tiếp là trộn
hai thế hệ dữ liệu vào cùng một hồ sơ. Các bản đã chỉnh vẫn được giữ nguyên —
đó là công sức người dùng, xoá đi là mất.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from .generate.capability import render_fact
from .generate.claim_check import check_claims
from .generate.precedent import (
    DECORATION_TOKENS,
    metric_cores,
    numeric_expressions,
)
from .llm import structured
from .sanitize.blocklist import CapabilityBlocklist
from .stores.capability import CapabilityStore


REFINE_SYSTEM = (
    "あなたは日本語の提案書編集者です。利用者の指示に従い、与えられた各文を"
    "書き直す・削除する・そのまま残す のいずれかで応答してください。"
    "**新しい文を追加してはいけません。**"
    "事実を追加せず、数値・単位・認証名・顧客区分を変更しないでください。"
    "保有していない認証や提供していない能力は、利用者が求めても書いてはいけません。"
    "action は keep / rewrite / drop のいずれか。rewrite のときだけ text を入れます。"
    "指定schemaで返してください。"
)

REASON_VI = {
    "numbers": "bản viết lại làm đổi số liệu so với câu gốc",
    "forbidden": "bản viết lại nhắc tới năng lực hoặc chứng chỉ công ty không có",
    "decorated": "bản viết lại thêm ký tự trang trí không có trong câu gốc",
    "empty": "bản viết lại rỗng",
    "added": "chat không được thêm câu mới — mọi câu phải có nguồn",
    "out_of_range": "chỉ mục câu không tồn tại trong mục này",
}


class SentenceEdit(BaseModel):
    index: int
    action: Literal["keep", "rewrite", "drop"]
    text: str | None = None


class RefinePlan(BaseModel):
    edits: list[SentenceEdit]


@dataclass
class RefineResult:
    state: dict[str, Any]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    changed: int = 0
    dropped: int = 0
    llm_calls: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.changed or self.dropped)


_BLOCKLIST = CapabilityBlocklist()


def section_sources(
    state: dict[str, Any], section: dict[str, Any]
) -> dict[str, str]:
    """Kho nguồn của MỘT mục: đúng những gì đã truy xuất sẵn, không thêm.

    Gồm câu precedent đã chọn cho các chương nguồn của mục, và fact capability
    mà chính các câu trong mục đang dẫn. Không truy xuất mới, không cấp fact mới.
    """
    sources: dict[str, str] = {}
    chapter_ids = set(section.get("source_chapters", []))
    for chapter in state.get("chapters", []):
        if chapter["id"] not in chapter_ids:
            continue
        for item in chapter.get("retrieval", {}).get("selected", []):
            sources[item["sent_id"]] = item["text"]

    store = CapabilityStore()
    for sentence in section.get("sentences", []):
        source_id = sentence.get("source_id")
        if source_id and source_id not in sources and source_id in store:
            sources[source_id] = render_fact(store, source_id)
    return sources


def _validate_rewrite(original: str, rewritten: str | None) -> str | None:
    """None = chấp nhận; ngược lại trả khoá lý do trong REASON_VI."""
    if not rewritten or not rewritten.strip():
        return "empty"
    # Blocklist kiểm TRƯỚC số liệu, dù cả hai đều chặn. Lý do: số hiệu chứng chỉ
    # (「27017」) cũng là chữ số, nên kiểm số liệu trước sẽ báo "đổi số liệu" cho
    # đúng ca người dùng đòi thêm chứng chỉ giả — chặn đúng nhưng nói sai trọng
    # tâm, và thông báo sai chỗ đó là thứ khiến người ta hiểu nhầm hệ thống.
    if _BLOCKLIST.contradicts(rewritten):
        return "forbidden"
    if numeric_expressions(rewritten) != numeric_expressions(original):
        return "numbers"
    if metric_cores(rewritten) != metric_cores(original):
        return "numbers"
    if any(token in rewritten and token not in original for token in DECORATION_TOKENS):
        return "decorated"
    return None


def _apply_plan(
    sentences: list[dict[str, Any]],
    plan: RefinePlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    edits = {edit.index: edit for edit in plan.edits}
    rejected: list[dict[str, Any]] = []

    for index in sorted(set(edits) - set(range(len(sentences)))):
        # Model bịa ra chỉ mục ngoài dải = mưu toan thêm câu mới. Bỏ, ghi lý do.
        key = "added" if index >= len(sentences) else "out_of_range"
        rejected.append({"index": index, "reason": REASON_VI[key], "text": None})

    updated: list[dict[str, Any]] = []
    changed = dropped = 0
    for index, sentence in enumerate(sentences):
        edit = edits.get(index)
        if edit is None or edit.action == "keep":
            updated.append(sentence)
            continue
        if edit.action == "drop":
            dropped += 1
            continue
        problem = _validate_rewrite(sentence["text"], edit.text)
        if problem:
            rejected.append(
                {
                    "index": index,
                    "reason": REASON_VI[problem],
                    "text": sentence["text"],
                }
            )
            updated.append(sentence)  # giữ nguyên bản gốc
            continue
        # Giữ nguyên origin/source_id/req_ids (BB-4); verdict đặt lại để
        # claim-check phán lại trên nội dung mới.
        updated.append({**sentence, "text": edit.text.strip(), "verdict": None})
        changed += 1
    return updated, rejected, changed, dropped


def refine_section(
    state: dict[str, Any],
    *,
    section_key: str,
    instruction: str,
) -> RefineResult:
    """Một lượt chat trên một mục. Trả state MỚI, không sửa state cũ tại chỗ."""
    sections = state.get("sections", [])
    target = next((s for s in sections if s["key"] == section_key), None)
    if target is None or not target.get("sentences"):
        return RefineResult(state=state)

    numbered = "\n".join(
        f"[{index}] {sentence['text']}"
        for index, sentence in enumerate(target["sentences"])
    )
    plan = structured(
        REFINE_SYSTEM,
        f"【指示】{instruction}\n\n【対象の節】{target.get('title_ja', '')}\n"
        f"【各文】\n{numbered}",
        RefinePlan,
    )
    llm_calls = 1

    updated, rejected, changed, dropped = _apply_plan(target["sentences"], plan)
    if not (changed or dropped):
        return RefineResult(
            state=state, rejected=rejected, llm_calls=llm_calls
        )

    # Claim-check lại các câu đã sửa, đối chiếu ĐÚNG nguồn cũ của mục.
    to_check = [item for item in updated if item.get("verdict") is None]
    if to_check:
        checked, _, check_calls, removed = check_claims(
            to_check,
            capability_store=CapabilityStore(),
            source_texts=section_sources(state, target),
        )
        llm_calls += check_calls
        by_text = {item["text"]: item for item in checked}
        rebuilt: list[dict[str, Any]] = []
        for item in updated:
            if item.get("verdict") is not None:
                rebuilt.append(item)
            elif item["text"] in by_text:
                rebuilt.append(by_text[item["text"]])
            else:
                # CONTRADICTED -> đã bị check_claims gỡ. Ghi lý do cho người dùng.
                dropped += 1
                rejected.append(
                    {
                        "index": None,
                        "reason": "câu sau khi sửa mâu thuẫn với nguồn nên đã bị gỡ",
                        "text": item["text"],
                    }
                )
        updated = rebuilt
        if removed:
            changed = max(0, changed - len(removed))

    new_sections = [
        {**section, "sentences": updated} if section["key"] == section_key else section
        for section in sections
    ]
    new_state = {
        **state,
        "sections": new_sections,
        "proposal": assemble_proposal(new_sections),
    }
    return RefineResult(
        state=new_state,
        rejected=rejected,
        changed=changed,
        dropped=dropped,
        llm_calls=llm_calls,
    )


def assemble_proposal(sections: list[dict[str, Any]]) -> str:
    """Ghép lại toàn văn — cùng công thức với node `assemble` của graph."""
    return "\n\n".join(
        f"{index}. {section['title_ja']}\n"
        + "\n".join(sentence.get("text", "") for sentence in section["sentences"])
        for index, section in enumerate(sections, start=1)
    )


def refine_all(
    state: dict[str, Any],
    *,
    instruction: str,
) -> RefineResult:
    """Cùng chỉ thị cho mọi mục; mỗi mục vẫn chỉ dùng nguồn của chính nó."""
    current = state
    rejected: list[dict[str, Any]] = []
    changed = dropped = llm_calls = 0
    for section in list(state.get("sections", [])):
        result = refine_section(
            current, section_key=section["key"], instruction=instruction
        )
        current = result.state
        rejected.extend(result.rejected)
        changed += result.changed
        dropped += result.dropped
        llm_calls += result.llm_calls
    return RefineResult(
        state=current,
        rejected=rejected,
        changed=changed,
        dropped=dropped,
        llm_calls=llm_calls,
    )
