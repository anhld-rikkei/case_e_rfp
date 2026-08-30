"""Chat-refine: làm theo chỉ thị người dùng, rồi dán nhãn (v1.6 · v1.7).

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

## Làm theo ý người dùng, rồi DÁN NHÃN trung thực (đổi triết lý ở v1.7)

Người dùng là human-in-the-loop và chịu trách nhiệm về nội dung họ yêu cầu. Nên
chat **làm theo chỉ thị**: viết lại tự do, **thêm câu mới**, bỏ câu, đổi số liệu.
Trước v1.7 những việc đó bị chặn; giờ chúng được thực hiện và **đánh dấu**.

Nguyên tắc thay thế cho "chặn": *câu nào không còn trung thành với nguồn của nó
thì phải mang nhãn của người dùng, không được đội lốt câu máy sinh có căn cứ.*
Cụ thể một câu trở thành `origin="user"` / `verdict="USER_PROVIDED"` khi:

  - người dùng yêu cầu **thêm mới** (không có nguồn nào để dẫn), hoặc
  - bản viết lại **đổi số liệu/metric** so với câu gốc, hoặc
  - claim-check phán **CONTRADICTED** sau khi sửa (không còn khớp nguồn cũ).

Câu chỉ được đổi cách viết mà giữ nguyên nguồn, số liệu và ý thì giữ nguyên
`origin`/`source_id`, chỉ gắn cờ `edited_by_chat` để hiển thị nhạt hơn.

## Lằn ranh DUY NHẤT không nhân nhượng: blocklist

Chứng chỉ và năng lực công ty không có (`ISO/IEC 27017`, `CMMI`, …) **không có
cửa nào chèn vào được, kể cả qua chat**. Đây là yêu cầu gốc của đề bài, không
phải một tuỳ chọn về trải nghiệm: một hồ sơ thầu tuyên bố sai chứng chỉ là hồ sơ
gây hậu quả pháp lý, và nhãn "người dùng tự thêm" không gột được việc công ty đã
nộp nó. Chặn ở hai chỗ: tiền kiểm chính chỉ thị, và kiểm mọi câu chat sinh ra.
`final_guard` vẫn chạy trước khi hiển thị và trước khi xuất.

## BB-4 vẫn nguyên vẹn cho output pipeline

`origin="user"` là **nhãn hậu-pipeline**, chỉ sống trong session app và file xuất.
`check.py` không biết tới nó và sẽ **fail loud** (`origin không hợp lệ: user`) nếu
ai đó đổ một state đã qua chat vào contract checker hoặc eval. Đó là hành vi
đúng: BB-4 ràng buộc thứ *hệ thống tự sinh*, còn đây là thứ *người dùng tự viết*
và đã được dán nhãn như vậy.

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

import re
import unicodedata
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
    "あなたは日本語の提案書編集者です。利用者の指示に忠実に従ってください。"
    "各文について keep / rewrite / drop を選び、rewrite のときだけ text を入れます。"
    "指示が新しい内容の追加を求める場合は added に新しい文を入れてください。"
    "数値の変更も指示があれば行って構いません。"
    "ただし、保有していない認証や提供していない能力（ISO/IEC 27017 など）は"
    "利用者が求めても絶対に書いてはいけません。"
    "【固定】が付いた文は利用者が自分で追加した文です。keep 以外を選んではいけません。"
    "説明や箇条書き記号を付けず、指定schemaで返してください。"
)

CAPABILITY_HINT = (
    "Nếu công ty thật sự có, hãy cập nhật `capability_sheet.json` rồi bấm "
    "**Nạp lại kho tri thức** và sinh lại hồ sơ."
)

REASON_VI = {
    "forbidden": (
        "yêu cầu thêm năng lực hoặc chứng chỉ **không có trong bảng năng lực** "
        f"— hệ thống không thêm nội dung không có căn cứ. {CAPABILITY_HINT}"
    ),
    "empty": "mô hình trả về câu rỗng nên bỏ qua thay đổi đó",
    "instruction_forbidden": (
        "chỉ thị yêu cầu thêm năng lực hoặc chứng chỉ **công ty không có** "
        f"({{terms}}) — hệ thống không thêm nội dung không có căn cứ. "
        f"{CAPABILITY_HINT}"
    ),
}


class SentenceEdit(BaseModel):
    index: int
    action: Literal["keep", "rewrite", "drop"]
    text: str | None = None


class RefinePlan(BaseModel):
    edits: list[SentenceEdit] = []
    added: list[str] = []


USER_ORIGIN = "user"
USER_VERDICT = "USER_PROVIDED"


def user_sentence(text: str, *, req_ids: list[str] | None = None) -> dict[str, Any]:
    """Câu do người dùng đưa vào — không có nguồn, và nói thẳng ra như vậy.

    Mặc định **được ghim**: lượt chat sau không được sửa hay xoá nó. Xem
    `_protected_indices` để biết vì sao mặc định là ghim chứ không phải không.
    """
    return {
        "text": text.strip(),
        "origin": USER_ORIGIN,
        "source_id": None,
        "req_ids": list(req_ids or []),
        "verdict": USER_VERDICT,
        "pinned": True,
    }


# Token "đặc trưng" để nhận ra chỉ thị có nhắm đích danh một câu ghim không.
# Latin/alnum (PL-300, BI, AWS) · katakana ≥3 · kanji ≥3. Ngưỡng dài cho tiếng
# Nhật vì từ 2 chữ (対応, 要件) xuất hiện khắp nơi và sẽ khớp nhầm liên tục.
_TARGET_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+|[ァ-ヶー]{3,}|[一-龯]{3,}")


def instruction_targets(text: str, instruction: str) -> bool:
    """Chỉ thị có nói đích danh nội dung của câu này không?

    Sai về phía nào cũng có giá, nên chọn hướng sai an toàn: **không nhận ra**
    thì câu vẫn được giữ nguyên và người dùng thấy thông báo "đã giữ lại N câu"
    kèm nút bỏ ghim. Ngược lại — nhận nhầm là có nhắm — thì câu người dùng thêm
    biến mất im lặng, đúng lỗi mà cơ chế ghim sinh ra để chặn.
    """
    normalized = unicodedata.normalize("NFKC", instruction)
    tokens = set(_TARGET_TOKEN_RE.findall(unicodedata.normalize("NFKC", text)))
    return any(token in normalized for token in tokens)


def _protected_indices(
    sentences: list[dict[str, Any]], instruction: str
) -> set[int]:
    """Câu nào lượt chat này KHÔNG được đụng tới.

    Câu người dùng thêm mặc định được ghim vì đã xảy ra đúng chuyện này: người
    dùng thêm một câu ở lượt 2, lượt 3 ra chỉ thị về mục khác, và model xoá luôn
    câu đó. Công sức người dùng không được biến mất vì một lượt chat nói về
    chuyện khác.
    """
    return {
        index
        for index, sentence in enumerate(sentences)
        if sentence.get("pinned")
        and not instruction_targets(sentence.get("text", ""), instruction)
    }


@dataclass
class RefineResult:
    state: dict[str, Any]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    changed: int = 0
    dropped: int = 0
    kept: int = 0
    added: int = 0
    restored: int = 0
    llm_calls: int = 0

    @property
    def touched(self) -> bool:
        return bool(self.changed or self.dropped or self.added)

    @property
    def blocked(self) -> int:
        return len(self.rejected)

    @property
    def outcome(self) -> str:
        """Ba kết cục phải nói khác nhau trên UI.

        Gộp `blocked` và `no_change` làm một là lỗi đã gặp thật: người dùng đòi
        thêm một chứng chỉ công ty không có, hệ thống từ chối đúng, nhưng màn
        hình chỉ ghi "không có thay đổi nào" — đọc ra thành tính năng hỏng.
        """
        if self.touched:
            return "changed"
        if self.rejected:
            return "blocked"
        return "no_change"


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


def _reject_reason(text: str | None) -> str | None:
    """Lý do TỪ CHỐI hẳn một câu chat sinh ra. Chỉ còn hai: rỗng và chuỗi cấm.

    Đổi số liệu và thêm câu mới không còn nằm ở đây — chúng được **thực hiện**
    rồi dán nhãn `user` (v1.7). Blocklist thì không nhân nhượng.
    """
    if not text or not text.strip():
        return "empty"
    if _BLOCKLIST.contradicts(text):
        return "forbidden"
    return None


def _drifted_from_source(original: str, rewritten: str) -> bool:
    """Bản viết lại còn trung thành với nguồn không?

    Đổi số liệu/metric nghĩa là câu không còn là thứ nguồn nói — vẫn cho qua
    theo ý người dùng, nhưng phải mang nhãn `user` chứ không được đội lốt câu
    có căn cứ.
    """
    return numeric_expressions(rewritten) != numeric_expressions(
        original
    ) or metric_cores(rewritten) != metric_cores(original)


def _strip_decoration(text: str) -> str:
    for token in DECORATION_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def instruction_conflicts(instruction: str) -> list[str]:
    """Chỉ thị có đòi thẳng một chuỗi cấm không? Regex, 0 lệnh gọi LLM.

    Bắt ở đây thì thông báo trỏ đúng vào *chỉ thị* thay vì vào câu bị từ chối,
    và không tốn lệnh gọi nào. Đây là lớp thêm, KHÔNG thay các lớp kiểm trên
    bản viết lại — chỉ thị vòng vo vẫn phải bị chặn ở đó.
    """
    return _BLOCKLIST.find(instruction)


def _apply_plan(
    sentences: list[dict[str, Any]],
    plan: RefinePlan,
    *,
    protected: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int, int, int]:
    edits = {edit.index: edit for edit in plan.edits}
    protected = protected or set()
    rejected: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    changed = dropped = kept = added = restored = 0

    # Chỉ mục ngoài dải: model muốn thêm câu. Từ v1.7 đó là việc hợp lệ — gom
    # vào cùng đường với `plan.added` thay vì từ chối.
    extra_texts = [
        edits[index].text
        for index in sorted(set(edits) - set(range(len(sentences))))
        if edits[index].action != "drop" and edits[index].text
    ]

    for index, sentence in enumerate(sentences):
        edit = edits.get(index)
        if index in protected:
            # Model không tuân lệnh giữ nguyên -> khôi phục, và ĐẾM để báo cho
            # người dùng. Không bao giờ mất im lặng.
            if edit is not None and edit.action != "keep":
                restored += 1
            updated.append(sentence)
            kept += 1
            continue
        if edit is None or edit.action == "keep":
            updated.append(sentence)
            kept += 1
            continue
        if edit.action == "drop":
            dropped += 1
            continue

        problem = _reject_reason(edit.text)
        if problem:
            rejected.append(
                {
                    "index": index,
                    "reason": REASON_VI[problem],
                    "text": sentence["text"],
                }
            )
            updated.append(sentence)  # giữ nguyên bản gốc
            kept += 1
            continue

        text = _strip_decoration(edit.text)
        if _drifted_from_source(sentence["text"], text):
            # Đã đổi số liệu -> không còn là thứ nguồn nói. Làm theo ý người
            # dùng, nhưng dán nhãn user thay vì để nó đội lốt câu có căn cứ.
            updated.append(user_sentence(text, req_ids=sentence.get("req_ids")))
        else:
            updated.append(
                {
                    **sentence,
                    "text": text,
                    "verdict": None,  # claim-check phán lại
                    "edited_by_chat": True,
                }
            )
        changed += 1

    for text in list(plan.added) + extra_texts:
        problem = _reject_reason(text)
        if problem:
            rejected.append({"index": None, "reason": REASON_VI[problem], "text": text})
            continue
        updated.append(user_sentence(_strip_decoration(text)))
        added += 1

    return updated, rejected, changed, dropped, kept, added, restored


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

    conflicts = instruction_conflicts(instruction)
    if conflicts:
        # Chặn trước khi gọi LLM: thông báo trỏ đúng vào chỉ thị, và không tốn
        # một lệnh gọi nào cho một yêu cầu chắc chắn bị từ chối.
        return RefineResult(
            state=state,
            rejected=[
                {
                    "index": None,
                    "reason": REASON_VI["instruction_forbidden"].format(
                        terms=", ".join(f"`{term}`" for term in conflicts)
                    ),
                    "text": None,
                }
            ],
            kept=len(target["sentences"]),
        )

    protected = _protected_indices(target["sentences"], instruction)
    # Nói cho model biết câu nào bị khoá. Đây chỉ là lời nhắc — thứ thật sự bảo
    # đảm là `protected` được ép trong `_apply_plan`, vì model có tuân hay không
    # là chuyện không kiểm soát được.
    numbered = "\n".join(
        f"[{index}]{'【固定】' if index in protected else ''} {sentence['text']}"
        for index, sentence in enumerate(target["sentences"])
    )
    plan = structured(
        REFINE_SYSTEM,
        f"【指示】{instruction}\n\n【対象の節】{target.get('title_ja', '')}\n"
        f"【各文】\n{numbered}",
        RefinePlan,
    )
    llm_calls = 1

    updated, rejected, changed, dropped, kept, added, restored = _apply_plan(
        target["sentences"], plan, protected=protected
    )
    if not (changed or dropped or added):
        return RefineResult(
            state=state,
            rejected=rejected,
            kept=kept,
            restored=restored,
            llm_calls=llm_calls,
        )

    # Claim-check lại các câu ĐÃ SỬA mà vẫn giữ nguồn. Từ v1.7 kết quả
    # CONTRADICTED không còn gỡ câu — nó chuyển câu sang nhãn `user`, vì câu đó
    # không còn là thứ nguồn nói nhưng vẫn là thứ người dùng yêu cầu.
    to_check = [item for item in updated if item.get("verdict") is None]
    if to_check:
        checked, _, check_calls, contradicted = check_claims(
            to_check,
            capability_store=CapabilityStore(),
            source_texts=section_sources(state, target),
        )
        llm_calls += check_calls
        by_text = {item["text"]: item for item in checked}
        contradicted_texts = {item["text"] for item in contradicted}
        rebuilt: list[dict[str, Any]] = []
        for item in updated:
            if item.get("verdict") is not None:
                rebuilt.append(item)
            elif item["text"] in by_text:
                rebuilt.append(by_text[item["text"]])
            elif item["text"] in contradicted_texts:
                rebuilt.append(
                    user_sentence(item["text"], req_ids=item.get("req_ids"))
                )
            else:
                rebuilt.append(item)
        updated = rebuilt

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
        kept=kept,
        added=added,
        restored=restored,
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
    changed = dropped = kept = added = restored = llm_calls = 0
    seen_reasons: set[str] = set()
    for section in list(state.get("sections", [])):
        result = refine_section(
            current, section_key=section["key"], instruction=instruction
        )
        current = result.state
        for item in result.rejected:
            # Chỉ thị đòi chuỗi cấm sẽ bị chặn ở MỌI mục với cùng một lý do —
            # in 5 lần y hệt nhau chỉ làm người đọc bỏ qua cả 5.
            if item.get("index") is None and item["reason"] in seen_reasons:
                continue
            seen_reasons.add(item["reason"])
            rejected.append(item)
        changed += result.changed
        dropped += result.dropped
        kept += result.kept
        added += result.added
        restored += result.restored
        llm_calls += result.llm_calls
    return RefineResult(
        state=current,
        rejected=rejected,
        changed=changed,
        dropped=dropped,
        kept=kept,
        added=added,
        restored=restored,
        llm_calls=llm_calls,
    )
