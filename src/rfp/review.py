"""Review loop: chỉ soi CHẤT LƯỢNG VĂN BẢN (Bước 6 v1.2 · multi-persona #10 v1.3).

## Ranh giới với guard — đọc trước khi sửa file này

Reviewer ở đây **không** được phép phán về compliance, chứng chỉ, số liệu hay
năng lực. Việc đó thuộc về ba lưới deterministic đã có:

  BB-1  capability sheet là trọng tài  -> claim_check.py (LLM, nhưng đối chiếu
        bằng chứng cố định, không phải "cảm nhận" của reviewer)
  BB-2  blocklist regex 2 lần          -> sanitize/blocklist.py + guard.py
  6.4-bis chống ảo giác lai            -> filter_hybrid_claims()

Judge LLM sai 5–10%. Với hồ sơ thầu, sai một lần là tuyên bố sai chứng chỉ. Nên
`ISSUE_TYPES` là **từ vựng đóng** thuần chất lượng, và `review_sections()` loại
bỏ mọi issue nằm ngoài tập đó thay vì tin lời model. Reviewer không có đường nào
bỏ qua `final_guard()`: guard chạy ở node `assemble`, sau khi review loop kết
thúc, trên bản hợp nhất cuối cùng.

## Vị trí trong graph — vì sao đặt SAU generate và TRƯỚC assemble

Ba lựa chọn từng cân nhắc, chọn phương án an toàn nhất:

1. Review *trong* `generate_per_section` — bị loại. Node đó đang được cache
   trọn gói (Bước 5); nhét vòng lặp có trạng thái vào trong sẽ khiến cùng một
   cache key ứng với nhiều kết quả khác nhau tuỳ số vòng đã chạy.
2. Review *sau* `assemble` — bị loại. Guard đã chạy ở `assemble`; sửa văn bản
   sau đó nghĩa là xuất bản thứ chưa qua guard. Vi phạm BB-2.
3. **Review giữa hai node đó** — chọn. Node `review` riêng biệt, không cache
   (kết quả phụ thuộc số vòng nên không phải hàm thuần của cache key), và guard
   vẫn là chốt chặn cuối cùng sau nó.

Vì `review` không cache, một lần chạy cache-hit sẽ dùng lại `sections` cũ nhưng
vẫn chạy lại review — tốn thêm lệnh gọi LLM. Đổi lại là tính đúng đắn: bản đã
review của lần trước không được ghi vào cache của `generate_per_section`, nên
cache key vẫn ứng với đúng một kết quả.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from config.settings import MAX_REVIEW_WORKERS
from pydantic import BaseModel

from .generate.precedent import (
    DECORATION_TOKENS,
    metric_cores,
    numeric_expressions,
)
from .llm import generate, structured
from .sanitize.blocklist import CapabilityBlocklist


# Từ vựng ĐÓNG, thuần chất lượng văn bản. Không có loại nào về chứng chỉ, số
# liệu hay năng lực — những thứ đó là việc của guard deterministic (xem trên).
ISSUE_TYPES = (
    "structure",          # thứ tự ý lộn xộn, thiếu câu dẫn
    "tone",               # giọng văn không phải văn phong hồ sơ thầu
    "redundancy",         # lặp ý giữa các câu trong cùng mục
    "unclear_reference",  # "これ" / "同様に" không rõ trỏ vào đâu
    "format",             # xuống dòng, dấu câu, độ dài câu
)
SEVERITIES = ("critical", "major", "minor")

_COMMON_RULES = (
    "認証・実績・数値・企業能力の正しさは別の仕組みが担当するため、"
    "それらについては一切指摘しないでください。"
    f"issue_type は次のいずれか: {', '.join(ISSUE_TYPES)}。"
    f"severity は次のいずれか: {', '.join(SEVERITIES)}。"
    "critical は提案書として提出できない水準の文章上の欠陥に限ります。"
    "問題がなければ issues を空にしてください。指定schemaで返してください。"
)

REVIEW_SYSTEM = (
    "あなたは提案書の日本語文章の校閲者です。**文章品質のみ**を見てください。"
    + _COMMON_RULES
)

# ── Persona (#10) ─────────────────────────────────────────────────────────
#
# Chỉ có model duy nhất (gpt-5.4-mini), nên phân hoá bằng PROMPT chứ không bằng
# model size — đúng ràng buộc của đề bài.
#
# ⚠ CỐ TÌNH KHÔNG CÓ PERSONA "COMPLIANCE".
# Đề bài gốc đề xuất ba persona: Compliance / Coverage / Quality. Hai persona
# sau an toàn, persona Compliance thì KHÔNG: nó sẽ phán về chứng chỉ giả và
# over-claim, tức giẫm đúng lên việc của BB-2 (blocklist regex 2 lần) và BB-1
# (capability sheet là trọng tài). Judge LLM sai 5–10%; ở đây sai một lần là hồ
# sơ tuyên bố sai chứng chỉ. Một reviewer "hiền" báo sạch không làm hồ sơ sạch
# hơn, nhưng nó tạo ra cảm giác đã có người canh — và đó là kiểu hỏng nguy hiểm
# nhất. Compliance ở lại tầng deterministic, không lên tầng LLM.
#
# Cả hai persona dưới đây vẫn bị giới hạn trong ISSUE_TYPES (từ vựng đóng thuần
# chất lượng), nên "Coverage" ở đây là *mạch lạc/đủ ý về mặt hành văn*, không
# phải requirement coverage — cái đó do coverage.py đo tất định.
PERSONAS = {
    "coverage": (
        "あなたは提案書の構成レビュアーです。各節が主題に対して抜けなく、"
        "順序立てて書かれているかだけを見てください。"
        "文の指す対象が曖昧、話が飛ぶ、導入文がないといった点を指摘します。"
        + _COMMON_RULES
    ),
    "quality": (
        "あなたは提案書の文体レビュアーです。日本語の提案書としての"
        "文体・敬体・簡潔さ・表記ゆれ・重複表現だけを見てください。"
        + _COMMON_RULES
    ),
}


class ReviewIssue(BaseModel):
    section_key: str
    severity: Literal["critical", "major", "minor"]
    issue_type: Literal[
        "structure", "tone", "redundancy", "unclear_reference", "format"
    ]
    original_text: str
    suggested_fix: str


class ReviewResult(BaseModel):
    issues: list[ReviewIssue]


def _section_text(section: dict[str, Any]) -> str:
    return "\n".join(
        sentence.get("text", "") for sentence in section.get("sentences", [])
    )


_SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2}


def deduplicate_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gộp theo (section_key, issue_type), giữ bản severity NẶNG NHẤT.

    Hai persona soi cùng một mục rất hay chỉ ra cùng một chỗ. Giữ bản nặng nhất
    chứ không phải bản gặp trước: hạ severity vì thứ tự chạy là biến một lỗi
    critical thành major một cách ngẫu nhiên.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in issues:
        key = (issue["section_key"], issue["issue_type"])
        current = best.get(key)
        if current is None or (
            _SEVERITY_RANK[issue["severity"]] < _SEVERITY_RANK[current["severity"]]
        ):
            best[key] = issue
    return list(best.values())


def _review_one(
    section: dict[str, Any],
    *,
    persona: str,
    system: str,
) -> list[dict[str, Any]]:
    user = (
        f"【章】{section.get('title_ja', '')}\n"
        f"【section_key】{section['key']}\n"
        f"【本文】\n{_section_text(section)}"
    )
    result = structured(system, user, ReviewResult)
    kept: list[dict[str, Any]] = []
    for issue in result.issues:
        # Model có thể trả section_key bịa hoặc issue_type ngoài từ vựng đóng.
        # Bỏ, không sửa hộ: đây là ranh giới compliance, không phải chỗ để
        # đoán ý model.
        if issue.section_key != section["key"]:
            continue
        if issue.issue_type not in ISSUE_TYPES:
            continue
        row = issue.model_dump()
        row["persona"] = persona
        kept.append(row)
    return kept


def review_sections(
    sections: list[dict[str, Any]],
    *,
    personas: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Chạy các persona SONG SONG trên từng mục rồi dedupe.

    Trả (issues, số lệnh gọi LLM). Mục rỗng không tốn lệnh gọi nào. Một persona
    lỗi không được làm hỏng cả vòng review: bỏ kết quả của nó, giữ phần còn lại
    — hồ sơ vẫn qua guard ở `assemble` nên bỏ sót một lượt soi văn phong không
    phải rủi ro compliance.
    """
    active = PERSONAS if personas is None else personas
    jobs = [
        (section, persona, system)
        for section in sections
        if _section_text(section).strip()
        for persona, system in active.items()
    ]
    if not jobs:
        return [], 0

    issues: list[dict[str, Any]] = []
    llm_calls = 0
    with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_REVIEW_WORKERS)) as pool:
        futures = {
            pool.submit(_review_one, section, persona=persona, system=system): persona
            for section, persona, system in jobs
        }
        for future in as_completed(futures):
            llm_calls += 1
            try:
                issues.extend(future.result())
            except Exception:
                continue

    # Sắp ổn định trước khi dedupe: ThreadPoolExecutor trả về theo thứ tự hoàn
    # thành, để nguyên thì cùng một đầu vào ra hai kết quả khác nhau giữa các
    # lần chạy.
    issues.sort(
        key=lambda item: (
            item["section_key"],
            item["issue_type"],
            _SEVERITY_RANK[item["severity"]],
            item.get("persona", ""),
        )
    )
    return deduplicate_issues(issues), llm_calls


def critical_section_keys(issues: list[dict[str, Any]]) -> list[str]:
    """Các mục có ít nhất một issue critical — đúng những mục được regen."""
    return list(
        dict.fromkeys(
            issue["section_key"]
            for issue in issues
            if issue["severity"] == "critical"
        )
    )


def score(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Điểm theo severity. Còn critical thì bản này không được coi là xong."""
    counts = {name: 0 for name in SEVERITIES}
    for issue in issues:
        counts[issue["severity"]] += 1
    return counts


def is_clean(issues: list[dict[str, Any]]) -> bool:
    return score(issues)["critical"] == 0


FIX_SYSTEM = (
    "あなたは日本語の提案書編集者です。指摘に従って与えられた一文だけを整えてください。"
    "事実を追加せず、数値・単位・認証名・顧客区分を変更しないでください。"
    "説明や箇条書き記号を付けず、完成した一文だけを返してください。"
)

_BLOCKLIST = CapabilityBlocklist()


def _safe_fix(text: str, issue: dict[str, Any]) -> tuple[str, int]:
    """Sửa một câu. Bản sửa đổi số liệu/thêm chuỗi cấm thì GIỮ NGUYÊN bản gốc.

    Cùng kiểu bảo vệ như `_safe_rewrite` của kênh precedent: LLM được phép làm
    mượt câu chữ, không được phép làm đổi nội dung đo đếm được. Kiểm blocklist ở
    đây là phòng thủ thêm tầng, KHÔNG thay `final_guard()` ở `assemble`.
    """
    rewritten = generate(
        FIX_SYSTEM,
        f"【指摘】{issue['issue_type']}: {issue['suggested_fix']}\n【原文】{text}",
    ).strip()
    if (
        not rewritten
        or numeric_expressions(rewritten) != numeric_expressions(text)
        or metric_cores(rewritten) != metric_cores(text)
        or any(
            token in rewritten and token not in text
            for token in DECORATION_TOKENS
        )
        or _BLOCKLIST.contradicts(rewritten)
    ):
        return text, 1
    return rewritten, 1


def apply_fixes(
    sections: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Chỉ sửa mục có issue critical; câu không bị chỉ ra giữ nguyên từng byte.

    Giữ nguyên `origin` / `source_id` / `req_ids` / `verdict` của câu — sửa câu
    chữ không được làm mất dấu vết nguồn (BB-4).
    """
    targets = set(critical_section_keys(issues))
    if not targets:
        return sections, 0, []

    issues_by_section: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        if issue["severity"] == "critical":
            issues_by_section.setdefault(issue["section_key"], []).append(issue)

    llm_calls = 0
    applied: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    for section in sections:
        if section["key"] not in targets:
            updated.append(section)
            continue
        section_issues = issues_by_section.get(section["key"], [])
        new_sentences = []
        for sentence in section.get("sentences", []):
            match = next(
                (
                    issue
                    for issue in section_issues
                    if issue["original_text"].strip()
                    and issue["original_text"].strip() in sentence.get("text", "")
                ),
                None,
            )
            if match is None:
                new_sentences.append(sentence)
                continue
            fixed, calls = _safe_fix(sentence["text"], match)
            llm_calls += calls
            if fixed != sentence["text"]:
                applied.append(
                    {
                        "section_key": section["key"],
                        "issue_type": match["issue_type"],
                        "before": sentence["text"],
                        "after": fixed,
                    }
                )
            new_sentences.append({**sentence, "text": fixed})
        updated.append({**section, "sentences": new_sentences})

    return updated, llm_calls, applied
