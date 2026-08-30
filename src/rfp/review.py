"""Review loop tối giản: chỉ soi CHẤT LƯỢNG VĂN BẢN (Bước 6, v1.2).

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

from typing import Any, Literal

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

REVIEW_SYSTEM = (
    "あなたは提案書の日本語文章の校閲者です。**文章品質のみ**を見てください。"
    "認証・実績・数値・企業能力の正しさは別の仕組みが担当するため、"
    "それらについては一切指摘しないでください。"
    f"issue_type は次のいずれか: {', '.join(ISSUE_TYPES)}。"
    f"severity は次のいずれか: {', '.join(SEVERITIES)}。"
    "critical は提案書として提出できない水準の文章上の欠陥に限ります。"
    "問題がなければ issues を空にしてください。指定schemaで返してください。"
)


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


def review_sections(
    sections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Trả (issues, số lệnh gọi LLM). Mục rỗng không tốn lệnh gọi nào."""
    issues: list[dict[str, Any]] = []
    llm_calls = 0

    for section in sections:
        body = _section_text(section)
        if not body.strip():
            continue
        user = (
            f"【章】{section.get('title_ja', '')}\n"
            f"【section_key】{section['key']}\n"
            f"【本文】\n{body}"
        )
        result = structured(REVIEW_SYSTEM, user, ReviewResult)
        llm_calls += 1
        for issue in result.issues:
            # Model có thể trả section_key bịa hoặc issue_type ngoài từ vựng
            # đóng. Bỏ, không sửa hộ: đây là ranh giới compliance, không phải
            # chỗ để đoán ý model.
            if issue.section_key != section["key"]:
                continue
            if issue.issue_type not in ISSUE_TYPES:
                continue
            issues.append(issue.model_dump())

    return issues, llm_calls


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
