"""Xuất hồ sơ ra markdown theo template 5 mục (#16, Bước 7 v1.3).

## Nhãn nháp phải đi theo file, không chỉ nằm trên UI

Banner "bản nháp" hiển thị trong app sẽ biến mất ngay khi người dùng bấm tải
về. Từ giây đó, file rời khỏi tầm kiểm soát của hệ thống và rất dễ được chuyển
tiếp như một hồ sơ hoàn chỉnh. Nên nhãn nháp + checklist người thật phải nằm
**trong chính nội dung file**, ở ngay đầu, không phải chỉ ở khung Streamlit.

## Guard và phần trích nguyên văn RFP

`final_guard()` chạy trên **câu do hệ sinh ra**, không chạy trên toàn file. Lý
do: bảng đối chiếu requirement trích **nguyên văn yêu cầu của bên mời thầu**, mà
một RFP hoàn toàn có thể *yêu cầu* ISO/IEC 27017 (golden case mutation có đúng
tình huống đó). Quét blocklist lên phần trích dẫn ấy sẽ báo động giả và chặn mất
một hồ sơ hợp lệ. Điều BB-2 cấm là **hệ thống tự nhận** có chứng chỉ đó — tức là
các câu trong `sections`, và đó đúng là thứ được quét ở đây.
"""
from __future__ import annotations

from typing import Any

from .guard import final_guard


DRAFT_BANNER_TITLE = "⚠ BẢN NHÁP DO HỆ THỐNG SINH — CHƯA ĐƯỢC NỘP"

DRAFT_BANNER = (
    f"> # {DRAFT_BANNER_TITLE}\n"
    ">\n"
    "> Tài liệu này do hệ thống sinh tự động từ RFP và hồ sơ thầu quá khứ.\n"
    "> **Chưa có người thật rà soát.** Không nộp, không gửi cho khách hàng và\n"
    "> không dùng làm bản cuối khi checklist bên dưới chưa được tick đủ.\n"
)

REVIEWER_CHECKLIST = (
    "## Checklist bắt buộc trước khi nộp\n"
    "\n"
    "Người phụ trách hồ sơ phải tự kiểm từng mục, không tin vào việc hệ thống\n"
    "đã có guard tự động:\n"
    "\n"
    "- [ ] Mọi chứng chỉ được nhắc tới đều đối chiếu đúng `capability_sheet.json`\n"
    "- [ ] Mọi số liệu (%, số người, số năm, SLA) truy được về câu nguồn ghi ở\n"
    "      bảng \"Nguồn từng câu\"\n"
    "- [ ] Không còn tên khách hàng cũ nào trong bản nộp\n"
    "- [ ] Các requirement ghi THIẾU ở bảng đối chiếu đã được bổ sung bằng tay\n"
    "      hoặc đã có quyết định chấp nhận không đáp ứng\n"
    "- [ ] Mục bị đánh dấu **Thiếu căn cứ** đã được viết lại bằng thông tin thật\n"
    "- [ ] Giọng văn và định dạng khớp mẫu hồ sơ của công ty\n"
    "- [ ] Người rà soát: ________________  Ngày: ____________\n"
)


def _origin_label(sentence: dict[str, Any]) -> str:
    origin = sentence.get("origin")
    if origin == "capability":
        return f"bảng năng lực · {sentence.get('source_id')}"
    if origin == "precedent":
        return f"hồ sơ cũ · {sentence.get('source_id')}"
    return "câu nối (không có nguồn)"


def _proposal_body(state: dict[str, Any]) -> str:
    """Phần văn bản do hệ sinh ra — đúng phần phải qua guard."""
    return "\n".join(
        sentence.get("text", "")
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
    )


def _sections_markdown(state: dict[str, Any]) -> list[str]:
    lines: list[str] = ["## Hồ sơ thầu (tiếng Nhật)", ""]
    for index, section in enumerate(state.get("sections", []), start=1):
        lines.append(f"### {index}. {section.get('title_ja', '')}")
        subtitle = section.get("title_vi", "")
        status = section.get("status") or "—"
        lines.append(f"*{subtitle} · trạng thái: `{status}`*")
        lines.append("")
        if section.get("note"):
            lines.append(f"> ⚠ {section['note']}")
            lines.append("")
        sentences = section.get("sentences", [])
        if not sentences:
            lines.append("*(mục này chưa có câu nào)*")
        for sentence in sentences:
            lines.append(sentence.get("text", ""))
        lines.append("")
    return lines


def _coverage_markdown(state: dict[str, Any]) -> list[str]:
    covered_by_req: dict[str, list[str]] = {}
    for section in state.get("sections", []):
        for sentence in section.get("sentences", []):
            if sentence.get("source_id") is None:
                continue
            for req_id in sentence.get("req_ids", []):
                label = _origin_label(sentence)
                if label not in covered_by_req.setdefault(req_id, []):
                    covered_by_req[req_id].append(label)

    lines = [
        "## Đối chiếu requirement",
        "",
        "| requirement | nội dung (nguyên văn RFP) | trạng thái | nguồn đáp ứng |",
        "|---|---|---|---|",
    ]
    total = 0
    covered = 0
    for chapter in state.get("chapters", []):
        for requirement in chapter.get("requirements", []):
            req_id = requirement["req_id"]
            sources = covered_by_req.get(req_id, [])
            total += 1
            covered += bool(sources)
            text = requirement["text"].replace("|", "\\|")
            status = "ĐÁP ỨNG" if sources else "**THIẾU**"
            lines.append(
                f"| {req_id} | {text} | {status} | {' · '.join(sources) or '—'} |"
            )
    lines.append("")
    lines.append(f"**Đáp ứng {covered}/{total} requirement.**")
    if covered < total:
        lines.append(
            f"Còn **{total - covered}** requirement chưa có bằng chứng — phải bổ "
            "sung bằng tay trước khi nộp."
        )
    lines.append("")
    return lines


def _provenance_markdown(state: dict[str, Any]) -> list[str]:
    lines = [
        "## Nguồn từng câu",
        "",
        "Mỗi câu trong hồ sơ đều phải truy được về một nguồn. Câu ghi *câu nối*",
        "là câu chuyển ý không mang thông tin sự thật.",
        "",
        "| # | mục | câu | nguồn | verdict |",
        "|---|---|---|---|---|",
    ]
    index = 0
    for section in state.get("sections", []):
        for sentence in section.get("sentences", []):
            index += 1
            text = sentence.get("text", "").replace("|", "\\|")
            lines.append(
                f"| {index} | {section.get('title_ja', '')} | {text} "
                f"| {_origin_label(sentence)} | {sentence.get('verdict') or '—'} |"
            )
    lines.append("")
    return lines


def _review_markdown(state: dict[str, Any]) -> list[str]:
    review = state.get("trace", {}).get("review") or {}
    if not review.get("enabled"):
        return []
    history = review.get("history", [])
    if not history:
        return []
    lines = ["## Vòng review tự động", ""]
    lines.append("| vòng | critical | major | minor | mục được sửa | câu đã sửa |")
    lines.append("|---|---|---|---|---|---|")
    for entry in history:
        score = entry.get("score", {})
        lines.append(
            f"| {entry.get('round')} | {score.get('critical', 0)} "
            f"| {score.get('major', 0)} | {score.get('minor', 0)} "
            f"| {', '.join(entry.get('critical_sections', [])) or '—'} "
            f"| {entry.get('fixes_applied', 0)} |"
        )
    lines.append("")
    lines.append(
        "> Vòng review chỉ soi **chất lượng văn bản**. Việc đúng/sai về chứng chỉ,"
    )
    lines.append(
        "> số liệu và năng lực do lớp kiểm tra tất định đảm nhiệm — và vẫn cần"
    )
    lines.append("> người thật xác nhận theo checklist ở đầu tài liệu.")
    lines.append("")
    return lines


def to_markdown(state: dict[str, Any]) -> str:
    """Dựng markdown hoàn chỉnh. Ném GuardViolation nếu phần sinh có chuỗi cấm."""
    final_guard(_proposal_body(state))

    rfp = state.get("rfp") or {}
    rfp_id = rfp.get("rfp_id", "") if isinstance(rfp, dict) else getattr(rfp, "rfp_id", "")

    lines: list[str] = [DRAFT_BANNER, ""]
    lines.append(f"# Hồ sơ thầu (bản nháp){f' — {rfp_id}' if rfp_id else ''}")
    lines.append("")
    lines.append(REVIEWER_CHECKLIST)
    lines.append("")
    lines.extend(_sections_markdown(state))
    lines.extend(_coverage_markdown(state))
    lines.extend(_review_markdown(state))
    lines.extend(_provenance_markdown(state))
    lines.append("---")
    lines.append("")
    lines.append(
        f"*{DRAFT_BANNER_TITLE} — nhắc lại ở cuối tài liệu để bản in nhiều trang "
        "không mất nhãn.*"
    )
    return "\n".join(lines) + "\n"


def export_filename(state: dict[str, Any]) -> str:
    rfp = state.get("rfp") or {}
    rfp_id = rfp.get("rfp_id", "") if isinstance(rfp, dict) else getattr(rfp, "rfp_id", "")
    return f"proposal_draft_{rfp_id or 'unknown'}.md"
