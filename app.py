from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import (
    PROMPT_VERSION,
    RFP_DIR,
    TRANSLATION_EFFORT,
    TRANSLATION_SYSTEM_PROMPT,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
)
from eval.golden.generator import (
    CHAPTER_TITLES,
    DEFAULT_OUTPUT_DIR as GOLDEN_OUTPUT_DIR,
    generate_combinatorial,
    generate_mutations,
    generate_paraphrases,
    load_base_case,
)
from eval.golden.runner import (
    DEFAULT_GOLDEN_DIR,
    assertion_rows,
    case_rows,
    discover_case_files,
    load_cases,
    run_case,
    run_cases,
    source_summary,
)
from eval.golden.schema import GoldenCase, load_case, save_case
from rfp.export import (
    DRAFT_BANNER_TITLE,
    REVIEWER_CHECKLIST,
    export_filename,
    to_markdown,
)
from config.display_vi import (
    ATTRIBUTE_COVERED_REASON,
    NO_EVIDENCE_REASON,
    ORIGIN_HINT,
    SKIP_LEGEND,
    SKIP_REASON_VI,
    ORIGIN_VI,
    RUN_LABEL_VI,
    SECTION_NOTE_VI,
    run_label,
    PERSONA_VI,
    RETRIEVAL_STAGE_VI,
    ROUTE_METHOD_VI,
    RUN_STATUS_VI,
    SECTION_STATUS_ICON,
    SECTION_STATUS_VI,
    SEVERITY_VI,
    ISSUE_TYPE_VI,
    STAGE_STATUS_ICON,
    STAGE_STATUS_VI,
    STAGE_VI,
    VERDICT_HINT,
    VERDICT_ICON,
    VERDICT_VI,
    label,
    to_raw,
)
from rfp.graph import PIPELINE_STAGES, stream_graph
from rfp.guard import GuardViolation
from rfp.llm import LLMUnavailable, MODEL as LLM_MODEL, generate


# Thứ tự hiển thị của luồng chạy. `ask_user` không nằm đây: nó là nhánh rẽ khi
# thiếu thông tin, hiện riêng bằng cảnh báo chứ không phải một bước của luồng
# thành công.
FLOW_STAGES = tuple(name for name in PIPELINE_STAGES if name != "ask_user")


st.set_page_config(
    page_title="RFP Proposal Studio",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def translate_once(content_hash: str, _content: str) -> str:
    """Cache the single post-pipeline translation call by exact content hash."""
    return generate(
        TRANSLATION_SYSTEM_PROMPT,
        _content,
        effort=TRANSLATION_EFFORT,
    )


def select_sample(text: str) -> None:
    st.session_state["rfp_input"] = text
    st.session_state["result_state"] = None
    st.session_state["translation"] = ""


def _load_uploaded_rfp(uploaded) -> str | None:
    try:
        return uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.sidebar.error(
            f"{uploaded.name} không phải UTF-8. RFP phải là .txt mã hoá UTF-8."
        )
        return None


def sidebar_controls() -> tuple[str, bool]:
    with st.sidebar:
        st.title("RFP đầu vào")
        uploaded = st.file_uploader(
            "Tải file RFP (.txt, UTF-8)",
            type=["txt"],
            key="rfp_upload",
        )
        if uploaded is not None:
            content = _load_uploaded_rfp(uploaded)
            # Chỉ nạp một lần cho mỗi file: nếu ghi đè mỗi lần chạy lại script
            # thì người dùng không sửa nổi nội dung trong ô text.
            if content is not None and st.session_state.get(
                "loaded_upload"
            ) != uploaded.name:
                st.session_state["loaded_upload"] = uploaded.name
                select_sample(content)
                st.rerun()
        text = st.text_area(
            "Dán RFP",
            key="rfp_input",
            height=430,
            placeholder="Dán nội dung RFP tiếng Nhật tại đây…",
        )
        st.markdown("#### RFP mẫu")
        for path in sorted(Path(RFP_DIR).glob("*.txt"))[:3]:
            sample_text = path.read_text(encoding="utf-8")
            st.button(
                f"Chọn {path.stem}",
                key=f"sample_{path.stem}",
                on_click=select_sample,
                args=(sample_text,),
                width="stretch",
            )
        submitted = st.button(
            "Nộp và sinh hồ sơ",
            type="primary",
            width="stretch",
        )
    return text, submitted


def source_label(sentence: dict[str, Any]) -> str:
    if sentence["origin"] == "capability":
        return f"bảng năng lực · {sentence['source_id']}"
    if sentence["origin"] == "precedent":
        return str(sentence["source_id"])
    return "câu nối"


def render_mapping(state: dict[str, Any]) -> None:
    st.subheader("Mỗi mục hồ sơ lấy từ chương nào của RFP")
    chapter_titles = {
        chapter["id"]: chapter["title"] for chapter in state.get("chapters", [])
    }
    rows = []
    for section in state.get("sections", []):
        source_ids = section["source_chapters"]
        rows.append(
            {
                "Mục hồ sơ": section["title_ja"],
                "Chương RFP nguồn": " · ".join(
                    f"{chapter_id} {chapter_titles.get(chapter_id, '')}".strip()
                    for chapter_id in source_ids
                )
                or "—",
                "Tình trạng": (
                    f"{SECTION_STATUS_ICON.get(section['status'], '')} "
                    f"{label(SECTION_STATUS_VI, section['status'])}"
                ).strip(),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_draft_banner(state: dict[str, Any]) -> None:
    """Banner nháp + checklist + nút tải. Nhãn nháp đi theo cả file xuất ra."""
    st.error(
        f"**{DRAFT_BANNER_TITLE}**  \n"
        "Chưa có người thật rà soát. Không nộp và không gửi khách hàng khi "
        "checklist bên dưới chưa tick đủ."
    )
    with st.expander("Checklist bắt buộc trước khi nộp", expanded=False):
        st.markdown(REVIEWER_CHECKLIST)
    try:
        markdown = to_markdown(state)
    except GuardViolation as violation:
        # Không bao giờ mở đường tải cho bản chưa qua guard.
        st.error(f"Không xuất được: final guard chặn — {violation}")
        return
    st.download_button(
        "⬇ Tải bản nháp (.md)",
        data=markdown.encode("utf-8"),
        file_name=export_filename(state),
        mime="text/markdown",
        width="stretch",
    )


def parse_evidence_note(note: str) -> list[dict[str, str]]:
    """Tách ghi chú `INSUFFICIENT_EVIDENCE: ...` thành từng yêu cầu.

    Chỉ là đường lui khi state không có sẵn dữ liệu cấu trúc (ví dụ state cũ
    dump từ phiên bản trước). Chuỗi có dạng:
        "INSUFFICIENT_EVIDENCE: 2.1 <câu> — <lý do>; 2.2 <câu> — <lý do>"
    Chuỗi lạ thì trả về nguyên văn trong `text`, không ném lỗi — đây là tầng
    hiển thị, vỡ ở đây không được phép làm chết cả trang.
    """
    if not note:
        return []
    body = note.split(":", 1)[1] if note.startswith("INSUFFICIENT_EVIDENCE") else note
    items: list[dict[str, str]] = []
    for chunk in body.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        head = chunk.split("—", 1)[0].strip()
        parts = head.split(None, 1)
        if len(parts) == 2 and any(char.isdigit() for char in parts[0]):
            items.append({"req_id": parts[0], "text": parts[1].strip()})
        else:
            items.append({"req_id": "", "text": head})
    return items


def evidence_gaps(state: dict[str, Any], section: dict[str, Any]) -> list[dict[str, str]]:
    """Các yêu cầu của một mục chưa có câu nào làm căn cứ.

    Dựng từ dữ liệu cấu trúc trong state (req_ids của từng câu) — chính xác
    hơn và không phụ thuộc câu chữ của ghi chú. Không dựng được thì mới parse
    chuỗi ghi chú.
    """
    covered = {
        req_id
        for sentence in section.get("sentences", [])
        if sentence.get("source_id") is not None
        for req_id in sentence.get("req_ids", [])
    }
    chapters = {chapter["id"]: chapter for chapter in state.get("chapters", [])}
    gaps = [
        {"req_id": requirement["req_id"], "text": requirement["text"]}
        for chapter_id in section.get("source_chapters", [])
        if chapter_id in chapters
        for requirement in chapters[chapter_id].get("requirements", [])
        if requirement["req_id"] not in covered
    ]
    if gaps:
        return gaps
    return parse_evidence_note(section.get("note") or "")


def render_section_note(state: dict[str, Any], section: dict[str, Any]) -> None:
    """Ghi chú của mục, đã Việt hoá — không lộ chuỗi enum thô ra giao diện."""
    note = section.get("note")
    if not note:
        return
    if section.get("status") == "INSUFFICIENT_EVIDENCE":
        gaps = evidence_gaps(state, section)
        lines = ["**⚠ Thiếu căn cứ** — các yêu cầu sau chưa có gì để dẫn:", ""]
        for gap in gaps:
            prefix = f"`{gap['req_id']}` " if gap["req_id"] else ""
            lines.append(f"- {prefix}{gap['text']}")
        lines.append("")
        lines.append(f"Lý do: {NO_EVIDENCE_REASON}.")
        st.warning("\n".join(lines))
        return
    st.warning(label(SECTION_NOTE_VI, note))


def render_proposal(state: dict[str, Any]) -> None:
    trace = state["trace"]
    grounding = trace.get("grounding", {"grounded": 0, "total": 0})
    render_draft_banner(state)
    st.success(
        f"{grounding['grounded']}/{grounding['total']} câu truy được về nguồn cụ thể "
        "(phần còn lại là câu nối, không mang thông tin sự thật)"
    )
    st.subheader("Hồ sơ tiếng Nhật")
    st.caption(
        "Nội dung hồ sơ giữ nguyên tiếng Nhật — đó là sản phẩm giao cho khách. "
        "Chỉ nhãn giao diện được dịch."
    )
    for index, section in enumerate(state.get("sections", []), start=1):
        status = section.get("status")
        st.markdown(f"### {index}. {section['title_ja']}")
        st.caption(
            f"{section['title_vi']} · {SECTION_STATUS_ICON.get(status, '')} "
            f"{label(SECTION_STATUS_VI, status)}"
        )
        render_section_note(state, section)
        for sentence in section["sentences"]:
            st.write(sentence["text"])
    st.divider()
    render_mapping(state)


def requirement_coverage(
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sources_by_req: dict[str, list[str]] = defaultdict(list)
    for section in state.get("sections", []):
        for sentence in section["sentences"]:
            if sentence.get("source_id") is None:
                continue
            label = source_label(sentence)
            for req_id in sentence.get("req_ids", []):
                if label not in sources_by_req[req_id]:
                    sources_by_req[req_id].append(label)

    section_by_chapter = {
        chapter_id: section
        for section in state.get("sections", [])
        for chapter_id in section["source_chapters"]
    }
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for chapter in state.get("chapters", []):
        for requirement in chapter["requirements"]:
            req_id = requirement["req_id"]
            sources = sources_by_req.get(req_id, [])
            row = {
                "Mã yêu cầu": req_id,
                "Nội dung yêu cầu": requirement["text"],
                "Tình trạng": "✅ Đã đáp ứng" if sources else "⚠️ Chưa có căn cứ",
                "Căn cứ đáp ứng": " · ".join(sources) or "—",
            }
            rows.append(row)
            if not sources:
                section = section_by_chapter.get(chapter["id"], {})
                missing.append(
                    {
                        "req_id": req_id,
                        "text": requirement["text"],
                        "section": section.get("title_ja", "—"),
                        # Lý do RIÊNG của yêu cầu này. Trước đây dán nguyên ghi
                        # chú gộp của cả mục vào từng dòng, nên 2.1/2.2/2.3 hiện
                        # ba đoạn giống hệt nhau.
                        "reason": NO_EVIDENCE_REASON,
                    }
                )
    return rows, missing


def render_coverage(state: dict[str, Any]) -> None:
    rows, missing = requirement_coverage(state)
    covered = len(rows) - len(missing)
    metric_col, status_col = st.columns([1, 3])
    with metric_col:
        st.metric("Đã đáp ứng", f"{covered}/{len(rows)}")
    with status_col:
        if missing:
            st.warning(
                f"Còn {len(missing)} yêu cầu chưa có căn cứ — cần người bổ sung "
                "bằng tay trước khi nộp."
            )
        else:
            st.success("Mọi yêu cầu của RFP đều đã có căn cứ.")
    st.subheader("Đối chiếu từng yêu cầu của RFP")
    st.dataframe(rows, width="stretch", hide_index=True)
    if missing:
        st.subheader("Yêu cầu chưa có căn cứ")
        st.caption(
            "Mỗi yêu cầu dưới đây cần người bổ sung bằng tay, hoặc ghi nhận là "
            "không đáp ứng."
        )
        st.dataframe(
            [
                {
                    "Mã yêu cầu": item["req_id"],
                    "Nội dung yêu cầu": item["text"],
                    "Thuộc mục": item["section"],
                    "Lý do thiếu căn cứ": item["reason"],
                }
                for item in missing
            ],
            width="stretch",
            hide_index=True,
        )


def sentence_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    """Giữ nguyên giá trị gốc ở `_origin`/`_verdict` để bộ lọc so sánh đúng."""
    return [
        {
            "Mục": section["title_ja"],
            "Câu (tiếng Nhật)": sentence["text"],
            "Nguồn": label(ORIGIN_VI, sentence["origin"]),
            "Mã nguồn": sentence["source_id"] or "—",
            "Kiểm chứng": (
                f"{VERDICT_ICON.get(sentence['verdict'], '')} "
                f"{label(VERDICT_VI, sentence['verdict'])}"
            ).strip(),
            "_origin": sentence["origin"],
            "_verdict": sentence["verdict"],
        }
        for section in state.get("sections", [])
        for sentence in section["sentences"]
    ]


def render_sources(state: dict[str, Any]) -> None:
    rows = sentence_rows(state)
    origins = list(dict.fromkeys(row["_origin"] for row in rows))
    verdicts = list(dict.fromkeys(row["_verdict"] for row in rows))
    filter_origin, filter_verdict = st.columns(2)
    with filter_origin:
        picked_origins = st.multiselect(
            "Lọc theo nguồn",
            [label(ORIGIN_VI, value) for value in origins],
            default=[label(ORIGIN_VI, value) for value in origins],
            key="source_origin_filter",
        )
    with filter_verdict:
        picked_verdicts = st.multiselect(
            "Lọc theo kết quả kiểm chứng",
            [label(VERDICT_VI, value) for value in verdicts],
            default=[label(VERDICT_VI, value) for value in verdicts],
            key="source_verdict_filter",
        )
    # Map NGƯỢC về giá trị gốc trước khi lọc — nhãn tiếng Việt chỉ là lớp áo.
    selected_origins = to_raw(ORIGIN_VI, picked_origins)
    selected_verdicts = to_raw(VERDICT_VI, picked_verdicts)
    filtered = [
        row
        for row in rows
        if row["_origin"] in selected_origins and row["_verdict"] in selected_verdicts
    ]
    st.dataframe(
        [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in filtered
        ],
        width="stretch",
        hide_index=True,
        height=600,
    )
    st.caption(f"Hiển thị {len(filtered)}/{len(rows)} câu.")
    with st.expander("Các nhãn này nghĩa là gì?"):
        st.markdown("**Nguồn của câu**")
        for key, text in ORIGIN_VI.items():
            st.markdown(f"- **{text}** — {ORIGIN_HINT[key]}")
        st.markdown("**Kết quả kiểm chứng**")
        for key, text in VERDICT_VI.items():
            st.markdown(f"- {VERDICT_ICON[key]} **{text}** — {VERDICT_HINT[key]}")


def translated_content(state: dict[str, Any]) -> tuple[str, str]:
    content = f"[RFP]\n{state['input_text']}\n\n[PROPOSAL]\n{state['proposal']}"
    # Key phải gồm model + phiên bản prompt dịch: bản cũ chỉ băm nội dung, nên
    # sửa TRANSLATION_SYSTEM_PROMPT hay đổi model vẫn trả về bản dịch cũ.
    key_material = "\n".join(
        (
            content,
            LLM_MODEL or "",
            PROMPT_VERSION,
            TRANSLATION_EFFORT,
            # Băm thẳng prompt: sửa câu chữ trong nó là đủ để cache miss, không
            # phải nhớ bump PROMPT_VERSION bằng tay.
            TRANSLATION_SYSTEM_PROMPT,
        )
    )
    content_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return content_hash, content


TRANSLATION_BLOCK_TITLE = {
    "RFP": "RFP (bản dịch)",
    "PROPOSAL": "Hồ sơ thầu (bản dịch)",
}

# Dòng tiêu đề: "Chương 3 Yêu cầu kỹ thuật" (RFP) hoặc "3. Thực tế triển khai"
# (mục hồ sơ). Dòng con "3.1 …" KHÔNG phải tiêu đề — nó là một yêu cầu.
_HEADING_RE = re.compile(r"^(?:Chương\s+\d+\b|\d+\.\s)")


def split_translation(text: str) -> list[dict[str, Any]]:
    """Tách bản dịch thành khối [RFP] / [PROPOSAL], mỗi khối giữ từng dòng.

    Bản dịch từ LLM **đã giữ đúng xuống dòng của bản gốc** (đo được: vào 48
    dấu, ra 48 dấu). Chữ dính liền một khối là do `st.markdown` gộp `\\n` đơn
    theo đúng ngữ nghĩa Markdown — nên đây là việc của tầng hiển thị, không
    phải sửa prompt dịch.

    Văn bản không có nhãn nào thì trả về một khối duy nhất `label=None`, để
    bản dịch lạ vẫn hiện ra được thay vì mất trắng.
    """
    if not text or not text.strip():
        return []
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        marker = line.strip()
        if marker in ("[RFP]", "[PROPOSAL]"):
            current = {"label": marker.strip("[]"), "lines": []}
            blocks.append(current)
            continue
        if current is None:
            current = {"label": None, "lines": []}
            blocks.append(current)
        current["lines"].append(line)
    for block in blocks:
        # Bỏ dòng trống ở hai đầu, giữ dòng trống giữa các đoạn.
        lines = block["lines"]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
    return [block for block in blocks if block["lines"]]


def is_translation_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line.strip()))


def render_translation(text: str) -> None:
    blocks = split_translation(text)
    if not blocks:
        st.info("Chưa có bản dịch.")
        return
    for index, block in enumerate(blocks):
        if index:
            st.divider()
        title = TRANSLATION_BLOCK_TITLE.get(block["label"])
        if title:
            st.subheader(title)
        for line in block["lines"]:
            if not line.strip():
                continue
            if is_translation_heading(line):
                st.markdown(f"**{line}**")
            else:
                # Mỗi dòng một lần gọi: st.markdown gộp \n đơn thành dấu cách,
                # nên nối cả khối rồi in một lần là ra đúng cục chữ dính liền.
                st.markdown(line)


def _status_state(status: str) -> str:
    if status == "running":
        return "running"
    if status in {"failed", "blocked"}:
        return "error"
    return "complete"


def _status_label(name: str, status: str, *, reason: str | None = None) -> str:
    suffix = label(STAGE_STATUS_VI, status)
    if reason:
        suffix = f"{suffix} ({reason})"
    return f"{STAGE_STATUS_ICON.get(status, '⚪')} {name} — {suffix}"


def skip_reason(state: dict[str, Any], stage_name: str) -> str | None:
    """Lý do một bước bị bỏ qua — ưu tiên suy từ state, thiếu thì tra bảng.

    `ask_user` bị bỏ qua khi `check_complete` không tìm thấy thiếu sót nào, và
    điều đó đọc thẳng được từ state. Các bước còn lại chỉ bị bỏ khi ngược lại —
    thiếu đầu vào — nên tra bảng theo loại node là đủ.
    """
    if stage_name == "ask_user" and not state.get("missing"):
        return SKIP_REASON_VI["ask_user"]
    return SKIP_REASON_VI.get(stage_name)


def _stage_statuses(
    state: dict[str, Any],
    *,
    failure: dict[str, str] | None,
) -> dict[str, str]:
    """Trạng thái từng bước để vẽ luồng, có tính tới lỗi/bị chặn.

    Bước đang chạy lúc pipeline dừng sẽ mang trạng thái lỗi; các bước sau nó
    quay về `pending` chứ không để `running` treo mãi.
    """
    stages = state.get("trace", {}).get("stages", {})
    statuses = {
        name: stages.get(name, {}).get("status", "pending") for name in FLOW_STAGES
    }
    if failure is None:
        return statuses

    failed_stage = failure["stage"]
    reached_failure = False
    for name in FLOW_STAGES:
        if name == failed_stage:
            statuses[name] = failure["kind"]
            reached_failure = True
        elif reached_failure:
            # Bước sau bước hỏng KHÔNG bao giờ được hiện "xong", kể cả khi state
            # cũ còn ghi completed. Một bước lỗi mà bước sau nó tích xanh là
            # hiển thị vô lý, và người đọc sẽ tin là hồ sơ vẫn chạy tới cuối.
            statuses[name] = "pending"
        elif statuses[name] == "running":
            statuses[name] = "completed"
    return statuses


def render_flow(
    state: dict[str, Any],
    target: Any,
    *,
    failure: dict[str, str] | None = None,
) -> None:
    """Luồng chạy 8 bước, cập nhật live theo state trả về từng node.

    Nguồn trạng thái là `trace.stages` trong chính state mà `stream_graph`
    yield ra sau mỗi node, nên đây là tiến độ thật chứ không phải hoạt ảnh
    phỏng đoán. Giới hạn: một node dài (sinh 5 mục) chỉ có hai mốc *bắt đầu* và
    *xong* — bên trong nó không phát tín hiệu, nên thanh tiến độ đứng yên trong
    lúc node đó chạy.
    """
    target.empty()
    statuses = _stage_statuses(state, failure=failure)
    done = sum(1 for value in statuses.values() if value == "completed")

    with target.container():
        st.progress(done / len(FLOW_STAGES), text=f"{done}/{len(FLOW_STAGES)} bước")
        for name in FLOW_STAGES:
            status = statuses[name]
            text = f"{STAGE_STATUS_ICON.get(status, '⚪')} **{STAGE_VI[name]}**"
            if status == "running":
                st.markdown(f"{text} — đang chạy…")
            elif status == "completed":
                st.markdown(f"{text}")
            elif status in {"failed", "blocked"}:
                st.markdown(f"{text} — {label(STAGE_STATUS_VI, status)}")
                if failure and failure.get("message"):
                    if status == "blocked":
                        # Bị chặn là hành vi ĐÚNG: thà không nộp còn hơn nộp hồ
                        # sơ tuyên bố sai chứng chỉ. Không tô như lỗi hệ thống.
                        st.warning(failure["message"], icon="🛑")
                    else:
                        st.error(failure["message"], icon="❌")
            elif status == "skipped":
                st.markdown(f"{text} — {label(STAGE_STATUS_VI, status)}")
            else:
                st.markdown(
                    f":gray[{STAGE_STATUS_ICON['pending']} {STAGE_VI[name]}]"
                )


def _running_stage(state: dict[str, Any] | None) -> str:
    """Bước đang chạy khi pipeline dừng — để tô đúng ô bị lỗi."""
    if not state:
        return FLOW_STAGES[0]
    stages = state.get("trace", {}).get("stages", {})
    for name in FLOW_STAGES:
        if stages.get(name, {}).get("status") == "running":
            return name
    # Không có bước nào "running" -> lỗi rơi vào bước ngay sau bước xong cuối.
    done = [
        name for name in FLOW_STAGES if stages.get(name, {}).get("status") == "completed"
    ]
    if not done:
        return FLOW_STAGES[0]
    last_index = FLOW_STAGES.index(done[-1])
    return FLOW_STAGES[min(last_index + 1, len(FLOW_STAGES) - 1)]


def render_flow_legend() -> None:
    st.caption(
        "  ·  ".join(
            f"{STAGE_STATUS_ICON[key]} {STAGE_STATUS_VI[key]}"
            for key in ("pending", "running", "completed", "skipped", "blocked", "failed")
        )
    )
    st.caption(
        f"{STAGE_STATUS_ICON['skipped']} {SKIP_LEGEND}.  ·  "
        f"{STAGE_STATUS_ICON['blocked']} guard chặn xuất bản là hành vi đúng: "
        "thà không nộp còn hơn nộp hồ sơ sai."
    )


def render_pipeline_status(state: dict[str, Any], target: Any) -> None:
    target.empty()
    stages = state.get("trace", {}).get("stages", {})
    overall_state = (
        "running"
        if any(item.get("status") == "running" for item in stages.values())
        else "complete"
    )
    with target.container():
        with st.status("Luồng chạy chi tiết", state=overall_state, expanded=True):
            for stage_name in PIPELINE_STAGES:
                status = stages.get(stage_name, {}).get("status", "pending")
                expanded = status == "running" or stage_name in {
                    "retrieve_per_chapter",
                    "generate_per_section",
                }
                with st.status(
                    _status_label(
                        STAGE_VI[stage_name],
                        status,
                        reason=(
                            skip_reason(state, stage_name)
                            if status == "skipped"
                            else None
                        ),
                    ),
                    state=_status_state(status),
                    expanded=expanded,
                ):
                    if stage_name == "retrieve_per_chapter":
                        for chapter in state.get("chapters", []):
                            retrieval_stages = chapter["retrieval"]["stages"]
                            precedent_skipped = retrieval_stages["query_embed"]["skipped"]
                            chapter_status = "skipped" if precedent_skipped else "completed"
                            with st.status(
                                _status_label(
                                    f"Chương {chapter['id']} · {chapter['title']}",
                                    chapter_status,
                                    reason=(
                                        ATTRIBUTE_COVERED_REASON
                                        if precedent_skipped
                                        else None
                                    ),
                                ),
                                state="complete",
                                expanded=False,
                            ):
                                for retrieval_name, values in retrieval_stages.items():
                                    item_status = (
                                        "skipped" if values.get("skipped") else "completed"
                                    )
                                    st.write(
                                        _status_label(
                                            label(RETRIEVAL_STAGE_VI, retrieval_name),
                                            item_status,
                                            reason=(
                                                ATTRIBUTE_COVERED_REASON
                                                if item_status == "skipped"
                                                else None
                                            ),
                                        )
                                    )
                    if stage_name == "generate_per_section":
                        for section in state.get("sections", []):
                            section_status = "completed" if section.get("status") else "pending"
                            with st.status(
                                _status_label(section["title_ja"], section_status),
                                state=_status_state(section_status),
                                expanded=False,
                            ):
                                st.write(
                                    label(
                                        SECTION_STATUS_VI,
                                        section.get("status"),
                                        unknown="Đang chờ sinh nội dung",
                                    )
                                )


def render_trace_metrics(state: dict[str, Any], target: Any) -> None:
    target.empty()
    trace = state.get("trace", {})
    with target.container():
        left, middle, right = st.columns(3)
        left.metric("Tổng lệnh gọi LLM", trace.get("llm_calls", 0))
        middle.metric(
            "Câu bị chặn vì số liệu lạ",
            len(trace.get("hybrid_blocked", [])),
            help="Câu có chỉ số định lượng không khớp nguyên văn nguồn nào — bị gỡ.",
        )
        right.metric(
            "Nguồn bị loại vì mâu thuẫn",
            len(trace.get("conflicts", [])),
            help="Hai câu nguồn nói khác nhau về cùng một chỉ số; giữ nguồn ưu tiên hơn.",
        )


def render_retrieval(state: dict[str, Any]) -> None:
    reference = state.get("reference_rfp", {})
    score = reference.get("score")
    score_text = "—" if score is None else f"{score:.4f}"
    st.write(
        f"Hồ sơ tham chiếu: **{reference.get('rfp_id') or 'không có'}** · "
        f"cách chọn: {label(ROUTE_METHOD_VI, reference.get('method', 'none'))} · "
        f"độ tương đồng: `{score_text}`"
    )
    rows = []
    for chapter in state.get("chapters", []):
        attribute = chapter["retrieval"]["stages"]["attribute"]
        for item in chapter["retrieval"]["selected"]:
            scores = item["scores"]
            rows.append(
                {
                    "Chương": chapter["id"],
                    "Mã câu nguồn": item["sent_id"],
                    "Câu nguồn (tiếng Nhật)": item["text"],
                    "Điểm tìm kiếm": round(scores["hybrid"], 4),
                    "Điểm sau chấm lại": round(scores["rerank"], 4),
                    "Đáp ứng yêu cầu": ", ".join(attribute["covered_req_ids"]) or "—",
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "Điểm tìm kiếm = độ khớp từ khoá + ngữ nghĩa. Điểm sau chấm lại có "
            "cộng thêm ưu tiên cùng ngành và cùng mục."
        )
    else:
        st.info(
            "Không lấy câu nào từ hồ sơ cũ — các chương tương ứng đã được đáp ứng "
            "bằng thông tin trong bảng năng lực công ty."
        )


LLM_STAGE_VI = {
    "parser": "Đọc RFP",
    "retrieval": "Truy xuất nguồn",
    "precedent_generation": "Viết lại câu từ hồ sơ cũ",
    "capability_generation": "Viết câu từ bảng năng lực",
    "claim_check": "Đối chiếu bằng chứng",
    "review": "Review chất lượng",
    "final_guard": "Kiểm tra cuối",
}


def render_review_rounds(state: dict[str, Any]) -> None:
    review = state.get("trace", {}).get("review") or {}
    if not review.get("enabled"):
        st.info("Vòng review đang tắt (`REVIEW_ENABLED = False`).")
        return
    history = review.get("history", [])
    if not history:
        st.info("Chưa chạy vòng review nào.")
        return
    st.dataframe(
        [
            {
                "Vòng": entry.get("round"),
                SEVERITY_VI["critical"]: entry.get("score", {}).get("critical", 0),
                SEVERITY_VI["major"]: entry.get("score", {}).get("major", 0),
                SEVERITY_VI["minor"]: entry.get("score", {}).get("minor", 0),
                "Mục phải sửa lại": ", ".join(entry.get("critical_sections", [])) or "—",
                "Số câu đã sửa": entry.get("fixes_applied", 0),
                "Người soi": ", ".join(
                    label(PERSONA_VI, name) for name in entry.get("personas", [])
                )
                or "—",
            }
            for entry in history
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Vòng review chỉ soi chất lượng văn bản. Việc đúng/sai về chứng chỉ, số "
        "liệu và năng lực do lớp kiểm tra tất định đảm nhiệm — "
        + " · ".join(f"{key}: {text}" for key, text in ISSUE_TYPE_VI.items())
    )


def render_trace_details(state: dict[str, Any]) -> None:
    trace = state["trace"]
    st.subheader("Số lệnh gọi LLM theo khâu")
    st.dataframe(
        [
            {"Khâu": label(LLM_STAGE_VI, name), "Số lệnh gọi": count}
            for name, count in trace.get("llm_calls_by_stage", {}).items()
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Đường đi: "
        + " → ".join(label(STAGE_VI, name) for name in trace.get("path", []))
    )
    st.subheader("Vòng review")
    render_review_rounds(state)
    st.subheader("Nguồn đã truy xuất")
    render_retrieval(state)
    conflicts = trace.get("conflicts", [])
    hybrids = trace.get("hybrid_blocked", [])
    if conflicts:
        st.warning(
            f"Đã loại {len(conflicts)} câu nguồn mâu thuẫn nhau về cùng một chỉ số."
        )
        st.dataframe(conflicts, width="stretch", hide_index=True)
    else:
        st.info("Không có câu nguồn nào mâu thuẫn nhau.")
    if hybrids:
        st.warning(
            f"Đã chặn {len(hybrids)} câu có số liệu không khớp nguyên văn nguồn nào."
        )
        st.dataframe(hybrids, width="stretch", hide_index=True)
    else:
        st.info("Không có câu nào bị chặn vì số liệu lạ.")


def _sync_golden_editor() -> None:
    cases = st.session_state.get("golden_preview_cases", [])
    index = st.session_state.get("golden_preview_index", 0)
    if cases and 0 <= index < len(cases):
        st.session_state["golden_rfp_editor"] = cases[index].rfp_text
        st.session_state["golden_trial_result"] = None


def _result_style(rows: list[dict[str, Any]], column: str) -> Any:
    frame = pd.DataFrame(rows)
    if frame.empty or column not in frame:
        return frame
    return frame.style.apply(
        lambda row: [
            "background-color: #5A1E25; color: #FFE9EC"
            if str(row[column]) == "FAIL"
            else ""
            for _ in row
        ],
        axis=1,
    )


def _golden_management_rows() -> tuple[list[Path], list[GoldenCase], list[dict[str, Any]]]:
    paths = discover_case_files(DEFAULT_GOLDEN_DIR)
    cases = [load_case(path) for path in paths]
    last_runs = st.session_state.get("golden_last_runs", {})
    rows = [
        {
            "case_id": case.case_id,
            "source": case.source,
            "#assertion": len(case.assertions),
            "needs_review": case.needs_review,
            "lần chạy cuối": last_runs.get(case.case_id, {}).get("time", "—"),
            "pass/fail": last_runs.get(case.case_id, {}).get("result", "—"),
        }
        for case in cases
    ]
    return paths, cases, rows


def render_golden() -> None:
    st.header("Sinh golden test")
    st.caption("Bộ test = RFP + điều kiện máy tự kiểm được; không lưu hồ sơ mẫu.")

    st.subheader("1. Cấu hình")
    mode_labels = {
        "Tổ hợp · 0 LLM": "combinatorial",
        "Mutation · 0 LLM": "mutation",
        "Paraphrase · LLM": "paraphrase",
    }
    config_left, config_middle, config_right = st.columns(3)
    with config_left:
        mode_label = st.radio("Mode", list(mode_labels), horizontal=False)
        mode = mode_labels[mode_label]
        industry = st.selectbox(
            "業種",
            ["製造業", "金融", "流通・小売", "公共", "医療"],
        )
    with config_middle:
        chapters = st.multiselect(
            "Chương",
            list(CHAPTER_TITLES),
            default=list(CHAPTER_TITLES),
        )
        out_scope_count = st.slider(
            "Số requirement ngoài năng lực",
            min_value=0,
            max_value=3,
            value=1,
            disabled=mode != "combinatorial",
        )
    with config_right:
        case_count = st.number_input(
            "Số ca cần sinh",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
        )
        base_paths = discover_case_files(DEFAULT_GOLDEN_DIR)
        base_path = st.selectbox(
            "Ca gốc",
            base_paths,
            format_func=lambda path: path.stem,
            disabled=mode == "combinatorial",
        ) if base_paths else None

    if st.button("Sinh preview", type="primary", key="golden_generate_preview"):
        if mode == "combinatorial":
            preview_cases = generate_combinatorial(
                int(case_count),
                industries=[industry],
                chapters=chapters,
                out_scope_count=out_scope_count,
            )
        elif mode == "mutation":
            preview_cases = generate_mutations(
                load_base_case(base_path),
                int(case_count),
            )
        else:
            preview_cases = generate_paraphrases(
                load_base_case(base_path),
                int(case_count),
            )
        st.session_state["golden_preview_cases"] = preview_cases
        st.session_state["golden_preview_index"] = 0
        st.session_state["golden_rfp_editor"] = preview_cases[0].rfp_text
        st.session_state["golden_trial_result"] = None

    preview_cases: list[GoldenCase] = st.session_state.get("golden_preview_cases", [])
    if preview_cases:
        selected_index = st.selectbox(
            "Ca đang preview",
            range(len(preview_cases)),
            key="golden_preview_index",
            format_func=lambda index: preview_cases[index].case_id,
            on_change=_sync_golden_editor,
        )
        selected_case = preview_cases[selected_index]
        if selected_case.needs_review:
            st.warning("🟠 Cần rà — ca paraphrase chưa được tính vào bảng §11.3.")

        st.subheader("2. Preview")
        preview_left, preview_right = st.columns(2)
        with preview_left:
            edited_rfp = st.text_area(
                "RFP sinh ra — có thể sửa tay",
                key="golden_rfp_editor",
                height=430,
            )
        with preview_right:
            st.dataframe(
                [
                    {
                        "kind": item.kind,
                        "target": item.target,
                        "reason": item.reason,
                    }
                    for item in selected_case.assertions
                ],
                width="stretch",
                hide_index=True,
                height=430,
            )

        working_case = replace(selected_case, rfp_text=edited_rfp)
        action_run, action_save, action_review = st.columns(3)
        with action_run:
            if st.button("Chạy thử", width="stretch", key="golden_run_one"):
                with st.spinner("Đang chạy graph và kiểm assertion…"):
                    trial = run_case(working_case)
                st.session_state["golden_trial_result"] = trial
                st.session_state.setdefault("golden_last_runs", {})[
                    working_case.case_id
                ] = {
                    "time": "phiên hiện tại",
                    "result": "PASS" if trial.passed else "FAIL",
                }
        with action_save:
            if st.button("Lưu", width="stretch", key="golden_save"):
                saved_path = save_case(
                    working_case,
                    GOLDEN_OUTPUT_DIR / f"{working_case.case_id}.json",
                )
                st.success(f"Đã lưu {saved_path.relative_to(ROOT_DIR)}")
        with action_review:
            if st.button(
                "Đã rà",
                width="stretch",
                key="golden_review",
                disabled=not working_case.needs_review,
            ):
                reviewed = replace(
                    working_case,
                    needs_review=False,
                    metadata={**working_case.metadata, "reviewed_by_human": True},
                )
                save_case(reviewed, GOLDEN_OUTPUT_DIR / f"{reviewed.case_id}.json")
                preview_cases[selected_index] = reviewed
                st.session_state["golden_preview_cases"] = preview_cases
                st.rerun()

        trial = st.session_state.get("golden_trial_result")
        if trial is not None and trial.case_id == working_case.case_id:
            st.subheader("3. Kết quả chạy thử")
            rows = assertion_rows(trial)
            st.dataframe(
                _result_style(rows, "PASS/FAIL"),
                width="stretch",
                hide_index=True,
            )
            if trial.passed:
                st.success("Tất cả assertion PASS.")
            else:
                st.error("Có assertion FAIL — xem reason trong bảng.")
    else:
        st.info("Chọn cấu hình và bấm “Sinh preview” để bắt đầu.")

    st.divider()
    st.subheader("4. Quản lý golden set")
    paths, cases, management_rows = _golden_management_rows()
    st.dataframe(management_rows, width="stretch", hide_index=True)
    generated_paths = [
        path
        for path in paths
        if path.resolve().is_relative_to(GOLDEN_OUTPUT_DIR.resolve())
    ]
    delete_col, delete_action = st.columns([3, 1])
    with delete_col:
        delete_path = st.selectbox(
            "Ca generated có thể xoá",
            generated_paths,
            format_func=lambda path: path.stem,
            disabled=not generated_paths,
        ) if generated_paths else None
    with delete_action:
        if st.button(
            "Xoá ca đã chọn",
            disabled=delete_path is None,
            key="golden_delete",
        ):
            resolved = delete_path.resolve()
            if not resolved.is_relative_to(GOLDEN_OUTPUT_DIR.resolve()):
                raise ValueError("Refusing to delete outside generated golden directory")
            resolved.unlink()
            st.rerun()

    st.divider()
    st.subheader("5. Regression")
    if st.button("Chạy toàn bộ golden set", key="golden_regression"):
        with st.spinner("Đang chạy regression toàn bộ golden set…"):
            regression = run_cases(cases)
        st.session_state["golden_regression_results"] = regression
    regression = st.session_state.get("golden_regression_results")
    if regression:
        regression_rows = case_rows(regression)
        st.dataframe(
            _result_style(regression_rows, "result"),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**Pass rate theo source — không tính ca needs_review=True**")
        st.dataframe(source_summary(regression), width="stretch", hide_index=True)
        failed = [
            result.case_id
            for result in regression
            if result.eligible_for_metrics and not result.passed
        ]
        if failed:
            st.error("Ca FAIL: " + ", ".join(failed))
        else:
            st.success("Không có ca đủ điều kiện nào FAIL.")


def render_empty_tabs(tabs: tuple[Any, ...]) -> None:
    # Tên biến tránh trùng hàm `label` đã import ở đầu file.
    hints = (
        "Dán hoặc tải RFP ở thanh bên rồi bấm **Nộp và sinh hồ sơ** — luồng chạy "
        "sẽ hiện ngay tại đây.",
        "Bảng đối chiếu từng yêu cầu của RFP sẽ hiện ở đây.",
        "Nguồn của từng câu sẽ hiện ở đây.",
        "Bản dịch tiếng Việt sẽ hiện ở đây sau bước dịch cuối.",
    )
    for tab, hint in zip(tabs[:4], hints):
        with tab:
            st.info(hint)


def _det_rows(runs: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Dòng deterministic của một cấu hình, ưu tiên bản `.det`.

    Giống hệt cách `eval/generate_report.py` chọn nguồn, để màn hình này và
    bảng §11.3 không ra hai con số khác nhau cho cùng một thứ.
    """
    for key in (f"{name}.det", name):
        rows = (runs.get(key) or {}).get("deterministic")
        if rows:
            return rows
    return []


def _sum_det(rows: list[dict[str, Any]], key: str) -> int | None:
    """Tổng một cột đếm; None nếu lượt đó chưa đo cột này (không phải 0)."""
    if not rows or any(key not in row for row in rows):
        return None
    return sum(int(row.get(key, 0)) for row in rows)


def _mean_det(rows: list[dict[str, Any]], key: str) -> float | None:
    # Bỏ dòng không sinh được mục nào (ca ask_user) — đúng như report làm.
    values = [
        float(row[key]) for row in rows if key in row and row.get("sections", 0) > 0
    ]
    return sum(values) / len(values) if values else None


def headline_numbers(runs: dict[str, Any]) -> list[dict[str, str]]:
    """Ba con số chính, đọc thẳng từ `eval/results` — không hardcode.

    Thiếu file nào thì bỏ dòng đó, không bịa số và cũng không hiện 0.
    """
    cards: list[dict[str, str]] = []

    off = _det_rows(runs, "force_precedent_k5_no_guard")
    on = _det_rows(runs, "force_precedent_k5_with_guard")
    fab_off, fab_on = _sum_det(off, "fabrication_count"), _sum_det(on, "fabrication_count")
    blocked = _sum_det(on, "guard_blocked_publish")
    if fab_off is not None and fab_on is not None:
        cards.append(
            {
                "title": "Lưới an toàn chặn được ảo giác",
                "value": f"{fab_off} → {fab_on}",
                "detail": (
                    f"Cùng ép lấy 5 câu nguồn và cùng tắt lọc lúc nạp, chỉ khác "
                    f"bật/tắt lưới an toàn: tắt thì **{fab_off} lần** chuỗi cấm lọt "
                    f"vào hồ sơ, bật thì **{fab_on}**"
                    + (
                        f" — và {blocked}/9 hồ sơ bị **chặn xuất bản** thay vì xuất ra bản sai."
                        if blocked
                        else "."
                    )
                ),
            }
        )

    dense = _mean_det(_det_rows(runs, "V0_A2_dense_only"), "coverage")
    hybrid = _mean_det(_det_rows(runs, "V1_A2_hybrid"), "coverage")
    if dense is not None and hybrid is not None:
        cards.append(
            {
                "title": "Thêm tìm theo từ khoá (BM25) đáp ứng được nhiều yêu cầu hơn",
                "value": f"{dense:.3f} → {hybrid:.3f}",
                "detail": (
                    "Tỉ lệ yêu cầu của RFP được đáp ứng. Đây mới là chỗ BM25 tạo "
                    "khác biệt — không phải ở các chỉ số RAGAS."
                ),
            }
        )

    cost = runs.get("review_cost", {}).get("runs", {})
    if cost.get("no_review") and cost.get("with_review"):
        def _avg(rows: list[dict[str, Any]]) -> float:
            return sum(row["product_tokens"] for row in rows) / len(rows)

        before, after = _avg(cost["no_review"]), _avg(cost["with_review"])
        cards.append(
            {
                "title": "Giá của vòng review chất lượng",
                "value": f"{before/1000:.1f}k → {after/1000:.1f}k token",
                "detail": (
                    f"Mỗi hồ sơ tốn thêm ~{(after - before)/1000:.1f}k token "
                    f"({after/before - 1:+.0%}) cho một lượt soi văn bản. "
                    "Tắt được bằng `REVIEW_ENABLED`."
                ),
            }
        )
    return cards


def render_headline_numbers(runs: dict[str, Any]) -> None:
    cards = headline_numbers(runs)
    if not cards:
        return
    st.subheader("Ba con số chính")
    for card in cards:
        st.metric(card["title"], card["value"])
        st.caption(card["detail"])


def render_eval() -> None:
    st.header("Kết quả đánh giá (RAGAS)")
    
    results_dir = ROOT_DIR / "eval" / "results"
    import json
    
    runs = []
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                runs.append((path.stem, data))
            except Exception:
                pass
                
    if not runs:
        st.info("Chưa có kết quả đánh giá nào. Chạy `python -m eval.run_ragas ...` rồi mở lại tab này.")
        return

    render_headline_numbers(dict(runs))
    st.divider()

    # Ước tính chi phí phải lấy từ một LƯỢT ĐÁNH GIÁ thật. `review_cost.json`
    # cũng nằm trong thư mục này nhưng là bản đo chi phí review, không có
    # `usage`/`deterministic` — đọc nhầm nó thì mọi ô hiện 0.
    eval_runs = [
        (name, data)
        for name, data in runs
        if isinstance(data.get("deterministic"), list)
        and data.get("deterministic")
        and isinstance(data.get("usage"), dict)
    ]
    if not eval_runs:
        st.info(
            "Có file kết quả nhưng chưa lượt nào đo được chi phí "
            "(thiếu `usage`/`deterministic`)."
        )
        return
    latest_name, latest_data = eval_runs[0]
    usage = latest_data.get("usage", {})
    total_seconds = usage.get("seconds", 0)
    # usage.seconds là tổng CẢ lần chạy. Lần chạy ablation có 9 RFP, nên lấy
    # thẳng số đó làm "thời gian/RFP" rồi nhân tiếp 9 là đếm 9 hai lần.
    # Số RFP của lần chạy = số dòng deterministic.
    rfp_count = max(1, len(latest_data.get("deterministic", [])))
    t_seconds = total_seconds / rfp_count
    seconds_by_stage = usage.get("seconds_by_stage", {})
    judge_seconds = seconds_by_stage.get("judge")
    product_seconds = (
        sum(v for k, v in seconds_by_stage.items() if k != "judge")
        if seconds_by_stage
        else None
    )

    st.subheader("Ước tính chi phí")
    st.caption(f"Dựa trên lượt gần nhất: **{run_label(latest_name)}** (`{latest_name}`)")
    
    # "TÁCH token thành 2 nhóm: sản phẩm (generate+structured) vs đo lường (judge)"
    tokens_by_stage = usage.get("tokens_by_stage", {})
    prod_toks = sum(v for k, v in tokens_by_stage.items() if k in ("generate", "structured"))
    judge_toks = tokens_by_stage.get("judge", 0)
    
    c1, c2, c3, c4 = st.columns(4)
    time_help = f"{total_seconds:.1f}s / {rfp_count} RFP trong `{latest_name}`"
    if product_seconds is not None:
        time_help += f" · sản phẩm {product_seconds:.1f}s · judge {judge_seconds or 0:.1f}s"
    else:
        time_help += " · lần chạy cũ chưa tách sản phẩm/judge"
    c1.metric("Thời gian/RFP", f"{t_seconds:.1f}s", help=time_help)
    c2.metric("Token Sản phẩm", f"{prod_toks:,}")
    c3.metric("Token Đo lường (Judge)", f"{judge_toks:,}")
    
    est_rfps = 9
    est_configs = 8
    total_est_seconds = t_seconds * est_rfps * est_configs
    est_h, est_m = divmod(total_est_seconds / 60, 60)
    c4.metric("Dự báo (9 RFP × 8 cấu hình)", f"{int(est_h)}h {int(est_m)}m")
    
    if COST_PER_1M_INPUT is not None and COST_PER_1M_OUTPUT is not None:
        p_cost = (prod_toks / 1e6) * COST_PER_1M_OUTPUT # Simplification: using output cost or mixed
        st.caption(f"Đơn giá đã cấu hình. (Cần chia input/output chi tiết hơn để ra số tiền chính xác)")
    else:
        st.caption("Chưa cấu hình đơn giá (COST_PER_1M_INPUT/OUTPUT = None). Không tính tiền.")

    st.divider()
    st.subheader("Lịch sử các lượt đánh giá")
    
    metric_labels = {
        "faithfulness": "Trung thành với nguồn↑",
        "answer_relevancy": "Đúng trọng tâm↑",
        "context_precision": "Nguồn lấy về có ích↑",
        "context_recall": "Lấy đủ nguồn cần↑",
        "noise_sensitivity": "Nhiễu↓",
        "context_entity_recall": "Bắt đủ thực thể↑",
    }

    rows = []
    for name, data in eval_runs:
        ragas = data.get("ragas", {})
        det = data.get("deterministic", [{}])[0] if data.get("deterministic") else {}
        usage_data = data.get("usage", {})
        tokens = usage_data.get("tokens_by_stage", {})
        product = sum(
            value for key, value in tokens.items() if key in ("generate", "structured")
        )

        row = {
            "Cấu hình": run_label(name),
            "Thời gian": f"{usage_data.get('seconds', 0):.1f}s",
            "Token sản phẩm / đo lường": f"{product:,} / {tokens.get('judge', 0):,}",
            "Mẫu chấm được / tổng": (
                f"{data.get('ragas_sample_count', 0)}/{data.get('sample_count', 0)}"
            ),
            "Đáp ứng / bỏ trống": (
                f"{det.get('coverage', 0):.2f} / {det.get('abstain_rate', 0):.2f}"
            ),
            "Câu có nguồn": f"{det.get('groundedness', 0):.3f}",
            "Trích dẫn đúng": f"{det.get('citation_accuracy', 0):.3f}",
            "Chuỗi cấm lọt": det.get("fabrication_count", 0),
            "Rò tên khách": det.get("client_leak_count", 0),
        }
        for metric, metric_label in metric_labels.items():
            values = ragas.get(metric, {})
            row[metric_label] = (
                f"{values.get('mean', 0):.3f} ± {values.get('std', 0):.3f}"
                if values
                else "—"
            )
        row["Mã lượt chạy"] = name
        rows.append(row)

    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "`—` = lượt đó chưa đo cột này, **không phải** đo ra 0. "
        "Cột *Mã lượt chạy* là tên file trong `eval/results/`."
    )

    other = [name for name, _ in runs if name not in {n for n, _ in eval_runs}]
    if other:
        st.caption(
            "Không phải lượt đánh giá, để riêng: "
            + " · ".join(f"`{name}` ({run_label(name)})" for name in other)
        )


def main() -> None:
    st.session_state.setdefault("rfp_input", "")
    st.session_state.setdefault("result_state", None)
    st.session_state.setdefault("translation", "")
    st.session_state.setdefault("failure", None)

    text, submitted = sidebar_controls()
    st.title("RFP Proposal Studio")
    st.caption("Sinh hồ sơ thầu tiếng Nhật, mỗi câu đều truy được về nguồn")
    tabs = st.tabs(
        [
            "Tổng quan",
            "Độ đáp ứng",
            "Nguồn từng câu",
            "Bản dịch",
            "Truy vết",
            "Sinh bộ test",
            "Kết quả đánh giá",
        ]
    )
    proposal_tab, coverage_tab, sources_tab, translation_tab, trace_tab, golden_tab, eval_tab = tabs

    with trace_tab:
        trace_metrics = st.empty()
        trace_status = st.empty()

    with proposal_tab:
        flow_area = st.empty()
        result_area = st.container()

    latest = st.session_state.get("result_state")
    failure = st.session_state.get("failure")

    if submitted:
        st.session_state["result_state"] = None
        st.session_state["translation"] = ""
        st.session_state["failure"] = None
        latest, failure = None, None
        try:
            for latest in stream_graph(text):
                render_flow(latest, flow_area)
                render_trace_metrics(latest, trace_metrics)
                render_pipeline_status(latest, trace_status)
        except GuardViolation as violation:
            # Bị chặn là hành vi ĐÚNG, không phải sự cố: thà không xuất hồ sơ
            # còn hơn xuất một hồ sơ tuyên bố sai chứng chỉ.
            failure = {
                "stage": "assemble",
                "kind": "blocked",
                "message": (
                    "Hồ sơ **bị chặn xuất bản do vi phạm quy tắc an toàn** — "
                    "nội dung sinh ra có nhắc tới năng lực hoặc chứng chỉ mà "
                    f"công ty không có ({violation}). Đây là hành vi đúng của hệ "
                    "thống: không xuất bản còn hơn xuất bản sai."
                ),
            }
        except LLMUnavailable as error:
            failure = {
                "stage": _running_stage(latest),
                "kind": "failed",
                "message": (
                    f"Nhà cung cấp LLM không phản hồi sau {error.attempts} lần gọi "
                    f"(lỗi {error.kind}). Phần đã sinh xong vẫn giữ nguyên; "
                    "chạy lại để tiếp tục."
                ),
            }
        except Exception as error:  # noqa: BLE001 — hiện lỗi thật, không nuốt
            failure = {
                "stage": _running_stage(latest),
                "kind": "failed",
                "message": f"Lỗi kỹ thuật: {type(error).__name__}: {error}",
            }
        st.session_state["result_state"] = latest
        st.session_state["failure"] = failure
        if failure is None and latest and latest.get("status") == "completed":
            content_hash, content = translated_content(latest)
            with st.spinner("Đang dịch RFP và hồ sơ sang tiếng Việt…"):
                st.session_state["translation"] = translate_once(content_hash, content)

    if latest is None:
        render_empty_tabs(tabs)
    else:
        render_flow(latest, flow_area, failure=failure)
        render_trace_metrics(latest, trace_metrics)
        render_pipeline_status(latest, trace_status)
        with proposal_tab:
            render_flow_legend()
        if failure is not None:
            with result_area:
                if failure["kind"] == "blocked":
                    st.warning(failure["message"], icon="🛑")
                else:
                    st.error(failure["message"], icon="❌")
            for tab in (coverage_tab, sources_tab, translation_tab):
                with tab:
                    st.info(
                        "Chưa có kết quả để hiển thị — lượt chạy vừa rồi "
                        f"{label(RUN_STATUS_VI, latest.get('status'), unknown='không hoàn tất')}."
                    )
        elif latest.get("status") == "ask_user":
            with result_area:
                st.warning(latest["message"])
            for tab in (coverage_tab, sources_tab, translation_tab):
                with tab:
                    st.info("Cần bổ sung đầu vào trước khi sinh kết quả.")
        elif latest.get("status") == "completed":
            with result_area:
                render_proposal(latest)
            with coverage_tab:
                render_coverage(latest)
            with sources_tab:
                render_sources(latest)
            with translation_tab:
                st.caption(
                    "Bản dịch tiếng Việt để đối chiếu. Bản nộp cho khách vẫn là "
                    "bản tiếng Nhật ở tab Tổng quan."
                )
                render_translation(st.session_state.get("translation", ""))
            with trace_tab:
                render_trace_details(latest)

    with golden_tab:
        render_golden()
        
    with eval_tab:
        render_eval()


if __name__ == "__main__":
    main()
