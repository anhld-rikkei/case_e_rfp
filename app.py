from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import (
    RFP_DIR,
    TRANSLATION_EFFORT,
    TRANSLATION_SYSTEM_PROMPT,
)
from rfp.graph import PIPELINE_STAGES, stream_graph
from rfp.llm import generate


STAGE_LABELS = {
    "parse_input": "Phân tích RFP",
    "check_complete": "Kiểm tra đầu vào",
    "ask_user": "Yêu cầu bổ sung",
    "route_reference_rfp": "Chọn RFP tham chiếu",
    "plan_sections": "Lập khung proposal",
    "retrieve_per_chapter": "Truy xuất bằng chứng",
    "generate_per_section": "Sinh 5 mục",
    "assemble": "Final guard & ghép bản",
}
STATUS_STYLES = {
    "completed": ("#22c55e", "white", "filled"),
    "running": ("#2563eb", "white", "filled"),
    "skipped": ("#d1d5db", "#4b5563", "filled,dashed"),
    "pending": ("white", "#6b7280", "rounded"),
}


st.set_page_config(
    page_title="RFP Proposal Studio",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def translate_once(content_hash: str, _content: str) -> str:
    """The content hash is the cache key; the full text is intentionally excluded."""
    return generate(
        TRANSLATION_SYSTEM_PROMPT,
        _content,
        effort=TRANSLATION_EFFORT,
    )


def progress_dot(trace: dict[str, Any]) -> str:
    stages = trace.get("stages", {})
    lines = [
        "digraph pipeline {",
        'rankdir="LR";',
        'graph [bgcolor="transparent", pad="0.2", nodesep="0.25", ranksep="0.35"];',
        'node [shape="box", fontname="Arial", fontsize="10", margin="0.12,0.08"];',
    ]
    for stage in PIPELINE_STAGES:
        status = stages.get(stage, {}).get("status", "pending")
        fill, font, style = STATUS_STYLES.get(status, STATUS_STYLES["pending"])
        label = STAGE_LABELS[stage].replace('"', '\\"')
        lines.append(
            f'"{stage}" [label="{label}", fillcolor="{fill}", '
            f'fontcolor="{font}", color="{fill}", style="{style}"];'
        )
    edges = [
        ("parse_input", "check_complete"),
        ("check_complete", "ask_user"),
        ("check_complete", "route_reference_rfp"),
        ("route_reference_rfp", "plan_sections"),
        ("plan_sections", "retrieve_per_chapter"),
        ("retrieve_per_chapter", "generate_per_section"),
        ("generate_per_section", "assemble"),
    ]
    lines.extend(f'"{left}" -> "{right}";' for left, right in edges)
    lines.append("}")
    return "\n".join(lines)


def source_label(sentence: dict[str, Any]) -> str:
    if sentence["origin"] == "capability":
        return f"[bảng năng lực · {sentence['source_id']}]"
    if sentence["origin"] == "precedent":
        return f"[{sentence['source_id']}]"
    return "[câu nối]"


def render_mapping(state: dict[str, Any]) -> None:
    st.subheader("Mục proposal ← chương RFP nguồn")
    chapter_titles = {
        chapter["id"]: chapter["title"] for chapter in state.get("chapters", [])
    }
    rows = []
    for section in state.get("sections", []):
        source_ids = section["source_chapters"]
        rows.append(
            {
                "Mục proposal": section["title_ja"],
                "Chương RFP nguồn": " · ".join(
                    f"{chapter_id} {chapter_titles.get(chapter_id, '')}".strip()
                    for chapter_id in source_ids
                )
                or "—",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_proposal(state: dict[str, Any]) -> None:
    st.subheader("Proposal tiếng Nhật")
    for index, section in enumerate(state.get("sections", []), start=1):
        st.markdown(f"### {index}. {section['title_ja']}")
        st.caption(
            f"{section['title_vi']} · Trạng thái: {section['status'] or '—'}"
        )
        if section.get("note"):
            st.warning(section["note"])
        for sentence in section["sentences"]:
            st.write(sentence["text"])
        with st.expander("Xem nguồn từng câu"):
            if not section["sentences"]:
                st.caption("Mục này chưa có câu nào.")
            for sentence in section["sentences"]:
                st.markdown(
                    f"**{source_label(sentence)}** `{sentence['verdict']}`  \n"
                    f"{sentence['text']}"
                )


def render_retrieval(state: dict[str, Any]) -> None:
    with st.expander("Vì sao chọn các đoạn này"):
        reference = state.get("reference_rfp", {})
        score = reference.get("score")
        score_text = "—" if score is None else f"{score:.4f}"
        st.write(
            f"RFP tham chiếu: **{reference.get('rfp_id') or 'không có'}** · "
            f"phương pháp: `{reference.get('method', 'none')}` · điểm: `{score_text}`"
        )
        for chapter in state.get("chapters", []):
            retrieval = chapter["retrieval"]
            attribute = retrieval["stages"]["attribute"]
            st.markdown(f"**{chapter['id']} · {chapter['title']}**")
            st.caption(
                "Attribute matches: "
                + (", ".join(attribute["covered_req_ids"]) or "không có")
            )
            rows = []
            for item in retrieval["selected"]:
                scores = item["scores"]
                rows.append(
                    {
                        "sent_id": item["sent_id"],
                        "văn bản": item["text"],
                        "hybrid": round(scores["hybrid"], 4),
                        "rerank": round(scores["rerank"], 4),
                        "industry": round(scores["industry_match"], 4),
                        "section": round(scores["same_section_prior"], 4),
                    }
                )
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info("Kênh precedent được bỏ qua; chương đã phủ bằng thuộc tính.")


def render_trace(state: dict[str, Any]) -> None:
    trace = state["trace"]
    with st.expander("Luồng xử lý"):
        st.metric("Tổng lệnh gọi LLM trong pipeline", trace["llm_calls"])
        call_rows = [
            {"Giai đoạn": name, "Số lệnh gọi": count}
            for name, count in trace.get("llm_calls_by_stage", {}).items()
        ]
        st.dataframe(call_rows, width="stretch", hide_index=True)
        st.caption("Đường đi: " + " → ".join(trace.get("path", [])))


def render_summary(state: dict[str, Any]) -> None:
    trace = state["trace"]
    grounding = trace.get("grounding", {"grounded": 0, "total": 0})
    st.success(
        f"{grounding['grounded']}/{grounding['total']} câu truy được về một câu nguồn đã verify"
    )
    st.subheader("Tổng kết")
    left, right = st.columns(2)
    with left:
        st.markdown("**Số chương theo trạng thái**")
        st.dataframe(
            [
                {"Trạng thái": name, "Số mục": count}
                for name, count in trace.get("section_statuses", {}).items()
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("**Tổng claim theo verdict**")
        st.dataframe(
            [
                {"Verdict": name, "Số claim": count}
                for name, count in trace.get("claim_verdicts", {}).items()
            ],
            width="stretch",
            hide_index=True,
        )


def render_guard_warnings(state: dict[str, Any]) -> None:
    trace = state["trace"]
    conflicts = trace.get("conflicts", [])
    hybrids = trace.get("hybrid_blocked", [])
    if conflicts:
        st.warning(f"5.7 đã loại {len(conflicts)} câu mâu thuẫn.")
        st.dataframe(conflicts, width="stretch", hide_index=True)
    else:
        st.info("5.7 không loại câu mâu thuẫn nào.")
    if hybrids:
        st.warning(f"6.4-bis đã chặn {len(hybrids)} câu lai.")
        st.dataframe(hybrids, width="stretch", hide_index=True)
    else:
        st.info("6.4-bis không chặn câu lai nào.")


def translated_content(state: dict[str, Any]) -> tuple[str, str]:
    content = f"[RFP]\n{state['input_text']}\n\n[PROPOSAL]\n{state['proposal']}"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content_hash, content


def render_result(state: dict[str, Any], translation: str) -> None:
    render_summary(state)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        render_proposal(state)
    with right:
        st.subheader("Bản dịch tiếng Việt")
        st.markdown(translation)
        render_mapping(state)
    render_retrieval(state)
    render_trace(state)
    render_guard_warnings(state)


def sample_picker() -> None:
    st.markdown("#### RFP mẫu — copy nhanh")
    sample_paths = sorted(Path(RFP_DIR).glob("*.txt"))[:3]
    columns = st.columns(3)
    for column, path in zip(columns, sample_paths):
        text = path.read_text(encoding="utf-8")
        with column:
            st.markdown(f"**{path.stem}**")
            st.code(text, language=None, height=180)
            if st.button("Dùng mẫu này", key=f"sample_{path.stem}", width="stretch"):
                st.session_state["rfp_input"] = text
                st.session_state["result_state"] = None
                st.session_state["translation"] = ""
                st.rerun()


def proposal_tab() -> None:
    st.title("RFP Proposal Studio")
    st.caption("Sinh hồ sơ thầu tiếng Nhật với nguồn truy vết cho từng câu")
    sample_picker()
    text = st.text_area(
        "Dán RFP",
        key="rfp_input",
        height=300,
        placeholder="Dán nội dung RFP tiếng Nhật tại đây…",
    )
    submitted = st.button("Nộp và sinh proposal", type="primary", width="stretch")

    progress = st.empty()
    if submitted:
        st.session_state["result_state"] = None
        st.session_state["translation"] = ""
        latest: dict[str, Any] | None = None
        with st.spinner("Đang chạy pipeline…"):
            for latest in stream_graph(text):
                progress.graphviz_chart(
                    progress_dot(latest["trace"]),
                    width="stretch",
                )
        st.session_state["result_state"] = latest
        if latest and latest.get("status") == "completed":
            content_hash, content = translated_content(latest)
            with st.spinner("Đang dịch RFP và proposal sang tiếng Việt…"):
                st.session_state["translation"] = translate_once(content_hash, content)

    state = st.session_state.get("result_state")
    if state is None:
        return
    progress.graphviz_chart(progress_dot(state["trace"]), width="stretch")
    if state.get("status") == "ask_user":
        st.warning(state["message"])
        return
    if state.get("status") == "completed":
        render_result(state, st.session_state.get("translation", ""))


def golden_tab() -> None:
    st.header("Sinh golden test")
    st.info("Khung trống — logic sinh golden test sẽ được bổ sung ở Bước 9-bis.")
    st.text_input("Tên test", disabled=True, placeholder="Bước 9-bis")
    st.button("Sinh golden test", disabled=True)


def main() -> None:
    st.session_state.setdefault("rfp_input", "")
    st.session_state.setdefault("result_state", None)
    st.session_state.setdefault("translation", "")
    proposal, golden = st.tabs(["Sinh proposal", "Sinh golden test"])
    with proposal:
        proposal_tab()
    with golden:
        golden_tab()


if __name__ == "__main__":
    main()
