from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import sys
from typing import Any

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import RFP_DIR, TRANSLATION_EFFORT, TRANSLATION_SYSTEM_PROMPT
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
STAGE_ICONS = {
    "completed": "✅",
    "running": "🔵",
    "skipped": "⏭️",
    "pending": "⚪",
}


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


def sidebar_controls() -> tuple[str, bool]:
    with st.sidebar:
        st.title("RFP đầu vào")
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
            "Nộp và sinh proposal",
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
                "Trạng thái": section["status"] or "—",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_proposal(state: dict[str, Any]) -> None:
    trace = state["trace"]
    grounding = trace.get("grounding", {"grounded": 0, "total": 0})
    st.success(
        f"{grounding['grounded']}/{grounding['total']} câu truy được về một câu nguồn đã verify"
    )
    st.subheader("Hồ sơ tiếng Nhật")
    for index, section in enumerate(state.get("sections", []), start=1):
        st.markdown(f"### {index}. {section['title_ja']}")
        st.caption(f"{section['title_vi']} · Trạng thái: {section['status'] or '—'}")
        if section.get("note"):
            st.warning(section["note"])
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
                "req_id": req_id,
                "nội dung": requirement["text"],
                "phủ/thiếu": "PHỦ" if sources else "THIẾU",
                "nguồn nào phủ": " · ".join(sources) or "—",
            }
            rows.append(row)
            if not sources:
                section = section_by_chapter.get(chapter["id"], {})
                missing.append(
                    {
                        "req_id": req_id,
                        "text": requirement["text"],
                        "section": section.get("title_ja", "—"),
                        "note": section.get("note") or "Không có note bổ sung.",
                    }
                )
    return rows, missing


def render_coverage(state: dict[str, Any]) -> None:
    rows, missing = requirement_coverage(state)
    covered = len(rows) - len(missing)
    metric_col, status_col = st.columns([1, 3])
    with metric_col:
        st.metric("Coverage", f"{covered}/{len(rows)}")
    with status_col:
        if missing:
            st.warning(f"Còn {len(missing)} requirement chưa có nguồn tương ứng.")
        else:
            st.success("Tất cả requirement đã có nguồn tương ứng.")
    st.subheader("Đối chiếu mọi requirement")
    st.dataframe(rows, width="stretch", hide_index=True)
    if missing:
        warning_lines = [
            f"- **{item['req_id']} · {item['text']}** — mục {item['section']}: {item['note']}"
            for item in missing
        ]
        st.warning("Requirement chưa phủ:\n\n" + "\n\n".join(warning_lines))


def sentence_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "mục": section["title_ja"],
            "câu": sentence["text"],
            "origin": sentence["origin"],
            "source_id": sentence["source_id"] or "—",
            "verdict": sentence["verdict"],
        }
        for section in state.get("sections", [])
        for sentence in section["sentences"]
    ]


def render_sources(state: dict[str, Any]) -> None:
    rows = sentence_rows(state)
    origins = list(dict.fromkeys(row["origin"] for row in rows))
    verdicts = list(dict.fromkeys(row["verdict"] for row in rows))
    filter_origin, filter_verdict = st.columns(2)
    with filter_origin:
        selected_origins = st.multiselect(
            "Lọc theo origin", origins, default=origins, key="source_origin_filter"
        )
    with filter_verdict:
        selected_verdicts = st.multiselect(
            "Lọc theo verdict", verdicts, default=verdicts, key="source_verdict_filter"
        )
    filtered = [
        row
        for row in rows
        if row["origin"] in selected_origins and row["verdict"] in selected_verdicts
    ]
    st.dataframe(filtered, width="stretch", hide_index=True, height=640)
    st.caption(f"Hiển thị {len(filtered)}/{len(rows)} câu.")


def translated_content(state: dict[str, Any]) -> tuple[str, str]:
    content = f"[RFP]\n{state['input_text']}\n\n[PROPOSAL]\n{state['proposal']}"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content_hash, content


def _status_state(status: str) -> str:
    return "running" if status == "running" else "complete"


def _status_label(name: str, status: str) -> str:
    suffix = {
        "completed": "xong",
        "running": "đang chạy",
        "skipped": "bỏ qua",
        "pending": "chờ",
    }.get(status, status)
    return f"{STAGE_ICONS.get(status, '⚪')} {name} — {suffix}"


def render_pipeline_status(state: dict[str, Any], target: Any) -> None:
    target.empty()
    stages = state.get("trace", {}).get("stages", {})
    overall_state = (
        "running"
        if any(item.get("status") == "running" for item in stages.values())
        else "complete"
    )
    with target.container():
        with st.status("Pipeline", state=overall_state, expanded=True):
            for stage_name in PIPELINE_STAGES:
                status = stages.get(stage_name, {}).get("status", "pending")
                expanded = status == "running" or stage_name in {
                    "retrieve_per_chapter",
                    "generate_per_section",
                }
                with st.status(
                    _status_label(STAGE_LABELS[stage_name], status),
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
                                ),
                                state="complete",
                                expanded=False,
                            ):
                                for retrieval_name, values in retrieval_stages.items():
                                    item_status = (
                                        "skipped" if values.get("skipped") else "completed"
                                    )
                                    st.write(_status_label(retrieval_name, item_status))
                    if stage_name == "generate_per_section":
                        for section in state.get("sections", []):
                            section_status = "completed" if section.get("status") else "pending"
                            with st.status(
                                _status_label(section["title_ja"], section_status),
                                state=_status_state(section_status),
                                expanded=False,
                            ):
                                st.write(section.get("status") or "Đang chờ sinh nội dung")


def render_trace_metrics(state: dict[str, Any], target: Any) -> None:
    target.empty()
    trace = state.get("trace", {})
    with target.container():
        left, middle, right = st.columns(3)
        left.metric("Tổng lệnh gọi LLM", trace.get("llm_calls", 0))
        middle.metric("hybrid_blocked", len(trace.get("hybrid_blocked", [])))
        right.metric("conflict_dropped", len(trace.get("conflicts", [])))


def render_retrieval(state: dict[str, Any]) -> None:
    reference = state.get("reference_rfp", {})
    score = reference.get("score")
    score_text = "—" if score is None else f"{score:.4f}"
    st.write(
        f"RFP tham chiếu: **{reference.get('rfp_id') or 'không có'}** · "
        f"method: `{reference.get('method', 'none')}` · score: `{score_text}`"
    )
    rows = []
    for chapter in state.get("chapters", []):
        attribute = chapter["retrieval"]["stages"]["attribute"]
        for item in chapter["retrieval"]["selected"]:
            scores = item["scores"]
            rows.append(
                {
                    "chương": chapter["id"],
                    "sent_id": item["sent_id"],
                    "văn bản": item["text"],
                    "hybrid": round(scores["hybrid"], 4),
                    "rerank": round(scores["rerank"], 4),
                    "attribute_req_ids": ", ".join(attribute["covered_req_ids"]),
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Không có precedent được chọn; các chương tương ứng dùng kênh thuộc tính.")


def render_trace_details(state: dict[str, Any]) -> None:
    trace = state["trace"]
    st.subheader("Số call theo giai đoạn")
    st.dataframe(
        [
            {"Giai đoạn": name, "Số lệnh gọi": count}
            for name, count in trace.get("llm_calls_by_stage", {}).items()
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption("Đường đi: " + " → ".join(trace.get("path", [])))
    st.subheader("Retrieval")
    render_retrieval(state)
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


def render_golden() -> None:
    st.header("Sinh golden test")
    st.info("Khung trống — logic sinh golden test sẽ được bổ sung ở Bước 9-bis.")
    st.text_input("Tên test", disabled=True, placeholder="Bước 9-bis")
    st.button("Sinh golden test", disabled=True)


def render_empty_tabs(tabs: tuple[Any, ...]) -> None:
    labels = (
        "Hồ sơ sẽ xuất hiện ở đây sau khi pipeline hoàn tất.",
        "Coverage requirement sẽ xuất hiện ở đây.",
        "Nguồn của từng câu sẽ xuất hiện ở đây.",
        "Bản dịch sẽ xuất hiện ở đây sau bước dịch cuối.",
    )
    for tab, label in zip(tabs[:4], labels):
        with tab:
            st.info(label)


def main() -> None:
    st.session_state.setdefault("rfp_input", "")
    st.session_state.setdefault("result_state", None)
    st.session_state.setdefault("translation", "")

    text, submitted = sidebar_controls()
    st.title("RFP Proposal Studio")
    st.caption("Sinh hồ sơ thầu tiếng Nhật với nguồn truy vết cho từng câu")
    tabs = st.tabs(
        [
            "Hồ sơ",
            "Độ đáp ứng",
            "Nguồn từng câu",
            "Bản dịch",
            "Trace",
            "Sinh golden test",
        ]
    )
    proposal_tab, coverage_tab, sources_tab, translation_tab, trace_tab, golden_tab = tabs

    with trace_tab:
        trace_metrics = st.empty()
        trace_status = st.empty()

    latest = st.session_state.get("result_state")
    if submitted:
        st.session_state["result_state"] = None
        st.session_state["translation"] = ""
        latest = None
        with st.spinner("Đang chạy pipeline…"):
            for latest in stream_graph(text):
                render_trace_metrics(latest, trace_metrics)
                render_pipeline_status(latest, trace_status)
        st.session_state["result_state"] = latest
        if latest and latest.get("status") == "completed":
            content_hash, content = translated_content(latest)
            with st.spinner("Đang dịch RFP và proposal sang tiếng Việt…"):
                st.session_state["translation"] = translate_once(content_hash, content)

    if latest is None:
        render_empty_tabs(tabs)
    else:
        render_trace_metrics(latest, trace_metrics)
        render_pipeline_status(latest, trace_status)
        if latest.get("status") == "ask_user":
            with proposal_tab:
                st.warning(latest["message"])
            for tab in (coverage_tab, sources_tab, translation_tab):
                with tab:
                    st.info("Cần bổ sung đầu vào trước khi sinh kết quả.")
        elif latest.get("status") == "completed":
            with proposal_tab:
                render_proposal(latest)
            with coverage_tab:
                render_coverage(latest)
            with sources_tab:
                render_sources(latest)
            with translation_tab:
                st.subheader("Bản dịch tiếng Việt của RFP và proposal")
                st.markdown(st.session_state.get("translation", ""))
            with trace_tab:
                render_trace_details(latest)

    with golden_tab:
        render_golden()


if __name__ == "__main__":
    main()
