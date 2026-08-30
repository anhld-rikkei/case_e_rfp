from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
from pathlib import Path
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
from rfp.graph import PIPELINE_STAGES, stream_graph
from rfp.llm import MODEL as LLM_MODEL, generate


STAGE_LABELS = {
    "parse_input": "Phân tích RFP",
    "check_complete": "Kiểm tra đầu vào",
    "ask_user": "Yêu cầu bổ sung",
    "route_reference_rfp": "Chọn RFP tham chiếu",
    "plan_sections": "Lập khung proposal",
    "retrieve_per_chapter": "Truy xuất bằng chứng",
    "generate_per_section": "Sinh 5 mục",
    "review": "Review chất lượng văn bản",
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
    st.caption("Golden = RFP + assertion máy kiểm được; không lưu proposal mẫu.")

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
    labels = (
        "Hồ sơ sẽ xuất hiện ở đây sau khi pipeline hoàn tất.",
        "Coverage requirement sẽ xuất hiện ở đây.",
        "Nguồn của từng câu sẽ xuất hiện ở đây.",
        "Bản dịch sẽ xuất hiện ở đây sau bước dịch cuối.",
    )
    for tab, label in zip(tabs[:4], labels):
        with tab:
            st.info(label)


def render_eval() -> None:
    st.header("Đo lường & Eval (RAGAS)")
    
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
        st.info("Chưa có kết quả chạy Eval. Chạy lệnh: `python -m eval.run_ragas ...` để xem kết quả.")
        return

    # Tóm tắt lần chạy gần nhất để nội suy
    latest_name, latest_data = runs[0]
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

    st.subheader("Ước tính chi phí (Dựa trên lần chạy gần nhất)")
    
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
    st.subheader("Lịch sử chạy Eval")
    
    rows = []
    for name, data in runs:
        ragas = data.get("ragas", {})
        det = data.get("deterministic", [{}])[0] if data.get("deterministic") else {}
        usage_data = data.get("usage", {})
        
        row = {
            "Run ID": name,
            "Thời gian": f"{usage_data.get('seconds', 0):.1f}s",
            "Token (SP/Đo)": f"{sum(v for k, v in usage_data.get('tokens_by_stage', {}).items() if k in ('generate', 'structured'))} / {usage_data.get('tokens_by_stage', {}).get('judge', 0)}",
            "Mẫu (có đáp án/tổng)": f"{data.get('ragas_sample_count', 0)}/{data.get('sample_count', 0)} (loại {data.get('unanswered_sample_count', 0)} không có answer)",
            "Coverage / Abstain": f"{det.get('coverage', 0):.2f} / {det.get('abstain_rate', 0):.2f}",
            "Groundedness": f"{det.get('groundedness', 0):.3f}",
            "Citation Acc": f"{det.get('citation_accuracy', 0):.3f}",
            "Fabrication": det.get("fabrication_count", 0),
            "Leak": det.get("client_leak_count", 0),
        }
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "noise_sensitivity", "context_entity_recall"]:
            m_data = ragas.get(metric, {})
            row[metric] = f"{m_data.get('mean', 0):.3f} ± {m_data.get('std', 0):.3f}" if m_data else "—"
            
        rows.append(row)
        
    st.dataframe(rows, width="stretch", hide_index=True)


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
            "Eval",
        ]
    )
    proposal_tab, coverage_tab, sources_tab, translation_tab, trace_tab, golden_tab, eval_tab = tabs

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
        
    with eval_tab:
        render_eval()


if __name__ == "__main__":
    main()
