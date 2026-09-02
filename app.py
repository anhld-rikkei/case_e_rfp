from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import html
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.settings import (
    PROMPT_VERSION,
    RELATED_SOURCES_TOP_N,
    SAMPLE_RFP_LIMIT,
    RFP_DIR,
    TRANSLATION_EFFORT,
    TRANSLATION_SYSTEM_PROMPT,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
)
from eval.golden.describe import describe_case
from eval.golden.coverage import (
    coverage_report,
    report_rows as coverage_rows,
    summary_line as coverage_summary,
)
from eval.golden.generator import (
    CHAPTER_TITLES,
    COVERAGE_AXES,
    DEFAULT_OUTPUT_DIR as GOLDEN_OUTPUT_DIR,
    generate_combinatorial,
    generate_for_coverage,
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
    CHECKLIST_ITEMS,
    export_filename,
    to_markdown,
)
from config.display_vi import (
    ATTRIBUTE_COVERED_REASON,
    BRANCH_VI,
    JOURNEY_LEGEND,
    REQ_BRANCH_VI,
    STRATEGY_TIER_VI,
    SOURCE_STRATEGY_VI,
    TIER_HINT,
    TIER_ICON,
    TIER_VI,
    EDITED_LABEL,
    MARK_LEGEND,
    NO_EVIDENCE_REASON,
    ORIGIN_HINT,
    USER_BLOCK_LABEL,
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
from rfp.freshness import (
    current_diff,
    load_manifest,
    quarantine_report,
    scan_sources,
    state_is_stale,
    write_manifest,
)
from rfp.graph import PIPELINE_STAGES, related_sources, stream_graph
from rfp.metrics import read_records, summarize
from rfp.guard import GuardViolation
from rfp.stores.sentence_index import SentenceIndex
from rfp.llm import LLMUnavailable, MODEL as LLM_MODEL, generate
from rfp.refine import refine_all, refine_section


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


def render_rfp_translation(text: str) -> None:
    """Dịch RFP đầu vào — giữ tính năng của tab Bản dịch cũ, đưa về cạnh ô dán.

    Dịch theo yêu cầu (chỉ khi mở expander và bấm), không tự chạy mỗi lần nhập.
    """
    if not st.button("Dịch RFP", key="translate_rfp", width="stretch"):
        cached = st.session_state.get("rfp_translation", "")
        if cached:
            render_translation(cached)
        return
    content = f"[RFP]\n{text}"
    key_material = "\n".join(
        (
            content,
            LLM_MODEL or "",
            PROMPT_VERSION,
            TRANSLATION_EFFORT,
            TRANSLATION_SYSTEM_PROMPT,
        )
    )
    content_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    with st.spinner("Đang dịch RFP…"):
        try:
            st.session_state["rfp_translation"] = translate_once(content_hash, content)
        except LLMUnavailable as error:
            st.warning(
                f"Chưa dịch được: nhà cung cấp LLM không phản hồi sau "
                f"{error.attempts} lần gọi."
            )
            return
    render_translation(st.session_state["rfp_translation"])


def _load_uploaded_rfp(uploaded) -> str | None:
    try:
        return uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.sidebar.error(
            f"{uploaded.name} không phải UTF-8. RFP phải là .txt mã hoá UTF-8."
        )
        return None


def _format_ingested_at(manifest: dict[str, Any] | None) -> str:
    if not manifest or not manifest.get("ingested_at"):
        return "chưa rõ"
    return time.strftime("%H:%M %d/%m/%Y", time.localtime(manifest["ingested_at"]))


def reload_knowledge_base() -> None:
    """Nạp lại kho: chạy ĐÚNG pipeline ingest, không đường tắt.

    `SentenceIndex.build()` là chính con đường mà pipeline thật dùng — quarantine
    blocklist và lọc rò rỉ tên khách hàng chạy nguyên vẹn bên trong nó. Không có
    phiên bản "nạp nhanh" bỏ qua hai lớp đó (BB-2).
    """
    index = SentenceIndex.build()
    counts = index.assert_clean()
    previous = load_manifest()
    write_manifest(index, files=scan_sources())
    st.session_state["reload_report"] = {
        "counts": counts,
        "previous": (previous or {}).get("counts"),
        "quarantine": quarantine_report(index),
    }
    # Bản đang xem sinh từ dữ liệu trước đó -> lượt chat tiếp theo bị chặn
    # (xem docstring freshness.py). Giữ nguyên bản, không xoá công sức người dùng.
    st.session_state["kb_reloaded_at"] = time.time()


def render_source_status() -> None:
    diff, manifest = current_diff()
    if not diff.has_manifest:
        st.info(
            "Chưa nạp kho tri thức lần nào trong phiên làm việc này. "
            "Bấm **Nạp lại kho tri thức** để ghi mốc đối chiếu."
        )
    elif not diff.changed:
        st.success(
            f"Dữ liệu nguồn khớp lần nạp gần nhất ({_format_ingested_at(manifest)})"
        )
    else:
        if diff.capability_changed:
            st.error(
                "🔴 **Bảng năng lực đã đổi.** Đây là trọng tài quyết định câu nào "
                "được phép nói — nạp lại trước khi sinh hồ sơ."
            )
        st.warning(
            f"⚠ Dữ liệu nguồn đã thay đổi so với lần nạp gần nhất "
            f"({_format_ingested_at(manifest)}) — {diff.total} file."
        )
        with st.expander(f"Xem {diff.total} thay đổi", expanded=False):
            for title, names in (
                ("Thêm", diff.added),
                ("Sửa", diff.modified),
                ("Xoá", diff.removed),
            ):
                if names:
                    st.markdown(f"**{title}** ({len(names)})")
                    for name in names:
                        st.markdown(f"- `{name}`")

    if st.button("Nạp lại kho tri thức", width="stretch", key="reload_kb"):
        with st.spinner("Đang nạp lại — parse, lọc rò rỉ, cách ly chuỗi cấm…"):
            reload_knowledge_base()
        st.rerun()

    report = st.session_state.get("reload_report")
    if report:
        counts = report["counts"]
        previous = report["previous"] or {}

        def _delta(key: str, previous_key: str) -> str:
            if previous_key not in previous:
                return ""
            change = counts[key] - previous[previous_key]
            return f" ({change:+d})" if change else " (không đổi)"

        st.success(
            f"Đã nạp: **{counts['indexed']}** câu vào kho"
            f"{_delta('indexed', 'indexed')} · "
            f"**{counts['capability_quarantine']}** câu bị cách ly vì chuỗi cấm"
            f"{_delta('capability_quarantine', 'quarantine')} · "
            f"**{counts['leak_quarantine']}** câu bị bỏ vì lộ tên khách hàng"
            f"{_delta('leak_quarantine', 'leak_dropped')}"
        )
        if report["quarantine"]:
            with st.expander(
                f"Vì sao {len(report['quarantine'])} câu bị loại", expanded=False
            ):
                st.dataframe(
                    [
                        {
                            "Mã câu": row["sent_id"],
                            "Loại": row["loai"],
                            "Lý do": row["ly_do"],
                            "Câu": row["text"],
                        }
                        for row in report["quarantine"]
                    ],
                    width="stretch",
                    hide_index=True,
                )


def render_case_brief(case: GoldenCase) -> None:
    """Ca này thử cái gì, đưa vào cái gì, máy chấm dựa trên điều gì.

    Dòng “3 điều kiện · must_cover · must_flag_insufficient” trước đây đúng
    nhưng vô dụng: người test đọc xong vẫn không biết phải nhìn vào đâu để
    chấm. Mỗi điều kiện giờ kèm câu yêu cầu gốc và lý do nó phải như vậy.
    """
    brief = describe_case(case)
    st.caption(f"**{brief.tier}** · {brief.purpose}")
    st.markdown("**Input**")
    st.markdown("\n".join(f"- {line}" for line in brief.inputs))
    st.markdown(f"**Output** — máy chấm {len(brief.checks)} điều kiện")
    st.markdown(
        "\n".join(
            f"- {check.icon} {check.what}  \n  <small>{check.why}</small>"
            for check in brief.checks
        ),
        unsafe_allow_html=True,
    )


def toggle_golden_picker() -> None:
    st.session_state["golden_picker_open"] = not st.session_state.get(
        "golden_picker_open"
    )


def render_golden_picker() -> None:
    """Chạy một ca golden qua ĐÚNG luồng sản phẩm.

    Tab "Sinh bộ test" chạy ca bằng runner và chỉ in PASS/FAIL. Muốn xem hồ sơ
    thật sự ra cái gì cho một ca biên thì phải nạp nó vào đây và bấm sinh như
    một RFP bình thường.
    """
    # Nút chứ không phải expander: nhãn nút tự canh giữa, khớp ngay với ba nút
    # "Chọn RFP-2025-00x" ngay trên. Canh giữa nhãn expander thì phải nhắm vào
    # DOM nội bộ của Streamlit — đã thử và trượt, vì expander ở bản này không
    # dựng bằng <summary>.
    st.button(
        "Chọn golden test",
        key="golden_picker_toggle",
        on_click=toggle_golden_picker,
        width="stretch",
    )
    if not st.session_state.get("golden_picker_open"):
        return
    cases = [load_case(path) for path in discover_case_files(DEFAULT_GOLDEN_DIR)]
    if not cases:
        st.caption("Chưa có ca nào trong golden set.")
        return
    by_id = {case.case_id: case for case in cases}
    picked = st.selectbox(
        "Chọn ca",
        list(by_id),
        key="golden_picker",
        label_visibility="collapsed",
        format_func=lambda case_id: (
            f"{case_id} · {by_id[case_id].metadata.get('tier', by_id[case_id].source)}"
        ),
    )
    render_case_brief(by_id[picked])
    st.button(
        "Nạp vào ô RFP",
        key="golden_picker_load",
        on_click=load_golden_into_input,
        args=(by_id[picked],),
        width="stretch",
    )


def render_rfp_upload() -> None:
    """Ô tải file RFP. Nạp xong thì nội dung nằm trong ô RFP như dán tay."""
    uploaded = st.file_uploader(
        "Tải file .txt (UTF-8)",
        type=["txt"],
        key="rfp_upload",
        label_visibility="collapsed",
    )
    if uploaded is None:
        return
    content = _load_uploaded_rfp(uploaded)
    # Chỉ nạp một lần cho mỗi file: nếu ghi đè mỗi lần chạy lại script thì
    # người dùng không sửa nổi nội dung trong ô text.
    if content is not None and st.session_state.get("loaded_upload") != uploaded.name:
        st.session_state["loaded_upload"] = uploaded.name
        select_sample(content)
        st.rerun()


def sidebar_controls() -> tuple[str, bool]:
    """Bảng trái, hai phần: kho dữ liệu nguồn và RFP của lượt này.

    Trước đây ô tải file nằm CHEN giữa hai phần đó, nên đọc từ trên xuống là
    một chuỗi việc không liên quan nhau. Nay mọi cách đưa RFP vào — dán, chọn
    mẫu, tải file — nằm chung một chỗ, ngay dưới ô RFP.
    """
    with st.sidebar:
        st.subheader("Dữ liệu nguồn")
        with st.expander("Tình trạng kho tri thức", expanded=False):
            render_source_status()

        st.divider()
        st.subheader("RFP")
        text = st.text_area(
            "RFP",
            key="rfp_input",
            height=380,
            label_visibility="collapsed",
            placeholder="Dán nội dung RFP tiếng Nhật tại đây…",
        )
        if text.strip():
            with st.expander("Bản dịch RFP (tiếng Việt)", expanded=False):
                render_rfp_translation(text)

        st.caption("Hoặc lấy sẵn từ:")
        with st.expander("RFP mẫu", expanded=False):
            for path in sorted(Path(RFP_DIR).glob("*.txt"))[:SAMPLE_RFP_LIMIT]:
                sample_text = path.read_text(encoding="utf-8")
                st.button(
                    f"Chọn {path.stem}",
                    key=f"sample_{path.stem}",
                    on_click=select_sample,
                    args=(sample_text,),
                    width="stretch",
                )
            render_golden_picker()
        with st.expander("Upload file RFP", expanded=False):
            render_rfp_upload()

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
    if sentence["origin"] == "user":
        return "✎ người dùng bổ sung (chưa kiểm chứng)"
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
    """Nhắc nháp, dạng một dòng chữ nhỏ.

    Khối đỏ to đã bỏ theo yêu cầu. Giữ lại đúng một dòng: nhãn "bản nháp" là
    thứ phân biệt tài liệu này với hồ sơ nộp được, và nó vẫn đi theo file xuất
    ra — bỏ hết trên màn hình thì chỉ còn file mang nhãn, người đọc màn hình
    không còn gì để biết.
    """
    st.caption(
        f"{DRAFT_BANNER_TITLE} · chưa có người thật rà soát — "
        "tick hết checklist trong **💬 Chat review** trước khi nộp."
    )
    if state_is_stale(state):
        st.warning(
            "⚠ **Hồ sơ này sinh từ dữ liệu nguồn đã cũ.** Dữ liệu nguồn đã thay "
            "đổi sau khi hồ sơ được sinh — nạp lại kho tri thức và sinh lại "
            "trước khi dùng. Cảnh báo này cũng nằm trong file tải về.",
            icon="⚠️",
        )


def render_download(state: dict[str, Any]) -> None:
    """Nút tải, đặt ở CUỐI tab.

    Tải về là việc làm sau cùng — đặt nó ngay đầu trang là mời người dùng tải
    trước khi đọc bất cứ thứ gì. Checklist chỉ hiện một chỗ (panel chat, nơi
    tick được); bản đầy đủ kèm dòng ký tên nằm trong chính file này.
    """
    try:
        markdown = to_markdown(state, stale=state_is_stale(state))
    except GuardViolation as violation:
        # Không bao giờ mở đường tải cho bản chưa qua guard.
        st.error(f"Không xuất được: final guard chặn — {violation}")
        return
    # Căn phải: nút cuối trang, không phải một khối nội dung.
    spacer, action = st.columns([4, 1])
    with action:
        st.download_button(
            "⬇ Tải file (.md)",
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
        render_related_sources(state, section)
        return
    st.warning(label(SECTION_NOTE_VI, note))


def _versions() -> list[dict[str, Any]]:
    return st.session_state.setdefault("versions", [])


def push_version(state: dict[str, Any], *, label: str, **extra: Any) -> None:
    """Thêm một bản MỚI. Không bao giờ ghi đè hay xoá bản nào trong session.

    Quay lại bản cũ rồi chat tiếp sẽ sinh bản mới nối vào cuối, có ghi rõ nó
    xuất phát từ bản nào — các bản sau vẫn còn nguyên để so và quay lại.
    """
    versions = _versions()
    versions.append({"state": state, "label": label, **extra})
    st.session_state["version_index"] = len(versions) - 1


def version_label(versions: list[dict[str, Any]], parent_index: int | None) -> str:
    label = f"v{len(versions) + 1}"
    if parent_index is not None and parent_index != len(versions) - 1:
        return f"{label} · từ {versions[parent_index]['label']}"
    return label


def displayed_state() -> dict[str, Any] | None:
    """Bản đang hiển thị — cũng là bản mà chat sẽ chỉnh và export sẽ tải về."""
    version = current_version()
    if version:
        return version["state"]
    latest = st.session_state.get("result_state")
    if latest and latest.get("status") == "completed":
        return latest
    return None


def current_version() -> dict[str, Any] | None:
    versions = _versions()
    index = st.session_state.get("version_index", len(versions) - 1)
    if not versions or not (0 <= index < len(versions)):
        return None
    return versions[index]


def version_diff(older: dict[str, Any], newer: dict[str, Any]) -> str:
    """Diff giữa hai bản, theo dòng. `difflib` là thư viện chuẩn — không thêm
    dependency, đúng luật repo."""
    import difflib

    return "\n".join(
        difflib.unified_diff(
            older.get("proposal", "").splitlines(),
            newer.get("proposal", "").splitlines(),
            fromfile="bản trước",
            tofile="bản này",
            lineterm="",
            n=1,
        )
    )


ALL_TARGET = "__all__"

# Bảng màu panel chat — cùng hệ với sơ đồ luồng và dark theme của app.
#
# KHÔNG dùng `:has()` và KHÔNG dùng <summary>. Đã tra bundle frontend của
# Streamlit 1.61: expander không dựng bằng <details>/<summary>, và khi truyền
# `avatar=` thì test-id đổi thành stChatMessageAvatarCustom. Hai selector kiểu
# đó khớp rỗng — code đổi mà màn hình y nguyên, không lỗi nào báo ra.
CHAT_DOCK_KEY = "chat_dock"

CHAT_CSS = f"""
<style>
:root {{ --chat-dock-w: 30rem; }}
/* Bảng chat neo mép phải, cao hết màn hình, cuộn riêng — một bảng độc lập
   như bảng RFP bên trái. Bề rộng đọc từ biến CSS để tay kéo đổi được cả bảng
   lẫn phần chừa chỗ của trang bằng một phép gán. */
.st-key-{CHAT_DOCK_KEY} {{
    position: fixed;
    top: 3.4rem;
    right: 0;
    bottom: 0;
    width: var(--chat-dock-w);
    overflow-y: auto;
    overflow-x: hidden;
    background: #12161C;
    border-left: 1px solid #2A323C;
    padding: .8rem 1.1rem 2rem 1.4rem;
    z-index: 50;
}}
/* Tay kéo: dải dọc sát mép trái bảng, đúng chỗ người dùng đưa chuột tới. */
.chat-dock-handle {{
    position: fixed;
    top: 3.4rem;
    bottom: 0;
    width: 7px;
    right: var(--chat-dock-w);
    cursor: col-resize;
    z-index: 51;
    background: transparent;
    transition: background .15s;
}}
.chat-dock-handle:hover, .chat-dock-handle.dragging {{ background: #2F8F63; }}

/* Câu văn là nút bấm để tra nguồn — nên nó phải TRÔNG như câu văn. Hỏng CSS
   thì tệ nhất là nó giống một cái nút, vẫn bấm được, vẫn ra nguồn. */
[data-testid="stPopoverButton"] {{
    background: transparent !important;
    border: none !important;
    padding: .15rem 0 !important;
    text-align: left !important;
    font-weight: 400 !important;
    justify-content: flex-start !important;
}}
[data-testid="stPopoverButton"]:hover {{
    background: #1B2129 !important;
    text-decoration: underline dotted;
}}
[data-testid="stChatMessage"] {{
    padding: .7rem .9rem;
    margin-bottom: .5rem;
    border-radius: 14px;
    background: #1B2129;
    border: 1px solid #2A323C;
}}
[data-testid="stChatMessage"] p {{ margin-bottom: .3rem; }}
[class*="st-key-chatturn_user"] /* Câu văn là nút bấm để tra nguồn — nên nó phải TRÔNG như câu văn. Hỏng CSS
   thì tệ nhất là nó giống một cái nút, vẫn bấm được, vẫn ra nguồn. */
[data-testid="stPopoverButton"] {{
    background: transparent !important;
    border: none !important;
    padding: .15rem 0 !important;
    text-align: left !important;
    font-weight: 400 !important;
    justify-content: flex-start !important;
}}
[data-testid="stPopoverButton"]:hover {{
    background: #1B2129 !important;
    text-decoration: underline dotted;
}}
[data-testid="stChatMessage"] {{
    flex-direction: row-reverse;
    background: #1E6F4C;
    border-color: #2F8F63;
}}
[class*="st-key-chatturn_user"] [data-testid="stChatMessage"] p {{
    color: #EAF6EF;
}}
</style>
"""

CHAT_RESERVE_CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    padding-right: calc(var(--chat-dock-w) + 2rem);
}
</style>
"""

# `resize: horizontal` của CSS đặt tay kéo ở góc dưới của khối — với bảng cao
# hết màn hình thì góc đó nằm tận đáy, không ai tìm ra. Dựng tay kéo riêng ở
# mép trái cho giống thanh kéo của sidebar.
#
# Chạy trong iframe của st.components nên phải với sang `window.parent`.
# MutationObserver là bắt buộc: Streamlit dựng lại DOM sau mỗi lần chạy script,
# gắn một lần lúc nạp thì lượt rerun sau là mất tay kéo.
CHAT_DRAG_JS = """
<script>
(function () {
  const doc = window.parent.document;
  const root = doc.documentElement;
  function attach() {
    const dock = doc.querySelector('.st-key-chat_dock');
    if (!dock) return;
    if (doc.querySelector('.chat-dock-handle')) return;
    const handle = doc.createElement('div');
    handle.className = 'chat-dock-handle';
    doc.body.appendChild(handle);
    let startX = 0, startW = 0, dragging = false;
    handle.addEventListener('mousedown', function (event) {
      dragging = true;
      startX = event.clientX;
      startW = dock.getBoundingClientRect().width;
      handle.classList.add('dragging');
      doc.body.style.userSelect = 'none';
      event.preventDefault();
    });
    doc.addEventListener('mousemove', function (event) {
      if (!dragging) return;
      const limit = window.parent.innerWidth * 0.7;
      const next = Math.min(Math.max(startW + (startX - event.clientX), 320), limit);
      root.style.setProperty('--chat-dock-w', next + 'px');
    });
    doc.addEventListener('mouseup', function () {
      dragging = false;
      handle.classList.remove('dragging');
      doc.body.style.userSelect = '';
    });
  }
  function cleanup() {
    if (!doc.querySelector('.st-key-chat_dock')) {
      const stale = doc.querySelector('.chat-dock-handle');
      if (stale) stale.remove();
    }
  }
  attach();
  new MutationObserver(function () { cleanup(); attach(); })
    .observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""

CHIP_STYLE = (
    "display:inline-block;margin:.15rem .2rem 0 0;padding:.1rem .45rem;"
    "border:1px solid {border};border-radius:999px;background:{fill};"
    "color:{text};font-size:.72rem;line-height:1.5"
)


def chip(text: str, *, tone: str = "muted") -> str:
    tones = {
        "muted": ("#2A323C", "#1B2129", "#8B97A6"),
        "block": ("#6B2B27", "#2A1A19", "#F0A8A2"),
        "pin": ("#5A4A1E", "#2A2416", "#E6C86A"),
    }
    border, fill, color = tones[tone]
    return (
        f'<span style="{CHIP_STYLE.format(border=border, fill=fill, text=color)}">'
        f"{_esc(text)}</span>"
    )


def refine_targets(state: dict[str, Any]) -> list[tuple[str, str]]:
    """(khoá, nhãn) cho ô chọn phạm vi chỉnh."""
    return [(ALL_TARGET, "Toàn bộ hồ sơ")] + [
        (section["key"], f"{index}. {section['title_ja']}")
        for index, section in enumerate(state.get("sections", []), start=1)
    ]


def instruction_suggestions(state: dict[str, Any]) -> list[dict[str, str]]:
    """Gợi ý chỉ thị, suy từ chính hồ sơ đang mở. Không gọi LLM.

    Ô chat trống là chỗ người dùng đứng hình: họ không biết chat này làm được
    gì. Ba gợi ý bấm-là-chạy nói điều đó nhanh hơn một đoạn hướng dẫn.
    """
    sections = [
        section for section in state.get("sections", []) if section.get("sentences")
    ]
    if not sections:
        return []
    by_size = sorted(
        sections,
        key=lambda section: (
            -sum(len(item.get("text", "")) for item in section["sentences"]),
            section["key"],
        ),
    )
    longest = by_size[0]
    suggestions = [
        {
            "instruction": f"Viết mục 「{longest['title_ja']}」 ngắn gọn hơn",
            "target": longest["key"],
        }
    ]
    if len(by_size) > 1:
        second = by_size[1]
        suggestions.append(
            {
                "instruction": (
                    f"Gộp các câu trùng ý trong mục 「{second['title_ja']}」"
                ),
                "target": second["key"],
            }
        )
    suggestions.append(
        {
            "instruction": "Viết lại toàn bộ hồ sơ với văn phong trang trọng hơn",
            "target": ALL_TARGET,
        }
    )
    return suggestions


def chat_turns(
    versions: list[dict[str, Any]],
    *,
    last: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Lịch sử chat dựng TỪ lịch sử phiên bản, không nuôi một bản sao riêng.

    Mỗi bản có chỉ thị là một cặp lượt: người nói, hệ thống đáp. Giữ hai kho
    song song rồi lệch nhau là cách chắc chắn nhất để chat kể sai chuyện đã xảy
    ra với tài liệu.
    """
    turns: list[dict[str, Any]] = []
    for version in versions:
        instruction = version.get("instruction")
        if not instruction:
            continue
        turns.append(
            {
                "role": "user",
                "text": instruction,
                "target": version.get("target", ""),
            }
        )
        turns.append(
            {
                "role": "assistant",
                "outcome": "changed",
                "label": version.get("label", ""),
                "counts": version.get("counts") or {},
                "rejected": version.get("rejected") or [],
                "restored": version.get("restored", 0),
            }
        )
    if last:
        turns.append(
            {"role": "user", "text": last["instruction"], "target": last["target"]}
        )
        turns.append({"role": "assistant", **last["reply"]})
    return turns


def chat_reply_text(turn: dict[str, Any]) -> str:
    counts = turn.get("counts") or {}
    if turn["outcome"] == "changed":
        label = turn.get("label", "")
        return f"Đã cập nhật — bản **{label}**. " + _counts_line(counts)
    if turn["outcome"] == "blocked":
        return (
            "**Không áp dụng thay đổi nào — lưới an toàn đã chặn.** "
            + _counts_line(counts)
        )
    if turn["outcome"] == "error":
        return turn.get("message", "Không gọi được mô hình. Hồ sơ giữ nguyên.")
    return (
        "**Không có thay đổi nào.** Chat chỉ viết lại, bỏ hoặc đổi thứ tự câu "
        "đã có — nó không thêm được nội dung chưa có bằng chứng."
    )


def chat_reply_chips(turn: dict[str, Any]) -> list[tuple[str, str]]:
    chips: list[tuple[str, str]] = []
    if turn.get("restored"):
        chips.append((f"📌 giữ lại {turn['restored']} câu bạn thêm", "pin"))
    for item in turn.get("rejected") or []:
        reason = re.sub(r"[*`]", "", str(item.get("reason", "")))
        chips.append((f"🛑 {reason[:90]}", "block"))
    return chips


def submit_instruction(
    state: dict[str, Any], *, target_key: str, target_label: str, instruction: str
) -> None:
    """Chạy một lượt chat rồi ghi kết quả vào lịch sử phiên bản.

    Mọi kết cục đều được ghi lại và hiện thành một lượt trả lời — kể cả lượt bị
    chặn. Im lặng khi bị chặn là đúng cái lỗi màn hình cũ đã mắc.
    """
    try:
        if target_key == ALL_TARGET:
            result = refine_all(state, instruction=instruction)
        else:
            result = refine_section(
                state, section_key=target_key, instruction=instruction
            )
    except LLMUnavailable as error:
        st.session_state["chat_last"] = {
            "instruction": instruction,
            "target": target_label,
            "reply": {
                "outcome": "error",
                "message": (
                    f"Nhà cung cấp LLM không phản hồi sau {error.attempts} lần gọi "
                    f"(lỗi {error.kind}). Hồ sơ giữ nguyên."
                ),
            },
        }
        return

    if result.outcome == "changed":
        parent_index = st.session_state.get("version_index", len(_versions()) - 1)
        push_version(
            result.state,
            label=version_label(_versions(), parent_index),
            instruction=instruction,
            target=target_label,
            rejected=result.rejected,
            counts=refine_counts(result),
            restored=result.restored,
            parent_index=parent_index,
        )
        st.session_state["translation"] = ""
        # Bản mới đã mang sẵn lượt trả lời; giữ thêm `chat_last` là in hai lần.
        st.session_state.pop("chat_last", None)
        return

    st.session_state["chat_last"] = {
        "instruction": instruction,
        "target": target_label,
        "reply": {
            "outcome": result.outcome,
            "counts": refine_counts(result),
            "rejected": result.rejected,
            "restored": result.restored,
        },
    }


def queue_instruction(instruction: str, target_key: str) -> None:
    st.session_state["chat_pending"] = {
        "instruction": instruction,
        "target_key": target_key,
    }


def open_chat_panel() -> None:
    st.session_state["chat_open"] = True


def close_chat_panel() -> None:
    st.session_state["chat_open"] = False


def chat_is_open() -> bool:
    return bool(st.session_state.get("chat_open"))


def scope_label(state: dict[str, Any], key: str) -> str:
    return dict(refine_targets(state)).get(key, "Toàn bộ hồ sơ")


def instructions_for(state: dict[str, Any], scope_key: str) -> list[str]:
    """Ba câu chỉ thị gợi sẵn cho phạm vi đang chọn. Tất định, 0 lệnh gọi LLM."""
    if scope_key == ALL_TARGET:
        return [
            "Viết lại toàn bộ hồ sơ với văn phong trang trọng hơn",
            "Rút ngắn những chỗ dài dòng",
            "Bỏ các câu trùng ý giữa các mục",
        ]
    title = next(
        (
            section["title_ja"]
            for section in state.get("sections", [])
            if section["key"] == scope_key
        ),
        scope_key,
    )
    return [
        f"Viết mục 「{title}」 ngắn gọn hơn",
        f"Gộp các câu trùng ý trong mục 「{title}」",
        f"Đưa câu có số liệu lên đầu mục 「{title}」",
    ]


def pick_scope(key: str) -> None:
    st.session_state["chat_scope"] = key
    st.session_state["chat_checklist_open"] = False


def clear_scope() -> None:
    st.session_state["chat_scope"] = None


def toggle_chat_checklist() -> None:
    st.session_state["chat_checklist_open"] = not st.session_state.get(
        "chat_checklist_open"
    )


def render_chat_toggle() -> None:
    """Nút mở chat, cạnh tiêu đề trang.

    KHÔNG vô hiệu hoá theo trạng thái hồ sơ. Bản trước có `disabled=` và nó
    luôn bật nhầm ở đúng lượt quan trọng nhất: nút vẽ ở đầu `main()`, còn
    `result_state` mãi cuối hàm mới được gán — nên ngay lượt sinh xong hồ sơ,
    nút vẫn đang mờ. Bấm không lên, không báo gì.
    """
    if chat_is_open():
        st.button(
            "✕ Đóng chat",
            key="chat_close_top",
            on_click=close_chat_panel,
            width="stretch",
        )
        return
    st.button(
        "💬 Chat",
        key="chat_open_button",
        type="primary",
        width="stretch",
        on_click=open_chat_panel,
        help="Chỉnh lại hồ sơ bằng chỉ thị, kèm checklist trước khi nộp",
    )


def render_chat_placeholder() -> None:
    """Chat mở khi chưa có hồ sơ: nói rõ còn thiếu bước nào, đừng im lặng."""
    with st.container(key="chatturn_assistant_empty"):
        with st.chat_message("assistant"):
            st.markdown(
                "Chưa có hồ sơ nào để chỉnh. Dán RFP ở bảng bên trái rồi bấm "
                "**Nộp và sinh hồ sơ** — xong tôi sẽ chỉnh giúp từng mục."
            )
    if st.session_state.get("chat_checklist_open"):
        render_checklist_bubble()
        st.button(
            "← Quay lại",
            key="chat_checklist_back",
            width="stretch",
            on_click=toggle_chat_checklist,
        )
        return
    # Checklist vẫn xem được: nó là thứ tham khảo trước khi bắt tay vào làm,
    # không phải phần thưởng sau khi sinh xong hồ sơ.
    st.button(
        "📋 Checklist trước khi nộp",
        key="chat_checklist_open_button",
        width="stretch",
        on_click=toggle_chat_checklist,
    )


def render_chat_dock() -> None:
    """Bảng chat neo mép phải — dựng ở NGOÀI tab và ở CUỐI `main()`.

    Ngoài tab: chat bấm được từ bất cứ tab nào, không riêng tab Tổng quan.
    Cuối `main()`: đọc `displayed_state()` sau khi cả lượt chạy đã xong, nên
    ngay lượt sinh hồ sơ nó đã thấy bản mới. Bảng `position: fixed` nên vị trí
    trong DOM không ảnh hưởng chỗ nó hiện.
    """
    if not chat_is_open():
        return
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.markdown(CHAT_RESERVE_CSS, unsafe_allow_html=True)
    with st.container(key=CHAT_DOCK_KEY):
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.subheader("Chat")
        with header_right:
            st.button(
                "✕", key="chat_close", on_click=close_chat_panel, help="Đóng chat"
            )
        state = displayed_state()
        if state is None:
            render_chat_placeholder()
        else:
            render_chat_panel(state)
    # `st.iframe` chứ không phải `st.components.v1.html`: bản này đã báo khai tử
    # API cũ (hạn 2026-06-01, đã qua). height tối thiểu là 1, nó từ chối 0.
    st.iframe(CHAT_DRAG_JS, height=1)


def render_checklist_bubble() -> None:
    """Checklist nằm TRONG hội thoại, như một câu trả lời tham khảo.

    Trước đây nó là một khối expander đứng riêng trên đầu bảng — một mục lạc
    lõng giữa khung chat. Ở đây nó là thứ chat đưa ra khi được hỏi.
    """
    with st.container(key="chatturn_assistant_checklist"):
        with st.chat_message("assistant"):
            ticked = sum(
                bool(st.session_state.get(f"checklist_{index}"))
                for index in range(len(CHECKLIST_ITEMS))
            )
            total = len(CHECKLIST_ITEMS)
            st.markdown(
                f"Trước khi nộp, tự kiểm {total} mục này — **{ticked}/{total}**:"
            )
            for index, (short, full) in enumerate(CHECKLIST_ITEMS):
                st.checkbox(short, key=f"checklist_{index}", help=full)
            if ticked < total:
                st.markdown(
                    chip("chưa tick đủ — chưa nộp được", tone="block"),
                    unsafe_allow_html=True,
                )


def render_chat_options(state: dict[str, Any], scope: str | None) -> None:
    """Lựa chọn bấm-là-chạy, thay cho ô select và khối gợi ý rời rạc.

    Người dùng không phải học trước là chat làm được gì: mỗi bước chỉ hiện đúng
    những nước đi kế tiếp, đánh số như một câu hỏi trong hội thoại.
    """
    if st.session_state.get("chat_checklist_open"):
        st.button(
            "← Quay lại",
            key="chat_checklist_back",
            width="stretch",
            on_click=toggle_chat_checklist,
        )
        return

    if scope is None:
        for index, (key, label) in enumerate(refine_targets(state), start=1):
            st.button(
                f"{index}. {label}",
                key=f"chat_scope_{key}",
                width="stretch",
                on_click=pick_scope,
                args=(key,),
            )
    else:
        for index, instruction in enumerate(instructions_for(state, scope)):
            st.button(
                instruction,
                key=f"chat_instruction_{index}",
                width="stretch",
                on_click=queue_instruction,
                args=(instruction, scope),
            )
        st.button(
            "← Đổi phần khác",
            key="chat_scope_reset",
            width="stretch",
            on_click=clear_scope,
        )
    st.button(
        "📋 Checklist trước khi nộp",
        key="chat_checklist_open_button",
        width="stretch",
        on_click=toggle_chat_checklist,
    )


def render_chat_panel(state: dict[str, Any]) -> None:
    """Ruột bảng chat. Phần khung và tiêu đề do `render_chat_dock` dựng."""
    if state_is_stale(state):
        st.warning(
            "Dữ liệu nguồn đã đổi sau khi hồ sơ này được sinh. Chạy lại hồ sơ "
            "trước khi chỉnh — chỉnh trên bản cũ là trộn hai thế hệ dữ liệu vào "
            "cùng một tài liệu.",
            icon="⚠️",
        )
        return

    scope = st.session_state.get("chat_scope")
    turns = chat_turns(_versions(), last=st.session_state.get("chat_last"))
    with st.container(height=420, autoscroll=True, key="chat_log", border=False):
        with st.container(key="chatturn_assistant_open"):
            with st.chat_message("assistant"):
                st.markdown(f"Hồ sơ đã sinh xong. {sentence_breakdown(state)}")
                st.markdown(
                    chip("chỉ viết lại trên căn cứ sẵn có")
                    + chip("không đổi số liệu"),
                    unsafe_allow_html=True,
                )
        for index, turn in enumerate(turns):
            # Key mang sẵn vai trong tên: CSS bắt lượt người dùng bằng
            # [class*="st-key-chatturn_user"], không cần `:has()`.
            with st.container(key=f"chatturn_{turn['role']}_{index}"):
                with st.chat_message(turn["role"]):
                    if turn["role"] == "user":
                        st.markdown(turn["text"])
                        if turn.get("target"):
                            st.markdown(chip(turn["target"]), unsafe_allow_html=True)
                        continue
                    st.markdown(chat_reply_text(turn))
                    chips = chat_reply_chips(turn)
                    if chips:
                        st.markdown(
                            "".join(chip(text, tone=tone) for text, tone in chips),
                            unsafe_allow_html=True,
                        )
        if st.session_state.get("chat_checklist_open"):
            render_checklist_bubble()
        elif scope is None:
            with st.container(key="chatturn_assistant_ask"):
                with st.chat_message("assistant"):
                    st.markdown("Bạn muốn chỉnh phần nào?")
        else:
            with st.container(key="chatturn_assistant_scope"):
                with st.chat_message("assistant"):
                    st.markdown(
                        f"Đang chỉnh **{scope_label(state, scope)}**. "
                        "Chọn một gợi ý hoặc tự gõ chỉ thị."
                    )

    render_chat_options(state, scope)

    typed = st.chat_input("Nhập tin nhắn…", key="chat_input")
    pending = st.session_state.pop("chat_pending", None)
    if typed:
        # Gõ thẳng mà chưa chọn phạm vi thì hiểu là cả hồ sơ — hỏi lại một
        # bước nữa chỉ để xác nhận điều hiển nhiên là bắt người dùng chờ.
        pending = {"instruction": typed, "target_key": scope or ALL_TARGET}
    if not pending:
        return

    target_key = pending["target_key"]
    with st.spinner("Đang chỉnh lại…"):
        submit_instruction(
            state,
            target_key=target_key,
            target_label=scope_label(state, target_key),
            instruction=pending["instruction"],
        )
    st.rerun()


def refine_counts(result: Any) -> dict[str, int]:
    # `added` không lên dòng tóm tắt nhưng phải lưu: `RefineResult.outcome` suy
    # từ changed/dropped/added, thiếu nó thì dựng lại lượt cũ ra sai kết cục.
    return {
        "changed": result.changed,
        "dropped": result.dropped,
        "kept": result.kept,
        "added": result.added,
        "blocked": result.blocked,
    }


def _counts_line(counts: dict[str, int]) -> str:
    return (
        f"**{counts['changed']}** câu sửa · "
        f"**{counts['dropped']}** câu bỏ · "
        f"**{counts['kept']}** câu giữ nguyên · "
        f"**{counts['blocked']}** thay đổi bị chặn"
    )


def _render_rejections(rejected: list[dict[str, Any]]) -> None:
    if not rejected:
        return
    for item in rejected:
        text = f"  \n  Câu liên quan: `{item['text'][:70]}…`" if item.get("text") else ""
        st.markdown(f"- {item['reason']}{text}")


def render_refine_outcome(result: Any) -> None:
    """Ba kết cục, ba thông báo khác nhau.

    Gộp "bị lưới an toàn chặn" và "model không đổi gì" làm một câu
    "không có thay đổi nào" là lỗi đã gặp thật: người dùng đòi thêm một chứng
    chỉ công ty không có, hệ thống từ chối đúng, nhưng màn hình đọc ra thành
    tính năng hỏng.
    """
    counts = refine_counts(result)
    if result.restored:
        st.info(
            f"📌 **Đã giữ lại {result.restored} câu bạn thêm trước đó.** Mô hình "
            "định sửa hoặc xoá chúng trong lượt này, nhưng chúng đang được ghim. "
            "Muốn cho phép sửa thì bấm **Bỏ ghim** ở câu tương ứng, hoặc nhắc "
            "đích danh nội dung đó trong chỉ thị."
        )
    if result.outcome == "changed":
        st.success(f"Đã cập nhật: {_counts_line(counts)}")
        if result.rejected:
            st.warning(
                "Một phần thay đổi bị lưới an toàn chặn, hồ sơ giữ nguyên "
                "những chỗ đó:",
                icon="🛑",
            )
            _render_rejections(result.rejected)
        return

    if result.outcome == "blocked":
        st.warning(
            f"**Không áp dụng thay đổi nào — lưới an toàn đã chặn.** "
            f"{_counts_line(counts)}",
            icon="🛑",
        )
        _render_rejections(result.rejected)
        return

    st.info(
        "**Không có thay đổi nào** — mô hình không đề xuất sửa câu nào trong "
        f"mục này. {_counts_line(counts)}  \n"
        "Chat chỉ **viết lại, bỏ hoặc đổi thứ tự** câu đã có. Nó không thêm "
        "được nội dung mới: mọi câu trong hồ sơ phải truy được về một nguồn cụ "
        "thể. Nếu bạn cần thêm một năng lực hay chứng chỉ, hãy cập nhật "
        "`capability_sheet.json` rồi bấm **Nạp lại kho tri thức** và sinh lại "
        "hồ sơ."
    )


def render_version_history() -> None:
    versions = _versions()
    if len(versions) <= 1:
        return
    st.divider()
    st.subheader("Lịch sử phiên bản")
    index = st.session_state.get("version_index", len(versions) - 1)
    labels = [
        f"{item['label']}"
        + (f" · {item.get('target', '')}" if item.get("instruction") else " · bản gốc")
        + (" · 📌 giữ lại câu ghim" if item.get("restored") else "")
        for item in versions
    ]
    picked = st.radio(
        "Bản đang hiển thị (cũng là bản sẽ tải về)",
        range(len(versions)),
        index=index,
        format_func=lambda i: labels[i],
        horizontal=True,
        key="version_picker",
    )
    if picked != index:
        st.session_state["version_index"] = picked
        st.session_state["translation"] = ""
        st.rerun()

    chosen = versions[picked]
    if chosen.get("instruction"):
        st.caption(f"Chỉ thị: “{chosen['instruction']}”")
    if chosen.get("counts"):
        st.caption(_counts_line(chosen["counts"]))
    if chosen.get("rejected"):
        st.warning("Thay đổi bị lưới an toàn chặn ở bản này:", icon="🛑")
        _render_rejections(chosen["rejected"])
    if picked > 0:
        with st.expander("Xem thay đổi so với bản trước", expanded=False):
            base = chosen.get("parent_index", picked - 1)
            diff = version_diff(versions[base]["state"], chosen["state"])
            st.code(diff or "(không có khác biệt trong phần văn bản)", language="diff")

    # Toàn văn bản đang chọn, ngay tại đây: đổi phiên bản là thấy ngay nội dung
    # của phiên bản đó, không phải suy từ diff.
    st.markdown(f"#### Bản đầy đủ — {chosen['label']}")
    render_mark_legend(
        [
            sentence
            for section in chosen["state"].get("sections", [])
            for sentence in section["sentences"]
        ]
    )
    render_bilingual_proposal(
        chosen["state"], key_prefix=f"version_{picked}", allow_unpin=True
    )


def sentence_mark(sentence: dict[str, Any]) -> str:
    """Mức đánh dấu của một câu: user / edited / plain.

    Hai mức đánh dấu thay vì ba: câu máy sinh nguyên bản không đánh dấu gì, nên
    chỉ cần phân biệt "người dùng đưa vào" (nền vàng) với "chat chỉnh cách viết
    nhưng giữ nguyên nguồn và số liệu" (viền trái). Ba mức màu trên cùng một
    trang văn bản tiếng Nhật đọc rất rối, mà mức thứ ba không mang thêm quyết
    định nào cho người rà soát.
    """
    if sentence.get("origin") == "user":
        return "user"
    if sentence.get("edited_by_chat"):
        return "edited"
    return "plain"


def group_by_mark(
    sentences: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Gộp các câu liền nhau cùng mức — nhiều câu user liền nhau thành MỘT vùng."""
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for sentence in sentences:
        mark = sentence_mark(sentence)
        if groups and groups[-1][0] == mark:
            groups[-1][1].append(sentence)
        else:
            groups.append((mark, [sentence]))
    return groups


def unpin_sentence(section_key: str, text: str) -> None:
    """Bỏ ghim một câu trong bản ĐANG hiển thị.

    Bỏ ghim là cho phép lượt chat sau sửa hoặc xoá câu đó — một quyết định của
    người dùng, nên nó phải là một cú bấm rõ ràng chứ không phải hệ quả phụ.
    """
    version = current_version()
    if not version:
        return
    for section in version["state"].get("sections", []):
        if section["key"] != section_key:
            continue
        for sentence in section.get("sentences", []):
            if sentence.get("text") == text:
                sentence["pinned"] = False


def sentence_source(state: dict[str, Any], text: str) -> dict[str, str] | None:
    """Tra thông tin nguồn của một câu trong hồ sơ đang hiển thị."""
    for row in sentence_rows(state):
        if row["Câu (tiếng Nhật)"] == text:
            return row
    return None


def render_source_lookup(
    state: dict[str, Any] | None, text: str, *, key: str
) -> None:
    """Chính CÂU đó là nút: bấm vào câu là hiện nguồn của nó.

    Bản trước để một nút kính lúp riêng ở cuối câu — thêm một thứ phải nhắm
    trúng, trong khi thứ người đọc đang nhìn chính là câu văn.
    """
    if state is None:
        st.write(text)
        return
    row = sentence_source(state, text)
    if row is None:
        st.write(text)
        return
    with st.popover(text, width="stretch"):
        st.markdown(f"**Nguồn:** {row['Nguồn']}")
        st.markdown(f"**Mã nguồn:** `{row['Mã nguồn']}`")
        st.markdown(f"**Kiểm chứng:** {row['Kiểm chứng']}")
        st.markdown(f"**Cách lấy nguồn:** {row['Cách lấy nguồn']}")
        if row["_req_ids"]:
            st.markdown(f"**Đáp ứng yêu cầu:** {' · '.join(row['_req_ids'])}")


def requirement_labels(state: dict[str, Any], section: dict[str, Any]) -> dict[str, str]:
    """Mã yêu cầu -> nguyên văn yêu cầu, lấy từ các chương RFP nuôi mục này."""
    chapters = {chapter["id"]: chapter for chapter in state.get("chapters", [])}
    labels: dict[str, str] = {}
    for chapter_id in section.get("source_chapters", []):
        chapter = chapters.get(chapter_id)
        if not chapter:
            continue
        for requirement in chapter.get("requirements", []):
            labels[requirement["req_id"]] = requirement.get("text", "")
    return labels


def requirement_groups(
    state: dict[str, Any], section: dict[str, Any]
) -> list[dict[str, Any]]:
    """Câu của mục, gom theo yêu cầu RFP mà câu đó đáp.

    Yêu cầu KHÔNG có câu nào đáp vẫn phải xuất hiện — chỗ trống mới là thứ
    người rà soát cần thấy, ẩn đi thì hồ sơ đọc như đã đủ.

    Chỉ số gốc của câu được giữ kèm: bản dịch khớp theo chỉ số, gom nhóm mà
    đánh mất chỉ số là hai cột lệch nhau ngay.
    """
    labels = requirement_labels(state, section)
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {
        req_id: [] for req_id in labels
    }
    loose: list[tuple[int, dict[str, Any]]] = []
    for index, sentence in enumerate(section.get("sentences", [])):
        req_ids = sentence.get("req_ids") or []
        if not req_ids:
            loose.append((index, sentence))
            continue
        # Câu đáp nhiều yêu cầu chỉ in MỘT lần, dưới yêu cầu đầu tiên; các mã
        # còn lại hiện trong ô tra nguồn của chính câu đó.
        buckets.setdefault(req_ids[0], []).append((index, sentence))
    groups = [
        {"req_id": req_id, "text": labels.get(req_id, ""), "items": items}
        for req_id, items in sorted(buckets.items())
    ]
    if loose:
        groups.append({"req_id": None, "text": "", "items": loose})
    return groups


def render_marked_sentences(
    sentences: list[dict[str, Any]],
    *,
    section_key: str = "",
    allow_unpin: bool = False,
    state: dict[str, Any] | None = None,
) -> None:
    for group_index, (mark, group) in enumerate(group_by_mark(sentences)):
        body = "  \n".join(item.get("text", "") for item in group)
        if mark == "user":
            pinned = [item for item in group if item.get("pinned")]
            label_line = USER_BLOCK_LABEL + (" · 📌 đã ghim" if pinned else "")
            # icon phải là emoji thật — Streamlit từ chối "✎" (ký tự dingbat).
            # Giữ ✎ trong nhãn chữ để khớp ký hiệu dùng ở file export.
            st.warning(f"{body}\n\n**{label_line}**", icon="✏️")
            if allow_unpin:
                for item_index, item in enumerate(pinned):
                    st.button(
                        f"📌 Bỏ ghim: {item['text'][:36]}…",
                        key=f"unpin_{section_key}_{group_index}_{item_index}",
                        on_click=unpin_sentence,
                        args=(section_key, item["text"]),
                        help=(
                            "Bỏ ghim để lượt chat sau được phép sửa hoặc xoá "
                            "câu này."
                        ),
                    )
        elif mark == "edited":
            st.markdown(
                f"> {body}\n>\n> *{EDITED_LABEL}*"
            )
        else:
            for item_index, item in enumerate(group):
                render_source_lookup(
                    state,
                    item.get("text", ""),
                    key=f"look_{section_key}_{group_index}_{item_index}",
                )


def render_full_proposal(state: dict[str, Any], *, allow_unpin: bool = False) -> None:
    """Toàn văn hồ sơ của một phiên bản, đủ bôi màu."""
    for index, section in enumerate(state.get("sections", []), start=1):
        st.markdown(f"**{index}. {section['title_ja']}**")
        render_marked_sentences(
            section["sentences"],
            section_key=section["key"],
            allow_unpin=allow_unpin,
        )


def split_translated_sections(translated: str) -> list[list[str]]:
    """Cắt bản dịch thành từng mục, dùng dòng tiêu đề `N. …` làm mốc.

    Dịch **một lần cho cả hồ sơ** rồi cắt, thay vì gọi LLM cho từng mục: rẻ hơn
    nhiều lần và giữ được giọng văn nhất quán giữa các mục.
    """
    blocks = split_translation(translated)
    lines = blocks[0]["lines"] if blocks else []
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if is_translation_heading(line):
            current = [line]
            sections.append(current)
        elif current is not None:
            current.append(line)
    return sections


def translate_proposal(state: dict[str, Any]) -> str:
    """Bản dịch của ĐÚNG phiên bản đang xem. Cache theo nội dung nên đổi phiên
    bản là dịch lại, còn quay lại bản cũ thì trúng cache."""
    body = state.get("proposal", "")
    content = f"[PROPOSAL]\n{body}"
    key_material = "\n".join(
        (
            content,
            LLM_MODEL or "",
            PROMPT_VERSION,
            TRANSLATION_EFFORT,
            TRANSLATION_SYSTEM_PROMPT,
        )
    )
    content_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return translate_once(content_hash, content)


def _user_count(section: dict[str, Any]) -> int:
    return sum(
        1 for item in section.get("sentences", []) if item.get("origin") == "user"
    )


def align_translation(
    sentences: list[dict[str, Any]], vi_lines: list[str]
) -> list[list[str]]:
    """Chia các dòng dịch của một mục cho từng câu tiếng Nhật.

    Trường hợp thường gặp là **khớp 1:1** — model dịch từng câu một và bộ tách
    giữ nguyên mỗi câu một dòng. Khi số dòng lệch (model gộp hoặc tách câu) thì
    chia đều theo tỉ lệ: sai vài dòng trong một mục thì vẫn đối chiếu được, còn
    đoán bừa từng câu thì đặt nhãn "người dùng bổ sung" sai chỗ — tệ hơn nhiều.
    """
    body = [line for line in vi_lines[1:] if line.strip()]
    count = len(sentences)
    if count == 0:
        return []
    if len(body) == count:
        return [[line] for line in body]
    chunks: list[list[str]] = [[] for _ in range(count)]
    if not body:
        return chunks
    for index, line in enumerate(body):
        chunks[min(index * count // len(body), count - 1)].append(line)
    return chunks


def render_marked_block(
    mark: str,
    lines: list[str],
    *,
    label_suffix: str = "",
) -> None:
    """Một khối văn bản với đúng kiểu đánh dấu — dùng cho CẢ hai cột ngôn ngữ."""
    body = "  \n".join(line for line in lines if line.strip())
    if not body:
        st.caption("*(chưa có bản dịch cho phần này)*")
        return
    if mark == "user":
        st.warning(f"{body}\n\n**{USER_BLOCK_LABEL}{label_suffix}**", icon="✏️")
    elif mark == "edited":
        st.markdown(f"> {body}\n>\n> *{EDITED_LABEL}*")
    else:
        for line in lines:
            if line.strip():
                st.write(line)


def render_requirement_heading(group: dict[str, Any]) -> None:
    """Nhãn yêu cầu đứng trên cụm câu đáp nó."""
    if group["req_id"] is None:
        st.caption("Câu chung của mục — không gắn yêu cầu cụ thể")
        return
    st.markdown(f"`{group['req_id']}` **{group['text']}**")


def render_bilingual_proposal(
    state: dict[str, Any],
    *,
    key_prefix: str,
    allow_unpin: bool = False,
) -> None:
    """Hồ sơ tiếng Nhật, kèm bản dịch song song theo TỪNG MỤC khi được bật.

    Trong mỗi mục, câu được gom theo YÊU CẦU RFP mà nó đáp (2.1, 2.2…), và yêu
    cầu chưa có câu nào đáp vẫn hiện — chỗ trống mới là thứ người rà soát cần
    thấy, ẩn đi thì hồ sơ đọc như đã đủ.

    Dịch **theo yêu cầu**: chỉ gọi LLM khi người dùng bật công tắc. Cache băm
    nội dung lo phần trùng lặp, nên bật/tắt hay quay lại một phiên bản cũ đều
    không tốn thêm lệnh gọi.
    """
    sections = state.get("sections", [])
    heading_column, toggle_column = st.columns([3, 2], vertical_alignment="center")
    with heading_column:
        st.subheader("Chi tiết")
    with toggle_column:
        show = st.toggle(
            "Hiện bản dịch tiếng Việt",
            key=f"{key_prefix}_bilingual",
            help="Dịch bản đang hiển thị để đối chiếu. Bản nộp vẫn là bản tiếng Nhật.",
        )
    render_mark_legend(
        [
            sentence
            for section in sections
            for sentence in section.get("sentences", [])
        ]
    )

    translated_sections: list[list[str]] = []
    if show:
        with st.spinner("Đang dịch bản đang hiển thị…"):
            try:
                translated_sections = split_translated_sections(
                    translate_proposal(state)
                )
            except LLMUnavailable as error:
                st.warning(
                    f"Chưa dịch được: nhà cung cấp LLM không phản hồi sau "
                    f"{error.attempts} lần gọi. Bản tiếng Nhật vẫn hiển thị bình thường."
                )
                show = False

    for index, section in enumerate(sections):
        sentences = section.get("sentences", [])
        status = section.get("status")
        vi_lines = translated_sections[index] if index < len(translated_sections) else []
        title = (
            f"**{index + 1}. {section['title_ja']}** "
            f"{SECTION_STATUS_ICON.get(status, '')}"
        )
        status_help = label(SECTION_STATUS_VI, status)

        if show:
            # Tên mục tiếng Việt nằm ở CỘT PHẢI, thẳng hàng với tên tiếng Nhật
            # bên trái — cùng quy ước với phần thân.
            title_left, title_right = st.columns(2)
            with title_left:
                st.markdown(title, help=status_help)
            with title_right:
                st.markdown(f"**{vi_lines[0] if vi_lines else section['title_vi']}**")
        else:
            st.markdown(title, help=status_help)

        # Ghi chú nói về CẢ MỤC, không thuộc bên nào, nên chiếm hết bề ngang.
        render_section_note(state, section)

        chunks = align_translation(sentences, vi_lines) if show else []
        for group in requirement_groups(state, section):
            render_requirement_heading(group)
            if not group["items"]:
                st.caption("⛔ Chưa có câu nào đáp yêu cầu này.")
                continue
            group_sentences = [sentence for _, sentence in group["items"]]
            if not show:
                render_marked_sentences(
                    group_sentences,
                    section_key=section["key"],
                    allow_unpin=allow_unpin,
                    state=state,
                )
                continue
            # Hai cột: chia theo TỪNG NHÓM đánh dấu, nên vùng vàng "người dùng
            # bổ sung" nằm cùng hàng ở cả hai ngôn ngữ.
            position = 0
            for group_index, (mark, marked) in enumerate(
                group_by_mark(group_sentences)
            ):
                indexes = [
                    group["items"][offset][0]
                    for offset in range(position, position + len(marked))
                ]
                position += len(marked)
                vi_chunk = [
                    line
                    for offset in indexes
                    if offset < len(chunks)
                    for line in chunks[offset]
                ]
                left, right = st.columns(2)
                with left:
                    render_marked_sentences(
                        marked,
                        section_key=f"{section['key']}_{group['req_id']}_{group_index}",
                        allow_unpin=allow_unpin and mark == "user",
                        state=state,
                    )
                with right:
                    render_marked_block(mark, vi_chunk)
        st.divider()


def render_mark_legend(sentences: list[dict[str, Any]]) -> None:
    marks = {sentence_mark(item) for item in sentences}
    if marks <= {"plain"}:
        return
    st.caption(MARK_LEGEND)


def render_proposal(state: dict[str, Any]) -> None:
    render_proposal_body(state)


def render_proposal_body(state: dict[str, Any]) -> None:
    render_draft_banner(state)
    st.success(sentence_breakdown(state))
    # KẾT QUẢ trước, CHI TIẾT sau: người mở tab cần thấy ngay bức tranh tổng —
    # mục nào đủ căn cứ, mục nào cần người bổ sung — rồi mới đọc văn bản.
    st.subheader("Kết quả")
    st.markdown(f"**{section_summary(state)}**")
    render_confidence(state)
    render_mapping(state)
    st.divider()

    render_bilingual_proposal(state, key_prefix="detail")
    render_version_history()
    render_download(state)


def count_phrase(total: int, parts: list[tuple[int, str]], *, unit: str) -> str:
    """Đếm tường minh từng loại, ẩn thành phần bằng 0.

    Dạng "9/10 … (phần còn lại là …)" bắt người đọc tự trừ để biết phần còn lại
    là bao nhiêu và gồm những gì — và khi có ba loại thì phép trừ đó sai. Liệt kê
    thẳng từng loại thì không phải suy luận gì.

    Tiếng Việt không biến đổi danh từ theo số, nên `unit` dùng nguyên cho cả 1
    lẫn nhiều; tham số vẫn để lộ ra để chỗ gọi tự quyết đơn vị.
    """
    if total <= 0:
        return f"Chưa có {unit} nào"
    shown = [f"{count} {text}" for count, text in parts if count]
    if not shown:
        return f"{total} {unit}"
    return f"{total} {unit}: " + " · ".join(shown)


def sentence_breakdown(state: dict[str, Any]) -> str:
    """Tóm tắt nguồn gốc các câu trong bản ĐANG xem.

    Đếm thẳng từ `sections` chứ không dùng `trace.grounding`: trace đếm
    "origin != bridge" là grounded, nên từ v1.7 nó tính cả câu người dùng bổ
    sung vào nhóm có nguồn — đúng cái mà nhãn v1.7 sinh ra để phân biệt.
    """
    sentences = [
        sentence
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
    ]
    origins = [sentence.get("origin") for sentence in sentences]
    return count_phrase(
        len(sentences),
        [
            (
                sum(1 for origin in origins if origin in ("capability", "precedent")),
                "câu truy được về nguồn cụ thể",
            ),
            (
                sum(1 for origin in origins if origin == "user"),
                "câu người dùng bổ sung (chưa kiểm chứng)",
            ),
            (
                sum(1 for origin in origins if origin == "bridge"),
                "câu nối (không mang thông tin sự thật)",
            ),
        ],
        unit="câu",
    )


def requirement_cell(info: dict[str, Any]) -> str:
    """Ô "Đáp ứng yêu cầu RFP" cho ba ca khác nhau.

    `0/0` đọc thành "đáp ứng kém" trong khi thật ra mục đó **không được giao**
    yêu cầu nào (会社概要 là mục cố định) — nên ca đó phải nói bằng chữ, không
    bằng phân số.
    """
    total = info.get("total_requirements", 0)
    covered = info.get("covered_requirements", 0)
    if total == 0:
        return "— không có yêu cầu"
    missing = info.get("missing_requirements") or []
    text = f"Đáp ứng {covered}/{total}"
    if missing:
        return f"{text} — thiếu: {' · '.join(missing)}"
    return text


def render_confidence(state: dict[str, Any]) -> None:
    """Điểm tin cậy + tầng của cả hồ sơ và từng mục.

    Điểm quy từ bằng chứng đã có, không có lệnh gọi LLM nào để chấm điểm — nên
    con số này tất định, chạy lại cùng state ra cùng kết quả.
    """
    overall = state.get("confidence")
    if not overall:
        return
    tier = overall.get("tier", "T3")
    st.markdown(
        f"{TIER_ICON.get(tier, '')} **{label(TIER_VI, tier)}** · độ tin cậy "
        f"**{overall.get('score', 0):.2f}** — {label(TIER_HINT, tier)}"
    )
    st.caption(
        "Điểm quy từ căn cứ sẵn có (kết quả kiểm chứng · điểm truy hồi · mức đáp ứng "
        "yêu cầu). Hồ sơ lấy điểm của **mục yếu nhất** — một mục hỏng thì cả hồ "
        "sơ chưa nộp được."
    )
    rows = [
        {
            "Mục": section["title_ja"],
            "Tầng": (
                f"{TIER_ICON.get(section['confidence']['tier'], '')} "
                f"{label(TIER_VI, section['confidence']['tier'])}"
            ),
            "Độ tin cậy": f"{section['confidence']['score']:.2f}",
            "Đáp ứng yêu cầu RFP": requirement_cell(section["confidence"]),
        }
        for section in state.get("sections", [])
        if section.get("confidence")
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "x/y = trong y yêu cầu RFP giao cho mục này, x đã có căn cứ thật."
        )
    render_requirement_table(state)


def render_requirement_table(state: dict[str, Any]) -> None:
    """Cùng dữ liệu, nhưng một dòng cho MỘT yêu cầu RFP.

    Bảng theo mục chỉ nói "đáp ứng 1/4" — muốn biết ba yêu cầu còn lại là gì
    thì phải đọc phần đuôi của ô đó. Bảng này gọi tên từng yêu cầu và nói ngay
    có mấy câu đang đáp nó.
    """
    rows = [
        {
            "Mục": section["title_ja"],
            "Yêu cầu": group["req_id"] or "—",
            "Nội dung yêu cầu": group["text"] or "(câu chung của mục)",
            "Số câu đáp": len(group["items"]),
            "Tình trạng": (
                "🟢 đã có câu đáp" if group["items"] else "🔴 chưa có căn cứ"
            ),
        }
        for section in state.get("sections", [])
        for group in requirement_groups(state, section)
        if group["req_id"] is not None
    ]
    if not rows:
        return
    with st.expander(f"Theo từng yêu cầu RFP — {len(rows)} yêu cầu", expanded=False):
        st.dataframe(rows, width="stretch", hide_index=True)


def render_related_sources(state: dict[str, Any], section: dict[str, Any]) -> None:
    """Tầng T3: nguồn 'có thể liên quan' cho mục thiếu căn cứ.

    Đây KHÔNG phải câu trả lời — chúng trượt vì không đủ liên quan hoặc bị cắt
    ở bước chọn. Hiện ra để người bổ sung có chỗ bắt đầu, và nhãn phải nói
    thẳng điều đó chứ không để người đọc tưởng hệ thống đã trả lời.
    """
    chapters = {chapter["id"]: chapter for chapter in state.get("chapters", [])}
    items: list[dict[str, Any]] = []
    for chapter_id in section.get("source_chapters", []):
        chapter = chapters.get(chapter_id)
        if not chapter:
            continue
        for item in related_sources(chapter.get("retrieval", {})):
            items.append({**item, "_chapter": chapter_id})
    # Điểm 0 nghĩa là KHÔNG liên quan chút nào — liệt kê chúng dưới nhãn "có
    # thể liên quan" là mời người rà soát đi tra một chỗ chắc chắn không có gì.
    items = [
        item
        for item in items
        if float((item.get("scores") or {}).get("rerank", 0.0)) > 0
    ]
    if not items:
        return
    items.sort(key=lambda item: -float((item.get("scores") or {}).get("rerank", 0.0)))
    items = items[:RELATED_SOURCES_TOP_N]
    with st.expander(
        f"🔎 {len(items)} câu trong hồ sơ cũ có nhắc tới chuyện tương tự",
        expanded=False,
    ):
        st.caption(
            "**Không phải câu trả lời.** Hệ thống đã loại chúng ở bước chọn "
            "nguồn vì chưa đủ sát yêu cầu, và **không** dùng để viết hồ sơ. "
            "Liệt kê ở đây để người bổ sung có chỗ bắt đầu tra. Điểm càng cao "
            "thì càng sát yêu cầu (thang 0–1)."
        )
        st.dataframe(
            [
                {
                    "Chương": item["_chapter"],
                    "Mã câu": item.get("sent_id"),
                    "Câu (tiếng Nhật)": item.get("text"),
                    "Điểm sát yêu cầu": round(
                        float((item.get("scores") or {}).get("rerank", 0.0)), 3
                    ),
                }
                for item in items
            ],
            width="stretch",
            hide_index=True,
        )


def section_summary(state: dict[str, Any]) -> str:
    """Một dòng tóm tắt tình trạng các mục — thứ đọc trước tiên."""
    sections = state.get("sections", [])
    counts: dict[str, int] = defaultdict(int)
    for section in sections:
        counts[section.get("status") or "—"] += 1
    parts = [
        f"{count}/{len(sections)} mục {label(SECTION_STATUS_VI, status).lower()}"
        if status == "OK"
        else f"{count} mục {label(SECTION_STATUS_VI, status).lower()}"
        for status, count in counts.items()
    ]
    return " · ".join(parts) if parts else "Chưa có mục nào"


def requirement_coverage(
    state: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sources_by_req: dict[str, list[str]] = defaultdict(list)
    for section in state.get("sections", []):
        for sentence in section["sentences"]:
            # Câu người dùng bổ sung không có source_id nhưng VẪN phải hiện
            # trong bảng đối chiếu — nếu không, một yêu cầu được đáp ứng bằng
            # nội dung người dùng tự viết sẽ trông như chưa ai đụng tới.
            if sentence.get("source_id") is None and sentence.get("origin") != "user":
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
    summary = count_phrase(
        len(rows),
        [
            (covered, "yêu cầu đã có căn cứ"),
            (len(missing), "yêu cầu chưa có căn cứ"),
        ],
        unit="yêu cầu",
    )
    if missing:
        st.warning(f"{summary} — phần chưa có căn cứ cần người bổ sung trước khi nộp.")
    else:
        st.success(f"{summary} — mọi yêu cầu của RFP đều đã có căn cứ.")
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


def _strategy_by_section(state: dict[str, Any]) -> dict[str, str]:
    """Mỗi mục lấy nguồn bằng cách nào — suy từ các chương nguồn của nó."""
    by_chapter = {
        chapter["id"]: chapter.get("retrieval", {}).get("source_strategy")
        for chapter in state.get("chapters", [])
    }
    out: dict[str, str] = {}
    for section in state.get("sections", []):
        names = [
            by_chapter[chapter_id]
            for chapter_id in section.get("source_chapters", [])
            if by_chapter.get(chapter_id)
        ]
        out[section["key"]] = " · ".join(
            dict.fromkeys(label(SOURCE_STRATEGY_VI, name) for name in names)
        ) or UNKNOWN_DASH
    return out


UNKNOWN_DASH = "—"


def sentence_rows(state: dict[str, Any]) -> list[dict[str, str]]:
    """Giữ nguyên giá trị gốc ở `_origin`/`_verdict` để bộ lọc so sánh đúng."""
    strategies = _strategy_by_section(state)
    return [
        {
            "Mục": section["title_ja"],
            "Cách lấy nguồn": strategies.get(section["key"], UNKNOWN_DASH),
            "Câu (tiếng Nhật)": sentence["text"],
            "Nguồn": label(ORIGIN_VI, sentence["origin"]),
            "Mã nguồn": sentence["source_id"] or "—",
            "Kiểm chứng": (
                f"{VERDICT_ICON.get(sentence['verdict'], '')} "
                f"{label(VERDICT_VI, sentence['verdict'])}"
            ).strip(),
            "_origin": sentence["origin"],
            "_verdict": sentence["verdict"],
            "_req_ids": list(sentence.get("req_ids") or []),
        }
        for section in state.get("sections", [])
        for sentence in section["sentences"]
    ]


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


# Sơ đồ luồng chạy: 4 ô một hàng, 2 hàng, nối bằng mũi tên. Trạng thái nằm
# NGAY TRONG ô ("đang chạy", "xong", "bỏ qua") nên không cần bảng chú giải
# riêng — người xem không phải đối chiếu icon với một dòng chữ ở cuối trang.
STAGE_TONE = {
    # (viền, nền, màu chữ trạng thái)
    "pending": ("#2A323C", "#161B22", "#6B7684"),
    "running": ("#D29922", "#2A2416", "#E6C86A"),
    "completed": ("#3FB950", "#16281F", "#7EE29A"),
    "skipped": ("#4B7BA8", "#16222E", "#8FB8DC"),
    "blocked": ("#F85149", "#2A1A19", "#F0A8A2"),
    "failed": ("#F85149", "#2A1A19", "#F0A8A2"),
}

_FLOW_BOX_W = 300
_FLOW_BOX_H = 62
_FLOW_GAP = 40
_FLOW_COLS = 4
_FLOW_X0 = 24
_FLOW_ROW_Y = (26, 152)


def _flow_box_xy(index: int) -> tuple[float, float]:
    row, column = divmod(index, _FLOW_COLS)
    return (
        _FLOW_X0 + column * (_FLOW_BOX_W + _FLOW_GAP),
        _FLOW_ROW_Y[row],
    )


def pipeline_svg(statuses: dict[str, str]) -> str:
    """Sơ đồ luồng chạy cho lượt này, trạng thái in thẳng trong từng ô."""
    out = [
        '<svg viewBox="0 0 1400 240" width="100%" '
        f'style="background:{_FLOW_BG};border-radius:12px">',
        _markers(),
    ]
    for index, name in enumerate(FLOW_STAGES):
        status = statuses.get(name, "pending")
        stroke, fill, text_color = STAGE_TONE.get(status, STAGE_TONE["pending"])
        x, y = _flow_box_xy(index)
        out.append(
            f'<rect x="{x}" y="{y}" width="{_FLOW_BOX_W}" height="{_FLOW_BOX_H}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        )
        out.append(
            _svg_text(
                x + _FLOW_BOX_W / 2,
                y + 26,
                f"{index + 1}. {STAGE_VI[name]}",
                size=13,
                weight="600",
            )
        )
        out.append(
            _svg_text(
                x + _FLOW_BOX_W / 2,
                y + 45,
                label(STAGE_STATUS_VI, status),
                size=11,
                color=text_color,
            )
        )
        if index + 1 >= len(FLOW_STAGES):
            continue
        next_x, next_y = _flow_box_xy(index + 1)
        if next_y == y:
            out.append(
                _edge(
                    f"M{x + _FLOW_BOX_W},{y + _FLOW_BOX_H / 2} H{next_x}",
                    key=f"step:{name}",
                )
            )
        else:
            # Xuống hàng: đi vòng qua khoảng giữa hai hàng rồi vào ô đầu hàng
            # dưới. Nối thẳng từ ô cuối hàng trên sang ô đầu hàng dưới sẽ cắt
            # ngang cả sơ đồ.
            middle = (y + _FLOW_BOX_H + next_y) / 2
            out.append(
                _edge(
                    f"M{x + _FLOW_BOX_W / 2},{y + _FLOW_BOX_H} "
                    f"V{middle - 10} Q{x + _FLOW_BOX_W / 2},{middle} "
                    f"{x + _FLOW_BOX_W / 2 - 10},{middle} "
                    f"H{next_x + _FLOW_BOX_W / 2 + 10} "
                    f"Q{next_x + _FLOW_BOX_W / 2},{middle} "
                    f"{next_x + _FLOW_BOX_W / 2},{middle + 10} V{next_y}",
                    key=f"step:{name}",
                )
            )
    out.append("</svg>")
    return "".join(out)


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
    *xong* — bên trong nó không phát tín hiệu, nên sơ đồ đứng yên trong lúc
    node đó chạy.
    """
    target.empty()
    statuses = _stage_statuses(state, failure=failure)
    done = sum(1 for value in statuses.values() if value == "completed")

    with target.container():
        st.progress(done / len(FLOW_STAGES), text=f"{done}/{len(FLOW_STAGES)} bước")
        st.markdown(pipeline_svg(statuses), unsafe_allow_html=True)
        present = set(statuses.values())
        if "skipped" in present:
            st.caption(f"{STAGE_STATUS_ICON['skipped']} {SKIP_LEGEND}.")
        if "blocked" in present:
            st.caption(
                f"{STAGE_STATUS_ICON['blocked']} guard chặn xuất bản là hành vi "
                "đúng: thà không nộp còn hơn nộp hồ sơ sai."
            )
        if not failure or not failure.get("message"):
            return
        stage = failure.get("stage")
        if stage and statuses.get(stage) == "blocked":
            # Bị chặn là hành vi ĐÚNG: thà không nộp còn hơn nộp hồ sơ tuyên bố
            # sai chứng chỉ. Không tô như lỗi hệ thống.
            st.warning(failure["message"], icon="🛑")
        else:
            st.error(failure["message"], icon="❌")


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
                            retrieval_stages = chapter.get("retrieval", {}).get(
                                "stages", {}
                            )
                            precedent_skipped = retrieval_stages.get(
                                "query_embed", {}
                            ).get("skipped", False)
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
        attribute = (
            chapter.get("retrieval", {}).get("stages", {}).get("attribute", {})
        )
        for item in chapter["retrieval"]["selected"]:
            scores = item["scores"]
            rows.append(
                {
                    "Chương": chapter["id"],
                    "Mã câu nguồn": item["sent_id"],
                    "Câu nguồn (tiếng Nhật)": item["text"],
                    "Điểm tìm kiếm (từ khoá + ngữ nghĩa)": round(
                        scores.get("hybrid", 0.0), 4
                    ),
                    "Điểm sau ưu tiên ngành/mục": round(
                        scores.get("rerank", 0.0), 4
                    ),
                    "Đáp ứng yêu cầu": ", ".join(
                        attribute.get("covered_req_ids", [])
                    ) or "—",
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


BRANCH_ORDER = {"human": 0, "warn": 1, "auto": 2}


def requirement_journey(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Hành trình của TỪNG yêu cầu RFP — kể chuyện theo yêu cầu, không theo node.

    Mọi thứ ở đây suy từ state đã có: `source_strategy` của chương, điểm tin cậy
    của câu dẫn, và danh sách nguồn bị loại. Không đo thêm, không gọi thêm.

    Nhánh của một yêu cầu:
      chưa có câu nào dẫn nó          -> 🔴 chuyển người
      có câu dẫn, mục đạt tầng T1     -> 🟢 tự trả lời
      có câu dẫn, mục ở T2/T3         -> 🟡 cảnh báo
    """
    from rfp.confidence import rerank_scores, sentence_confidence

    scores = rerank_scores(state)
    chapters = {chapter["id"]: chapter for chapter in state.get("chapters", [])}
    section_by_chapter: dict[str, dict[str, Any]] = {}
    for section in state.get("sections", []):
        for chapter_id in section.get("source_chapters", []):
            section_by_chapter[chapter_id] = section

    rows: list[dict[str, Any]] = []
    for chapter in state.get("chapters", []):
        retrieval = chapter.get("retrieval", {})
        strategy = retrieval.get("source_strategy")
        section = section_by_chapter.get(chapter["id"], {})
        tier = (section.get("confidence") or {}).get("tier")
        for requirement in chapter.get("requirements", []):
            req_id = requirement["req_id"]
            supporting = [
                sentence
                for sentence in section.get("sentences", [])
                if req_id in sentence.get("req_ids", [])
                and sentence.get("origin") in ("capability", "precedent")
            ]
            if not supporting:
                branch = "human"
                score = 0.0
                sources = [
                    item.get("sent_id", "")
                    for item in related_sources(retrieval)
                ]
                source_text = (
                    f"🔎 {' · '.join(sources)} (không phải câu trả lời)"
                    if sources
                    else "— chưa có nguồn nào"
                )
            else:
                # Nhánh theo bằng chứng của CHÍNH yêu cầu này, không theo tầng
                # của mục: một yêu cầu đã có precedent chống lưng không nên bị
                # tô vàng chỉ vì yêu cầu anh em trong cùng mục còn thiếu.
                branch = (
                    "auto"
                    if any(item.get("origin") == "precedent" for item in supporting)
                    else "warn"
                )
                score = max(
                    sentence_confidence(item, rerank_by_source=scores) or 0.0
                    for item in supporting
                )
                source_text = " · ".join(
                    dict.fromkeys(
                        str(item.get("source_id")) for item in supporting
                    )
                )
            rows.append(
                {
                    "req_id": req_id,
                    "text": requirement["text"],
                    "strategy": strategy,
                    "score": round(score, 2),
                    "branch": branch,
                    "sources": source_text,
                }
            )
    # Đỏ trước, vàng giữa, xanh cuối: người đọc thấy ngay chỗ cần đến mình.
    rows.sort(key=lambda row: (BRANCH_ORDER[row["branch"]], row["req_id"]))
    return rows


# Bảng màu lấy thẳng từ dark theme của app (.streamlit/config.toml) — sơ đồ
# nằm trong trang tối nên phải cùng nền, một thẻ trắng giữa trang tối đọc như
# ảnh dán vào chứ không như một phần của app.
_FLOW_BG = "#12161C"
_FLOW_CARD = "#1B2129"
_FLOW_LINE = "#3A4552"
_FLOW_TEXT = "#E6EAF0"
_FLOW_SUB = "#8B97A6"

# Ba kết quả cuối, gọi bằng đúng việc người dùng phải làm với nó.
# (viền, nền, tiêu đề, mô tả)
_BRANCH_STYLE = {
    "auto": (
        "#3FB950", "#16281F",
        "Dùng được ngay",
        "có nguồn dẫn · vẫn nên liếc lại nguồn",
    ),
    "warn": (
        "#D29922", "#2A2416",
        "Dùng được, phải kiểm lại",
        "chỉ dựa vào bảng năng lực công ty",
    ),
    "human": (
        "#F85149", "#2A1A19",
        "Người phải bổ sung",
        "chưa tìm được căn cứ nào cho yêu cầu này",
    ),
}
_EDGE_LABEL = {
    "auto": "điểm cao",
    "warn": "điểm giữa",
    "human": "điểm thấp",
}

# Cách hệ thống THẬT SỰ tìm căn cứ cho một yêu cầu. Không phải chuỗi "thử lần
# lượt": kênh đối chiếu bảng năng lực chạy trước và có thể phủ đủ luôn, còn khi
# phải tìm trong hồ sơ cũ thì từ khoá và ngữ nghĩa chạy CÙNG LÚC rồi chấm điểm
# lại — vẽ thành hai bước nối tiếp là mô tả sai hệ thống.
_TIER_CHAIN = (
    (
        "capability-only",
        "Đối chiếu bảng năng lực công ty",
        "khớp thẳng, không cần tìm ở đâu nữa",
    ),
    (
        "hybrid-bm25+dense",
        "Tìm trong hồ sơ thầu cũ",
        "từ khoá + ngữ nghĩa cùng lúc, rồi chọn câu sát nhất",
    ),
    (
        "dense-only",
        "Tìm bằng ngữ nghĩa (khi tắt từ khoá)",
        "chỉ dùng khi cấu hình tắt tìm theo từ khoá",
    ),
    (
        "fallback-listing",
        "Không tìm được — chỉ liệt kê nguồn gần đúng",
        "KHÔNG coi đó là câu trả lời",
    ),
)


_MARKER_IDS = {
    _FLOW_LINE: "ar-line",
    "#3FB950": "ar-auto",
    "#D29922": "ar-warn",
    "#F85149": "ar-human",
}


def _markers() -> str:
    """Một marker cho mỗi màu cạnh.

    Dùng chung một marker đen cho mọi cạnh thì đầu mũi tên trên cạnh dày trông
    như vệt mực đè lên hộp — lỗi nhìn thấy ngay khi cạnh sankey dày lên.
    """
    return "<defs>" + "".join(
        f'<marker id="{name}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        for color, name in _MARKER_IDS.items()
    ) + "</defs>"


def _esc(text):
    return html.escape(str(text))


def _svg_text(x, y, text, *, size=12, color=None, weight="400", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" fill="{color or _FLOW_TEXT}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        f'font-family="Segoe UI,Helvetica,Arial,sans-serif">{_esc(text)}</text>'
    )


def _card(x, y, w, h, title, sub, *, fill=None, stroke=None, active=True,
          title_color=None):
    # Hộp không dùng tới lượt này vẫn phải đọc được. Trên nền tối, mờ tới 0.3
    # là chữ biến mất hẳn chứ không còn là "mờ đi".
    opacity = 1.0 if active else 0.42
    parts = [
        f'<g opacity="{opacity:.2f}">',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{fill or _FLOW_CARD}" stroke="{stroke or _FLOW_LINE}" '
        f'stroke-width="1.4"/>',
        _svg_text(x + w / 2, y + (24 if sub else h / 2 + 4), title, size=12.5,
                  weight="600", color=title_color),
    ]
    if sub:
        parts.append(_svg_text(x + w / 2, y + 41, sub, size=10, color=_FLOW_SUB))
    parts.append("</g>")
    return "".join(parts)


# Nét nối: MỘT độ dày cho mọi cạnh. Bản trước vẽ dày theo lưu lượng (sankey);
# nhìn ra chỉ thấy nét to nét nhỏ so le chứ không đọc được thành thông tin, mà
# số yêu cầu thì đã in sẵn trên từng hộp. Chỉ đường đang được tô sáng mới dày
# hơn — đó là khác biệt CÓ nghĩa với người xem.
_W_EDGE = 1.7
_W_LIT = 2.8


def _edge(d, *, color=None, active=True, key="", lit=False, head=True):
    """Một nét nối. `d` dựng sẵn để điểm chạm rơi đúng giữa mép hộp."""
    color = color or _FLOW_LINE
    marker = (
        f' marker-end="url(#{_MARKER_IDS.get(color, "ar-line")})"' if head else ""
    )
    return (
        f'<path d="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{_W_LIT if lit else _W_EDGE:.1f}" '
        f'opacity="{0.95 if active else 0.15:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker} '
        f'data-edge="{key}"/>'
    )


def _elbow(x1, y1, corner_x, y2, x2, radius=10):
    """Gấp khúc ngang → dọc → ngang, bo góc.

    Dùng gấp khúc chứ không dùng đường cong tự do: đường cong nối hai hộp lệch
    hàng nhau thì đầu mút đâm xiên vào mép, nhìn như trượt ra ngoài hộp. Gấp
    khúc thì luôn chạm vuông góc vào giữa mép.
    """
    if abs(y2 - y1) < 1:
        return f"M{x1},{y1} H{x2}"
    step = radius if y2 > y1 else -radius
    return (
        f"M{x1},{y1} H{corner_x - radius} "
        f"Q{corner_x},{y1} {corner_x},{y1 + step} "
        f"V{y2 - step} Q{corner_x},{y2} {corner_x + radius},{y2} H{x2}"
    )


def flow_svg(rows, *, blocked, highlight=None, checked=0, reviewed=0):
    """Sơ đồ ĐÚNG luồng hệ thống cho lượt chạy này, số thật trên từng hộp.

    Vẽ theo thứ tự việc mà code thật sự làm: tìm căn cứ -> viết câu gắn nguồn ->
    kiểm lại từng câu -> soi văn bản -> chấm độ tin cậy -> ba kết quả -> chặn
    an toàn -> hồ sơ.

    Bố cục neo vào ba hàng cố định để mọi điểm chạm thẳng hàng nhau. Chỗ nhiều
    nhánh chụm về một đích thì gom qua một trục dọc rồi mới đi tiếp — bốn mũi
    tên cùng đâm vào một điểm là chỗ trông rối nhất của bản trước.
    """
    row_top, row_mid, row_low = 89, 199, 309
    total = len(rows) or 1
    by_tier = {key: 0 for key, _, _ in _TIER_CHAIN}
    for row in rows:
        by_tier[row["strategy"]] = by_tier.get(row["strategy"], 0) + 1
    by_branch = {name: 0 for name in ("auto", "warn", "human")}
    for row in rows:
        by_branch[row["branch"]] += 1
    picked = next((row for row in rows if row["req_id"] == highlight), None)

    out = [
        f'<svg viewBox="0 0 1480 400" width="100%" '
        f'style="background:{_FLOW_BG};border-radius:12px">',
        _markers(),
    ]

    # 1. Đầu vào
    out.append(_card(20, 168, 110, 62, f"{len(rows)} yêu cầu", "tách từ RFP"))
    out.append(_edge(f"M130,{row_mid} H176", key="in"))

    # 2. Tìm căn cứ — bốn cách, gom vào một trục rồi mới sang bước sau
    out.append(
        # Khối canh giữa đúng hàng giữa (199) để mũi tên từ đầu vào chạm vào
        # giữa mép trái chứ không lệch xuống vài pixel.
        f'<rect x="176" y="40" width="330" height="318" rx="12" fill="none" '
        f'stroke="{_FLOW_LINE}" stroke-width="1.4"/>'
    )
    out.append(_svg_text(341, 64, "① Tìm căn cứ cho từng yêu cầu", size=12.5,
                         weight="600"))
    tier_y = (80, 148, 216, 284)
    for index, (key, title, sub) in enumerate(_TIER_CHAIN):
        count = by_tier.get(key, 0)
        on_path = bool(picked and picked["strategy"] == key)
        out.append(
            _card(194, tier_y[index], 294, 54, f"{title} · {count}", sub,
                  stroke=_BRANCH_STYLE["auto"][0] if on_path else None,
                  active=count > 0)
        )
        out.append(
            _edge(f"M488,{tier_y[index] + 27} H524",
                  color=_BRANCH_STYLE["auto"][0] if on_path else None,
                  active=count > 0 and (picked is None or on_path),
                  lit=on_path, head=False, key=f"out:{key}")
        )
    out.append(
        _edge(f"M524,{tier_y[3] + 27} V99 Q524,{row_top} 534,{row_top} H566",
              key="to-generate")
    )
    out.append(
        _svg_text(176, 376,
                  "hồ sơ ghi lại mục nào tìm bằng cách nào — xem cột "
                  "\u201cCách lấy nguồn\u201d",
                  size=10, color=_FLOW_SUB, anchor="start")
    )

    # 3. Viết câu -> kiểm lại -> soi văn bản: một cột dọc, chạm giữa mép
    out.append(_card(566, 62, 210, 54, "② Viết câu, gắn nguồn",
                     "mỗi câu một mã nguồn cụ thể"))
    out.append(_edge("M671,116 V152", key="gen"))
    out.append(_card(566, 152, 210, 54, "③ Kiểm lại từng câu",
                     f"{checked} câu · đối chiếu bằng chứng"))
    out.append(_edge("M671,206 V242", key="check"))
    out.append(_card(566, 242, 210, 54, "④ Soi lại văn bản",
                     f"{reviewed} vòng · bố cục và văn phong"))
    out.append(_edge(_elbow(776, 269, 806, row_mid, 816), key="review"))

    # 4. Chấm điểm
    out.append(
        _card(816, 168, 154, 62, "⑤ Chấm độ tin cậy",
              f"{picked['score']:.2f} · {picked['req_id']}" if picked
              else "0–1 · ghi log mọi lượt")
    )

    # 5. Ba kết quả — toả ra từ một trục, rồi gom lại vào chốt chặn
    out.append(_edge(f"M970,{row_mid} H990", key="to-branch", head=False))
    out.append(_edge(f"M990,{row_top} V{row_low}", key="branch-bus", head=False))
    branch_y = {"auto": 58, "warn": 168, "human": 278}
    for name, y in branch_y.items():
        centre = y + 31
        count = by_branch[name]
        stroke, fill, title, sub = _BRANCH_STYLE[name]
        on_path = bool(picked and picked["branch"] == name)
        active = count > 0 and (picked is None or on_path)
        out.append(
            _edge(f"M990,{centre} H1060", color=stroke, active=active,
                  lit=on_path, key=f"branch:{name}")
        )
        # Nhãn nằm gọn trong đoạn thẳng trục -> hộp, phía trên nét nối.
        out.append(
            _svg_text(1025, centre - 8, _EDGE_LABEL[name], size=9.5, color=stroke)
        )
        out.append(
            _card(1060, y, 246, 62, f"{title} · {count} ({count / total:.0%})",
                  sub, fill=fill, stroke=stroke, active=count > 0,
                  title_color=stroke)
        )
        out.append(
            _edge(f"M1306,{centre} H1326", color=stroke, active=active,
                  lit=on_path, head=False, key=f"guard:{name}")
        )
    out.append(_edge(f"M1326,{row_top} V{row_low}", key="guard-bus", head=False))
    out.append(_edge(f"M1326,{row_mid} H1344", key="to-guard"))
    out.append(
        _svg_text(1060, 376,
                  "câu không đạt điểm cao đều kèm ghi chú phải đối chiếu tài "
                  "liệu gốc",
                  size=10, color=_FLOW_SUB, anchor="start")
    )

    # 6. Chặn an toàn + hồ sơ
    out.append(
        f'<circle cx="1372" cy="{row_mid}" r="28" fill="{_FLOW_CARD}" '
        f'stroke="{_FLOW_LINE}" stroke-width="1.4"/>'
    )
    out.append(_svg_text(1372, 195, "⑥ Chặn", size=10.5, weight="600"))
    out.append(_svg_text(1372, 209, f"{blocked} câu", size=9.5, color=_FLOW_SUB))
    out.append(_edge(f"M1400,{row_mid} H1416", key="publish"))
    out.append(_card(1416, 168, 58, 62, "Hồ sơ", "nháp"))
    out.append("</svg>")
    return "".join(out)


def render_journey_flow(state: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Sơ đồ + chọn một yêu cầu để tô sáng đường đi của đúng nó."""
    ids = [row["req_id"] for row in rows]
    picked = st.selectbox(
        "Tô sáng đường đi của một yêu cầu",
        ["(toàn bộ)"] + ids,
        key="journey_highlight",
    )
    highlight = None if picked == "(toàn bộ)" else picked
    trace = state.get("trace", {})
    checked = sum(
        1
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
        if sentence.get("verdict")
    )
    reviewed = (trace.get("review") or {}).get("rounds", 0)
    st.markdown(
        flow_svg(
            rows,
            blocked=len(trace.get("hybrid_blocked", [])),
            highlight=highlight,
            checked=checked,
            reviewed=reviewed,
        ),
        unsafe_allow_html=True,
    )
    if highlight:
        row = next(row for row in rows if row["req_id"] == highlight)
        st.markdown(
            f"**{row['req_id']}** · {row['text']}\n\n"
            f"Tầng trả lời: **{label(STRATEGY_TIER_VI, row['strategy'])}** · "
            f"độ tin cậy **{row['score']:.2f}** · "
            f"{label(REQ_BRANCH_VI, row['branch'])} · "
            f"nguồn dẫn: {row['sources']}"
        )
    st.caption(JOURNEY_LEGEND)


def render_journey_summary(state: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    trace = state.get("trace", {})
    counts = {name: 0 for name in ("auto", "warn", "human")}
    for row in rows:
        counts[row["branch"]] += 1
    total = len(rows) or 1

    columns = st.columns(6)
    for column, key in zip(columns[:3], ("auto", "warn", "human")):
        column.metric(
            label(REQ_BRANCH_VI, key),
            f"{counts[key]}",
            f"{counts[key] / total:.0%}",
            delta_color="off",
        )
    columns[3].metric("Lệnh gọi LLM", trace.get("llm_calls", 0))
    usage = trace.get("usage") or {}
    columns[4].metric(
        "Câu bị chặn vì số liệu lạ",
        len(trace.get("hybrid_blocked", [])),
        help="Câu có chỉ số định lượng không khớp nguyên văn nguồn nào — bị gỡ.",
    )
    columns[5].metric(
        "Nguồn bị loại vì mâu thuẫn",
        len(trace.get("conflicts", [])),
        help="Hai câu nguồn nói khác nhau về cùng một chỉ số; giữ nguồn ưu tiên hơn.",
    )
    seconds = usage.get("seconds")
    tokens = usage.get("tokens_product")
    if seconds is not None or tokens is not None:
        st.caption(
            f"Lượt chạy này: {seconds:.1f}s · {tokens:,} token sản phẩm"
            if seconds is not None and tokens is not None
            else "Lượt chạy này: chưa đo được thời gian/token"
        )


def render_requirement_journey(state: dict[str, Any]) -> None:
    rows = requirement_journey(state)
    if not rows:
        st.info("Chưa có yêu cầu nào để hiển thị.")
        return
    render_journey_summary(state, rows)
    st.subheader("Luồng xử lý lượt chạy này")
    render_journey_flow(state, rows)
    st.subheader("Hành trình từng yêu cầu RFP")
    st.dataframe(
        [
            {
                "Yêu cầu": f"{row['req_id']} · {row['text']}",
                "Tầng trả lời": label(STRATEGY_TIER_VI, row["strategy"]),
                "Độ tin cậy": f"{row['score']:.2f}",
                "Nhánh": label(REQ_BRANCH_VI, row["branch"]),
                "Nguồn dẫn": row["sources"],
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )


def render_trace_details(state: dict[str, Any]) -> None:
    """Khối chi tiết kỹ thuật — gấp sẵn, dưới bảng hành trình.

    Thứ tự có chủ đích: hành trình từng yêu cầu trả lời câu hỏi *"tôi phải làm
    gì tiếp"*, còn các khối dưới đây trả lời *"hệ thống đã chạy thế nào"*. Người
    mở tab cần cái trước, người gỡ lỗi cần cái sau.
    """
    trace = state["trace"]

    with st.expander("Chi tiết kỹ thuật theo node", expanded=False):
        st.caption(
            "Đường đi: "
            + " → ".join(label(STAGE_VI, name) for name in trace.get("path", []))
        )
        holder = st.empty()
        render_pipeline_status(state, holder)

    with st.expander("Số lệnh gọi theo khâu", expanded=False):
        st.dataframe(
            [
                {"Khâu": label(LLM_STAGE_VI, name), "Số lệnh gọi": count}
                for name, count in trace.get("llm_calls_by_stage", {}).items()
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Khâu ghi **0 lệnh gọi** là khâu chạy thuần code/regex — lưới an "
            "toàn (blocklist, lọc rò rỉ, kiểm số liệu) **không tốn tiền LLM**, "
            "nên bật chúng không có lý do gì để tiếc."
        )

    with st.expander("Vòng review", expanded=False):
        render_review_rounds(state)

    with st.expander("Nguồn đã truy xuất", expanded=False):
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


def render_golden_coverage(
    cases: list[GoldenCase], *, added: list[GoldenCase] | None = None
) -> None:
    """Bảng độ phủ. Cột "còn thiếu" gọi tên đích danh, không chỉ nói còn mấy cái.

    Có `added` thì so hai bảng: cột "sau khi thêm" cho thấy lô sắp sinh vá được
    trục nào — người test thấy tác dụng TRƯỚC khi lưu file.
    """
    axes = coverage_report(cases)
    rows = coverage_rows(axes)
    if added:
        after = {axis.key: axis for axis in coverage_report([*cases, *added])}
        for row, axis in zip(rows, axes):
            target = after[axis.key]
            row["sau khi thêm"] = (
                f"{target.hit}/{target.total}" + (" ✅" if target.is_full else "")
            )
    st.dataframe(rows, width="stretch", hide_index=True)
    gaps = [axis for axis in axes if not axis.is_full]
    if gaps:
        st.warning(
            f"{coverage_summary(axes)} — còn hở: "
            + " · ".join(axis.label for axis in gaps),
            icon="⚠️",
        )
    else:
        st.success(coverage_summary(axes), icon="✅")


def load_golden_into_input(case: GoldenCase) -> None:
    """Nạp RFP của một ca golden vào ô nhập, để chạy nó qua đúng luồng sản phẩm."""
    select_sample(case.rfp_text)
    st.session_state["loaded_golden_case"] = case.case_id


def render_golden() -> None:
    st.header("Sinh golden test")
    st.caption("Bộ test = RFP + điều kiện máy tự kiểm được; không lưu hồ sơ mẫu.")

    saved_cases = [load_case(path) for path in discover_case_files(DEFAULT_GOLDEN_DIR)]
    st.subheader("0. Độ phủ của bộ test hiện tại")
    st.caption(
        "Đếm số ca không nói lên điều gì — 39 ca đều đánh vào một góc thì vẫn là "
        "một góc. Bảng dưới đếm theo trục: mỗi thứ đề bài bắt hệ thống làm đúng "
        "có ít nhất một ca canh nó không."
    )
    render_golden_coverage(saved_cases)

    st.divider()
    st.subheader("1. Cấu hình")
    mode_labels = {
        "Theo tiêu chí phủ · 0 LLM": "coverage",
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
            disabled=mode == "coverage",
        )
    with config_middle:
        chapters = st.multiselect(
            "Chương",
            list(CHAPTER_TITLES),
            default=list(CHAPTER_TITLES),
            disabled=mode == "coverage",
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
            disabled=mode == "coverage",
        )
        base_paths = discover_case_files(DEFAULT_GOLDEN_DIR)
        base_path = st.selectbox(
            "Ca gốc",
            base_paths,
            format_func=lambda path: path.stem,
            disabled=mode == "combinatorial",
        ) if base_paths else None

    selected_axes = list(COVERAGE_AXES)
    if mode == "coverage":
        st.markdown("**Tiêu chí cần phủ** — chỉ sinh đúng phần còn thiếu")
        selected_axes = []
        axis_columns = st.columns(2)
        # Đi theo thứ tự của bảng độ phủ phía trên, không theo thứ tự nội bộ của
        # bộ sinh: hai danh sách cạnh nhau mà xếp khác nhau thì người test phải
        # dò từng dòng để biết ô tick nào ứng với dòng nào.
        for index, axis in enumerate(coverage_report(saved_cases)):
            with axis_columns[index % 2]:
                mark = "" if axis.is_full else "  ⚠"
                if st.checkbox(
                    f"{axis.label}{mark}  ·  _{axis.tier}_",
                    value=True,
                    key=f"golden_axis_{axis.key}",
                ):
                    selected_axes.append(axis.key)
        if not selected_axes:
            st.info("Chọn ít nhất một tiêu chí.")

    if st.button(
        "Sinh preview",
        type="primary",
        key="golden_generate_preview",
        disabled=mode == "coverage" and not selected_axes,
    ):
        if mode == "coverage":
            preview_cases = generate_for_coverage(
                selected_axes,
                existing=saved_cases,
            )
            if not preview_cases:
                st.success(
                    "Bộ test hiện tại đã phủ đủ các tiêu chí được chọn — "
                    "không cần sinh thêm ca nào.",
                    icon="✅",
                )
        elif mode == "combinatorial":
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
        if preview_cases:
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
        tier = selected_case.metadata.get("tier")
        axis = selected_case.metadata.get("axis")
        if tier:
            st.caption(f"Tầng **{tier}** · sinh cho tiêu chí `{axis}`")
        if any(case.metadata.get("axis") for case in preview_cases):
            with st.expander(
                f"Lô này vá được gì — {len(preview_cases)} ca", expanded=True
            ):
                render_golden_coverage(saved_cases, added=preview_cases)
                if st.button(
                    "Lưu cả lô", key="golden_save_batch", type="primary"
                ):
                    for case in preview_cases:
                        save_case(case, GOLDEN_OUTPUT_DIR / f"{case.case_id}.json")
                    st.success(f"Đã lưu {len(preview_cases)} ca.")
                    st.rerun()

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
        "Hành trình xử lý từng yêu cầu sẽ hiện ở đây.",
    )
    for tab, hint in zip(tabs[:3], hints):
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


def render_ops_metrics() -> None:
    """Thống kê vận hành từ nhật ký mỗi lượt chạy (v1.8).

    Đọc `cache/metrics.jsonl` — không đo lại gì, không gọi gì.
    """
    st.subheader("Thống kê vận hành")
    records = read_records()
    summary = summarize(records)
    if not summary.get("runs"):
        st.info(
            "Chưa có lượt chạy nào được ghi nhật ký. Sinh một hồ sơ rồi quay "
            "lại đây, hoặc xem `cache/metrics.jsonl`."
        )
        return

    st.caption(f"Dựa trên **{summary['runs']}** lượt chạy đã ghi nhận.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Tự trả lời",
        f"{summary['auto_rate']:.0%}",
        help="Tỉ lệ lượt chạy đạt tầng T1 — đủ căn cứ và có hồ sơ quá khứ chống lưng.",
    )
    col2.metric(
        "Chuyển người",
        f"{summary['handover_rate']:.0%}",
        help="Gồm cả 'cần người xem lại' (T2) và 'chuyển người xử lý' (T3).",
    )
    col3.metric(
        "p50 / p95 thời gian",
        f"{summary['p50_seconds']:.0f}s / {summary['p95_seconds']:.0f}s",
    )
    col4.metric("Token sản phẩm / lượt", f"{summary['tokens_product_avg']:,.0f}")

    if summary["failed_rate"]:
        st.warning(
            f"{summary['failed_rate']:.0%} lượt chạy **không ra được hồ sơ** "
            "(guard chặn xuất bản hoặc nhà cung cấp LLM không phản hồi). "
            "Những lượt này KHÔNG được tính là tự trả lời."
        )

    st.dataframe(
        [
            {"Nhánh": label(BRANCH_VI, name), "Số lượt": count}
            for name, count in summary["by_branch"].items()
        ],
        width="stretch",
        hide_index=True,
    )
    with st.expander("20 lượt gần nhất", expanded=False):
        st.dataframe(
            [
                {
                    "RFP": item.get("rfp_id") or "—",
                    "Nhánh": label(BRANCH_VI, item.get("branch")),
                    "Độ tin cậy": item.get("score"),
                    "Giây": item.get("seconds"),
                    "Token sản phẩm": item.get("tokens_product"),
                }
                for item in records[-20:][::-1]
            ],
            width="stretch",
            hide_index=True,
        )


def render_eval() -> None:
    render_ops_metrics()
    st.divider()
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
    st.session_state.setdefault("versions", [])
    st.session_state.setdefault("version_index", 0)

    text, submitted = sidebar_controls()
    # Nút Chat đặt cạnh tiêu đề, không nằm lẫn trong thân hồ sơ: muốn chat thì
    # bấm được ngay, không phải cuộn đi tìm.
    title_left, title_right = st.columns([4, 1], vertical_alignment="center")
    with title_left:
        st.title("RFP Proposal Studio")
        st.caption("Sinh hồ sơ thầu tiếng Nhật, mỗi câu đều truy được về nguồn")
    with title_right:
        render_chat_toggle()
    tabs = st.tabs(
        [
            "Tổng quan",
            "Độ đáp ứng",
            "Truy vết",
            "Sinh bộ test",
            "Kết quả đánh giá",
        ]
    )
    proposal_tab, coverage_tab, trace_tab, golden_tab, eval_tab = tabs

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
            # Bản gốc là v1 của lịch sử phiên bản; mọi lượt chat đẩy thêm bản mới.
            # KHÔNG dịch tự động ở đây: bản dịch lấy theo yêu cầu ở khối song ngữ,
            # nên một lượt sinh không còn kéo theo một lệnh gọi dịch bắt buộc.
            st.session_state["versions"] = []
            push_version(latest, label="v1")

    if latest is None:
        render_empty_tabs(tabs)
    else:
        render_flow(latest, flow_area, failure=failure)
        if failure is not None:
            with result_area:
                if failure["kind"] == "blocked":
                    st.warning(failure["message"], icon="🛑")
                else:
                    st.error(failure["message"], icon="❌")
            for tab in (coverage_tab,):
                with tab:
                    st.info(
                        "Chưa có kết quả để hiển thị — lượt chạy vừa rồi "
                        f"{label(RUN_STATUS_VI, latest.get('status'), unknown='không hoàn tất')}."
                    )
        elif latest.get("status") == "ask_user":
            with result_area:
                st.warning(latest["message"])
            for tab in (coverage_tab,):
                with tab:
                    st.info("Cần bổ sung đầu vào trước khi sinh kết quả.")
        elif latest.get("status") == "completed":
            # Bản đang hiển thị = bản đang chọn trong lịch sử, và đó cũng là bản
            # được export. Chưa có lịch sử (state nạp lại từ phiên cũ) thì dùng
            # thẳng kết quả pipeline.
            shown = displayed_state() or latest
            with result_area:
                render_proposal(shown)
            with coverage_tab:
                render_coverage(shown)
            with trace_tab:
                render_requirement_journey(shown)
                st.divider()
                render_trace_details(shown)

    with golden_tab:
        render_golden()
        
    with eval_tab:
        render_eval()

    # Sau cùng và ngoài mọi tab: xem mục "render_chat_dock".
    render_chat_dock()


if __name__ == "__main__":
    main()
