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


def sidebar_controls() -> tuple[str, bool]:
    with st.sidebar:
        st.title("RFP đầu vào")
        with st.expander("Dữ liệu nguồn", expanded=False):
            render_source_status()
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
        if text.strip():
            with st.expander("Bản dịch RFP (tiếng Việt)", expanded=False):
                render_rfp_translation(text)
        st.markdown("#### RFP mẫu")
        for path in sorted(Path(RFP_DIR).glob("*.txt"))[:SAMPLE_RFP_LIMIT]:
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
    """Banner nháp + checklist + nút tải. Nhãn nháp đi theo cả file xuất ra."""
    st.error(
        f"**{DRAFT_BANNER_TITLE}**  \n"
        "Chưa có người thật rà soát. Không nộp và không gửi khách hàng khi "
        "checklist bên dưới chưa tick đủ."
    )
    stale = state_is_stale(state)
    if stale:
        st.warning(
            "⚠ **Hồ sơ này sinh từ dữ liệu nguồn đã cũ.** Dữ liệu nguồn đã thay "
            "đổi sau khi hồ sơ được sinh — nạp lại kho tri thức và sinh lại "
            "trước khi dùng. Cảnh báo này cũng nằm trong file tải về.",
            icon="⚠️",
        )
    with st.expander("Checklist bắt buộc trước khi nộp", expanded=False):
        st.markdown(REVIEWER_CHECKLIST)
    try:
        markdown = to_markdown(state, stale=stale)
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


def render_chat_refine(state: dict[str, Any]) -> None:
    """Ô chat chỉnh hồ sơ + lịch sử phiên bản."""
    st.subheader("Chỉnh lại bằng chỉ thị")
    st.caption(
        "Chat chỉnh **cách viết trên căn cứ sẵn có** — không thêm được nội dung "
        "chưa có bằng chứng, và không đổi được số liệu."
    )

    if state_is_stale(state):
        st.warning(
            "Dữ liệu nguồn đã thay đổi sau khi hồ sơ này được sinh. Chạy lại hồ "
            "sơ trước khi chỉnh tiếp — chỉnh trên bản cũ là trộn hai thế hệ dữ "
            "liệu vào cùng một tài liệu.",
            icon="⚠️",
        )
        return

    sections = state.get("sections", [])
    options = ["Toàn bộ hồ sơ"] + [
        f"{index}. {section['title_ja']}"
        for index, section in enumerate(sections, start=1)
    ]
    picked = st.selectbox("Chỉnh phần nào", options, key="refine_target")
    instruction = st.text_area(
        "Chỉ thị",
        key="refine_instruction",
        placeholder="ví dụ: viết phần bảo mật ngắn gọn hơn",
        height=80,
    )

    if st.button("Gửi chỉ thị", type="primary", key="refine_submit"):
        if not instruction.strip():
            st.warning("Chưa nhập chỉ thị.")
            return
        with st.spinner("Đang chỉnh lại…"):
            try:
                if picked == options[0]:
                    result = refine_all(state, instruction=instruction)
                else:
                    section = sections[options.index(picked) - 1]
                    result = refine_section(
                        state, section_key=section["key"], instruction=instruction
                    )
            except LLMUnavailable as error:
                st.error(
                    f"Nhà cung cấp LLM không phản hồi sau {error.attempts} lần gọi "
                    f"(lỗi {error.kind}). Hồ sơ giữ nguyên."
                )
                return
        if result.outcome == "changed":
            parent_index = st.session_state.get("version_index", len(_versions()) - 1)
            push_version(
                result.state,
                label=version_label(_versions(), parent_index),
                instruction=instruction,
                target=picked,
                rejected=result.rejected,
                counts=refine_counts(result),
                restored=result.restored,
                parent_index=parent_index,
            )
            st.session_state["translation"] = ""
            st.rerun()
        render_refine_outcome(result)


def refine_counts(result: Any) -> dict[str, int]:
    return {
        "changed": result.changed,
        "dropped": result.dropped,
        "kept": result.kept,
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


def render_marked_sentences(
    sentences: list[dict[str, Any]],
    *,
    section_key: str = "",
    allow_unpin: bool = False,
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
            for item in group:
                st.write(item.get("text", ""))


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


def render_bilingual_proposal(
    state: dict[str, Any],
    *,
    key_prefix: str,
    allow_unpin: bool = False,
) -> None:
    """Hồ sơ tiếng Nhật, kèm bản dịch song song theo TỪNG MỤC khi được bật.

    Dịch **theo yêu cầu**: chỉ gọi LLM khi người dùng bật công tắc, không dịch
    tự động sau mỗi lượt chat. Cache băm nội dung lo phần trùng lặp, nên bật/tắt
    hay quay lại một phiên bản cũ đều không tốn thêm lệnh gọi.
    """
    sections = state.get("sections", [])
    show = st.toggle(
        "Hiện bản dịch tiếng Việt",
        key=f"{key_prefix}_bilingual",
        help="Dịch bản đang hiển thị để đối chiếu. Bản nộp vẫn là bản tiếng Nhật.",
    )
    translated_sections: list[list[str]] = []
    stacked = False
    if show:
        stacked = st.checkbox(
            "Xếp dọc (màn hình hẹp)",
            key=f"{key_prefix}_stacked",
            help="Hai cột quá chật thì xếp tiếng Nhật trước, tiếng Việt sau.",
        )
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

        # Phần chỉ có MỘT bên (tiêu đề, trạng thái, cảnh báo thiếu căn cứ) nằm
        # NGOÀI cặp cột và chiếm hết bề ngang. Để chúng trong cột trái sẽ đẩy
        # thân văn bản bên trái tụt xuống, và cả mục lệch nhau từ dòng đầu.
        st.markdown(f"**{index + 1}. {section['title_ja']}**")
        vi_heading = vi_lines[0] if vi_lines else section["title_vi"]
        st.caption(
            f"{vi_heading} · {SECTION_STATUS_ICON.get(status, '')} "
            f"{label(SECTION_STATUS_VI, status)}"
        )
        render_section_note(state, section)

        if not (show and not stacked):
            # Một cột: giữ nguyên thứ tự đọc Nhật -> Việt trong từng mục.
            render_marked_sentences(
                sentences, section_key=section["key"], allow_unpin=allow_unpin
            )
            if show:
                render_marked_block("plain", vi_lines[1:])
            continue

        # Hai cột: chia theo TỪNG NHÓM đánh dấu, nên vùng vàng "người dùng bổ
        # sung" nằm cùng hàng ở cả hai ngôn ngữ.
        chunks = align_translation(sentences, vi_lines)
        position = 0
        for group_index, (mark, group) in enumerate(group_by_mark(sentences)):
            vi_chunk = [
                line
                for offset in range(position, position + len(group))
                if offset < len(chunks)
                for line in chunks[offset]
            ]
            position += len(group)
            left, right = st.columns(2)
            with left:
                render_marked_block(mark, [item.get("text", "") for item in group])
                if mark == "user" and allow_unpin:
                    for item_index, item in enumerate(
                        [item for item in group if item.get("pinned")]
                    ):
                        st.button(
                            f"📌 Bỏ ghim: {item['text'][:30]}…",
                            key=f"unpin_{key_prefix}_{section['key']}_{group_index}_{item_index}",
                            on_click=unpin_sentence,
                            args=(section["key"], item["text"]),
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
    render_draft_banner(state)
    st.success(sentence_breakdown(state))
    # KẾT QUẢ trước, CHI TIẾT sau: người mở tab cần thấy ngay bức tranh tổng —
    # mục nào đủ căn cứ, mục nào cần người bổ sung — rồi mới đọc văn bản.
    st.subheader("Kết quả")
    st.markdown(f"**{section_summary(state)}**")
    render_confidence(state)
    render_mapping(state)
    st.divider()

    st.subheader("Chi tiết hồ sơ")
    st.caption(
        "Nội dung hồ sơ giữ nguyên tiếng Nhật — đó là sản phẩm giao cho khách. "
        "Chỉ nhãn giao diện được dịch."
    )
    render_mark_legend(
        [
            sentence
            for section in state.get("sections", [])
            for sentence in section["sentences"]
        ]
    )
    render_bilingual_proposal(state, key_prefix="detail")
    st.divider()
    render_chat_refine(state)
    render_version_history()


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
    if not items:
        return
    items = items[:RELATED_SOURCES_TOP_N]
    with st.expander(
        f"🔎 {len(items)} nguồn có thể liên quan — **không phải câu trả lời**",
        expanded=False,
    ):
        st.caption(
            "Những câu này đã bị loại ở bước chọn nguồn vì không đủ liên quan. "
            "Hệ thống **không** dùng chúng để viết hồ sơ; liệt kê ở đây chỉ để "
            "người bổ sung có chỗ bắt đầu tra."
        )
        st.dataframe(
            [
                {
                    "Chương": item["_chapter"],
                    "Mã câu": item.get("sent_id"),
                    "Câu (tiếng Nhật)": item.get("text"),
                    "Điểm": round(
                        float((item.get("scores") or {}).get("rerank", 0.0)), 4
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
    st.title("RFP Proposal Studio")
    st.caption("Sinh hồ sơ thầu tiếng Nhật, mỗi câu đều truy được về nguồn")
    tabs = st.tabs(
        [
            "Tổng quan",
            "Độ đáp ứng",
            "Nguồn từng câu",
            "Truy vết",
            "Sinh bộ test",
            "Kết quả đánh giá",
        ]
    )
    proposal_tab, coverage_tab, sources_tab, trace_tab, golden_tab, eval_tab = tabs

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
        with proposal_tab:
            render_flow_legend()
        if failure is not None:
            with result_area:
                if failure["kind"] == "blocked":
                    st.warning(failure["message"], icon="🛑")
                else:
                    st.error(failure["message"], icon="❌")
            for tab in (coverage_tab, sources_tab):
                with tab:
                    st.info(
                        "Chưa có kết quả để hiển thị — lượt chạy vừa rồi "
                        f"{label(RUN_STATUS_VI, latest.get('status'), unknown='không hoàn tất')}."
                    )
        elif latest.get("status") == "ask_user":
            with result_area:
                st.warning(latest["message"])
            for tab in (coverage_tab, sources_tab):
                with tab:
                    st.info("Cần bổ sung đầu vào trước khi sinh kết quả.")
        elif latest.get("status") == "completed":
            # Bản đang hiển thị = bản đang chọn trong lịch sử, và đó cũng là bản
            # được export. Chưa có lịch sử (state nạp lại từ phiên cũ) thì dùng
            # thẳng kết quả pipeline.
            version = current_version()
            shown = version["state"] if version else latest
            with result_area:
                render_proposal(shown)
            with coverage_tab:
                render_coverage(shown)
            with sources_tab:
                render_sources(shown)
            with trace_tab:
                render_requirement_journey(shown)
                st.divider()
                render_trace_details(shown)

    with golden_tab:
        render_golden()
        
    with eval_tab:
        render_eval()


if __name__ == "__main__":
    main()
