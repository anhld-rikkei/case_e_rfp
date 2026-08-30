"""Test tầng hiển thị của app.py (v1.4) bằng AppTest — headless, không mạng.

Chỉ kiểm **tầng render**: tên tab, nhãn tiếng Việt, cột bảng, và ba trạng thái
của luồng chạy (bình thường / bị chặn / lỗi). Không chạy pipeline thật, state
đầu vào là dữ liệu dựng sẵn.
"""
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from config.display_vi import (  # noqa: E402
    ATTRIBUTE_COVERED_REASON,
    NO_EVIDENCE_REASON,
    SECTION_STATUS_VI,
    SKIP_LEGEND,
    STAGE_VI,
    VERDICT_VI,
)

APP_PATH = str(PROJECT_ROOT / "app.py")

EXPECTED_TABS = [
    "Tổng quan",
    "Độ đáp ứng",
    "Nguồn từng câu",
    "Bản dịch",
    "Truy vết",
    "Sinh bộ test",
    "Kết quả đánh giá",
]


def _state() -> dict[str, Any]:
    return {
        "input_text": "評価用RFP",
        "rfp": {"rfp_id": "RFP-TEST"},
        "status": "completed",
        "proposal": "本文です。",
        "reference_rfp": {"rfp_id": "RFP-TEST", "method": "industry", "score": None},
        "chapters": [
            {
                "id": "3",
                "title": "技術要件",
                "requirements": [
                    {"req_id": "3.1", "text": "基幹システム構築に対応できること。"},
                    {"req_id": "3.2", "text": "24時間監視に対応できること。"},
                ],
                "retrieval": {
                    "selected": [],
                    "stages": {
                        "attribute": {"skipped": False, "covered_req_ids": ["3.1"]},
                        "query_embed": {"skipped": True},
                        "rerank": {"skipped": True},
                        "mmr": {"skipped": True},
                    },
                },
            }
        ],
        "sections": [
            {
                "key": "technical",
                "title_ja": "技術要件への対応",
                "title_vi": "Đáp ứng yêu cầu kỹ thuật",
                "source_chapters": ["3"],
                "status": "OK",
                "note": "",
                "sentences": [
                    {
                        "text": "基幹システム構築の実績があります。",
                        "origin": "capability",
                        "source_id": "capabilities:基幹システム構築",
                        "req_ids": ["3.1"],
                        "verdict": "VERIFIED",
                    },
                    {
                        "text": "つなぎの文です。",
                        "origin": "bridge",
                        "source_id": None,
                        "req_ids": [],
                        "verdict": "UNVERIFIABLE",
                    },
                ],
            }
        ],
        "trace": {
            "llm_calls": 5,
            "llm_calls_by_stage": {"parser": 1, "review": 2},
            "retrieval_stats": {},
            "dedup": {"before": 2, "after": 2},
            "path": ["parse_input", "assemble"],
            "conflicts": [],
            "hybrid_blocked": [],
            "grounding": {"grounded": 1, "total": 2},
            "stages": {
                name: {"status": "completed"}
                for name in ("parse_input", "check_complete", "route_reference_rfp")
            },
            "review": {
                "enabled": True,
                "rounds": 1,
                "history": [
                    {
                        "round": 1,
                        "score": {"critical": 0, "major": 1, "minor": 0},
                        "critical_sections": [],
                        "fixes_applied": 0,
                        "personas": ["coverage", "quality"],
                    }
                ],
            },
        },
    }


def _render(body, *args: Any) -> AppTest:
    # AppTest chép NGUYÊN VĂN source của hàm ra file tạm rồi chạy, nên hàm không
    # thấy biến module. Mọi thứ nó cần phải đi qua `args`.
    at = AppTest.from_function(
        body, default_timeout=180, args=(str(PROJECT_ROOT), *args)
    )
    at.run()
    return at


# ── App chạy được và tab đúng tên ─────────────────────────────────────────

def test_app_starts_without_exception_and_has_vietnamese_tabs() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == EXPECTED_TABS


def test_submit_button_is_vietnamese() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert "Nộp và sinh hồ sơ" in [button.label for button in at.button]


# ── Các bảng dùng tiêu đề cột tiếng Việt ──────────────────────────────────

def _columns_of(at: AppTest) -> list[list[str]]:
    return [list(frame.value.columns) for frame in at.dataframe]


def test_result_tables_use_vietnamese_headers() -> None:
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_proposal(state)
        app.render_coverage(state)
        app.render_sources(state)
        app.render_trace_details(state)

    at = _render(body, state)
    assert not at.exception
    columns = _columns_of(at)
    assert ["Mã yêu cầu", "Nội dung yêu cầu", "Tình trạng", "Căn cứ đáp ứng"] in columns
    assert ["Mục", "Câu (tiếng Nhật)", "Nguồn", "Mã nguồn", "Kiểm chứng"] in columns
    # Không cột nào còn tên enum thô
    flat = {name for group in columns for name in group}
    assert not {"origin", "verdict", "source_id", "req_id"} & flat


def test_internal_filter_columns_are_hidden_from_table() -> None:
    """`_origin`/`_verdict` chỉ để lọc, không được lộ ra bảng."""
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_sources(state)

    at = _render(body, state)
    flat = {name for group in _columns_of(at) for name in group}
    assert not any(name.startswith("_") for name in flat)


def test_sentence_rows_keep_raw_values_for_filtering() -> None:
    """Nhãn hiển thị là lớp áo; giá trị gốc phải còn nguyên để lọc."""
    import app

    rows = app.sentence_rows(_state())
    assert rows[0]["_origin"] == "capability"
    assert rows[0]["_verdict"] == "VERIFIED"
    assert rows[0]["Nguồn"] == "Năng lực công ty"
    assert VERDICT_VI["VERIFIED"] in rows[0]["Kiểm chứng"]


# ── Nội dung hồ sơ vẫn là tiếng Nhật ──────────────────────────────────────

def test_japanese_proposal_text_is_not_translated() -> None:
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_proposal(state)

    at = _render(body, state)
    # Tiêu đề mục render bằng markdown, nhãn trạng thái bằng caption.
    rendered = " ".join(
        item.value for item in list(at.markdown) + list(at.caption)
    )
    assert "技術要件への対応" in rendered  # tiêu đề mục giữ tiếng Nhật
    assert "基幹システム構築の実績があります。" in " ".join(
        item.value for item in at.markdown
    )
    assert SECTION_STATUS_VI["OK"] in rendered  # nhãn trạng thái đã dịch


# ── Cảnh báo thiếu căn cứ: không lộ chuỗi enum thô ────────────────────────

NOTE = (
    "INSUFFICIENT_EVIDENCE: 2.1 電子申請フォームを提供すること。 — "
    "対応する能力・先行事例の根拠がありません; "
    "2.2 進捗照会が可能なこと。 — 対応する能力・先行事例の根拠がありません"
)


def test_parse_note_splits_each_requirement() -> None:
    import app

    items = app.parse_evidence_note(NOTE)
    assert [item["req_id"] for item in items] == ["2.1", "2.2"]
    assert items[0]["text"] == "電子申請フォームを提供すること。"
    assert items[1]["text"] == "進捗照会が可能なこと。"
    # Không dòng nào còn dính chuỗi enum thô hay lý do tiếng Nhật
    for item in items:
        assert "INSUFFICIENT_EVIDENCE" not in item["text"]
        assert "根拠がありません" not in item["text"]


@pytest.mark.parametrize(
    "weird",
    [
        "",
        "INSUFFICIENT_EVIDENCE:",
        "INSUFFICIENT_EVIDENCE: không có dấu gạch",
        "chuỗi lạ hoàn toàn",
        "INSUFFICIENT_EVIDENCE: 2.1 câu — lý do;;; ",
    ],
)
def test_parse_note_survives_weird_strings(weird: str) -> None:
    """Tầng hiển thị vỡ ở đây không được phép làm chết cả trang."""
    import app

    items = app.parse_evidence_note(weird)
    assert isinstance(items, list)
    for item in items:
        assert set(item) == {"req_id", "text"}


def test_evidence_gaps_prefers_structured_state_over_note() -> None:
    """Có dữ liệu cấu trúc thì dựng từ đó, không phụ thuộc câu chữ ghi chú."""
    import app

    state = _state()
    state["sections"][0]["status"] = "INSUFFICIENT_EVIDENCE"
    state["sections"][0]["note"] = "INSUFFICIENT_EVIDENCE: rác không parse được"
    gaps = app.evidence_gaps(state, state["sections"][0])
    # 3.1 đã có câu dẫn, 3.2 thì chưa
    assert [gap["req_id"] for gap in gaps] == ["3.2"]
    assert gaps[0]["text"] == "24時間監視に対応できること。"


def test_evidence_gaps_falls_back_to_note_without_chapters() -> None:
    import app

    section = {
        "key": "s",
        "source_chapters": [],
        "sentences": [],
        "status": "INSUFFICIENT_EVIDENCE",
        "note": NOTE,
    }
    gaps = app.evidence_gaps({"chapters": []}, section)
    assert [gap["req_id"] for gap in gaps] == ["2.1", "2.2"]


def test_section_note_never_shows_raw_enum_string() -> None:
    state = _state()
    state["sections"][0]["status"] = "INSUFFICIENT_EVIDENCE"
    state["sections"][0]["note"] = NOTE

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_proposal(state)

    at = _render(body, state)
    text = " ".join(
        item.value
        for item in list(at.markdown) + list(at.caption) + list(at.warning)
    )
    assert "INSUFFICIENT_EVIDENCE" not in text
    assert "Thiếu căn cứ" in text
    assert "24時間監視に対応できること。" in text  # câu yêu cầu giữ tiếng Nhật


def test_attribute_only_note_is_translated() -> None:
    state = _state()
    state["sections"][0]["status"] = "ATTRIBUTE_ONLY"
    state["sections"][0]["note"] = (
        "参照可能な先行事例がないため、能力表のみで回答しました。"
    )

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_proposal(state)

    at = _render(body, state)
    warnings = " ".join(item.value for item in at.warning)
    assert "hồ sơ quá khứ" in warnings


def test_missing_requirements_get_their_own_reason() -> None:
    """Trước đây dán nguyên ghi chú gộp của cả mục vào từng yêu cầu."""
    import app

    state = _state()
    _, missing = app.requirement_coverage(state)
    assert [item["req_id"] for item in missing] == ["3.2"]
    assert missing[0]["reason"] == NO_EVIDENCE_REASON
    assert "INSUFFICIENT_EVIDENCE" not in missing[0]["reason"]


# ── Tab Bản dịch: giữ cấu trúc, không dồn thành một khối ──────────────────

TRANSLATION = (
    "[RFP]\n"
    "Tài liệu yêu cầu mua sắm\n"
    "Ngành mua sắm: Sản xuất\n"
    "\n"
    "Chương 1 Tổng quan mua sắm\n"
    "1.1 Mục đích là tái cấu trúc hệ thống lõi.\n"
    "1.2 Thời hạn hợp đồng là 18 tháng.\n"
    "\n"
    "[PROPOSAL]\n"
    "1. Tổng quan công ty\n"
    "Chúng tôi đáp ứng các yêu cầu.\n"
    "\n"
    "2. Tổng quan đề xuất\n"
    "Chúng tôi hỗ trợ xuyên suốt.\n"
)


def test_split_translation_separates_rfp_and_proposal() -> None:
    import app

    blocks = app.split_translation(TRANSLATION)
    assert [block["label"] for block in blocks] == ["RFP", "PROPOSAL"]
    assert blocks[0]["lines"][0] == "Tài liệu yêu cầu mua sắm"
    assert blocks[1]["lines"][0] == "1. Tổng quan công ty"
    # Nhãn [RFP]/[PROPOSAL] không còn nằm trong nội dung
    for block in blocks:
        assert "[RFP]" not in block["lines"]
        assert "[PROPOSAL]" not in block["lines"]


def test_split_translation_keeps_each_line_separate() -> None:
    """Đây chính là lỗi cũ: mọi dòng bị dồn thành một khối chữ."""
    import app

    rfp = app.split_translation(TRANSLATION)[0]["lines"]
    assert "1.1 Mục đích là tái cấu trúc hệ thống lõi." in rfp
    assert "1.2 Thời hạn hợp đồng là 18 tháng." in rfp


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Chương 1 Tổng quan mua sắm", True),
        ("1. Tổng quan công ty", True),
        ("1.1 Mục đích là tái cấu trúc.", False),  # dòng con, không phải tiêu đề
        ("Chúng tôi đáp ứng các yêu cầu.", False),
        ("", False),
    ],
)
def test_translation_heading_detection(line: str, expected: bool) -> None:
    import app

    assert app.is_translation_heading(line) is expected


@pytest.mark.parametrize("weird", ["", "   ", "\n\n", "không có nhãn nào cả"])
def test_split_translation_survives_weird_input(weird: str) -> None:
    import app

    blocks = app.split_translation(weird)
    assert isinstance(blocks, list)
    # Văn bản không có nhãn vẫn phải hiện được, không mất trắng
    if weird.strip():
        assert blocks and blocks[0]["label"] is None


def test_render_translation_emits_one_element_per_line() -> None:
    def body(root, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_translation(text)

    at = _render(body, TRANSLATION)
    assert not at.exception
    values = [item.value for item in at.markdown]
    # Mỗi dòng là một phần tử riêng -> không dồn thành một khối
    assert "1.1 Mục đích là tái cấu trúc hệ thống lõi." in values
    assert "1.2 Thời hạn hợp đồng là 18 tháng." in values
    assert "**Chương 1 Tổng quan mua sắm**" in values
    headers = [item.value for item in at.subheader]
    assert "RFP (bản dịch)" in headers
    assert "Hồ sơ thầu (bản dịch)" in headers


# ── Tab Truy vết: "bỏ qua" phải nói lý do ─────────────────────────────────

def test_skip_reason_for_ask_user_when_input_complete() -> None:
    import app

    assert app.skip_reason({"missing": []}, "ask_user") == (
        "RFP đã đủ thông tin bắt buộc"
    )


def test_skip_reason_falls_back_to_stage_table() -> None:
    import app

    assert app.skip_reason({}, "generate_per_section") is not None
    assert app.skip_reason({}, "khong_ton_tai") is None


def test_status_label_shows_reason_in_parentheses() -> None:
    import app

    text = app._status_label("Hỏi thêm người dùng", "skipped", reason="RFP đã đủ")
    assert text == "⏭️ Hỏi thêm người dùng — bỏ qua (RFP đã đủ)"


def test_trace_tab_explains_every_skip() -> None:
    """Không dòng 'bỏ qua' nào được để trống lý do."""
    state = _state()
    state["missing"] = []
    state["trace"]["stages"] = {
        "parse_input": {"status": "completed"},
        "check_complete": {"status": "completed"},
        "ask_user": {"status": "skipped"},
        "retrieve_per_chapter": {"status": "completed"},
    }

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st
        import app

        holder = st.empty()
        app.render_pipeline_status(state, holder)

    at = _render(body, state)
    assert not at.exception
    # Nhãn "bỏ qua" nằm ở label của st.status; gom cả markdown lẫn label lại.
    combined = " ".join(
        [item.value for item in at.markdown]
        + [str(getattr(item, "label", "")) for item in at.status]
    )
    assert "RFP đã đủ thông tin bắt buộc" in combined
    # Chương đã phủ bằng bảng năng lực -> nói rõ lý do, không để trống
    assert ATTRIBUTE_COVERED_REASON in combined


def test_legend_explains_skip_is_not_an_error() -> None:
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_flow_legend()

    at = _render(body)
    captions = " ".join(item.value for item in at.caption)
    assert SKIP_LEGEND in captions
    assert "⏭️" in captions


# ── Chat-refine: ba kết cục phải đọc ra ba nghĩa khác nhau ───────────────

def _outcome_text(kind: str) -> str:
    def body(root, kind):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        from rfp.refine import RefineResult

        if kind == "changed":
            result = RefineResult(state={}, changed=2, kept=3)
        elif kind == "blocked":
            result = RefineResult(
                state={},
                kept=4,
                rejected=[
                    {
                        "index": None,
                        "reason": (
                            "chỉ thị yêu cầu thêm năng lực hoặc chứng chỉ "
                            "**công ty không có** — hãy cập nhật "
                            "`capability_sheet.json` rồi bấm **Nạp lại kho**"
                        ),
                        "text": None,
                    }
                ],
            )
        else:
            result = RefineResult(state={}, kept=4)
        app.render_refine_outcome(result)

    at = _render(body, kind)
    assert not at.exception
    return " ".join(
        item.value
        for item in list(at.markdown)
        + list(at.info)
        + list(at.success)
        + list(at.warning)
    )


def test_refine_outcome_changed_reports_counts() -> None:
    text = _outcome_text("changed")
    assert "Đã cập nhật" in text
    assert "**2** câu sửa" in text and "**3** câu giữ nguyên" in text


def test_refine_outcome_blocked_explains_why_and_how_to_fix() -> None:
    """Lỗi đã gặp thật: bị chặn mà chỉ hiện 'không có thay đổi nào'."""
    text = _outcome_text("blocked")
    assert "lưới an toàn đã chặn" in text.lower()
    assert "capability_sheet.json" in text
    assert "Nạp lại kho" in text
    # Không được đọc thành "model không đổi gì"
    assert "mô hình không đề xuất" not in text


def test_refine_outcome_no_change_is_distinct_from_blocked() -> None:
    text = _outcome_text("no_change")
    assert "mô hình không đề xuất" in text
    assert "không thêm được nội dung mới" in text
    assert "lưới an toàn đã chặn" not in text.lower()


def test_all_three_outcomes_show_the_counts_line() -> None:
    for kind in ("changed", "blocked", "no_change"):
        assert "câu giữ nguyên" in _outcome_text(kind), kind


# ── Bôi màu theo nguồn gốc (v1.7) ────────────────────────────────────────

def _user_sentence(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "origin": "user",
        "source_id": None,
        "req_ids": [],
        "verdict": "USER_PROVIDED",
    }


def test_sentence_mark_three_way_classification() -> None:
    import app

    machine = {"text": "a", "origin": "precedent", "source_id": "S1"}
    edited = {**machine, "edited_by_chat": True}
    assert app.sentence_mark(machine) == "plain"
    assert app.sentence_mark(edited) == "edited"
    assert app.sentence_mark(_user_sentence("b")) == "user"


def test_adjacent_user_sentences_group_into_one_block() -> None:
    """Nhiều câu user liền nhau gộp thành MỘT vùng vàng."""
    import app

    sentences = [
        {"text": "m1", "origin": "precedent", "source_id": "S1"},
        _user_sentence("u1"),
        _user_sentence("u2"),
        {"text": "m2", "origin": "capability", "source_id": "c"},
    ]
    groups = app.group_by_mark(sentences)
    assert [mark for mark, _ in groups] == ["plain", "user", "plain"]
    assert len(groups[1][1]) == 2


def test_user_block_renders_with_warning_and_label() -> None:
    from config.display_vi import USER_BLOCK_LABEL

    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_marked_sentences(
            [
                {"text": "máy sinh", "origin": "precedent", "source_id": "S1"},
                {
                    "text": "người dùng thêm",
                    "origin": "user",
                    "source_id": None,
                    "verdict": "USER_PROVIDED",
                },
            ]
        )

    at = _render(body)
    assert not at.exception
    warnings = " ".join(item.value for item in at.warning)
    assert "người dùng thêm" in warnings
    assert USER_BLOCK_LABEL in warnings
    # Câu máy sinh KHÔNG nằm trong vùng vàng
    assert "máy sinh" not in warnings


def test_edited_sentence_marked_more_lightly_than_user() -> None:
    from config.display_vi import EDITED_LABEL

    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_marked_sentences(
            [
                {
                    "text": "đã chỉnh",
                    "origin": "precedent",
                    "source_id": "S1",
                    "edited_by_chat": True,
                }
            ]
        )

    at = _render(body)
    text = " ".join(item.value for item in at.markdown)
    assert "đã chỉnh" in text and EDITED_LABEL in text
    assert not at.warning  # không dùng vùng vàng cho mức nhẹ


def test_legend_hidden_when_nothing_is_marked() -> None:
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_mark_legend(
            [{"text": "a", "origin": "precedent", "source_id": "S1"}]
        )

    at = _render(body)
    assert not at.caption


def test_legend_shown_when_user_content_exists() -> None:
    from config.display_vi import MARK_LEGEND

    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_mark_legend(
            [
                {
                    "text": "a",
                    "origin": "user",
                    "source_id": None,
                    "verdict": "USER_PROVIDED",
                }
            ]
        )

    at = _render(body)
    assert MARK_LEGEND in " ".join(item.value for item in at.caption)


def test_user_sentence_appears_in_requirement_coverage() -> None:
    """Yêu cầu được đáp ứng bằng nội dung người dùng vẫn phải hiện ra."""
    import app

    state = _state()
    state["sections"][0]["sentences"].append(
        {
            "text": "người dùng bổ sung cho 3.2",
            "origin": "user",
            "source_id": None,
            "req_ids": ["3.2"],
            "verdict": "USER_PROVIDED",
        }
    )
    rows, missing = app.requirement_coverage(state)
    row = next(item for item in rows if item["Mã yêu cầu"] == "3.2")
    assert "người dùng bổ sung" in row["Căn cứ đáp ứng"]
    assert [item["req_id"] for item in missing] == []


# ── Tab Kết quả đánh giá ──────────────────────────────────────────────────

def test_headline_numbers_read_from_results_not_hardcoded() -> None:
    import app

    runs = {
        "force_precedent_k5_no_guard.det": {
            "deterministic": [{"fabrication_count": 9, "sections": 5}]
        },
        "force_precedent_k5_with_guard.det": {
            "deterministic": [
                {"fabrication_count": 0, "guard_blocked_publish": 3, "sections": 5}
            ]
        },
        "V0_A2_dense_only.det": {"deterministic": [{"coverage": 0.10, "sections": 5}]},
        "V1_A2_hybrid.det": {"deterministic": [{"coverage": 0.90, "sections": 5}]},
        "review_cost": {
            "runs": {
                "no_review": [{"product_tokens": 1000}],
                "with_review": [{"product_tokens": 3000}],
            }
        },
    }
    cards = app.headline_numbers(runs)
    values = [card["value"] for card in cards]
    assert "9 → 0" in values
    assert "0.100 → 0.900" in values
    assert "1.0k → 3.0k token" in values


def test_headline_skips_cards_without_data() -> None:
    """Thiếu file thì bỏ dòng, không bịa số và cũng không hiện 0."""
    import app

    assert app.headline_numbers({}) == []


def test_headline_mean_ignores_rows_without_sections() -> None:
    """Ca ask_user (sections=0) không được kéo trung bình xuống — như report."""
    import app

    runs = {
        "V0_A2_dense_only.det": {
            "deterministic": [
                {"coverage": 0.40, "sections": 5},
                {"coverage": 0.00, "sections": 0},
            ]
        },
        "V1_A2_hybrid.det": {"deterministic": [{"coverage": 0.60, "sections": 5}]},
    }
    card = next(card for card in app.headline_numbers(runs) if "BM25" in card["title"])
    assert card["value"] == "0.400 → 0.600"


def test_run_label_translates_config_ids() -> None:
    from config.display_vi import run_label

    assert run_label("V4_V5_A2").startswith("Cấu hình đề xuất")
    assert "TẮT" in run_label("force_precedent_k5_no_guard")
    assert run_label("V1_A2_hybrid.det").endswith("chỉ đo tất định")
    assert "cùng tập mẫu" in run_label("V0_A2_dense_only_intersect")
    assert run_label("ten_la_hoac_moi") == "ten_la_hoac_moi"


# ── Luồng chạy: ba trạng thái ─────────────────────────────────────────────

def _flow_text(failure: dict[str, str] | None) -> str:
    state = _state()

    def body(root, state, failure):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st
        import app

        holder = st.empty()
        app.render_flow(state, holder, failure=failure)
        app.render_flow_legend()

    at = _render(body, state, failure)
    assert not at.exception
    parts = [item.value for item in at.markdown]
    parts += [item.value for item in at.warning]
    parts += [item.value for item in at.error]
    parts += [item.value for item in at.caption]
    return " ".join(parts)


def test_flow_lists_every_stage_in_vietnamese() -> None:
    text = _flow_text(None)
    for name in ("parse_input", "generate_per_section", "review", "assemble"):
        assert STAGE_VI[name] in text
    # `ask_user` là nhánh rẽ, không thuộc luồng chạy thành công
    assert STAGE_VI["ask_user"] not in text


def test_blocked_flow_reads_as_safety_stop_not_technical_error() -> None:
    text = _flow_text(
        {
            "stage": "assemble",
            "kind": "blocked",
            "message": "Hồ sơ bị chặn xuất bản do vi phạm quy tắc an toàn.",
        }
    )
    assert "bị chặn xuất bản" in text
    assert "🛑" in text
    # Không được đọc thành hỏng hóc kỹ thuật
    assert "Lỗi kỹ thuật" not in text


def test_failed_flow_reads_as_technical_error() -> None:
    text = _flow_text(
        {
            "stage": "generate_per_section",
            "kind": "failed",
            "message": "Lỗi kỹ thuật: TimeoutError",
        }
    )
    assert "Lỗi kỹ thuật" in text
    assert "❌" in text


def test_stages_after_failure_never_show_as_completed() -> None:
    """Kể cả khi state cũ ghi completed — bước sau bước hỏng không được tích xanh.

    Một bước lỗi mà bước sau nó hiện ✅ khiến người đọc tin hồ sơ vẫn chạy tới
    cuối. Đây là lỗi hiển thị bắt được khi dump giao diện thật.
    """
    import app

    state = _state()
    state["trace"]["stages"] = {
        name: {"status": "completed"} for name in app.FLOW_STAGES
    }
    statuses = app._stage_statuses(
        state, failure={"stage": "generate_per_section", "kind": "failed"}
    )
    assert statuses["generate_per_section"] == "failed"
    assert statuses["review"] == "pending"
    assert statuses["assemble"] == "pending"
    assert statuses["parse_input"] == "completed"  # bước trước vẫn giữ


def test_stages_after_failure_are_not_left_running() -> None:
    import app

    state = _state()
    state["trace"]["stages"]["generate_per_section"] = {"status": "running"}
    statuses = app._stage_statuses(
        state, failure={"stage": "generate_per_section", "kind": "failed"}
    )
    assert statuses["generate_per_section"] == "failed"
    assert statuses["review"] == "pending"
    assert statuses["assemble"] == "pending"
    assert "running" not in statuses.values()
