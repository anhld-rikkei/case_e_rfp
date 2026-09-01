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
    assert [
        "Mục",
        "Cách lấy nguồn",
        "Câu (tiếng Nhật)",
        "Nguồn",
        "Mã nguồn",
        "Kiểm chứng",
    ] in columns
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


# ── Ghim câu + lịch sử phiên bản (v1.7) ──────────────────────────────────

def _pinned_state() -> dict[str, Any]:
    state = _state()
    state["sections"][0]["sentences"].append(
        {
            "text": "BIツールとPL-300資格で対応します。",
            "origin": "user",
            "source_id": None,
            "req_ids": ["3.1"],
            "verdict": "USER_PROVIDED",
            "pinned": True,
        }
    )
    return state


def test_pinned_user_block_shows_pin_icon() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_marked_sentences(
            state["sections"][0]["sentences"], section_key="technical"
        )

    at = _render(body, _pinned_state())
    warnings = " ".join(item.value for item in at.warning)
    assert "📌" in warnings


def test_unpin_button_appears_only_where_allowed() -> None:
    def body(root, state, allow):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_marked_sentences(
            state["sections"][0]["sentences"],
            section_key="technical",
            allow_unpin=allow,
        )

    with_button = _render(body, _pinned_state(), True)
    assert any("Bỏ ghim" in b.label for b in with_button.button)
    without = _render(body, _pinned_state(), False)
    assert not any("Bỏ ghim" in b.label for b in without.button)


def test_full_proposal_renders_every_section_with_marks() -> None:
    """Bản đầy đủ của phiên bản đang chọn phải hiện, không chỉ diff."""

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_full_proposal(state)

    at = _render(body, _pinned_state())
    assert not at.exception
    text = " ".join(item.value for item in at.markdown)
    assert "技術要件への対応" in text  # tiêu đề mục
    warnings = " ".join(item.value for item in at.warning)
    assert "BIツールとPL-300資格で対応します。" in warnings  # vùng vàng user


def test_version_label_marks_lineage_when_branching() -> None:
    import app

    versions = [{"label": "v1"}, {"label": "v2"}, {"label": "v3"}]
    # Chat từ bản mới nhất -> nhãn thường
    assert app.version_label(versions, 2) == "v4"
    # Quay lại v2 rồi chat -> ghi rõ xuất phát từ đâu
    assert app.version_label(versions, 1) == "v4 · từ v2"


def test_push_version_never_deletes_later_versions() -> None:
    import app
    import streamlit as st

    st.session_state.clear()
    st.session_state["versions"] = []
    app.push_version({"proposal": "a"}, label="v1")
    app.push_version({"proposal": "b"}, label="v2")
    app.push_version({"proposal": "c"}, label="v3")
    st.session_state["version_index"] = 1  # quay lại v2
    app.push_version({"proposal": "d"}, label="v4 · từ v2", parent_index=1)

    labels = [item["label"] for item in st.session_state["versions"]]
    assert labels == ["v1", "v2", "v3", "v4 · từ v2"]  # v3 còn nguyên
    assert st.session_state["version_index"] == 3


def test_unpin_sentence_clears_the_flag() -> None:
    import app
    import streamlit as st

    state = _pinned_state()
    st.session_state.clear()
    st.session_state["versions"] = [{"state": state, "label": "v1"}]
    st.session_state["version_index"] = 0

    app.unpin_sentence("technical", "BIツールとPL-300資格で対応します。")
    assert state["sections"][0]["sentences"][-1]["pinned"] is False


def test_restored_count_surfaces_in_outcome_message() -> None:
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        from rfp.refine import RefineResult

        app.render_refine_outcome(RefineResult(state={}, changed=1, restored=2))

    at = _render(body)
    text = " ".join(item.value for item in list(at.info) + list(at.success))
    assert "giữ lại 2 câu" in text
    assert "Bỏ ghim" in text


# ── Tổng quan: Kết quả trước, Chi tiết sau (v1.7) ────────────────────────

def test_no_translation_tab_anymore() -> None:
    """Tab Bản dịch đã thành khối song ngữ tại chỗ."""
    assert "Bản dịch" not in EXPECTED_TABS


def test_section_summary_counts_by_status() -> None:
    import app

    state = _state()
    state["sections"] = [
        {"key": "a", "status": "OK", "sentences": []},
        {"key": "b", "status": "OK", "sentences": []},
        {"key": "c", "status": "INSUFFICIENT_EVIDENCE", "sentences": []},
    ]
    summary = app.section_summary(state)
    assert "2/3 mục" in summary
    assert "1 mục" in summary and "thiếu căn cứ" in summary.lower()


def test_section_summary_handles_empty_state() -> None:
    import app

    assert app.section_summary({"sections": []}) == "Chưa có mục nào"


def test_overview_puts_result_block_before_detail() -> None:
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_proposal(state)

    at = _render(body, state)
    assert not at.exception
    headers = [item.value for item in at.subheader]
    assert headers.index("Kết quả") < headers.index("Chi tiết hồ sơ")
    # Bảng ánh xạ mục ← chương nằm trong khối Kết quả
    assert any(
        "Mục hồ sơ" in list(frame.value.columns) for frame in at.dataframe
    )


# ── Dòng tóm tắt: đếm tường minh, ẩn thành phần bằng 0 ───────────────────

def test_count_phrase_lists_every_non_zero_part() -> None:
    import app

    assert app.count_phrase(
        10, [(9, "câu có nguồn"), (1, "câu nối")], unit="câu"
    ) == "10 câu: 9 câu có nguồn · 1 câu nối"


def test_count_phrase_hides_zero_parts() -> None:
    """Không được hiện '0 câu nối'."""
    import app

    text = app.count_phrase(9, [(9, "câu có nguồn"), (0, "câu nối")], unit="câu")
    assert text == "9 câu: 9 câu có nguồn"
    assert "0 câu" not in text


def test_count_phrase_handles_single_item() -> None:
    import app

    assert app.count_phrase(1, [(1, "câu có nguồn")], unit="câu") == (
        "1 câu: 1 câu có nguồn"
    )


def test_count_phrase_handles_empty() -> None:
    import app

    assert app.count_phrase(0, [(0, "câu có nguồn")], unit="câu") == "Chưa có câu nào"


def test_count_phrase_without_any_breakdown() -> None:
    import app

    assert app.count_phrase(3, [], unit="câu") == "3 câu"


def test_sentence_breakdown_counts_three_kinds() -> None:
    import app

    state = _state()  # 1 capability + 1 bridge
    state["sections"][0]["sentences"].append(
        {
            "text": "người dùng thêm",
            "origin": "user",
            "source_id": None,
            "req_ids": [],
            "verdict": "USER_PROVIDED",
        }
    )
    text = app.sentence_breakdown(state)
    assert text.startswith("3 câu:")
    assert "1 câu truy được về nguồn cụ thể" in text
    assert "1 câu người dùng bổ sung (chưa kiểm chứng)" in text
    assert "1 câu nối (không mang thông tin sự thật)" in text


def test_sentence_breakdown_omits_missing_kinds() -> None:
    import app

    state = _state()
    state["sections"][0]["sentences"] = [
        {"text": "a", "origin": "precedent", "source_id": "S1", "verdict": "VERIFIED"}
    ]
    text = app.sentence_breakdown(state)
    assert text == "1 câu: 1 câu truy được về nguồn cụ thể"
    assert "câu nối" not in text and "người dùng" not in text


def test_sentence_breakdown_does_not_count_user_as_grounded() -> None:
    """trace.grounding đếm origin != bridge nên gộp nhầm câu user vào nhóm
    có nguồn — chính thứ mà nhãn v1.7 sinh ra để phân biệt."""
    import app

    state = _state()
    state["sections"][0]["sentences"] = [
        {
            "text": "u",
            "origin": "user",
            "source_id": None,
            "verdict": "USER_PROVIDED",
        }
    ]
    text = app.sentence_breakdown(state)
    assert "truy được về nguồn" not in text
    assert "1 câu người dùng bổ sung" in text


def test_sentence_breakdown_on_empty_proposal() -> None:
    import app

    assert app.sentence_breakdown({"sections": []}) == "Chưa có câu nào"


def test_coverage_summary_uses_explicit_counts() -> None:
    import app

    state = _state()  # 3.1 có căn cứ, 3.2 thì không

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_coverage(state)

    at = _render(body, state)
    text = " ".join(item.value for item in list(at.warning) + list(at.success))
    assert "2 yêu cầu:" in text
    assert "1 yêu cầu đã có căn cứ" in text
    assert "1 yêu cầu chưa có căn cứ" in text


# ── Song ngữ theo từng mục ───────────────────────────────────────────────

def test_bilingual_off_by_default_makes_no_llm_call() -> None:
    """Không bật công tắc thì không tốn một lệnh gọi dịch nào."""
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        def explode(*args, **kwargs):
            raise AssertionError("dịch khi chưa bật toggle")

        app.translate_proposal = explode
        app.render_bilingual_proposal(state, key_prefix="t")

    at = _render(body, state)
    assert not at.exception
    assert any("Hiện bản dịch tiếng Việt" in t.label for t in at.toggle)


def test_split_translated_sections_aligns_by_heading() -> None:
    import app

    translated = "[PROPOSAL]\n1. Tổng quan công ty\nCâu A.\n\n2. Đề xuất\nCâu B.\nCâu C.\n"
    sections = app.split_translated_sections(translated)
    assert len(sections) == 2
    assert sections[0][0] == "1. Tổng quan công ty"
    assert sections[1][1:] == ["Câu B.", "Câu C."]


def test_split_translated_sections_survives_empty() -> None:
    import app

    assert app.split_translated_sections("") == []


def test_bilingual_renders_both_languages_per_section() -> None:
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st

        app.translate_proposal = lambda s: "[PROPOSAL]\n1. Đáp ứng yêu cầu kỹ thuật\nCâu đã dịch.\n"
        st.session_state["t_bilingual"] = True
        app.render_bilingual_proposal(state, key_prefix="t")

    at = _render(body, state)
    assert not at.exception
    text = " ".join(item.value for item in at.markdown)
    assert "技術要件への対応" in text  # cột tiếng Nhật
    assert "Câu đã dịch." in text  # cột tiếng Việt


def test_translated_column_keeps_user_label() -> None:
    """Nhãn trung thực không được rớt khi qua ngôn ngữ khác — và phải nằm
    CÙNG HÀNG với vùng vàng bên cột tiếng Nhật."""
    state = _state()
    state["sections"][0]["sentences"].append(
        {
            "text": "người dùng thêm",
            "origin": "user",
            "source_id": None,
            "req_ids": [],
            "verdict": "USER_PROVIDED",
        }
    )

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st

        # 3 câu Nhật -> 3 dòng dịch: khớp 1:1
        app.translate_proposal = lambda s: "[PROPOSAL]\n1. Đáp ứng yêu cầu kỹ thuật\nCâu dịch 1.\nCâu dịch 2.\nCâu dịch 3.\n"
        st.session_state["t2_bilingual"] = True
        app.render_bilingual_proposal(state, key_prefix="t2")

    at = _render(body, state)
    assert not at.exception
    warnings = [item.value for item in at.warning]
    # Vùng vàng xuất hiện ở CẢ HAI cột, mỗi bên mang nhãn ✎
    user_blocks = [text for text in warnings if "Người dùng bổ sung" in text]
    assert len(user_blocks) == 2
    assert any("người dùng thêm" in text for text in user_blocks)
    assert any("Câu dịch 3." in text for text in user_blocks)


def test_align_translation_maps_one_to_one_when_counts_match() -> None:
    import app

    sentences = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    lines = ["1. Tiêu đề", "A.", "B.", "C."]
    assert app.align_translation(sentences, lines) == [["A."], ["B."], ["C."]]


def test_align_translation_spreads_when_counts_differ() -> None:
    """Model gộp/tách câu thì chia đều — sai vài dòng còn hơn đoán bừa."""
    import app

    sentences = [{"text": "a"}, {"text": "b"}]
    chunks = app.align_translation(sentences, ["1. T", "A.", "B.", "C.", "D."])
    assert [len(c) for c in chunks] == [2, 2]
    assert sum(len(c) for c in chunks) == 4


def test_align_translation_handles_missing_translation() -> None:
    import app

    chunks = app.align_translation([{"text": "a"}, {"text": "b"}], [])
    assert chunks == [[], []]
    assert app.align_translation([], ["1. T", "A."]) == []


def test_bilingual_puts_one_sided_elements_outside_columns() -> None:
    """Tiêu đề / trạng thái / cảnh báo thiếu căn cứ phải ở NGOÀI cặp cột.

    Để chúng trong cột trái sẽ đẩy thân văn bản bên trái tụt xuống và cả mục
    lệch nhau từ dòng đầu — đúng lỗi đang sửa.
    """
    state = _state()
    state["sections"][0]["status"] = "INSUFFICIENT_EVIDENCE"
    state["sections"][0]["note"] = NOTE

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st

        app.translate_proposal = lambda s: "[PROPOSAL]\n1. Đáp ứng yêu cầu kỹ thuật\nCâu dịch 1.\nCâu dịch 2.\n"
        st.session_state["align_bilingual"] = True
        app.render_bilingual_proposal(state, key_prefix="align")

    at = _render(body, state)
    assert not at.exception
    # Cảnh báo thiếu căn cứ hiện đúng MỘT lần (không nhân đôi vào hai cột)
    warnings = [item.value for item in at.warning]
    assert sum("Thiếu căn cứ" in text for text in warnings) == 1
    # Tiêu đề mục và dòng trạng thái cũng chỉ một lần
    assert sum("技術要件への対応" in item.value for item in at.markdown) == 1
    assert sum(
        "Đủ căn cứ" in item.value or "Thiếu căn cứ" in item.value
        for item in at.caption
    ) == 1


def test_stacked_layout_keeps_japanese_first() -> None:
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st

        app.translate_proposal = lambda s: "[PROPOSAL]\n1. Đáp ứng yêu cầu kỹ thuật\nCâu dịch 1.\nCâu dịch 2.\n"
        st.session_state["st_bilingual"] = True
        st.session_state["st_stacked"] = True
        app.render_bilingual_proposal(state, key_prefix="st")

    at = _render(body, state)
    assert not at.exception
    text = " ".join(item.value for item in at.markdown)
    assert "技術要件への対応" in text


def test_bilingual_failure_does_not_hide_japanese() -> None:
    """LLM chết thì vẫn phải thấy bản tiếng Nhật."""
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st
        from rfp.llm import LLMUnavailable

        def boom(_s):
            raise LLMUnavailable("timeout", 4, RuntimeError("x"))

        app.translate_proposal = boom
        st.session_state["t3_bilingual"] = True
        app.render_bilingual_proposal(state, key_prefix="t3")

    at = _render(body, state)
    assert not at.exception
    assert any("Chưa dịch được" in w.value for w in at.warning)
    assert "技術要件への対応" in " ".join(item.value for item in at.markdown)


# ── Tin cậy, source_strategy, tầng T3 (v1.8) ─────────────────────────────

def _scored_state() -> dict[str, Any]:
    from rfp.confidence import annotate

    state = _state()
    state["chapters"][0]["retrieval"]["source_strategy"] = "hybrid-bm25+dense"
    state["chapters"][0]["retrieval"]["candidates"] = [
        {"sent_id": "C1", "text": "gần đúng 1", "scores": {"rerank": 0.6}},
        {"sent_id": "C2", "text": "gần đúng 2", "scores": {"rerank": 0.4}},
    ]
    return annotate(state)


def test_requirement_cell_no_requirements_says_so_in_words() -> None:
    """0/0 đọc thành 'đáp ứng kém' — nhưng mục đó KHÔNG được giao yêu cầu nào."""
    import app

    cell = app.requirement_cell(
        {"total_requirements": 0, "covered_requirements": 0, "missing_requirements": []}
    )
    assert cell == "— không có yêu cầu"
    assert "0/0" not in cell


def test_requirement_cell_lists_missing_ids() -> None:
    import app

    cell = app.requirement_cell(
        {
            "total_requirements": 4,
            "covered_requirements": 1,
            "missing_requirements": ["2.1", "2.2", "3.2"],
        }
    )
    assert cell == "Đáp ứng 1/4 — thiếu: 2.1 · 2.2 · 3.2"


def test_requirement_cell_when_fully_met() -> None:
    import app

    cell = app.requirement_cell(
        {"total_requirements": 3, "covered_requirements": 3, "missing_requirements": []}
    )
    assert cell == "Đáp ứng 3/3"
    assert "thiếu" not in cell


def test_confidence_table_uses_coverage_vocabulary_of_the_other_tab() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_confidence(state)

    at = _render(body, _scored_state())
    columns = [list(frame.value.columns) for frame in at.dataframe]
    assert ["Mục", "Tầng", "Độ tin cậy", "Đáp ứng yêu cầu RFP"] in columns
    captions = " ".join(item.value for item in at.caption)
    assert "trong y yêu cầu RFP giao cho mục này, x đã có căn cứ thật" in captions


def test_no_user_facing_coverage_jargon_left() -> None:
    """Thuật ngữ 'phủ/coverage' giữ trong eval kỹ thuật, không lên giao diện."""
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    for marker in ('"Yêu cầu đã phủ"', "đã phủ", "chưa phủ", "độ phủ"):
        assert marker not in source, marker


def test_confidence_block_shows_tier_and_score() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_confidence(state)

    at = _render(body, _scored_state())
    assert not at.exception
    text = " ".join(item.value for item in at.markdown)
    assert "độ tin cậy" in text
    assert any(
        list(frame.value.columns)
        == ["Mục", "Tầng", "Độ tin cậy", "Đáp ứng yêu cầu RFP"]
        for frame in at.dataframe
    )


def test_confidence_block_hidden_when_not_scored() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_confidence(state)

    at = _render(body, _state())
    assert not at.markdown and not at.dataframe


def test_sources_table_shows_how_each_section_got_its_evidence() -> None:
    import app

    rows = app.sentence_rows(_scored_state())
    assert rows[0]["Cách lấy nguồn"] == "Từ khoá + ngữ nghĩa"


def test_related_sources_block_says_it_is_not_an_answer() -> None:
    """Nhãn phải nói thẳng, không để người đọc tưởng hệ thống đã trả lời."""
    state = _scored_state()
    section = state["sections"][0]

    def body(root, state, section):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_related_sources(state, section)

    at = _render(body, state, section)
    assert not at.exception
    captions = " ".join(item.value for item in at.caption)
    assert "không đủ liên quan" in captions
    assert "**không** dùng chúng" in captions
    assert any(list(f.value.columns) == ["Chương", "Mã câu", "Câu (tiếng Nhật)", "Điểm"]
               for f in at.dataframe)


def test_related_sources_silent_when_nothing_left() -> None:
    state = _scored_state()
    state["chapters"][0]["retrieval"]["candidates"] = []

    def body(root, state, section):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_related_sources(state, section)

    at = _render(body, state, state["sections"][0])
    assert not at.dataframe and not at.caption


def test_ops_metrics_reports_nothing_without_runs(tmp_path: Path) -> None:
    """Không có dữ liệu thì nói 'chưa có', không bịa 0%."""

    def body(root, path):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.read_records = lambda *a, **k: []
        app.render_ops_metrics()

    at = _render(body, str(tmp_path))
    assert any("Chưa có lượt chạy nào" in item.value for item in at.info)


def test_ops_metrics_shows_auto_and_handover_rates() -> None:
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.read_records = lambda *a, **k: [
            {"branch": "auto", "seconds": 10, "tokens_product": 100, "rfp_id": "R1"},
            {"branch": "human_takeover", "seconds": 30, "tokens_product": 300,
             "rfp_id": "R2"},
        ]
        app.render_ops_metrics()

    at = _render(body)
    assert not at.exception
    labels = {item.label: item.value for item in at.metric}
    assert labels["Tự trả lời"] == "50%"
    assert labels["Chuyển người"] == "50%"


def test_ops_metrics_warns_about_failed_runs() -> None:
    """Guard chặn / provider sập không được lẫn vào 'tự trả lời'."""

    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.read_records = lambda *a, **k: [
            {"branch": "auto", "seconds": 1, "tokens_product": 1},
            {"branch": "failed", "seconds": 1, "tokens_product": 1},
        ]
        app.render_ops_metrics()

    at = _render(body)
    warnings = " ".join(item.value for item in at.warning)
    assert "không ra được hồ sơ" in warnings
    assert "KHÔNG được tính là tự trả lời" in warnings


# ── Tab Truy vết: hành trình từng yêu cầu (v1.8) ─────────────────────────

def _journey_state() -> dict[str, Any]:
    """RFP 3 yêu cầu -> đủ ba nhánh: đáp ứng tốt / cảnh báo / chưa có căn cứ."""
    from rfp.confidence import annotate

    state = {
        "chapters": [
            {
                "id": "3",
                "title": "技術要件",
                "requirements": [
                    {"req_id": "3.1", "text": "yc một"},
                    {"req_id": "3.2", "text": "yc hai"},
                ],
                "retrieval": {
                    "source_strategy": "hybrid-bm25+dense",
                    "selected": [
                        {"sent_id": "S1", "text": "x", "scores": {"rerank": 0.9}}
                    ],
                    "candidates": [
                        {"sent_id": "C9", "text": "gần đúng", "scores": {"rerank": 0.5}}
                    ],
                },
            },
            {
                "id": "4",
                "title": "セキュリティ要件",
                "requirements": [{"req_id": "4.1", "text": "yc ba"}],
                "retrieval": {
                    "source_strategy": "capability-only",
                    "selected": [],
                    "candidates": [],
                },
            },
        ],
        "sections": [
            {
                "key": "technical",
                "title_ja": "技術要件",
                "source_chapters": ["3"],
                "status": "OK",
                "sentences": [
                    {
                        "text": "a",
                        "origin": "precedent",
                        "source_id": "S1",
                        "req_ids": ["3.1"],
                        "verdict": "VERIFIED",
                    }
                ],
            },
            {
                "key": "security",
                "title_ja": "セキュリティ",
                "source_chapters": ["4"],
                "status": "ATTRIBUTE_ONLY",
                "sentences": [
                    {
                        "text": "b",
                        "origin": "capability",
                        "source_id": "capabilities:x",
                        "req_ids": ["4.1"],
                        "verdict": "VERIFIED",
                    }
                ],
            },
        ],
        "trace": {"llm_calls": 7, "hybrid_blocked": [], "conflicts": [], "path": []},
    }
    return annotate(state)


def test_flow_svg_shows_all_three_branches_with_counts() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    svg = app.flow_svg(rows, blocked=0)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    for name in ("🟢 Tự trả lời", "🟡 Cảnh báo", "🔴 Chuyển người"):
        assert name in svg
    # Mỗi nhánh 1/3 yêu cầu
    assert svg.count("1 · 33%") == 3


def test_flow_svg_shows_all_four_retrieval_tiers() -> None:
    import app

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    for name in ("Khớp bảng năng lực", "T1 ngữ nghĩa", "T2 +từ khoá",
                 "T3 chỉ liệt kê nguồn"):
        assert name in svg


def test_flow_svg_dims_edges_with_no_requirements() -> None:
    """Cạnh 0 yêu cầu vẫn vẽ nhưng mờ — người xem cần thấy nhánh đó tồn tại."""
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    svg = app.flow_svg(rows, blocked=0)
    # dense-only không có yêu cầu nào trong fixture -> cạnh của nó phải mờ
    dim = re.findall(r'opacity="0\.18"[^>]*data-edge="in:dense-only"', svg)
    dim += re.findall(r'data-edge="in:dense-only"[^>]*', svg)
    assert any("0.18" in item for item in dim)


def test_flow_edge_width_grows_with_traffic() -> None:
    """Sankey đơn giản: cạnh nhiều yêu cầu phải dày hơn cạnh ít."""
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    rows = rows + [dict(rows[0])] * 5  # dồn thêm vào nhánh human
    svg = app.flow_svg(rows, blocked=0)
    widths = {
        key: float(width)
        for width, key in re.findall(
            r'stroke-width="([\d.]+)"[^>]*data-edge="(branch:[a-z]+)"', svg
        )
    }
    assert widths["branch:human"] > widths["branch:auto"]


def test_flow_svg_reports_guard_even_when_zero() -> None:
    import app

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    assert "0 chặn" in svg
    svg2 = app.flow_svg(app.requirement_journey(_journey_state()), blocked=3)
    assert "3 chặn" in svg2


def test_flow_highlight_marks_only_the_selected_path() -> None:
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    plain = app.flow_svg(rows, blocked=0)
    lit = app.flow_svg(rows, blocked=0, highlight="4.1")

    assert plain != lit
    # 4.1 đi qua "capability-only" và nhánh warn -> cạnh đó sáng
    active = re.findall(r'opacity="0\.90"[^>]*data-edge="in:capability-only"', lit)
    assert active
    # Cạnh của nhánh không được chọn bị mờ đi
    assert re.findall(r'opacity="0\.18"[^>]*data-edge="branch:auto"', lit)
    # Điểm tin cậy của chính yêu cầu đó hiện trên hình
    assert f"{rows[0]['score']:.2f}" in lit or "1.00" in lit


def test_flow_renders_and_shows_one_line_for_selected_requirement() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        import streamlit as st

        st.session_state["journey_highlight"] = "4.1"
        rows = app.requirement_journey(state)
        app.render_journey_flow(state, rows)

    at = _render(body, _journey_state())
    assert not at.exception
    text = " ".join(item.value for item in at.markdown)
    assert "<svg" in text
    assert "Tầng trả lời:" in text and "độ tin cậy" in text


def test_legend_appears_once_under_the_diagram() -> None:
    """Gộp về một chỗ, không lặp hai nơi."""
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_requirement_journey(state)

    at = _render(body, _journey_state())
    from config.display_vi import JOURNEY_LEGEND

    captions = [item.value for item in at.caption]
    assert sum(JOURNEY_LEGEND == text for text in captions) == 1


def test_journey_has_one_row_per_requirement() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    assert len(rows) == 3
    assert {row["req_id"] for row in rows} == {"3.1", "3.2", "4.1"}


def test_journey_covers_all_three_branches() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    by_id = {row["req_id"]: row for row in rows}
    assert by_id["3.1"]["branch"] == "auto"    # có precedent, mục T1
    assert by_id["4.1"]["branch"] == "warn"    # chỉ bảng năng lực, mục T2
    assert by_id["3.2"]["branch"] == "human"   # chưa có câu nào dẫn


def test_journey_sorts_red_first_then_yellow_then_green() -> None:
    """Người đọc phải thấy ngay chỗ cần đến mình."""
    import app

    branches = [row["branch"] for row in app.requirement_journey(_journey_state())]
    assert branches == ["human", "warn", "auto"]


def test_uncovered_requirement_lists_related_sources_as_not_an_answer() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    row = next(item for item in rows if item["req_id"] == "3.2")
    assert "C9" in row["sources"]
    assert "không phải câu trả lời" in row["sources"]


def test_covered_requirement_shows_its_source_id() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    row = next(item for item in rows if item["req_id"] == "3.1")
    assert row["sources"] == "S1"
    assert row["score"] > 0


def test_journey_table_renders_with_legend_and_three_colours() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_requirement_journey(state)

    at = _render(body, _journey_state())
    assert not at.exception
    columns = [list(frame.value.columns) for frame in at.dataframe]
    assert [
        "Yêu cầu",
        "Tầng trả lời",
        "Độ tin cậy",
        "Nhánh",
        "Nguồn dẫn",
    ] in columns
    captions = " ".join(item.value for item in at.caption)
    assert "🟢" in captions and "🟡" in captions and "🔴" in captions
    # Dải tóm tắt có đủ ba nhánh
    labels = {item.label for item in at.metric}
    assert "🟢 Tự trả lời" in labels and "🔴 Chuyển người" in labels


def test_journey_summary_percentages_match_row_counts() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_requirement_journey(state)

    at = _render(body, _journey_state())
    values = {item.label: item.value for item in at.metric}
    assert values["🟢 Tự trả lời"] == "1"
    assert values["🟡 Cảnh báo"] == "1"
    assert values["🔴 Chuyển người"] == "1"


def test_trace_details_are_collapsed_expanders() -> None:
    """Khối kỹ thuật không bỏ, chỉ gấp lại phía dưới."""
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_trace_details(state)

    at = _render(body, _journey_state())
    assert not at.exception
    labels = [str(item.label) for item in at.expander]
    for name in (
        "Chi tiết kỹ thuật theo node",
        "Số lệnh gọi theo khâu",
        "Vòng review",
        "Nguồn đã truy xuất",
    ):
        assert any(name in text for text in labels), name


def test_zero_call_stage_note_says_guards_are_free() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_trace_details(state)

    at = _render(body, _journey_state())
    captions = " ".join(item.value for item in at.caption)
    assert "không tốn tiền LLM" in captions


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
