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
    SECTION_STATUS_ICON,
    SECTION_STATUS_VI,
    SKIP_LEGEND,
    STAGE_VI,
    VERDICT_VI,
)

APP_PATH = str(PROJECT_ROOT / "app.py")

EXPECTED_TABS = [
    "Tổng quan",
    "Độ đáp ứng",
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

def _sentence_buttons(at: AppTest) -> list[str]:
    """Câu văn giờ CHÍNH LÀ nút bấm tra nguồn — nhãn nằm ở proto."""
    return [item.proto.popover.label for item in at.get("popover")]


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
        app.render_trace_details(state)

    at = _render(body, state)
    assert not at.exception
    columns = _columns_of(at)
    assert ["Mã yêu cầu", "Nội dung yêu cầu", "Tình trạng", "Căn cứ đáp ứng"] in columns
    # Không cột nào còn tên enum thô
    flat = {name for group in columns for name in group}
    assert not {"origin", "verdict", "source_id", "req_id"} & flat


def test_internal_keys_never_reach_the_source_lookup() -> None:
    """`_origin`/`_verdict` là khoá nội bộ, không được lộ ra chỗ người đọc."""
    import app

    state = _state()
    row = app.sentence_source(state, state["sections"][0]["sentences"][0]["text"])
    assert row is not None
    shown = {
        "Mục",
        "Cách lấy nguồn",
        "Câu (tiếng Nhật)",
        "Nguồn",
        "Mã nguồn",
        "Kiểm chứng",
    }
    assert shown <= set(row)
    assert {key for key in row if key.startswith("_")} == {
        "_origin",
        "_verdict",
        "_req_ids",
    }

    def body(root, state, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, text, key="k")

    at = _render(body, state, state["sections"][0]["sentences"][0]["text"])
    assert not at.exception
    body_text = " ".join(item.value for item in at.markdown)
    assert "_origin" not in body_text and "_verdict" not in body_text


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
    rendered = " ".join(
        item.value for item in list(at.markdown) + list(at.caption)
    )
    assert "技術要件への対応" in rendered  # tiêu đề mục giữ tiếng Nhật
    # Câu hồ sơ giữ nguyên tiếng Nhật; nó là nhãn của nút tra nguồn.
    assert "基幹システム構築の実績があります。" in _sentence_buttons(at)
    # Trạng thái đi cùng tiêu đề dưới dạng chấm màu; chữ nằm ở tooltip, và
    # bảng "Kết quả" ngay trên đã ghi rõ bằng chữ cho từng mục.
    title = next(
        item for item in at.markdown if "技術要件への対応" in item.value
    )
    assert SECTION_STATUS_ICON["OK"] in title.value
    assert SECTION_STATUS_VI["OK"] in (title.proto.help or "")


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


def _flow_captions(state: dict[str, Any]) -> str:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app

        app.render_flow(state, st.empty())

    at = _render(body, state)
    assert not at.exception
    return " ".join(item.value for item in at.caption)


def test_skip_is_explained_only_when_a_step_is_actually_skipped() -> None:
    """Chú giải theo ngữ cảnh, không phải bảng chú thích cố định cuối trang.

    Bảng cố định bắt người xem đối chiếu icon với một dòng chữ ở cuối trang,
    kể cả khi lượt chạy không có trạng thái nào như vậy.
    """
    plain = _state()
    plain["trace"] = {"stages": {"parse_input": {"status": "completed"}}}
    assert SKIP_LEGEND not in _flow_captions(plain)

    skipped = _state()
    skipped["trace"] = {"stages": {"review": {"status": "skipped"}}}
    captions = _flow_captions(skipped)
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
    assert headers.index("Kết quả") < headers.index("Chi tiết")
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
    # Tiêu đề mục kèm chấm trạng thái cũng chỉ một lần
    titles = [item for item in at.markdown if "技術要件への対応" in item.value]
    assert len(titles) == 1
    assert SECTION_STATUS_ICON["INSUFFICIENT_EVIDENCE"] in titles[0].value


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


# Màn "Sinh bộ test" có người đọc khác: người sửa code, không phải người làm
# hồ sơ thầu. Ở đó "độ phủ" là tên đúng của thứ đang đo, không phải thuật ngữ
# rò rỉ ra ngoài. Lệnh cấm dưới đây vẫn giữ nguyên cho mọi màn còn lại.
GOLDEN_SCREEN_FUNCTIONS = {
    "render_golden",
    "render_case_brief",
    "render_golden_coverage",
    "render_golden_picker",
    "load_golden_into_input",
    "_golden_management_rows",
}


def _app_source_outside_golden_screen() -> str:
    import ast

    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    skipped: set[int] = set()
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in GOLDEN_SCREEN_FUNCTIONS
        ):
            skipped.update(range(node.lineno - 1, node.end_lineno))
    return "".join(
        line for index, line in enumerate(lines) if index not in skipped
    )


def test_no_user_facing_coverage_jargon_left() -> None:
    """Thuật ngữ 'phủ/coverage' giữ trong eval kỹ thuật, không lên màn hồ sơ."""
    source = _app_source_outside_golden_screen()
    for marker in ('"Yêu cầu đã phủ"', "đã phủ", "chưa phủ", "độ phủ"):
        assert marker not in source, marker


def test_the_jargon_scan_still_covers_the_proposal_screens() -> None:
    """Chốt chặn cho chính bài trên: bỏ sót màn hồ sơ thì bài kia thành vô nghĩa."""
    source = _app_source_outside_golden_screen()
    for kept in ("def render_confidence", "def render_proposal", "def render_mapping"):
        assert kept in source, kept
    assert "def render_golden_coverage" not in source


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
    assert "Không phải câu trả lời" in captions
    assert "chưa đủ sát yêu cầu" in captions
    assert "**không** dùng để viết hồ sơ" in captions
    # Thang điểm phải nói ra, nếu không thì con số 0.4 không đọc được thành gì
    assert "0–1" in captions
    assert any(
        list(frame.value.columns)
        == ["Chương", "Mã câu", "Câu (tiếng Nhật)", "Điểm sát yêu cầu"]
        for frame in at.dataframe
    )


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
                    {"req_id": "3.3", "text": "yc bốn"},
                ],
                "retrieval": {
                    "source_strategy": "hybrid-bm25+dense",
                    "selected": [
                        {"sent_id": "S1", "text": "x", "scores": {"rerank": 0.9}},
                        # rerank 0 -> điểm 0.70: nằm giữa hai ngưỡng, đúng nhánh
                        # "phải kiểm lại". Không có dòng này thì fixture chỉ còn
                        # hai nhánh và mấy bài đếm phần trăm mất chỗ dựa.
                        {"sent_id": "S2", "text": "y", "scores": {"rerank": 0.0}},
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
                    },
                    {
                        "text": "c",
                        "origin": "precedent",
                        "source_id": "S2",
                        "req_ids": ["3.3"],
                        "verdict": "VERIFIED",
                    },
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
    for name in ("Dùng được ngay", "Dùng được, phải kiểm lại",
                 "Người phải bổ sung"):
        assert name in svg
    # 4 yêu cầu: 2 điểm cao, 1 điểm giữa, 1 điểm thấp
    assert svg.count("2 (50%)") == 1
    assert svg.count("1 (25%)") == 2


def test_flow_svg_describes_how_the_system_really_finds_evidence() -> None:
    """Vẽ đúng cách code chạy: từ khoá và ngữ nghĩa chạy CÙNG LÚC.

    Vẽ thành chuỗi "thử T1 rồi mới thử T2" là mô tả sai hệ thống — code chạy
    hybrid một lượt rồi chấm điểm lại, không có bước lùi nào.
    """
    import app

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    for name in (
        "Đối chiếu bảng năng lực công ty",
        "Tìm trong hồ sơ thầu cũ",
        "Không tìm được — chỉ liệt kê nguồn gần đúng",
    ):
        assert name in svg
    # Tên thật của kỹ thuật, theo yêu cầu "chỉ rõ truy xuất kiểu gì"
    assert "BM25 + vector ngữ nghĩa" in svg


def test_flow_svg_shows_the_processing_steps_in_order() -> None:
    import app

    svg = app.flow_svg(
        app.requirement_journey(_journey_state()), blocked=0, checked=9, reviewed=1
    )
    for step in (
        "① Tìm căn cứ cho từng yêu cầu",
        "② Viết câu, gắn nguồn",
        "③ Kiểm lại từng câu",
        "④ Soi lại văn bản",
        "⑤ Chấm độ tin cậy",
    ):
        assert step in svg
    assert "9 câu" in svg and "1 vòng" in svg


def test_flow_svg_keeps_plain_titles_and_puts_technique_in_the_subline() -> None:
    """Tiêu đề nói việc, dòng phụ nói kỹ thuật.

    Người đọc hỏi cả hai câu: "hệ thống làm gì" và "làm bằng cách nào". Tên
    thật của kỹ thuật (BM25, MMR) nằm ở dòng phụ; tên nội bộ của code
    (`source_strategy`, `hybrid-bm25+dense`) thì không bao giờ được lên màn
    hình — nó chỉ có nghĩa với người đọc source.
    """
    import app

    import re

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    # Chỉ soi CHỮ NGƯỜI ĐỌC THẤY. Khoá máy đọc trong data-edge vẫn giữ tên gốc
    # để test và log truy được về source_strategy.
    visible = " ".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    for internal in ("source_strategy", "sankey", "hybrid-bm25", "capability-only"):
        assert internal.lower() not in visible.lower(), internal
    # Kỹ thuật thật thì phải gọi tên
    assert "BM25" in visible and "MMR" in visible


def test_flow_svg_marks_where_the_llm_is_called() -> None:
    """Câu hỏi đầu tiên của người xem sơ đồ, mà trước đây phải tự đoán."""
    import app

    import re

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    visible = " ".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    # Ba bước gọi LLM: viết câu, kiểm lại từng câu, soi lại văn bản
    assert visible.count(app.LLM_BADGE) == 3
    # Tìm căn cứ và chấm điểm thì KHÔNG: BM25 là thống kê, vector do mô hình
    # nhúng chạy tại chỗ, chấm điểm là số học.
    assert visible.count(app.NO_LLM_BADGE) == 2


def test_flow_svg_dims_edges_with_no_requirements() -> None:
    """Cạnh 0 yêu cầu vẫn vẽ nhưng mờ — người xem cần thấy nhánh đó tồn tại."""
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    svg = app.flow_svg(rows, blocked=0)
    # dense-only không có yêu cầu nào trong fixture -> cạnh của nó phải mờ
    assert re.findall(r'opacity="0\.15"[^>]*data-edge="out:dense-only"', svg)


def _svg_endpoint(path: str) -> tuple[float, float]:
    """Điểm cuối của một path chỉ gồm M/H/V/Q."""
    x = y = 0.0
    tokens = path.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("M"):
            x, y = (float(value) for value in token[1:].split(","))
        elif token.startswith("H"):
            x = float(token[1:])
        elif token.startswith("V"):
            y = float(token[1:])
        elif token.startswith("Q"):
            index += 1
            x, y = (float(value) for value in tokens[index].split(","))
        index += 1
    return x, y


def test_flow_edges_all_have_the_same_width() -> None:
    """Mọi nét nối dày như nhau.

    Bản cũ vẽ dày theo lưu lượng: nhìn ra chỉ thấy nét to nét nhỏ so le, mà số
    yêu cầu thì đã in trên từng hộp rồi. Độ dày giờ chỉ còn mang MỘT nghĩa —
    đường đang được tô sáng.
    """
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    rows = rows + [dict(rows[0])] * 5  # dồn thêm vào một nhánh
    svg = app.flow_svg(rows, blocked=0)
    widths = {float(width) for width in re.findall(r'stroke-width="([\d.]+)" o', svg)}
    assert widths == {app._W_EDGE}


def test_flow_highlighted_path_is_the_only_thicker_edge() -> None:
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    svg = app.flow_svg(rows, blocked=0, highlight="4.1")
    thick = re.findall(
        rf'stroke-width="{app._W_LIT:.1f}"[^>]*data-edge="([^"]+)"', svg
    )
    assert set(thick) == {"out:capability-only", "branch:auto", "guard:auto"}


def test_flow_arrows_land_on_the_middle_of_a_box_edge() -> None:
    """Điểm chạm phải rơi đúng giữa mép hộp, không trượt ra ngoài.

    Lỗi thật đã gặp: cạnh gom từ khối "tìm căn cứ" đâm vào sườn hộp ③ thay vì
    vào hộp ② — vừa lệch chỗ chạm vừa vẽ sai luồng (viết câu xong mới kiểm).
    """
    import app
    import re

    svg = app.flow_svg(app.requirement_journey(_journey_state()), blocked=0)
    anchors = set()
    for match in re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg
    ):
        x, y, w, h = (float(value) for value in match)
        anchors |= {
            (x, y + h / 2),
            (x + w, y + h / 2),
            (x + w / 2, y),
            (x + w / 2, y + h),
        }
    for match in re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"', svg):
        cx, cy, r = (float(value) for value in match)
        anchors |= {(cx - r, cy), (cx + r, cy)}

    heads = re.findall(r'<path d="([^"]+)"[^>]*marker-end', svg)
    assert len(heads) >= 8
    for path in heads:
        assert _svg_endpoint(path) in anchors, path


def test_flow_svg_reports_guard_even_when_zero() -> None:
    import app

    rows = app.requirement_journey(_journey_state())
    assert "0 câu" in app.flow_svg(rows, blocked=0)
    assert "3 câu" in app.flow_svg(rows, blocked=3)


def test_flow_highlight_marks_only_the_selected_path() -> None:
    import app
    import re

    rows = app.requirement_journey(_journey_state())
    plain = app.flow_svg(rows, blocked=0)
    lit = app.flow_svg(rows, blocked=0, highlight="4.1")

    assert plain != lit
    # 4.1 đi qua "capability-only" và — theo phép chia mới — nhánh auto
    active = re.findall(r'opacity="0\.95"[^>]*data-edge="out:capability-only"', lit)
    assert active
    # Cạnh của nhánh không được chọn bị mờ đi
    assert re.findall(r'opacity="0\.15"[^>]*data-edge="branch:warn"', lit)
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
    assert len(rows) == 4
    assert {row["req_id"] for row in rows} == {"3.1", "3.2", "3.3", "4.1"}


def test_journey_branch_follows_the_score_not_the_source_kind() -> None:
    """Lỗi thật: cạnh dán nhãn "điểm cao/giữa/thấp" mà nhánh lại quyết bằng
    "có precedent hay không".

    Câu lấy thẳng từ bảng năng lực có điểm 1.00 — cao nhất thang — vẫn rơi vào
    nhánh "điểm giữa". Với kho dữ liệu này phần lớn câu đến từ bảng năng lực,
    nên nhánh "điểm cao" gần như không bao giờ có ai: lúc nào cũng 0%.
    """
    import app

    rows = app.requirement_journey(_journey_state())
    by_id = {row["req_id"]: row for row in rows}
    # Mỗi dòng phải khớp đúng phép chia theo điểm
    for row in rows:
        assert row["branch"] == app.score_branch(row["score"]), row
    # Câu từ bảng năng lực (điểm 1.00) giờ vào đúng nhánh điểm cao
    assert by_id["4.1"]["score"] >= 0.75
    assert by_id["4.1"]["branch"] == "auto"
    assert by_id["3.2"]["branch"] == "human"   # chưa có câu nào dẫn
    assert by_id["3.3"]["branch"] == "warn"    # precedent điểm giữa


def test_score_branch_uses_the_same_thresholds_as_the_tiers() -> None:
    """Một phép chia cho cả bảng Kết quả lẫn sơ đồ — hai chỗ tự chia riêng là
    lúc màn hình nói hai chuyện khác nhau về cùng một câu."""
    import app
    from config.settings import CONFIDENCE_T_HIGH, CONFIDENCE_T_LOW

    assert app.score_branch(CONFIDENCE_T_HIGH) == "auto"
    assert app.score_branch(CONFIDENCE_T_HIGH - 0.01) == "warn"
    assert app.score_branch(CONFIDENCE_T_LOW) == "warn"
    assert app.score_branch(CONFIDENCE_T_LOW - 0.01) == "human"
    assert app.score_branch(0.0) == "human"


def test_journey_sorts_red_first_then_yellow_then_green() -> None:
    """Người đọc phải thấy ngay chỗ cần đến mình."""
    import app

    branches = [row["branch"] for row in app.requirement_journey(_journey_state())]
    assert branches == ["human", "warn", "auto", "auto"]


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
    assert "🟢 Dùng được ngay" in labels and "🔴 Người phải bổ sung" in labels


def test_journey_summary_percentages_match_row_counts() -> None:
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_requirement_journey(state)

    at = _render(body, _journey_state())
    values = {item.label: item.value for item in at.metric}
    assert values["🟢 Dùng được ngay"] == "2"
    assert values["🟡 Phải kiểm lại"] == "1"
    assert values["🔴 Người phải bổ sung"] == "1"


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

    at = _render(body, state, failure)
    assert not at.exception
    parts = [item.value for item in at.markdown]
    parts += [item.value for item in at.warning]
    parts += [item.value for item in at.error]
    parts += [item.value for item in at.caption]
    return " ".join(parts)


def _flow_kinds(failure: dict[str, str]) -> set[str]:
    """Thông báo hỏng rơi vào khối nào — error hay warning."""
    state = _state()

    def body(root, state, failure):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app

        app.render_flow(state, st.empty(), failure=failure)

    at = _render(body, state, failure)
    kinds = set()
    if at.error:
        kinds.add("error")
    if at.warning:
        kinds.add("warning")
    return kinds


def test_flow_lists_every_stage_in_vietnamese() -> None:
    text = _flow_text(None)
    for name in ("parse_input", "generate_per_section", "review", "assemble"):
        assert STAGE_VI[name] in text
    # `ask_user` là nhánh rẽ, không thuộc luồng chạy thành công
    assert STAGE_VI["ask_user"] not in text


def test_flow_is_a_diagram_not_a_bullet_list() -> None:
    import app
    import re

    statuses = {name: "completed" for name in app.FLOW_STAGES}
    statuses["review"] = "running"
    svg = app.pipeline_svg(statuses)
    boxes = re.findall(r"<rect ", svg)
    assert len(boxes) == len(app.FLOW_STAGES)
    # Có mũi tên nối giữa các bước, kể cả chỗ xuống hàng
    arrows = re.findall(r'data-edge="step:([a-z_]+)"', svg)
    assert len(arrows) == len(app.FLOW_STAGES) - 1
    assert app.FLOW_STAGES[app.FLOW_STAGES.index("review")] in svg


def test_each_step_carries_its_own_status_word() -> None:
    """Trạng thái nằm TRONG ô — đó là lý do bỏ được bảng chú giải."""
    import app

    statuses = {name: "pending" for name in app.FLOW_STAGES}
    statuses["parse_input"] = "completed"
    statuses["check_complete"] = "running"
    statuses["review"] = "skipped"
    svg = app.pipeline_svg(statuses)
    for word in ("xong", "đang chạy", "bỏ qua", "chưa tới"):
        assert f">{word}</text>" in svg, word


def test_every_status_has_its_own_colour() -> None:
    import app

    tones = {value[0] for value in app.STAGE_TONE.values()}
    # blocked và failed cùng đỏ là cố ý; còn lại phải phân biệt được
    assert len(tones) == len(app.STAGE_TONE) - 1


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
    # Và phải là khối cảnh báo, không phải khối lỗi
    assert _flow_kinds({"stage": "assemble", "kind": "blocked",
                        "message": "x"}) == {"warning"}


def test_failed_flow_reads_as_technical_error() -> None:
    text = _flow_text(
        {
            "stage": "generate_per_section",
            "kind": "failed",
            "message": "Lỗi kỹ thuật: TimeoutError",
        }
    )
    assert "Lỗi kỹ thuật" in text
    # Icon nằm ở tham số `icon=` của st.error, không lọt vào chuỗi giá trị —
    # nên kiểm đúng thứ phân biệt hai ca: lỗi kỹ thuật vào khối ĐỎ, bị guard
    # chặn vào khối CẢNH BÁO.
    assert _flow_kinds({"stage": "generate_per_section", "kind": "failed",
                        "message": "x"}) == {"error"}


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


# ── Màn "Sinh bộ test": bảng độ phủ + sinh theo tiêu chí ──────────────────

def test_golden_coverage_table_names_every_gap() -> None:
    """Bộ rỗng thì bảng phải gọi tên đích danh từng thứ còn thiếu."""
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_golden_coverage([])

    at = _render(body)
    frame = at.dataframe[0].value
    assert list(frame.columns) == ["tầng", "trục", "phủ", "đạt", "còn thiếu"]
    assert set(frame["đạt"]) == {"⚠"}
    # Ba tầng của đề bài đều có mặt
    assert set(frame["tầng"]) == {"phổ biến", "biên", "cấm-sai"}
    text = " ".join(frame["còn thiếu"])
    from eval.golden.coverage import leaked_client_names, out_scope_terms

    for term in (*out_scope_terms(), *leaked_client_names()):
        assert term in text, term
    assert at.warning


def test_golden_coverage_table_shows_what_a_batch_would_fix() -> None:
    """Cột 'sau khi thêm' — thấy tác dụng TRƯỚC khi lưu file."""
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        from eval.golden.generator import generate_for_coverage

        app.render_golden_coverage([], added=generate_for_coverage(existing=[]))

    at = _render(body)
    frame = at.dataframe[0].value
    assert "sau khi thêm" in frame.columns
    assert all(value.endswith("✅") for value in frame["sau khi thêm"])
    # Bảng bên trái vẫn là hiện trạng: chưa lưu thì vẫn còn hở.
    assert set(frame["đạt"]) == {"⚠"}


def test_golden_coverage_table_is_all_green_when_nothing_is_missing() -> None:
    def body(root):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app
        from eval.golden.generator import generate_for_coverage

        app.render_golden_coverage(generate_for_coverage(existing=[]))

    at = _render(body)
    assert not at.warning
    assert at.success
    assert set(at.dataframe[0].value["đạt"]) == {"✅"}


def test_golden_tab_defaults_to_the_criteria_driven_mode() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    modes = [option for option in at.radio[0].options]
    assert modes[0] == "Theo tiêu chí phủ · 0 LLM"
    assert at.radio[0].value == "Theo tiêu chí phủ · 0 LLM"
    # Mỗi trục một ô tick, bật sẵn
    labels = [box.label for box in at.checkbox]
    assert any("Tên khách hàng cũ" in label for label in labels)
    assert any("6 chương RFP" in label for label in labels)


def test_generating_by_criteria_only_makes_what_is_missing() -> None:
    """Bấm sinh ở chế độ tiêu chí: ra đúng lô vá chỗ hở, không sinh bừa."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.button(key="golden_generate_preview").click().run()
    cases = at.session_state["golden_preview_cases"]
    assert cases
    # Bộ golden thật hiện chỉ hở trục rò rỉ khách hàng
    assert {case.metadata["axis"] for case in cases} == {"client_leak"}
    assert all(case.source == "coverage" for case in cases)
    assert not at.exception


COLLAPSED_LABEL = 2  # LabelVisibilityMessage.LabelVisibilityOptions.COLLAPSED


def test_golden_picker_is_a_button_like_the_sample_rfp_ones() -> None:
    """Cùng kiểu với ba nút "Chọn RFP-2025-00x" ngay trên nó.

    Trước đây là expander nên nhãn dính mép trái. Canh giữa nhãn expander phải
    nhắm vào DOM nội bộ của Streamlit, mà bản này không dựng expander bằng
    <summary> — CSS khớp rỗng, sửa xong màn hình y nguyên. Nút thì tự canh giữa.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [item.label for item in at.sidebar.button]
    assert "Chọn golden test" in labels
    assert "Chọn golden test" not in [item.label for item in at.sidebar.expander]
    # Chưa bấm thì chưa bung danh sách
    assert not [item for item in at.sidebar.selectbox if item.key == "golden_picker"]


def test_golden_picker_opens_and_hides_its_redundant_label() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["golden_picker_open"] = True
    at.run()
    picker = next(
        item for item in at.sidebar.selectbox if item.key == "golden_picker"
    )
    # Nhãn "Chọn ca" thừa khi nút đã tên là "Chọn golden test". Streamlit vẫn
    # giữ label trong cây (cho trình đọc màn hình) nên phải kiểm đúng cờ ẩn.
    assert picker.proto.label_visibility.value == COLLAPSED_LABEL


def test_sidebar_explains_what_the_selected_case_checks() -> None:
    """Người chấm phải hiểu ca này thử gì mà không cần mở file JSON.

    Dòng cũ ("3 điều kiện · must_cover · must_flag_insufficient") đúng nhưng vô
    dụng: đọc xong vẫn không biết nhìn vào đâu để chấm.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["golden_picker_open"] = True
    at.run()
    shown = " ".join(item.value for item in at.sidebar.markdown) + " ".join(
        item.value for item in at.sidebar.caption
    )
    assert "Input" in shown
    assert "Output" in shown
    for jargon in ("must_cover", "must_flag_insufficient", "must_not_contain"):
        assert jargon not in shown, jargon


def test_sidebar_can_load_a_golden_case_into_the_rfp_box() -> None:
    """Chạy một ca golden qua đúng luồng sản phẩm, không chỉ qua runner."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["golden_picker_open"] = True
    at.run()
    assert "Nạp vào ô RFP" in [button.label for button in at.sidebar.button]
    at.button(key="golden_picker_load").click().run()

    from eval.golden.runner import DEFAULT_GOLDEN_DIR, discover_case_files
    from eval.golden.schema import load_case

    picked = at.session_state["golden_picker"]
    expected = load_case(
        next(
            path
            for path in discover_case_files(DEFAULT_GOLDEN_DIR)
            if load_case(path).case_id == picked
        )
    )
    assert at.session_state["rfp_input"] == expected.rfp_text
    # Nạp RFP mới thì kết quả cũ phải bị xoá, nếu không màn hình trộn hai lượt
    assert at.session_state["result_state"] is None


# ── Chat review: panel trái thay khối "Chỉnh lại bằng chỉ thị" ────────────

def test_refine_targets_put_the_whole_document_first() -> None:
    import app

    targets = app.refine_targets(_state())
    assert targets[0] == (app.ALL_TARGET, "Toàn bộ hồ sơ")
    assert all(key for key, _ in targets[1:])
    assert len({key for key, _ in targets}) == len(targets)


def test_instruction_suggestions_point_at_real_sections() -> None:
    """Gợi ý phải bám mục có thật, và không gọi LLM lần nào."""
    import app

    state = _state()
    keys = {section["key"] for section in state["sections"]}
    suggestions = app.instruction_suggestions(state)
    assert suggestions
    for item in suggestions:
        assert item["target"] in keys | {app.ALL_TARGET}
        assert item["instruction"].strip()
    # Tất định: cùng đầu vào, cùng gợi ý
    assert app.instruction_suggestions(state) == suggestions


def test_no_suggestions_when_there_is_nothing_to_edit() -> None:
    import app

    assert app.instruction_suggestions({"sections": []}) == []
    assert app.instruction_suggestions({"sections": [{"key": "a", "sentences": []}]}) == []


def test_chat_history_is_built_from_version_history() -> None:
    """Lịch sử chat suy TỪ lịch sử phiên bản — không nuôi bản sao thứ hai.

    Hai kho song song rồi lệch nhau là cách chắc chắn nhất để chat kể sai
    chuyện đã thật sự xảy ra với tài liệu.
    """
    import app

    versions = [
        {"label": "v1", "state": {}},  # bản gốc, không có chỉ thị
        {
            "label": "v2",
            "state": {},
            "instruction": "viết ngắn hơn",
            "target": "3. 技術要件",
            "counts": {"changed": 2, "dropped": 0, "kept": 5, "blocked": 0},
            "rejected": [],
            "restored": 0,
        },
    ]
    turns = app.chat_turns(versions)
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "viết ngắn hơn"
    assert turns[0]["target"] == "3. 技術要件"
    assert turns[1]["outcome"] == "changed"
    assert "v2" in app.chat_reply_text(turns[1])


def test_chat_history_appends_the_turn_that_made_no_version() -> None:
    """Lượt bị chặn KHÔNG sinh bản mới — vẫn phải hiện trong chat.

    Im lặng khi bị chặn đúng là lỗi màn hình cũ đã mắc: người dùng đòi một
    chứng chỉ công ty không có, hệ thống từ chối đúng, màn hình không nói gì.
    """
    import app

    turns = app.chat_turns(
        [],
        last={
            "instruction": "thêm chứng chỉ ISO/IEC 27017",
            "target": "Toàn bộ hồ sơ",
            "reply": {
                "outcome": "blocked",
                "counts": {"changed": 0, "dropped": 0, "kept": 4, "blocked": 1},
                "rejected": [{"reason": "chứng chỉ **công ty không có**", "text": None}],
                "restored": 0,
            },
        },
    )
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert "lưới an toàn đã chặn" in app.chat_reply_text(turns[1]).lower()
    chips = app.chat_reply_chips(turns[1])
    assert chips and chips[0][1] == "block"
    # Markdown thô trong lý do phải bị gỡ, chip không render markdown
    assert "**" not in chips[0][0]


def test_three_outcomes_read_differently_in_chat() -> None:
    import app

    changed = app.chat_reply_text(
        {"outcome": "changed", "label": "v2", "counts": {"changed": 1, "dropped": 0, "kept": 1, "blocked": 0}}
    )
    blocked = app.chat_reply_text({"outcome": "blocked", "counts": {"changed": 0, "dropped": 0, "kept": 1, "blocked": 1}})
    nothing = app.chat_reply_text({"outcome": "no_change", "counts": {}})
    assert changed != blocked != nothing != changed
    assert "chặn" in blocked and "chặn" not in nothing


def test_pinned_sentences_show_as_a_chip() -> None:
    import app

    chips = app.chat_reply_chips({"outcome": "changed", "restored": 2, "rejected": []})
    assert any("📌" in text and "2" in text for text, _ in chips)


def test_refine_counts_keep_added_so_a_turn_replays_correctly() -> None:
    """`outcome` suy từ changed/dropped/added — thiếu `added` là dựng lại sai."""
    import app
    from rfp.refine import RefineResult

    counts = app.refine_counts(RefineResult(state={}, added=3))
    assert counts["added"] == 3
    replayed = RefineResult(
        state={},
        changed=counts["changed"],
        dropped=counts["dropped"],
        added=counts["added"],
    )
    assert replayed.outcome == "changed"


# ── Panel trái: hai chế độ, không bao giờ cả hai ─────────────────────────

def _app_with_result(**session: Any) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["result_state"] = _state()
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    # Streamlit vẫn dựng xong phần trước chỗ lỗi, nên kiểm phần tử KHÔNG bắt
    # được ngoại lệ. Đã sập bẫy này một lần: `st.iframe(height=0)` ném lỗi mà
    # cả bộ test vẫn xanh.
    assert not at.exception, at.exception
    return at


def test_a_single_button_opens_chat_and_the_old_section_is_gone() -> None:
    """Một nút, không phải cả một mục ở cuối tab.

    Bản trước để nguyên mục "Review & chỉnh lại" chỉ để chứa một nút — phải
    cuộn hết hồ sơ mới thấy chỗ mở chat.
    """
    at = _app_with_result()
    labels = [button.label for button in at.button]
    assert "💬 Chat" in labels
    assert "Gửi chỉ thị" not in labels
    headings = " ".join(item.value for item in at.subheader)
    assert "Review & chỉnh lại" not in headings


def test_chat_opens_on_the_right_and_leaves_the_rfp_panel_alone() -> None:
    """Bảng trái vẫn là ô nhập RFP — người dùng cần giữ nó để đối chiếu."""
    at = _app_with_result()
    at.button(key="chat_open_button").click().run()
    assert at.session_state["chat_open"] is True
    # Sidebar không bị chiếm
    assert "RFP" in [item.value for item in at.sidebar.subheader]
    assert "RFP" in [item.label for item in at.sidebar.text_area]
    assert not at.sidebar.chat_input
    # Chat nằm trong thân trang
    assert at.chat_input
    assert "Chat" in [item.value for item in at.subheader]


def test_chat_panel_closes_from_its_own_button() -> None:
    at = _app_with_result(chat_open=True)
    assert at.chat_input
    at.button(key="chat_close").click().run()
    assert at.session_state["chat_open"] is False
    assert not at.chat_input


def _chat_css(at: AppTest) -> str:
    return " ".join(item.value for item in at.markdown)


def test_chat_dock_is_pinned_and_has_a_real_drag_handle() -> None:
    """Bảng riêng neo mép phải, kéo bằng dải dọc ở mép trái.

    `resize: horizontal` của CSS đặt tay kéo ở GÓC DƯỚI của khối — bảng cao hết
    màn hình thì góc đó nằm tận đáy, không ai tìm ra. Đó là lý do lần trước kéo
    không được. Nay dựng dải kéo riêng, đúng chỗ như thanh kéo của sidebar.
    """
    at = _app_with_result(chat_open=True)
    css = _chat_css(at)
    assert ".st-key-chat_dock" in css
    assert "position: fixed" in css
    # Bề rộng đi qua một biến CSS để tay kéo đổi được cả bảng lẫn phần chừa chỗ
    assert "--chat-dock-w" in css
    assert "var(--chat-dock-w)" in css
    assert "cursor: col-resize" in css
    assert "stMainBlockContainer" in css and "padding-right" in css

    import app

    # Dải kéo phải bám lại sau mỗi lượt rerun: Streamlit dựng lại DOM, gắn một
    # lần lúc nạp là lượt sau mất tay kéo.
    assert "MutationObserver" in app.CHAT_DRAG_JS
    assert "window.parent" in app.CHAT_DRAG_JS
    for guard in ("Math.min", "Math.max"):
        assert guard in app.CHAT_DRAG_JS, guard


def test_chat_dock_reserves_no_space_when_it_is_closed() -> None:
    css = _chat_css(_app_with_result())
    assert "padding-right" not in css


def test_chat_styling_never_depends_on_has_or_the_avatar_testid() -> None:
    """Chốt lại đúng hai lỗi đã mắc — cả hai đều "sửa code mà màn hình y nguyên".

    1. Streamlit 1.61 không dựng expander bằng <details>/<summary>.
    2. Truyền `avatar=` làm test-id đổi thành stChatMessageAvatarCustom, nên
       `:has([data-testid="stChatMessageAvatarUser"])` khớp rỗng.
    Không selector nào được dựa vào hai thứ đó nữa.
    """
    css = _chat_css(_app_with_result(chat_open=True))
    assert ":has(" not in css
    assert "stChatMessageAvatarUser" not in css
    assert "summary" not in css
    # Cách thay thế: class `st-key-` mà Streamlit gắn cho container có key
    assert '[class*="st-key-chatturn_user"]' in css


def test_chat_asks_which_part_before_anything_else() -> None:
    """Bước đầu là một câu hỏi có đánh số, không phải một ô select rời rạc.

    Người dùng không phải học trước chat làm được gì — mỗi bước chỉ hiện đúng
    những nước đi kế tiếp.
    """
    at = _app_with_result(chat_open=True)
    labels = [button.label for button in at.button]
    assert any(label.startswith("1. Toàn bộ hồ sơ") for label in labels)
    assert any(label.startswith("2. ") for label in labels)
    # Ô select "Chỉnh phần nào" đã bỏ
    assert "Chỉnh phần nào" not in [item.label for item in at.selectbox]
    # Chưa chọn phạm vi thì chưa gợi ý chỉ thị
    assert not [
        button.key
        for button in at.button
        if (button.key or "").startswith("chat_instruction_")
    ]


def test_choosing_a_part_switches_to_instruction_suggestions() -> None:
    at = _app_with_result(chat_open=True)
    at.button(key="chat_scope___all__").click().run()
    assert at.session_state["chat_scope"] == "__all__"
    keys = [
        button.key
        for button in at.button
        if (button.key or "").startswith("chat_instruction_")
    ]
    assert len(keys) == 3
    assert "← Đổi phần khác" in [button.label for button in at.button]
    at.button(key="chat_scope_reset").click().run()
    assert at.session_state["chat_scope"] is None


def test_instructions_are_deterministic_and_scoped() -> None:
    import app

    state = _state()
    whole = app.instructions_for(state, app.ALL_TARGET)
    assert whole == app.instructions_for(state, app.ALL_TARGET)
    section_key = state["sections"][0]["key"]
    scoped = app.instructions_for(state, section_key)
    assert scoped != whole
    title = state["sections"][0]["title_ja"]
    assert all(title in instruction for instruction in scoped)


def test_typing_without_choosing_a_part_means_the_whole_document() -> None:
    """Hỏi lại một bước nữa chỉ để xác nhận điều hiển nhiên là bắt người dùng chờ."""
    import app
    from rfp.refine import RefineResult

    seen = {}
    original = app.refine_all
    app.refine_all = lambda state, instruction: (
        seen.update(instruction=instruction) or RefineResult(state=state, changed=1)
    )
    try:
        def body(root, state):
            import sys as _s

            _s.path[:0] = [root + "/src", root]
            import streamlit as st

            import app as _app

            st.session_state.setdefault("versions", [{"label": "v1", "state": state}])
            if not st.session_state.get("seeded"):
                st.session_state["seeded"] = True
                st.session_state["chat_pending"] = {
                    "instruction": "viết ngắn hơn",
                    "target_key": _app.ALL_TARGET,
                }
            _app.render_chat_panel(state)

        at = _render(body, _state())
        assert not at.exception
        assert seen["instruction"] == "viết ngắn hơn"
        assert at.session_state["versions"][1]["target"] == "Toàn bộ hồ sơ"
    finally:
        app.refine_all = original


# ── Checklist trong panel chat ───────────────────────────────────────────

def _checklist_boxes(at: AppTest) -> list[str]:
    return [
        item.label
        for item in at.checkbox
        if (item.key or "").startswith("checklist_")
    ]


def test_checklist_lives_inside_the_conversation() -> None:
    """Checklist là thứ chat đưa ra khi được hỏi, không phải một khối đứng riêng."""
    at = _app_with_result(chat_open=True)
    assert "📋 Checklist trước khi nộp" in [button.label for button in at.button]
    # Chưa hỏi thì chưa hiện. Lọc theo key: tab "Sinh bộ test" cũng có ô tick,
    # `at.checkbox` gộp cả trang chứ không riêng bảng chat.
    assert not _checklist_boxes(at)

    at.button(key="chat_checklist_open_button").click().run()
    from rfp.export import CHECKLIST_ITEMS

    shown = _checklist_boxes(at)
    assert shown == [short for short, _ in CHECKLIST_ITEMS]
    assert "← Quay lại" in [button.label for button in at.button]


def test_checklist_is_the_short_form_of_the_export_one() -> None:
    """Một nguồn, hai cách hiện — lệch nhau là tick đủ mà bản nộp vẫn thiếu."""
    from rfp.export import CHECKLIST_ITEMS, CHECKLIST_SIGNOFF, REVIEWER_CHECKLIST

    at = _app_with_result(chat_open=True, chat_checklist_open=True)
    shown = _checklist_boxes(at)
    for short, full in CHECKLIST_ITEMS:
        assert short in shown, short
        assert full in REVIEWER_CHECKLIST, full
    # Dòng ký tên là thứ của bản in, không phải ô tick trên màn hình
    assert CHECKLIST_SIGNOFF in REVIEWER_CHECKLIST
    assert not any("Người rà soát" in label for label in shown)


def test_checklist_counts_and_warns_until_every_box_is_ticked() -> None:
    from rfp.export import CHECKLIST_ITEMS

    total = len(CHECKLIST_ITEMS)
    at = _app_with_result(chat_open=True, chat_checklist_open=True)
    body = " ".join(item.value for item in at.markdown)
    assert f"**0/{total}**" in body
    assert "chưa tick đủ" in body

    ticked = {f"checklist_{index}": True for index in range(total)}
    done = _app_with_result(chat_open=True, chat_checklist_open=True, **ticked)
    done_body = " ".join(item.value for item in done.markdown)
    assert f"**{total}/{total}**" in done_body
    assert "chưa tick đủ" not in done_body


def test_a_click_on_a_suggestion_runs_a_turn_and_records_a_version() -> None:
    """Vòng đầy đủ: bấm gợi ý -> chạy -> thành một bản mới + một lượt chat.

    Giả lập `refine_*` để không gọi LLM; cái đang kiểm là đường dây nối, không
    phải chất lượng câu chữ.
    """
    import app
    from rfp.refine import RefineResult

    original = (app.refine_all, app.refine_section)
    app.refine_all = lambda state, instruction: RefineResult(
        state=state, changed=2, kept=3
    )
    app.refine_section = lambda state, section_key, instruction: RefineResult(
        state=state, changed=1, kept=4
    )
    try:
        def body(root, state):
            import sys as _s

            _s.path[:0] = [root + "/src", root]
            import streamlit as st

            import app as _app

            st.session_state.setdefault("versions", [{"label": "v1", "state": state}])
            # Nạp chỉ thị ĐÚNG MỘT LẦN. `setdefault` ở đây sẽ nạp lại sau mỗi
            # lượt rerun mà panel gọi, thành vòng lặp vô tận — trong app thật
            # chỉ có cú bấm nút đặt khoá này, không ai đặt lại.
            if not st.session_state.get("seeded"):
                st.session_state["seeded"] = True
                st.session_state["chat_pending"] = {
                    "instruction": "viết ngắn hơn",
                    "target_key": _app.ALL_TARGET,
                }
            _app.render_chat_panel(state)

        at = _render(body, _state())
        assert not at.exception
        versions = at.session_state["versions"]
        assert len(versions) == 2
        assert versions[1]["instruction"] == "viết ngắn hơn"
        assert versions[1]["counts"]["changed"] == 2
        # Lượt đã thành bản mới thì không được giữ thêm bản nháp -> in hai lần
        assert "chat_last" not in at.session_state
        turns = app.chat_turns(versions)
        assert [turn["role"] for turn in turns] == ["user", "assistant"]
    finally:
        app.refine_all, app.refine_section = original


def test_a_blocked_turn_leaves_no_new_version_but_still_answers() -> None:
    import app
    from rfp.refine import RefineResult

    original = app.refine_all
    app.refine_all = lambda state, instruction: RefineResult(
        state=state,
        kept=4,
        rejected=[{"reason": "chứng chỉ công ty không có", "text": None}],
    )
    try:
        def body(root, state):
            import sys as _s

            _s.path[:0] = [root + "/src", root]
            import streamlit as st

            import app as _app

            st.session_state.setdefault("versions", [{"label": "v1", "state": state}])
            if not st.session_state.get("seeded"):
                st.session_state["seeded"] = True
                st.session_state["chat_pending"] = {
                    "instruction": "thêm ISO/IEC 27017",
                    "target_key": _app.ALL_TARGET,
                }
            _app.render_chat_panel(state)

        at = _render(body, _state())
        assert not at.exception
        assert len(at.session_state["versions"]) == 1  # không sinh bản mới
        last = at.session_state["chat_last"]
        assert last["reply"]["outcome"] == "blocked"
    finally:
        app.refine_all = original


# ── Banner nháp và nút tải ───────────────────────────────────────────────

def _ordered(at: AppTest) -> list[tuple[str, str]]:
    """Danh sách phẳng (loại, nhãn) theo đúng thứ tự hiện trên trang."""
    out = []
    for element in at.main:
        label = getattr(element, "label", None)
        if label is None:
            label = str(getattr(element, "value", ""))
        out.append((element.type, str(label)))
    return out


def _tab_slice(at: AppTest, name: str) -> list[tuple[str, str]]:
    """Chỉ lấy phần tử của MỘT tab.

    `at.main` gộp phẳng cả sáu tab, nên "không còn gì phía sau" tính trên cả
    danh sách sẽ luôn sai — phần sau là nội dung của tab kế tiếp.
    """
    order = _ordered(at)
    start = next(
        index for index, (kind, label) in enumerate(order)
        if kind == "tab" and label == name
    )
    later = [
        index for index, (kind, _) in enumerate(order)
        if kind == "tab" and index > start
    ]
    return order[start : later[0] if later else len(order)]


def test_draft_notice_is_no_longer_a_red_block() -> None:
    """Khối đỏ bỏ theo yêu cầu — nhưng nhãn "bản nháp" phải còn trên màn hình."""
    from rfp.export import DRAFT_BANNER_TITLE

    at = _app_with_result()
    reds = " ".join(item.value for item in at.error)
    assert DRAFT_BANNER_TITLE not in reds
    captions = " ".join(item.value for item in at.caption)
    assert DRAFT_BANNER_TITLE in captions
    assert "Chat review" in captions


def test_download_button_sits_at_the_very_bottom_of_the_tab() -> None:
    """Tải về là việc sau cùng.

    Đặt nút tải ngay đầu trang là mời người dùng tải trước khi đọc bất cứ thứ gì.
    """
    order = _tab_slice(_app_with_result(), "Tổng quan")
    downloads = [
        index for index, (kind, _) in enumerate(order) if kind == "download_button"
    ]
    assert len(downloads) == 1
    at_index = downloads[0]
    # Không còn khối nội dung nào phía sau nó TRONG tab này
    for kind, label in order[at_index + 1 :]:
        assert kind not in {"subheader", "dataframe", "toggle"}, (kind, label)
    detail = next(
        index for index, (_, label) in enumerate(order) if label == "Chi tiết"
    )
    assert at_index > detail


def test_download_still_refuses_a_draft_the_guard_would_block() -> None:
    """Không bao giờ mở đường tải cho bản chưa qua guard."""
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app
        from rfp.guard import GuardViolation

        original = _app.to_markdown
        _app.to_markdown = lambda *a, **k: (_ for _ in ()).throw(
            GuardViolation("ISO/IEC 27017")
        )
        try:
            _app.render_download(state)
        finally:
            _app.to_markdown = original

    at = _render(body, _state())
    assert not at.download_button
    assert any("final guard chặn" in item.value for item in at.error)


def test_chat_button_sits_next_to_the_page_title() -> None:
    """Bấm chat được ngay, không phải cuộn hết hồ sơ mới thấy nút."""
    order = _tab_slice(_app_with_result(), "Tổng quan")
    at = _app_with_result()
    flat = _ordered(at)
    title = next(index for index, (kind, _) in enumerate(flat) if kind == "title")
    button = next(
        index
        for index, (kind, label) in enumerate(flat)
        if kind == "button" and label == "💬 Chat"
    )
    # Nút đứng ngay sau tiêu đề, trước cả hàng tab
    tabs = next(index for index, (kind, _) in enumerate(flat) if kind == "tab")
    assert title < button < tabs
    assert not any(label == "💬 Chat" for _, label in order)


def test_chat_button_is_never_disabled() -> None:
    """Bấm chat được bất cứ lúc nào.

    Lỗi thật đã gặp: nút có `disabled=` theo `displayed_state()`, mà nút vẽ ở
    ĐẦU `main()` còn `result_state` mãi cuối hàm mới gán — nên đúng lượt sinh
    xong hồ sơ, nút vẫn mờ. Bấm không lên và không báo gì.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    chat = next(item for item in at.button if item.key == "chat_open_button")
    assert not chat.disabled


def test_chat_opens_without_a_proposal_and_says_what_is_missing() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["chat_open"] = True
    at.run()
    assert not at.exception
    body = " ".join(item.value for item in at.markdown)
    assert "Chưa có hồ sơ nào để chỉnh" in body
    assert "Nộp và sinh hồ sơ" in body
    # Checklist vẫn tham khảo được trước khi bắt tay vào làm
    assert "📋 Checklist trước khi nộp" in [item.label for item in at.button]


def test_chat_dock_lives_outside_the_tabs() -> None:
    """Mở từ tab nào cũng thấy chat, không riêng tab Tổng quan."""
    at = _app_with_result(chat_open=True)
    overview = _tab_slice(at, "Tổng quan")
    assert not any(kind == "chat_input" for kind, _ in overview)
    assert at.chat_input


def test_attribute_only_label_says_what_it_rests_on() -> None:
    """"Chỉ có thông tin công ty" không nói được gì cho người đọc hồ sơ."""
    from config import display_vi

    label = display_vi.SECTION_STATUS_VI["ATTRIBUTE_ONLY"]
    assert "bảng năng lực" in label
    assert "Chỉ có thông tin công ty" != label


def test_attribute_only_section_does_not_repeat_itself() -> None:
    """Ghi chú ngay dưới đã nói rõ lý do — cụm trạng thái ở trên là thừa."""
    state = _state()
    state["sections"][0]["status"] = "ATTRIBUTE_ONLY"
    state["sections"][0]["note"] = (
        "参照可能な先行事例がないため、能力表のみで回答しました。"
    )

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_bilingual_proposal(state, key_prefix="t")

    at = _render(body, state)
    assert not at.exception
    title = next(item for item in at.markdown if "技術要件への対応" in item.value)
    assert SECTION_STATUS_ICON["ATTRIBUTE_ONLY"] in title.value
    # Không nhắc lại cụm trạng thái thành chữ ngay cạnh tiêu đề
    assert "bảng năng lực công ty" not in title.value
    # Lý do thật vẫn còn, ở hộp ghi chú
    warnings = " ".join(item.value for item in at.warning)
    assert "hồ sơ quá khứ" in warnings


def test_section_title_translation_appears_only_with_the_toggle() -> None:
    """Tên mục là tiếng Nhật — bản dịch tên mục đi cùng lúc với bản dịch thân."""
    state = _state()

    def body(root, state, translate):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app

        app.translate_proposal = lambda s: (
            "[PROPOSAL]\n1. Đáp ứng yêu cầu kỹ thuật\nCâu dịch 1.\n"
        )
        st.session_state["t_bilingual"] = translate
        app.render_bilingual_proposal(state, key_prefix="t")

    off = _render(body, state, False)
    assert not off.exception
    assert not any(
        "Đáp ứng yêu cầu kỹ thuật" in item.value for item in off.markdown
    )

    on = _render(body, state, True)
    assert not on.exception
    # Tên mục tiếng Việt in đậm như tên tiếng Nhật, ở cột phải
    assert any(
        item.value.strip() == "**1. Đáp ứng yêu cầu kỹ thuật**"
        for item in on.markdown
    )


def test_the_tab_ends_with_one_small_download_button() -> None:
    """Ba vạch kẻ chồng nhau rồi một nút to hết bề ngang — đã bỏ."""
    at = _app_with_result()
    order = _tab_slice(at, "Tổng quan")
    download = next(
        index for index, (kind, _) in enumerate(order) if kind == "download_button"
    )
    # Không còn vạch kẻ ngay trước nút tải
    assert order[download - 1][0] != "divider"
    assert order[download][1] == "⬇ Tải file (.md)"
    button = next(item for item in at.download_button)
    assert button.proto.use_container_width is False


def test_sidebar_has_two_parts_and_groups_every_way_in() -> None:
    """Hai phần: kho dữ liệu nguồn, và RFP của lượt này.

    Ô tải file trước đây nằm CHEN giữa hai phần đó — đọc từ trên xuống là một
    chuỗi việc không liên quan nhau. Nay mọi cách đưa RFP vào nằm chung một
    chỗ, ngay dưới ô RFP.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    headings = [item.value for item in at.sidebar.subheader]
    assert headings == ["Dữ liệu nguồn", "RFP"]

    blocks = [item.label for item in at.sidebar.expander]
    for name in ("Tình trạng kho tri thức", "RFP mẫu", "Upload file RFP"):
        assert name in blocks, name
    # Khu chọn nhanh đứng SAU ô RFP
    assert blocks.index("RFP mẫu") > blocks.index("Tình trạng kho tri thức")


def test_sidebar_upload_still_fills_the_rfp_box() -> None:
    """Gộp lại thành khu riêng nhưng vẫn phải nạp được nội dung vào ô RFP."""
    import app

    assert callable(app.render_rfp_upload)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    uploads = [item for item in at.sidebar.file_uploader]
    assert len(uploads) == 1


# ── Tra nguồn ngay tại câu ────────────────────────────────────────────────

def test_the_sentence_itself_is_the_button() -> None:
    """Bấm vào câu là ra nguồn — không còn nút kính lúp riêng.

    Nút riêng ở cuối câu là thêm một thứ phải nhắm trúng, trong khi thứ người
    đọc đang nhìn chính là câu văn.
    """
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_bilingual_proposal(state, key_prefix="look")

    at = _render(body, state)
    assert not at.exception
    plain = [
        sentence["text"]
        for section in state["sections"]
        for sentence in section["sentences"]
    ]
    assert _sentence_buttons(at) == plain
    assert "🔍" not in " ".join(item.label for item in at.button)


def test_the_lookup_shows_the_same_fields_the_old_tab_did() -> None:
    import app

    state = _state()
    text = state["sections"][0]["sentences"][0]["text"]

    def body(root, state, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, text, key="k")

    at = _render(body, state, text)
    assert _sentence_buttons(at) == [text]
    shown = " ".join(item.value for item in at.markdown)
    row = app.sentence_source(state, text)
    for field in ("Nguồn", "Mã nguồn", "Kiểm chứng", "Cách lấy nguồn"):
        assert field in shown, field
        assert row[field] in shown, field


def test_lookup_is_silent_for_a_sentence_it_cannot_place() -> None:
    """Câu lạ thì không vẽ kính lúp — thà không có nút còn hơn nút rỗng."""
    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, "câu không có trong hồ sơ", key="k")
        _app.render_source_lookup(None, "bất kỳ", key="k2")

    at = _render(body, _state())
    assert not at.exception
    assert not at.get("popover")
    # Không tra được nguồn thì vẫn phải in câu ra, đừng nuốt mất nội dung
    written = " ".join(item.value for item in at.markdown)
    assert "câu không có trong hồ sơ" in written
    assert "bất kỳ" in written


def test_source_tab_is_gone() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert "Nguồn từng câu" not in [tab.label for tab in at.tabs]


# ── Bóc tách theo từng yêu cầu RFP ───────────────────────────────────────

def test_sentences_are_grouped_under_the_requirement_they_answer() -> None:
    import app

    state = _state()
    groups = app.requirement_groups(state, state["sections"][0])
    by_id = {group["req_id"]: group for group in groups}
    # 3.1 có câu đáp, 3.2 không — cả hai đều phải có mặt
    assert by_id["3.1"]["items"]
    assert by_id["3.2"]["items"] == []
    assert by_id["3.2"]["text"]  # nguyên văn yêu cầu, không chỉ mã
    # Câu không gắn yêu cầu nào xếp cuối, không bị bỏ rơi
    assert groups[-1]["req_id"] is None
    assert groups[-1]["items"]


def test_a_requirement_with_no_sentence_is_shown_not_hidden() -> None:
    """Chỗ trống mới là thứ người rà soát cần thấy."""
    state = _state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_bilingual_proposal(state, key_prefix="gap")

    at = _render(body, state)
    assert not at.exception
    captions = [item.value for item in at.caption]
    # Yêu cầu trống và lý do nằm CÙNG một dòng — tách ra là mỗi chỗ thiếu
    # chiếm hai dòng mà không thêm chữ nào.
    assert any(
        line.startswith("Yêu cầu 3.2") and "Chưa có dữ liệu đáp ứng yêu cầu" in line
        for line in captions
    )
    # Yêu cầu có câu đáp thì nói luôn lấy từ đâu
    assert any(
        line.startswith("Yêu cầu 3.1") and "lấy từ" in line for line in captions
    )
    # Màn hình chính KHÔNG trích nguyên văn RFP — đây là chỗ đọc hồ sơ.
    shown = " ".join([item.value for item in at.markdown] + captions)
    assert "24時間監視に対応できること。" not in shown


def test_grouping_keeps_the_original_index_for_translation_alignment() -> None:
    """Gom nhóm mà đánh mất chỉ số gốc là hai cột lệch nhau ngay."""
    import app

    state = _state()
    groups = app.requirement_groups(state, state["sections"][0])
    seen = sorted(index for group in groups for index, _ in group["items"])
    assert seen == list(range(len(state["sections"][0]["sentences"])))


def test_result_block_also_breaks_down_per_requirement() -> None:
    state = _scored_state()

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app

        app.render_confidence(state)

    at = _render(body, state)
    assert not at.exception
    columns = [list(frame.value.columns) for frame in at.dataframe]
    assert [
        "Mục",
        "Yêu cầu",
        "Nội dung yêu cầu",
        "Số câu đáp",
        "Tình trạng",
    ] in columns


def test_related_sources_drop_the_zero_score_ones() -> None:
    """Điểm 0 = không liên quan chút nào. Liệt kê chúng là mời đi tra chỗ trống."""
    import app

    state = _state()
    state["chapters"][0]["retrieval"] = {
        "selected": [],
        "candidates": [
            {"sent_id": "A", "text": "câu 0 điểm", "scores": {"rerank": 0.0}},
            {"sent_id": "B", "text": "câu có điểm", "scores": {"rerank": 0.4}},
        ],
    }

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_related_sources(state, state["sections"][0])

    at = _render(body, state)
    assert not at.exception
    frames = [frame.value for frame in at.dataframe]
    assert frames
    texts = list(frames[0]["Câu (tiếng Nhật)"])
    assert "câu có điểm" in texts
    assert "câu 0 điểm" not in texts
    assert "Điểm sát yêu cầu" in frames[0].columns


def test_related_sources_hidden_when_everything_scores_zero() -> None:
    state = _state()
    state["chapters"][0]["retrieval"] = {
        "selected": [],
        "candidates": [
            {"sent_id": "A", "text": "câu 0 điểm", "scores": {"rerank": 0.0}}
        ],
    }

    def body(root, state):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_related_sources(state, state["sections"][0])

    at = _render(body, state)
    assert not at.exception
    assert not at.dataframe
    assert not at.get("expander")


def test_lookup_quotes_the_whole_requirement_text() -> None:
    """Lúc người đọc hỏi "câu này đáp cái gì" thì câu hỏi phải hiện đủ chữ."""
    state = _state()
    text = state["sections"][0]["sentences"][0]["text"]

    def body(root, state, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, text, key="k")

    at = _render(body, state, text)
    assert not at.exception
    shown = " ".join(item.value for item in at.markdown)
    assert "Đáp ứng yêu cầu" in shown
    assert "`3.1` 基幹システム構築に対応できること。" in shown


def test_sentence_styling_is_loaded_even_when_chat_is_closed() -> None:
    """Lỗi thật: CSS câu-là-nút nằm trong CHAT_CSS, chỉ nhả khi mở chat.

    Chat đóng thì câu hiện thành ô có viền, và sửa CSS bao nhiêu lần cũng không
    thấy đổi vì nó chưa từng được nạp.
    """
    at = _app_with_result()
    # `at.session_state` không có `.get` — proxy của AppTest chỉ hỗ trợ `in`.
    assert "chat_open" not in at.session_state
    css = " ".join(item.value for item in at.markdown)
    assert "stPopover" in css
    for rule in ("border: none", "text-align: left", "background: transparent"):
        assert rule in css, rule


def test_requirement_line_names_where_the_answer_came_from() -> None:
    import app

    state = _state()
    groups = app.requirement_groups(state, state["sections"][0])
    covered = next(group for group in groups if group["items"])
    assert app.group_origins(covered) == "Năng lực công ty"
    loose = next(group for group in groups if group["req_id"] is None)
    assert app.group_origins(loose) == "Câu nối"


def test_bridge_sentence_explains_it_is_not_model_written() -> None:
    """Câu hỏi thật của người dùng: "không kiểm chứng được thì sinh ra từ đâu?"

    Câu nối là chuỗi cố định trong `merge.py`, không do mô hình sinh và không
    có nguồn — nên ô tra nguồn phải nói thẳng, đừng để người đọc tự suy.
    """
    state = _state()
    bridge = next(
        sentence
        for section in state["sections"]
        for sentence in section["sentences"]
        if sentence["origin"] == "bridge"
    )

    def body(root, state, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, text, key="k")

    at = _render(body, state, bridge["text"])
    assert not at.exception
    notes = " ".join(item.value for item in at.info)
    assert "câu mẫu cố định" in notes
    assert "không do mô hình sinh ra" in notes
    assert "không có gì để kiểm chứng" in notes


def test_bridge_text_really_is_a_fixed_template() -> None:
    """Chốt lời giải thích trên bằng chính code: đổi cách sinh thì bài này gãy."""
    from rfp.generate.merge import BRIDGE_TEXTS

    assert BRIDGE_TEXTS
    assert all(isinstance(text, str) and text for text in BRIDGE_TEXTS.values())


def test_section_notes_read_as_a_short_chain() -> None:
    """Tách ý, xuống dòng, mũi tên — không phải một khối chữ liền."""
    from config.display_vi import SECTION_NOTE_VI

    for note in SECTION_NOTE_VI.values():
        assert "→" in note
        lines = [line for line in note.split("\n") if line.strip()]
        assert len(lines) >= 2
        assert all(len(line) < 120 for line in lines), note


def test_each_flow_box_carries_its_own_detail_on_hover() -> None:
    """Chi tiết của bước nằm NGAY trên ô đó, không phải một danh sách cuối trang.

    Dùng `<title>` của SVG: trình duyệt tự hiện tooltip, không JS, không CSS —
    nên không có gì để hỏng.
    """
    import app
    import re

    statuses = {name: "completed" for name in app.FLOW_STAGES}
    statuses["review"] = "running"
    svg = app.pipeline_svg(statuses, _state())
    titles = re.findall(r"<title>(.*?)</title>", svg, re.S)
    assert len(titles) == len(app.FLOW_STAGES)
    joined = " ".join(titles)
    # Mỗi tooltip nói tên bước, trạng thái, và có gọi LLM không
    assert "Đọc RFP — xong" in joined
    assert "Review chất lượng — đang chạy" in joined
    assert joined.count(app.LLM_BADGE) == 3
    assert joined.count(app.NO_LLM_BADGE) == len(app.FLOW_STAGES) - 3


def test_llm_map_covers_every_stage_in_the_flow() -> None:
    """Thiếu một bước là tooltip của bước đó im lặng về chuyện gọi LLM."""
    import app

    assert set(app.STAGE_USES_LLM) >= set(app.FLOW_STAGES)


# ── Lùi / tiến như Ctrl+Z, Ctrl+Y ────────────────────────────────────────

def _versioned_app(count: int = 3) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["result_state"] = _state()
    at.session_state["versions"] = [
        {"label": f"v{index + 1}", "state": _state()} for index in range(count)
    ]
    at.session_state["version_index"] = count - 1
    at.run()
    assert not at.exception, at.exception
    return at


def test_undo_and_redo_walk_the_version_history() -> None:
    """Đi trên chính `versions` chứ không nuôi một ngăn xếp undo riêng.

    Mọi thay đổi trong studio (chat, thay câu) đều đã đẩy một bản mới vào đó;
    hai kho song song chỉ để lệch nhau.
    """
    at = _versioned_app()
    assert at.session_state["version_index"] == 2

    at.button(key="undo_button").click().run()
    assert at.session_state["version_index"] == 1
    at.button(key="undo_button").click().run()
    assert at.session_state["version_index"] == 0

    at.button(key="redo_button").click().run()
    assert at.session_state["version_index"] == 1


def test_undo_is_disabled_at_the_first_version_redo_at_the_last() -> None:
    at = _versioned_app()
    buttons = {item.key: item for item in at.button}
    assert buttons["redo_button"].disabled      # đang ở bản cuối
    assert not buttons["undo_button"].disabled

    # Đi tới bản đầu bằng đúng đường người dùng đi. Đặt thẳng `version_index`
    # thì radio "Bản đang hiển thị" ghi đè lại ngay ở lượt sau — chính cái bẫy
    # mà `step_version` phải xử lý.
    at.button(key="undo_button").click().run()
    at.button(key="undo_button").click().run()
    assert at.session_state["version_index"] == 0
    buttons = {item.key: item for item in at.button}
    assert buttons["undo_button"].disabled      # đang ở bản đầu
    assert not buttons["redo_button"].disabled


def test_stepping_out_of_range_changes_nothing() -> None:
    import app

    at = _versioned_app(count=1)
    assert at.session_state["version_index"] == 0
    buttons = {item.key: item for item in at.button}
    assert buttons["undo_button"].disabled and buttons["redo_button"].disabled


def test_switching_version_drops_the_stale_translation() -> None:
    """Bản dịch thuộc về NỘI DUNG — giữ lại là hai ngôn ngữ nói hai chuyện."""
    at = _versioned_app()
    at.session_state["translation"] = "bản dịch cũ"
    at.run()
    at.button(key="undo_button").click().run()
    assert at.session_state["translation"] == ""


# ── Bảng điểm nguồn ở tab Tổng quan ──────────────────────────────────────

def test_source_scores_are_available_next_to_the_verdict() -> None:
    """Bảng chấm điểm nguồn nằm ngay dưới bảng Kết quả, không ở tab khác."""
    at = _app_with_result()
    labels = [item.label for item in at.get("expander")]
    assert "Điểm của từng câu nguồn đã lấy" in labels


# ── Thay câu bằng lựa chọn khác ──────────────────────────────────────────

def _state_with_alternatives() -> dict[str, Any]:
    state = _state()
    state["chapters"][0]["retrieval"] = {
        "selected": [],
        "candidates": [
            {"sent_id": "ALT-1", "text": "câu thay thế một", "scores": {"rerank": 0.6}},
            {"sent_id": "ALT-0", "text": "câu không liên quan", "scores": {"rerank": 0.0}},
        ],
    }
    return state


def test_alternatives_drop_the_zero_score_candidates() -> None:
    import app

    state = _state_with_alternatives()
    options = app.alternative_sources(state, state["sections"][0])
    assert [item["sent_id"] for item in options] == ["ALT-1"]


def test_popover_offers_a_replacement_button() -> None:
    state = _state_with_alternatives()
    text = state["sections"][0]["sentences"][0]["text"]

    def body(root, state, text):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import app as _app

        _app.render_source_lookup(state, text, key="k")

    at = _render(body, state, text)
    assert not at.exception
    shown = " ".join(item.value for item in at.markdown)
    assert "Thay bằng câu nguồn khác" in shown
    assert "câu thay thế một" in shown
    assert "Thay câu này vào" in [item.label for item in at.button]


def test_replacing_creates_a_new_version_and_relabels_the_sentence() -> None:
    """Ghi thành BẢN MỚI: sửa tại chỗ thì nút lùi không lùi lại được.

    Câu thay vào chưa qua bước kiểm lại nên không được gắn VERIFIED — điểm tin
    cậy tụt theo, và đó là sự thật.
    """
    import app

    state = _state_with_alternatives()
    old_text = state["sections"][0]["sentences"][0]["text"]
    candidate = app.alternative_sources(state, state["sections"][0])[0]

    def body(root, state, old_text, candidate):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app as _app

        st.session_state["versions"] = [{"label": "v1", "state": state}]
        st.session_state["version_index"] = 0
        _app.replace_sentence(old_text, candidate)

    at = _render(body, state, old_text, candidate)
    assert not at.exception
    versions = at.session_state["versions"]
    assert len(versions) == 2
    new_state = versions[1]["state"]
    replaced = new_state["sections"][0]["sentences"][0]
    assert replaced["text"] == "câu thay thế một"
    assert replaced["source_id"] == "ALT-1"
    assert replaced["origin"] == "precedent"
    assert replaced["verdict"] == "UNVERIFIABLE"
    assert replaced["replaced_by_user"] is True
    # Bản cũ KHÔNG bị sửa theo — nếu không thì lùi lại cũng ra bản mới
    assert versions[0]["state"]["sections"][0]["sentences"][0]["text"] == old_text


def test_replaced_sentence_is_still_traceable_and_scored() -> None:
    """Thay xong vẫn phải tra được nguồn và có điểm, như mọi câu khác."""
    import app

    state = _state_with_alternatives()
    old_text = state["sections"][0]["sentences"][0]["text"]
    candidate = app.alternative_sources(state, state["sections"][0])[0]

    def body(root, state, old_text, candidate):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app as _app

        st.session_state["versions"] = [{"label": "v1", "state": state}]
        st.session_state["version_index"] = 0
        _app.replace_sentence(old_text, candidate)
        new_state = st.session_state["versions"][1]["state"]
        _app.render_source_lookup(new_state, "câu thay thế một", key="k2")

    at = _render(body, state, old_text, candidate)
    assert not at.exception
    shown = " ".join(item.value for item in at.markdown)
    assert "ALT-1" in shown
    assert "Nguồn:" in shown and "Kiểm chứng:" in shown
    # Mục vẫn được chấm điểm lại sau khi thay
    new_state = at.session_state["versions"][1]["state"]
    assert new_state["sections"][0].get("confidence")


def test_replacement_is_blocked_when_it_names_a_certificate_we_lack() -> None:
    """BB-2: blocklist kiểm bằng regex, kể cả với câu người dùng tự chọn."""
    import app

    state = _state_with_alternatives()
    old_text = state["sections"][0]["sentences"][0]["text"]
    bad = {"sent_id": "BAD-1", "text": "ISO/IEC 27017認証を取得しています。",
           "scores": {"rerank": 0.9}}

    def body(root, state, old_text, bad):
        import sys as _s

        _s.path[:0] = [root + "/src", root]
        import streamlit as st

        import app as _app

        st.session_state["versions"] = [{"label": "v1", "state": state}]
        st.session_state["version_index"] = 0
        _app.replace_sentence(old_text, bad)

    at = _render(body, state, old_text, bad)
    assert not at.exception
    assert len(at.session_state["versions"]) == 1  # không sinh bản mới
    assert "lưới an toàn chặn" in at.session_state["replace_error"]


def test_undo_keeps_the_version_picker_in_step() -> None:
    """Radio "Bản đang hiển thị" và nút lùi phải nói cùng một con số.

    Radio có `key="version_picker"`; giá trị widget đó thắng tham số `index` ở
    lượt chạy sau, nên không đồng bộ thì bấm lùi xong màn hình nhảy về chỗ cũ
    mà không báo gì.
    """
    at = _versioned_app()
    at.button(key="undo_button").click().run()
    assert at.session_state["version_index"] == 1
    assert at.session_state["version_picker"] == 1
