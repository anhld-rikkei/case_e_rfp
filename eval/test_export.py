"""Test xuất hồ sơ markdown + nhãn nháp (Bước 7, v1.3).

Cách ly: thuần hàm, không Streamlit, không mạng.
"""
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from rfp.export import (  # noqa: E402
    DRAFT_BANNER_TITLE,
    REVIEWER_CHECKLIST,
    export_filename,
    to_markdown,
)
from rfp.guard import GuardViolation  # noqa: E402


def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "rfp": {"rfp_id": "RFP-2025-001"},
        "chapters": [
            {
                "id": "3",
                "title": "技術要件",
                "requirements": [
                    {"req_id": "3.1", "text": "基幹システム構築に対応できること。"},
                    {"req_id": "3.2", "text": "24時間監視に対応できること。"},
                ],
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
                    }
                ],
            }
        ],
        "trace": {"review": {"enabled": False}},
    }
    state.update(overrides)
    return state


# ── Nhãn nháp phải nằm TRONG file, không chỉ trên UI ───────────────────────

def test_export_starts_with_draft_banner() -> None:
    markdown = to_markdown(_state())
    assert DRAFT_BANNER_TITLE in markdown.split("\n")[0]


def test_export_repeats_banner_at_end() -> None:
    """Bản in nhiều trang không được mất nhãn ở trang cuối."""
    markdown = to_markdown(_state())
    assert DRAFT_BANNER_TITLE in markdown.rsplit("\n---", 1)[-1]


def test_export_contains_full_reviewer_checklist() -> None:
    markdown = to_markdown(_state())
    for line in REVIEWER_CHECKLIST.strip().split("\n"):
        assert line.strip() in markdown


def test_checklist_has_unticked_boxes_and_signature_line() -> None:
    assert "- [ ]" in REVIEWER_CHECKLIST
    assert "- [x]" not in REVIEWER_CHECKLIST.lower()
    assert "Người rà soát" in REVIEWER_CHECKLIST


# ── Nội dung hồ sơ ────────────────────────────────────────────────────────

def test_export_contains_all_five_section_titles() -> None:
    sections = [
        {
            "key": f"s{i}",
            "title_ja": f"第{i}節",
            "title_vi": f"Muc {i}",
            "source_chapters": [],
            "status": "OK",
            "note": "",
            "sentences": [],
        }
        for i in range(1, 6)
    ]
    markdown = to_markdown(_state(sections=sections))
    for i in range(1, 6):
        assert f"### {i}. 第{i}節" in markdown


def test_export_shows_coverage_and_flags_missing_requirement() -> None:
    markdown = to_markdown(_state())
    assert "| 3.1 |" in markdown and "ĐÁP ỨNG" in markdown
    # 3.2 không có câu nào phủ -> phải bị đánh dấu THIẾU, không im lặng bỏ qua
    assert "| 3.2 |" in markdown and "**THIẾU**" in markdown
    assert "Đáp ứng 1/2 requirement" in markdown
    assert "Còn **1** requirement chưa có bằng chứng" in markdown


def test_export_lists_provenance_for_every_sentence() -> None:
    markdown = to_markdown(_state())
    assert "## Nguồn từng câu" in markdown
    assert "bảng năng lực · capabilities:基幹システム構築" in markdown
    assert "VERIFIED" in markdown


def test_export_marks_bridge_sentence_as_sourceless() -> None:
    state = _state()
    state["sections"][0]["sentences"].append(
        {
            "text": "つなぎの文です。",
            "origin": "bridge",
            "source_id": None,
            "req_ids": [],
            "verdict": "UNVERIFIABLE",
        }
    )
    markdown = to_markdown(state)
    assert "câu nối (không có nguồn)" in markdown


def test_export_surfaces_section_note_as_warning() -> None:
    state = _state()
    state["sections"][0]["note"] = "3.2 chưa có bằng chứng."
    state["sections"][0]["status"] = "INSUFFICIENT_EVIDENCE"
    markdown = to_markdown(state)
    assert "> ⚠ 3.2 chưa có bằng chứng." in markdown
    assert "`INSUFFICIENT_EVIDENCE`" in markdown


def test_export_includes_review_rounds_when_enabled() -> None:
    state = _state()
    state["trace"] = {
        "review": {
            "enabled": True,
            "history": [
                {
                    "round": 1,
                    "score": {"critical": 1, "major": 2, "minor": 0},
                    "critical_sections": ["technical"],
                    "fixes_applied": 1,
                }
            ],
        }
    }
    markdown = to_markdown(state)
    assert "## Vòng review tự động" in markdown
    assert "| 1 | 1 | 2 | 0 | technical | 1 |" in markdown


def test_export_omits_review_table_when_disabled() -> None:
    assert "## Vòng review tự động" not in to_markdown(_state())


# ── Guard: không xuất được bản có chuỗi cấm ───────────────────────────────

def test_export_blocked_when_generated_sentence_has_forbidden_term() -> None:
    state = _state()
    state["sections"][0]["sentences"][0]["text"] = "当社はISO 27017認証を取得済み"
    with pytest.raises(GuardViolation, match="blocklist"):
        to_markdown(state)


def test_export_allows_rfp_requirement_quoting_forbidden_term() -> None:
    """RFP có quyền YÊU CẦU 27017; cấm là hệ thống tự nhận có nó.

    Quét blocklist lên phần trích nguyên văn RFP sẽ báo động giả và chặn mất
    một hồ sơ hợp lệ — đúng tình huống của golden case mutation 27017.
    """
    state = _state()
    state["chapters"][0]["requirements"].append(
        {"req_id": "3.3", "text": "ISO/IEC 27017認証を保有していることが望ましい。"}
    )
    markdown = to_markdown(state)
    assert "ISO/IEC 27017認証を保有していることが望ましい。" in markdown
    assert "| 3.3 |" in markdown and "**THIẾU**" in markdown


def test_export_blocked_when_client_name_leaks() -> None:
    state = _state()
    state["sections"][0]["sentences"][0]["text"] = "新生証券様向けの実績があります。"
    with pytest.raises(GuardViolation, match="client name leaked"):
        to_markdown(state)


# ── Tên file ──────────────────────────────────────────────────────────────

def test_export_filename_uses_rfp_id() -> None:
    assert export_filename(_state()) == "proposal_draft_RFP-2025-001.md"


def test_export_filename_falls_back_without_rfp() -> None:
    assert export_filename({"sections": []}) == "proposal_draft_unknown.md"
