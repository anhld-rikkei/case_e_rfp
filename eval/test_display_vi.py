"""Test nhãn hiển thị tiếng Việt (v1.4).

Trọng tâm: mapping **không sót enum nào**, và nhãn hiển thị **không rò ngược**
vào dữ liệu. Mọi giá trị kỳ vọng lấy từ nguồn canonical trong code, không chép
tay — thêm một enum mới mà quên dịch là test đỏ ngay.
"""
import sys
import typing
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from config import display_vi  # noqa: E402
from config.display_vi import label, to_raw  # noqa: E402
from rfp.check import VALID_ORIGINS  # noqa: E402
from rfp.generate.claim_check import ClaimVerdict  # noqa: E402
from rfp.generate.coverage import SectionStatus  # noqa: E402
from rfp.graph import PIPELINE_STAGES  # noqa: E402
from rfp.review import ISSUE_TYPES, PERSONAS, SEVERITIES, ReviewIssue  # noqa: E402


def _literal_values(annotation) -> set[str]:
    return set(typing.get_args(annotation))


# ── Không sót enum: mọi giá trị canonical đều có nhãn ─────────────────────

def test_origin_mapping_covers_all_valid_origins() -> None:
    """Phủ đủ enum canonical, cộng nhãn chỉ-UI của chat-refine (v1.7).

    `user` không nằm trong `check.py:VALID_ORIGINS` vì pipeline không bao giờ
    sinh ra nó — nó là nhãn hậu-pipeline. Nhưng UI vẫn phải dịch được.
    """
    expected = VALID_ORIGINS | display_vi.UI_ONLY_ORIGINS
    assert set(display_vi.ORIGIN_VI) == expected
    assert set(display_vi.ORIGIN_HINT) == expected


def test_ui_only_origins_are_not_pipeline_origins() -> None:
    """Nhãn chỉ-UI không được lẫn vào enum mà contract checker chấp nhận."""
    assert display_vi.UI_ONLY_ORIGINS & VALID_ORIGINS == set()


def test_verdict_mapping_covers_all_claim_verdicts() -> None:
    verdicts = _literal_values(ClaimVerdict.model_fields["verdict"].annotation)
    expected = verdicts | display_vi.UI_ONLY_VERDICTS
    assert set(display_vi.VERDICT_VI) == expected
    assert set(display_vi.VERDICT_ICON) == expected
    assert set(display_vi.VERDICT_HINT) == expected


def test_user_provided_verdict_says_it_is_unverified() -> None:
    """Nhãn phải nói thẳng là chưa kiểm chứng, không được nghe như đã duyệt."""
    text = display_vi.VERDICT_VI["USER_PROVIDED"]
    assert "chưa kiểm chứng" in text
    assert "không kiểm chứng" in display_vi.VERDICT_HINT["USER_PROVIDED"]


def test_section_status_mapping_covers_all_statuses() -> None:
    statuses = _literal_values(SectionStatus)
    assert set(display_vi.SECTION_STATUS_VI) == statuses
    assert set(display_vi.SECTION_STATUS_ICON) == statuses


def test_stage_mapping_covers_every_pipeline_stage() -> None:
    assert set(display_vi.STAGE_VI) == set(PIPELINE_STAGES)


def test_stage_status_mapping_has_icon_for_every_status() -> None:
    assert set(display_vi.STAGE_STATUS_VI) == set(display_vi.STAGE_STATUS_ICON)


def test_blocked_is_distinct_from_failed() -> None:
    """Guard chặn là hành vi ĐÚNG, không phải lỗi kỹ thuật — phải khác nhãn."""
    assert display_vi.STAGE_STATUS_VI["blocked"] != display_vi.STAGE_STATUS_VI["failed"]
    assert display_vi.STAGE_STATUS_ICON["blocked"] != display_vi.STAGE_STATUS_ICON["failed"]
    assert "chặn" in display_vi.STAGE_STATUS_VI["blocked"]


def test_review_mappings_cover_severity_issue_type_and_persona() -> None:
    assert set(display_vi.SEVERITY_VI) == set(SEVERITIES)
    assert set(display_vi.ISSUE_TYPE_VI) == set(ISSUE_TYPES)
    assert set(display_vi.PERSONA_VI) == set(PERSONAS)


def test_issue_type_mapping_matches_schema_literal() -> None:
    """Đối chiếu cả với schema pydantic, không chỉ với hằng số ISSUE_TYPES."""
    schema_types = _literal_values(ReviewIssue.model_fields["issue_type"].annotation)
    assert set(display_vi.ISSUE_TYPE_VI) == schema_types


def test_run_status_mapping_covers_every_status_graph_can_emit() -> None:
    # Các giá trị state["status"] mà graph.py thực sự gán.
    emitted = {
        "completed",
        "ask_user",
        "partial",
        "guard_blocked",
        "needs_input",
        "ready",
    }
    assert emitted <= set(display_vi.RUN_STATUS_VI)


def test_retrieval_stage_mapping_covers_four_stages() -> None:
    assert set(display_vi.RETRIEVAL_STAGE_VI) == {
        "attribute",
        "query_embed",
        "rerank",
        "mmr",
    }


# ── Nhãn không rỗng, không trùng nhau ─────────────────────────────────────

@pytest.mark.parametrize(
    "mapping_name",
    [
        "ORIGIN_VI",
        "VERDICT_VI",
        "SECTION_STATUS_VI",
        "STAGE_VI",
        "STAGE_STATUS_VI",
        "SEVERITY_VI",
        "ISSUE_TYPE_VI",
        "RUN_STATUS_VI",
        "ROUTE_METHOD_VI",
        "RETRIEVAL_STAGE_VI",
    ],
)
def test_labels_are_non_empty_and_unique(mapping_name: str) -> None:
    mapping = getattr(display_vi, mapping_name)
    values = list(mapping.values())
    assert all(value.strip() for value in values)
    # Hai giá trị khác nhau mà cùng một nhãn thì người đọc không phân biệt được.
    assert len(set(values)) == len(values)


# ── label(): giá trị lạ hiện nguyên văn, không nuốt thành "—" ─────────────

def test_label_translates_known_key() -> None:
    assert label(display_vi.VERDICT_VI, "VERIFIED") == "Có nguồn xác thực"


def test_label_shows_unknown_value_verbatim() -> None:
    """Enum mới chưa dịch phải lộ ra, không được giấu thành '—'."""
    assert label(display_vi.VERDICT_VI, "BRAND_NEW") == "BRAND_NEW"


def test_label_handles_none_and_empty() -> None:
    assert label(display_vi.VERDICT_VI, None) == display_vi.UNKNOWN
    assert label(display_vi.VERDICT_VI, "") == display_vi.UNKNOWN


# ── to_raw(): bộ lọc phải map NGƯỢC về giá trị gốc ────────────────────────

def test_to_raw_maps_labels_back_to_enum_values() -> None:
    labels = ["Năng lực công ty", "Hồ sơ quá khứ"]
    assert to_raw(display_vi.ORIGIN_VI, labels) == ["capability", "precedent"]


def test_to_raw_round_trips_every_value() -> None:
    for mapping in (
        display_vi.ORIGIN_VI,
        display_vi.VERDICT_VI,
        display_vi.SECTION_STATUS_VI,
    ):
        raw_values = list(mapping)
        labels = [mapping[value] for value in raw_values]
        assert to_raw(mapping, labels) == raw_values


def test_to_raw_keeps_untranslated_value_as_is() -> None:
    """Giá trị gốc chưa có bản dịch vẫn phải lọc đúng."""
    assert to_raw(display_vi.ORIGIN_VI, ["capability", "LẠ"]) == ["capability", "LẠ"]
