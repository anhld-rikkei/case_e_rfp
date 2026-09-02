"""Bảng độ phủ golden set + chế độ sinh theo tiêu chí phủ.

Cả file này không gọi LLM lần nào.
"""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from eval.golden import coverage as cov  # noqa: E402
from eval.golden.generator import (  # noqa: E402
    COVERAGE_AXES,
    generate_combinatorial,
    generate_for_coverage,
)
from eval.golden.runner import load_cases  # noqa: E402
from eval.golden.schema import Assertion, GoldenCase, load_case  # noqa: E402


def _axis(axes, key):
    return next(item for item in axes if item.key == key)


def _case(text, assertions, case_id="tmp-001"):
    return GoldenCase(
        case_id=case_id,
        rfp_text=text,
        source="combinatorial",
        needs_review=False,
        assertions=assertions,
    )


# ── vũ trụ của mỗi trục lấy từ nguồn chân lý, không chép tay ───────────────

def test_universes_come_from_the_capability_sheet() -> None:
    data = cov.capability_data()
    assert set(cov.in_scope_terms()) == set(data["capabilities"]) | set(
        data["certifications_held"]
    )
    assert set(cov.out_scope_terms()) == set(data["capabilities_NOT_offered"]) | set(
        data["certifications_NOT_held"]
    )
    # Ngành KHÔNG phục vụ phải nằm trong vũ trụ, nếu không thì trục này chỉ
    # kiểm đường sáng và ca 医療 biến mất khỏi bộ test.
    assert cov.UNSERVED_INDUSTRY in cov.industries()
    assert cov.UNSERVED_INDUSTRY not in data["industries_served"]


def test_leaked_client_names_are_scanned_from_the_real_corpus() -> None:
    from config.templates_ja import COMMON_CLIENT_NAMES

    names = cov.leaked_client_names()
    assert len(names) == 7
    # Tên chung không được lọt vào vũ trụ: chặn 大手金融機関 là chặn nhầm.
    assert not set(names) & set(COMMON_CLIENT_NAMES)


# ── phép đếm phải trung thực ───────────────────────────────────────────────

def test_empty_set_covers_nothing() -> None:
    axes = cov.coverage_report([])
    assert all(axis.hit == 0 for axis in axes)
    assert all(axis.missing == axis.universe for axis in axes)
    assert not any(axis.is_full for axis in axes)


def test_term_in_rfp_without_an_assertion_does_not_count() -> None:
    """Từ có mặt trong RFP giả mà không ca nào canh kết quả thì CHƯA được test.

    Đây là chỗ dễ tự lừa nhất: đếm "từ xuất hiện" thì bảng báo phủ đủ trong khi
    không ai kiểm hệ thống trả lời cái gì cho từ đó.
    """
    term = cov.in_scope_terms()[0]
    text = f"発注業種：金融\n\n第3章 技術要件\n3.1 {term}に対応できること。\n"

    silent = _case(text, [Assertion("must_route", "completed", "chỉ kiểm rẽ nhánh")])
    assert term not in _axis(cov.coverage_report([silent]), "in_scope").covered

    guarded = _case(text, [Assertion("must_cover", "3.1", "phải trả lời được")])
    assert term in _axis(cov.coverage_report([guarded]), "in_scope").covered


def test_blocklist_axis_reads_must_not_contain_targets() -> None:
    term = cov.out_scope_terms()[0]
    case = _case(
        "発注業種：金融\n\n第3章 技術要件\n3.1 何かに対応できること。\n",
        [Assertion("must_not_contain", term, "không được tuyên bố")],
    )
    axis = _axis(cov.coverage_report([case]), "blocklist")
    assert axis.covered == (term,)
    # Cùng ca đó KHÔNG phủ trục out_scope: chặn đầu ra khác với hỏi ở đầu vào.
    assert _axis(cov.coverage_report([case]), "out_scope").hit == 0


def test_edge_case_axis_detects_the_three_shapes() -> None:
    no_security = _case(
        "発注業種：金融\n\n第3章 技術要件\n3.1 x。\n",
        [Assertion("must_cover", "3.1", "r")],
    )
    no_industry = _case(
        "第3章 技術要件\n3.1 x。\n", [Assertion("must_cover", "3.1", "r")]
    )
    no_number = _case(
        "発注業種：金融\n技術要件\n3.1 x。\n", [Assertion("must_cover", "3.1", "r")]
    )
    axis = _axis(cov.coverage_report([no_security, no_industry, no_number]), "edge_cases")
    assert axis.is_full


def test_report_rows_name_the_missing_items() -> None:
    """Ô 'còn thiếu' phải gọi tên, không chỉ nói còn mấy cái."""
    rows = cov.report_rows(cov.coverage_report([]))
    leak_row = next(row for row in rows if "khách hàng" in row["trục"])
    for name in cov.leaked_client_names():
        assert name in leak_row["còn thiếu"]
    assert leak_row["đạt"] == "⚠"

    full = cov.report_rows(cov.coverage_report(generate_for_coverage(existing=[])))
    assert {row["đạt"] for row in full} == {"✅"}
    assert {row["còn thiếu"] for row in full} == {"—"}


# ── sinh theo tiêu chí phủ ────────────────────────────────────────────────

def test_generating_from_scratch_covers_every_axis() -> None:
    cases = generate_for_coverage(existing=[])
    axes = cov.coverage_report(cases)
    assert all(axis.is_full for axis in axes), [
        (axis.key, axis.missing) for axis in axes if not axis.is_full
    ]


def test_generation_is_deterministic() -> None:
    first = generate_for_coverage(existing=[])
    second = generate_for_coverage(existing=[])
    assert [case.rfp_text for case in first] == [case.rfp_text for case in second]
    assert [case.case_id for case in first] == [case.case_id for case in second]


def test_generating_twice_adds_nothing() -> None:
    """Bấm hai lần không đẻ ra ca trùng — đã phủ rồi thì không sinh nữa."""
    cases = generate_for_coverage(existing=[])
    assert generate_for_coverage(existing=cases) == []


def test_selecting_one_axis_fills_only_that_axis() -> None:
    cases = generate_for_coverage(["client_leak"], existing=[])
    axes = cov.coverage_report(cases)
    assert _axis(axes, "client_leak").is_full
    assert not _axis(axes, "edge_cases").is_full


def test_unknown_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="Trục phủ không có"):
        generate_for_coverage(["khong-co-truc-nay"], existing=[])


def test_every_axis_can_be_filled_on_its_own() -> None:
    for key in COVERAGE_AXES:
        cases = generate_for_coverage([key], existing=[])
        assert _axis(cov.coverage_report(cases), key).is_full, key


def test_generated_cases_carry_tier_and_axis() -> None:
    """Ba tầng theo đề bài: phổ biến · biên · cấm-sai."""
    cases = generate_for_coverage(existing=[])
    tiers = {case.metadata["tier"] for case in cases}
    assert tiers == {cov.TIER_COMMON, cov.TIER_EDGE, cov.TIER_FORBIDDEN}
    assert all(case.metadata["axis"] in COVERAGE_AXES for case in cases)
    assert all(case.source == "coverage" for case in cases)
    assert all(not case.needs_review for case in cases)


def test_coverage_set_is_smaller_than_the_current_golden_set() -> None:
    """15 ca chọn theo trục phủ hơn 39 ca sinh mù — đây là lý do có chế độ này."""
    cases = generate_for_coverage(existing=[])
    assert len(cases) < len(load_cases())


# ── không được làm hỏng cái đang chạy ─────────────────────────────────────

def test_combinatorial_output_is_unchanged() -> None:
    """30 ca đã commit phải tái sinh y hệt sau khi thêm khoá `chapter`."""
    saved = [
        load_case(path)
        for path in sorted(
            (PROJECT_ROOT / "synthetic" / "golden_test_set" / "generated").glob("*.json")
        )
    ]
    fresh = generate_combinatorial(len(saved))
    assert [case.rfp_text for case in fresh] == [case.rfp_text for case in saved]
    assert [
        [(item.kind, item.target) for item in case.assertions] for case in fresh
    ] == [[(item.kind, item.target) for item in case.assertions] for case in saved]


def test_explicit_chapter_beats_the_default_placement() -> None:
    """Không có khoá này thì 4/6 chương không bao giờ có ca."""
    from eval.golden.generator import _preferred_chapter

    capability = {"term": "RPA導入", "category": "capability"}
    assert _preferred_chapter(capability) == "技術要件"
    assert _preferred_chapter({**capability, "chapter": "納期・体制"}) == "納期・体制"


def test_current_golden_set_still_has_exactly_one_gap() -> None:
    """Số liệu cơ sở: 39 ca hiện tại phủ 7/8 trục, hở đúng trục rò rỉ khách hàng.

    Bài này là cái mốc — vá xong thì nó đổi, và đổi thì phải cố ý.
    """
    axes = cov.coverage_report(load_cases())
    gaps = [axis.key for axis in axes if not axis.is_full]
    assert gaps == ["client_leak"]


# ── Chú thích ca test: người chấm phải hiểu mà không cần đọc code ─────────

def test_brief_explains_every_assertion_in_plain_words() -> None:
    """Mỗi điều kiện phải kèm câu yêu cầu gốc và lý do — không để trơ tên kỹ thuật."""
    from eval.golden.describe import describe_case

    case = _case(
        "発注業種：金融\n\n第3章 技術要件\n3.1 ネットワーク構築・保守に対応できること。\n"
        "\n第4章 セキュリティ要件\n4.1 ISO/IEC 27018認証を有すること。\n",
        [
            Assertion("must_cover", "3.1", "trong năng lực"),
            Assertion("must_flag_insufficient", "4.1", "ngoài năng lực"),
            Assertion("must_not_contain", "ISO/IEC 27018", "không được tuyên bố"),
        ],
    )
    brief = describe_case(case)
    assert brief.tier == "cấm-sai"

    # Điều kiện phải nhắc lại nguyên văn yêu cầu, không chỉ nói "Mục 4.1"
    assert "ISO/IEC 27018認証を有すること。" in brief.checks[1].what
    assert "ネットワーク構築・保守" in brief.checks[0].what
    # Và phải nói VÌ SAO
    assert "KHÔNG có chứng chỉ" in brief.checks[2].why

    text = " ".join(check.what + check.why for check in brief.checks)
    for jargon in ("must_cover", "must_flag_insufficient", "must_not_contain"):
        assert jargon not in text, jargon


def test_brief_summarises_what_goes_in() -> None:
    from eval.golden.describe import describe_case

    normal = describe_case(
        _case(
            "発注業種：金融\n\n第3章 技術要件\n3.1 x。\n",
            [Assertion("must_cover", "3.1", "r")],
        )
    )
    assert any("金融" in line for line in normal.inputs)
    assert any("1 yêu cầu" in line for line in normal.inputs)

    # Ca biên: phải nói RÕ là cố ý thiếu, không để người test tưởng lỗi dữ liệu
    missing = describe_case(
        _case("第3章 技術要件\n3.1 x。\n", [Assertion("must_cover", "3.1", "r")])
    )
    assert any("cố ý" in line for line in missing.inputs)


def test_brief_names_the_route_and_ask_targets_in_plain_words() -> None:
    from eval.golden.describe import describe_case

    brief = describe_case(
        _case(
            "第3章 技術要件\n3.1 x。\n",
            [
                Assertion("must_ask_user", "industry", "thiếu ngành"),
                Assertion("must_route", "completed", "vẫn phải chạy xong"),
                Assertion("must_route", "industry:医療", "giữ đúng ngành"),
            ],
        )
    )
    what = [check.what for check in brief.checks]
    assert "業種" in what[0] and "industry" not in what[0]
    assert "chạy xong" in what[1] and "completed" not in what[1]
    assert "医療" in what[2]


def test_brief_purpose_reads_as_a_sentence_for_every_saved_case() -> None:
    """Không ca nào rơi vào nhánh mặc định trống rỗng."""
    from eval.golden.describe import describe_case

    for case in load_cases():
        brief = describe_case(case)
        assert brief.purpose.endswith((".", ")")), case.case_id
        assert brief.tier in {"phổ biến", "biên", "cấm-sai"}, case.case_id
        assert len(brief.checks) == len(case.assertions)
        assert "RFP nền" not in brief.purpose, case.case_id


def test_leak_check_explains_it_is_a_client_name() -> None:
    from eval.golden.describe import describe_case

    name = cov.leaked_client_names()[0]
    brief = describe_case(
        _case(
            "発注業種：金融\n\n第3章 技術要件\n3.1 x。\n",
            [Assertion("must_not_contain", name, "tên khách hàng")],
        )
    )
    assert "khách hàng" in brief.checks[0].why
    assert "rò rỉ" in brief.checks[0].why
