"""Test review loop (Bước 6, v1.2).

Cách ly: mock `structured`/`generate`, không gọi mạng, không build index.
Trọng tâm là ba chốt cứng: reviewer không đụng compliance, guard vẫn chạy sau
review loop (BB-2), và chỉ mục có issue critical mới bị sửa.
"""
import sys
import time
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.graph as graph_module  # noqa: E402
import rfp.review as review_module  # noqa: E402
from rfp.guard import GuardViolation  # noqa: E402
from rfp.review import (  # noqa: E402
    ISSUE_TYPES,
    ReviewIssue,
    ReviewResult,
    apply_fixes,
    critical_section_keys,
    is_clean,
    review_sections,
    score,
)


def _sentence(text: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "text": text,
        "origin": "precedent",
        "source_id": "PROP-001-S01",
        "req_ids": ["3.1"],
        "verdict": "VERIFIED",
    }
    base.update(overrides)
    return base


def _section(key: str, sentences: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key": key,
        "title_ja": "技術要件",
        "title_vi": "Yêu cầu kỹ thuật",
        "source_chapters": ["3"],
        "sentences": sentences,
        "status": "OK",
        "note": "",
    }


def _issue(section_key: str, severity: str, original: str, **kw: Any) -> dict[str, Any]:
    issue = {
        "section_key": section_key,
        "severity": severity,
        "issue_type": kw.get("issue_type", "tone"),
        "original_text": original,
        "suggested_fix": kw.get("suggested_fix", "文体を整えてください"),
    }
    return issue


# ── Ranh giới compliance: từ vựng đóng, không có loại nào về chứng chỉ ──────

def test_issue_types_contain_no_compliance_category() -> None:
    assert set(ISSUE_TYPES) == {
        "structure",
        "tone",
        "redundancy",
        "unclear_reference",
        "format",
    }


def test_review_prompt_forbids_compliance_judgement() -> None:
    assert "認証" in review_module.REVIEW_SYSTEM
    assert "指摘しないでください" in review_module.REVIEW_SYSTEM


def test_review_drops_issue_for_other_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model trả section_key bịa -> bỏ, không đoán ý model."""
    sections = [_section("technical", [_sentence("文A")])]

    def fake_structured(system, user, model_cls):
        return ReviewResult(
            issues=[
                ReviewIssue(
                    section_key="mục-không-tồn-tại",
                    severity="critical",
                    issue_type="tone",
                    original_text="文A",
                    suggested_fix="x",
                )
            ]
        )

    monkeypatch.setattr(review_module, "structured", fake_structured)
    issues, calls = review_sections(sections)
    assert issues == []
    assert calls == len(review_module.PERSONAS)  # mỗi persona soi 1 lần


def test_empty_section_costs_no_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def fake_structured(system, user, model_cls):
        called["n"] += 1
        return ReviewResult(issues=[])

    monkeypatch.setattr(review_module, "structured", fake_structured)
    issues, calls = review_sections([_section("empty", [])])
    assert issues == [] and calls == 0 and called["n"] == 0


# ── Severity + chọn mục regen ──────────────────────────────────────────────

def test_score_counts_by_severity() -> None:
    issues = [
        _issue("a", "critical", "x"),
        _issue("a", "major", "y"),
        _issue("b", "minor", "z"),
    ]
    assert score(issues) == {"critical": 1, "major": 1, "minor": 1}
    assert not is_clean(issues)
    assert is_clean([_issue("a", "major", "y")])


def test_only_critical_sections_are_regenerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sections = [
        _section("technical", [_sentence("技術の文です。")]),
        _section("security", [_sentence("セキュリティの文です。")]),
    ]
    issues = [
        _issue("technical", "critical", "技術の文です。"),
        _issue("security", "major", "セキュリティの文です。"),  # major -> khong sua
    ]
    monkeypatch.setattr(
        review_module, "generate", lambda *a, **k: "技術の文を整えました。"
    )

    assert critical_section_keys(issues) == ["technical"]
    updated, calls, applied = apply_fixes(sections, issues)

    assert calls == 1
    assert updated[0]["sentences"][0]["text"] == "技術の文を整えました。"
    # Mục major giữ nguyên TỪNG BYTE, kể cả object
    assert updated[1] == sections[1]
    assert [item["section_key"] for item in applied] == ["technical"]


def test_no_critical_means_no_llm_call_and_no_change() -> None:
    sections = [_section("technical", [_sentence("文A")])]
    updated, calls, applied = apply_fixes(sections, [_issue("technical", "minor", "文A")])
    assert updated is sections and calls == 0 and applied == []


def test_fix_preserves_provenance_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """BB-4: sửa câu chữ không được làm mất dấu vết nguồn."""
    sections = [_section("technical", [_sentence("元の文です。")])]
    monkeypatch.setattr(review_module, "generate", lambda *a, **k: "整えた文です。")
    updated, _, _ = apply_fixes(
        sections, [_issue("technical", "critical", "元の文です。")]
    )
    sentence = updated[0]["sentences"][0]
    assert sentence["origin"] == "precedent"
    assert sentence["source_id"] == "PROP-001-S01"
    assert sentence["req_ids"] == ["3.1"]
    assert sentence["verdict"] == "VERIFIED"


# ── Guard số liệu khi sửa câu ──────────────────────────────────────────────

def test_fix_rejected_when_numbers_change(monkeypatch: pytest.MonkeyPatch) -> None:
    original = "在庫精度を20%向上した実績があります。"
    sections = [_section("technical", [_sentence(original)])]
    monkeypatch.setattr(
        review_module,
        "generate",
        lambda *a, **k: "在庫精度を55%向上した実績があります。",
    )
    updated, calls, applied = apply_fixes(
        sections, [_issue("technical", "critical", original)]
    )
    assert updated[0]["sentences"][0]["text"] == original  # giữ bản gốc
    assert applied == [] and calls == 1


def test_fix_rejected_when_it_introduces_forbidden_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "セキュリティ体制を整備しています。"
    sections = [_section("security", [_sentence(original)])]
    monkeypatch.setattr(
        review_module,
        "generate",
        lambda *a, **k: "ISO/IEC 27017認証に基づく体制を整備しています。",
    )
    updated, _, applied = apply_fixes(
        sections, [_issue("security", "critical", original)]
    )
    assert updated[0]["sentences"][0]["text"] == original
    assert applied == []


def test_fix_rejected_when_empty_or_decorated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = "文Aです。"
    sections = [_section("technical", [_sentence(original)])]
    for bad in ("", "   ", "【整えた文】文Aです。"):
        monkeypatch.setattr(review_module, "generate", lambda *a, **k: bad)
        updated, _, _ = apply_fixes(
            sections, [_issue("technical", "critical", original)]
        )
        assert updated[0]["sentences"][0]["text"] == original


# ── Điều kiện dừng của vòng lặp ────────────────────────────────────────────

def _route(rounds: int, critical: int, fixes: int, enabled: bool = True) -> str:
    state = {
        "review_rounds": rounds,
        "trace": {
            "review": {
                "enabled": enabled,
                "history": [
                    {
                        "round": rounds,
                        "score": {"critical": critical, "major": 0, "minor": 0},
                        "fixes_applied": fixes,
                    }
                ],
            }
        },
    }
    return graph_module.route_after_review(state)


def test_loop_stops_when_no_critical_left() -> None:
    assert _route(rounds=1, critical=0, fixes=1) == "done"


def test_loop_repeats_while_critical_and_progress() -> None:
    assert _route(rounds=1, critical=2, fixes=1) == "again"


def test_loop_stops_at_max_rounds() -> None:
    assert graph_module.MAX_REVIEW_ROUNDS == 3
    assert _route(rounds=3, critical=2, fixes=1) == "done"


def test_loop_stops_when_round_fixed_nothing() -> None:
    """Còn critical nhưng không sửa nổi câu nào -> lặp thêm chỉ tốn lệnh gọi."""
    assert _route(rounds=1, critical=2, fixes=0) == "done"


def test_loop_skipped_when_review_disabled() -> None:
    assert _route(rounds=0, critical=5, fixes=0, enabled=False) == "done"


def test_review_node_disabled_makes_no_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module, "REVIEW_ENABLED", False)

    def explode(*args, **kwargs):
        raise AssertionError("review bị tắt mà vẫn gọi LLM")

    monkeypatch.setattr(graph_module, "review_sections", explode)
    result = graph_module.review(
        {"sections": [_section("technical", [_sentence("文A")])], "trace": {}}
    )
    assert result["review_rounds"] == 0
    assert result["trace"]["review"]["enabled"] is False


# ── BB-2: guard chạy SAU review loop, reviewer không có đường vòng ─────────

def test_guard_runs_after_review_loop_in_graph_topology() -> None:
    """review -> assemble; assemble là nơi final_guard chạy, và nó ở cuối."""
    assert graph_module.PIPELINE_STAGES.index("review") < (
        graph_module.PIPELINE_STAGES.index("assemble")
    )
    assert graph_module._next_stage("generate_per_section", {}) == "review"
    assert graph_module._next_stage("review", {}) == "assemble"


def test_assemble_still_blocks_forbidden_text_after_review() -> None:
    """Dù review đã chạy, câu chứa chuỗi cấm vẫn không xuất bản được."""
    state = {
        "sections": [
            {
                "title_ja": "認証・コンプライアンス",
                "sentences": [{"text": "当社はISO 27017認証を取得済み"}],
            }
        ],
        "trace": {
            "llm_calls": 0,
            "retrieval_stats": {},
            "dedup": {"before": 0, "after": 0},
            "path": ["review"],
        },
    }
    with pytest.raises(GuardViolation, match="blocklist"):
        graph_module.assemble(state)


# ── #10 multi-persona ─────────────────────────────────────────────────────

def test_no_compliance_persona_exists() -> None:
    """Chốt cứng: Compliance ở lại tầng deterministic, không lên tầng LLM."""
    assert set(review_module.PERSONAS) == {"coverage", "quality"}
    for system in review_module.PERSONAS.values():
        assert "指摘しないでください" in system  # cấm phán về chứng chỉ/số liệu


def test_every_persona_is_bound_to_closed_issue_vocabulary() -> None:
    for system in review_module.PERSONAS.values():
        for issue_type in ISSUE_TYPES:
            assert issue_type in system


def test_personas_run_for_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_structured(system, user, model_cls):
        persona = next(
            name for name, text in review_module.PERSONAS.items() if text == system
        )
        section_key = user.split("【section_key】")[1].split("\n")[0]
        seen.append((persona, section_key))
        return ReviewResult(issues=[])

    monkeypatch.setattr(review_module, "structured", fake_structured)
    sections = [
        _section("technical", [_sentence("文A")]),
        _section("security", [_sentence("文B")]),
    ]
    issues, calls = review_sections(sections)

    assert calls == 4  # 2 mục × 2 persona
    assert sorted(seen) == [
        ("coverage", "security"),
        ("coverage", "technical"),
        ("quality", "security"),
        ("quality", "technical"),
    ]
    assert issues == []


def test_dedupe_merges_same_section_and_issue_type() -> None:
    issues = [
        {**_issue("technical", "major", "文A", issue_type="tone"), "persona": "quality"},
        {
            **_issue("technical", "critical", "文A", issue_type="tone"),
            "persona": "coverage",
        },
        {
            **_issue("technical", "minor", "文A", issue_type="format"),
            "persona": "quality",
        },
    ]
    merged = review_module.deduplicate_issues(issues)

    assert len(merged) == 2  # (technical,tone) gộp 1, (technical,format) giữ
    tone = next(i for i in merged if i["issue_type"] == "tone")
    # Giữ bản NẶNG nhất, không phải bản gặp trước
    assert tone["severity"] == "critical"


def test_dedupe_keeps_issues_from_different_sections() -> None:
    issues = [
        {**_issue("technical", "major", "x", issue_type="tone"), "persona": "quality"},
        {**_issue("security", "major", "y", issue_type="tone"), "persona": "quality"},
    ]
    assert len(review_module.deduplicate_issues(issues)) == 2


def test_result_is_stable_regardless_of_completion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ThreadPool trả theo thứ tự hoàn thành; kết quả phải không phụ thuộc nó."""

    def make_fake(delay_first: bool):
        def fake_structured(system, user, model_cls):
            persona = next(
                name for name, text in review_module.PERSONAS.items() if text == system
            )
            if (persona == "coverage") is delay_first:
                time.sleep(0.02)
            return ReviewResult(
                issues=[
                    ReviewIssue(
                        section_key="technical",
                        severity="major" if persona == "quality" else "critical",
                        issue_type="tone",
                        original_text="文A",
                        suggested_fix=f"fix tu {persona}",
                    )
                ]
            )

        return fake_structured

    sections = [_section("technical", [_sentence("文A")])]

    monkeypatch.setattr(review_module, "structured", make_fake(True))
    first = review_sections(sections)[0]
    monkeypatch.setattr(review_module, "structured", make_fake(False))
    second = review_sections(sections)[0]

    assert first == second
    assert first[0]["severity"] == "critical"


def test_one_failing_persona_does_not_kill_the_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_structured(system, user, model_cls):
        if system == review_module.PERSONAS["coverage"]:
            raise RuntimeError("persona lỗi")
        return ReviewResult(
            issues=[
                ReviewIssue(
                    section_key="technical",
                    severity="major",
                    issue_type="tone",
                    original_text="文A",
                    suggested_fix="x",
                )
            ]
        )

    monkeypatch.setattr(review_module, "structured", fake_structured)
    issues, calls = review_sections([_section("technical", [_sentence("文A")])])

    assert calls == 2
    assert [issue["persona"] for issue in issues] == ["quality"]


def test_apply_fixes_still_only_touches_critical_after_dedupe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BB-4 và ràng buộc 'chỉ critical' không đổi khi có nhiều persona."""
    sections = [
        _section("technical", [_sentence("技術の文です。")]),
        _section("security", [_sentence("セキュリティの文です。")]),
    ]
    issues = review_module.deduplicate_issues(
        [
            {
                **_issue("technical", "critical", "技術の文です。"),
                "persona": "coverage",
            },
            {
                **_issue("security", "major", "セキュリティの文です。"),
                "persona": "quality",
            },
        ]
    )
    monkeypatch.setattr(review_module, "generate", lambda *a, **k: "整えた文です。")
    updated, calls, applied = apply_fixes(sections, issues)

    assert calls == 1
    assert updated[1] == sections[1]
    sentence = updated[0]["sentences"][0]
    assert sentence["source_id"] == "PROP-001-S01"
    assert sentence["req_ids"] == ["3.1"]


def test_review_node_records_score_and_rounds_in_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sections = [_section("technical", [_sentence("文Aです。")])]
    monkeypatch.setattr(
        graph_module,
        "review_sections",
        lambda s: ([_issue("technical", "critical", "文Aです。")], 1),
    )
    monkeypatch.setattr(
        graph_module,
        "apply_fixes",
        lambda s, i: (s, 1, [{"section_key": "technical"}]),
    )

    result = graph_module.review({"sections": sections, "trace": {}})

    review_trace = result["trace"]["review"]
    assert result["review_rounds"] == 1
    assert review_trace["rounds"] == 1
    assert review_trace["history"][-1]["score"]["critical"] == 1
    assert review_trace["history"][-1]["critical_sections"] == ["technical"]
    assert result["trace"]["llm_calls"] == 2  # 1 review + 1 fix
