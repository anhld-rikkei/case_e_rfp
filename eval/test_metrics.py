"""Test nhật ký đo lường + timeout + source_strategy (v1.8)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.graph as graph_module  # noqa: E402
from rfp.metrics import (  # noqa: E402
    BRANCH_AUTO,
    BRANCH_FAILED,
    BRANCH_HUMAN,
    BRANCH_REVIEW,
    branch_of,
    build_record,
    log_run,
    percentile,
    read_records,
    summarize,
)


def _usage(**overrides: Any) -> SimpleNamespace:
    base = {
        "calls": 20,
        "seconds": 18.5,
        "tokens_by_stage": {"generate": 3000, "structured": 2000, "judge": 90000},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _state(status: str = "completed", tier: str = "T1") -> dict[str, Any]:
    return {
        "status": status,
        "rfp": {"rfp_id": "RFP-2025-001"},
        "sections": [{"key": "a"}, {"key": "b"}],
        "confidence": {"score": 0.9, "tier": tier, "by_tier": {"T1": 2}},
    }


# ── Nhánh: tự trả lời vs chuyển người ─────────────────────────────────────

@pytest.mark.parametrize(
    "tier,expected",
    [("T1", BRANCH_AUTO), ("T2", BRANCH_REVIEW), ("T3", BRANCH_HUMAN)],
)
def test_branch_follows_tier(tier: str, expected: str) -> None:
    assert branch_of(_state(tier=tier)) == expected


@pytest.mark.parametrize("status", ["partial", "guard_blocked"])
def test_failed_runs_are_their_own_branch(status: str) -> None:
    """Guard chặn và provider sập KHÔNG được tính là 'tự trả lời'."""
    assert branch_of(_state(status=status)) == BRANCH_FAILED


def test_ask_user_counts_as_handover() -> None:
    assert branch_of(_state(status="ask_user")) == BRANCH_HUMAN


def test_missing_confidence_defaults_to_handover() -> None:
    """Không biết thì chuyển người — không mặc định 'tự trả lời được'."""
    assert branch_of({"status": "completed"}) == BRANCH_HUMAN


# ── Bản ghi ───────────────────────────────────────────────────────────────

def test_record_separates_product_and_judge_tokens() -> None:
    record = build_record(_state(), usage=_usage(), seconds=12.25, now=100.0)
    assert record["tokens_product"] == 5000
    assert record["tokens_judge"] == 90000
    assert record["seconds"] == 12.25
    assert record["at"] == 100.0
    assert record["rfp_id"] == "RFP-2025-001"


def test_record_works_without_usage() -> None:
    record = build_record(_state(), usage=None, seconds=1.0)
    assert record["tokens_product"] == 0 and record["llm_calls"] == 0


# ── Ghi/đọc file ──────────────────────────────────────────────────────────

def test_log_appends_one_json_line_per_run(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    log_run(_state(), usage=_usage(), seconds=1.0, path=path, enabled=True)
    log_run(_state(tier="T3"), usage=_usage(), seconds=2.0, path=path, enabled=True)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["tier"] == "T3"


def test_log_disabled_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    log_run(_state(), path=path, enabled=False)
    assert not path.exists()


def test_log_failure_never_breaks_the_run(tmp_path: Path) -> None:
    """Hồ sơ quan trọng hơn nhật ký: ghi lỗi thì nuốt, không ném."""
    directory = tmp_path / "as-file"
    directory.write_text("không phải thư mục", encoding="utf-8")
    record = log_run(_state(), path=directory / "x.jsonl", enabled=True)
    assert record is not None  # vẫn trả bản ghi, không nổ


def test_read_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"a":1}\nkhông phải json\n{"b":2}\n', encoding="utf-8")
    assert len(read_records(path)) == 2


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_records(tmp_path / "chua-co.jsonl") == []


# ── Thống kê ──────────────────────────────────────────────────────────────

def test_percentile_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 0.50) == 3.0
    assert percentile(values, 0.95) == 10.0
    assert percentile([], 0.5) is None


def test_summary_reports_auto_vs_handover_rate() -> None:
    records = [
        {"branch": BRANCH_AUTO, "seconds": 10, "tokens_product": 100},
        {"branch": BRANCH_AUTO, "seconds": 20, "tokens_product": 200},
        {"branch": BRANCH_REVIEW, "seconds": 30, "tokens_product": 300},
        {"branch": BRANCH_HUMAN, "seconds": 40, "tokens_product": 400},
    ]
    summary = summarize(records)
    assert summary["runs"] == 4
    assert summary["auto_rate"] == 0.5
    assert summary["handover_rate"] == 0.5
    assert summary["p50_seconds"] == 20.0
    assert summary["tokens_product_avg"] == 250.0


def test_summary_on_empty_log_does_not_invent_numbers() -> None:
    assert summarize([]) == {"runs": 0}


# ── source_strategy ───────────────────────────────────────────────────────

def test_strategy_capability_only_when_search_skipped() -> None:
    retrieval = {"stages": {"query_embed": {"skipped": True}}, "selected": []}
    assert graph_module.source_strategy(retrieval) == "capability-only"


def test_strategy_fallback_when_nothing_selected() -> None:
    retrieval = {"stages": {"query_embed": {"skipped": False}}, "selected": []}
    assert graph_module.source_strategy(retrieval) == "fallback-listing"


def test_strategy_hybrid_when_bm25_on(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = {"stages": {"query_embed": {"skipped": False}}, "selected": [{"x": 1}]}
    monkeypatch.setattr(graph_module, "RETRIEVAL_USE_BM25", True)
    assert graph_module.source_strategy(retrieval) == "hybrid-bm25+dense"
    monkeypatch.setattr(graph_module, "RETRIEVAL_USE_BM25", False)
    assert graph_module.source_strategy(retrieval) == "dense-only"


# ── Tầng T3: nguồn "có thể liên quan" ────────────────────────────────────

def test_related_sources_exclude_chosen_and_rank_by_score() -> None:
    retrieval = {
        "selected": [{"sent_id": "A"}],
        "candidates": [
            {"sent_id": "A", "scores": {"rerank": 0.9}},
            {"sent_id": "B", "scores": {"rerank": 0.5}},
            {"sent_id": "C", "scores": {"rerank": 0.8}},
        ],
    }
    related = graph_module.related_sources(retrieval, limit=2)
    assert [item["sent_id"] for item in related] == ["C", "B"]


def test_related_sources_respect_limit_and_empty() -> None:
    assert graph_module.related_sources({"selected": [], "candidates": []}) == []


# ── Timeout: huỷ êm về partial ───────────────────────────────────────────

def test_timeout_raises_response_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(graph_module, "RESPONSE_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(graph_module.ResponseTimeout) as excinfo:
        graph_module.with_timeout("slow", lambda _s: time.sleep(5), {})
    assert excinfo.value.kind == "timeout"
    assert excinfo.value.node == "slow"


def test_timeout_is_an_llm_unavailable_so_partial_path_reuses_it() -> None:
    """Kế thừa LLMUnavailable để đi chung đường huỷ êm đã có."""
    from rfp.llm import LLMUnavailable

    assert issubclass(graph_module.ResponseTimeout, LLMUnavailable)


def test_fast_node_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "RESPONSE_TIMEOUT_SECONDS", 5)
    assert graph_module.with_timeout("fast", lambda s: {"ok": s}, 1) == {"ok": 1}


def test_timeout_disabled_runs_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "RESPONSE_TIMEOUT_SECONDS", 0)
    assert graph_module.with_timeout("x", lambda s: s + 1, 1) == 2
