"""Test điểm tin cậy + phân tầng (v1.8). Cách ly: thuần hàm, không mạng."""
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from config import settings  # noqa: E402
from rfp.confidence import (  # noqa: E402
    TIER_AUTO,
    TIER_HUMAN,
    TIER_REVIEW,
    annotate,
    section_confidence,
    sentence_confidence,
    tier_of,
)


def _sentence(**overrides: Any) -> dict[str, Any]:
    base = {
        "text": "文です。",
        "origin": "precedent",
        "source_id": "S1",
        "req_ids": ["3.1"],
        "verdict": "VERIFIED",
    }
    base.update(overrides)
    return base


def _state(sentences: list[dict[str, Any]], *, requirements: int = 1) -> dict[str, Any]:
    return {
        "chapters": [
            {
                "id": "3",
                "requirements": [
                    {"req_id": f"3.{i}", "text": "…"} for i in range(1, requirements + 1)
                ],
                "retrieval": {
                    "selected": [
                        {"sent_id": "S1", "text": "x", "scores": {"rerank": 1.0}}
                    ]
                },
            }
        ],
        "sections": [
            {
                "key": "technical",
                "title_ja": "技術要件",
                "source_chapters": ["3"],
                "status": "OK",
                "sentences": sentences,
            }
        ],
    }


# ── Bất biến của ngưỡng: mặc định không đổi hành vi ───────────────────────

def test_threshold_invariants_hold() -> None:
    """Hai bất đẳng thức ép bản đồ tầng khớp trạng thái mục.

    Phá một trong hai là mục thiếu căn cứ có thể rơi vào tầng tự trả lời —
    nên đây là test, không phải lời dặn trong comment.
    """
    assert settings.INCOMPLETE_COVERAGE_FACTOR < settings.CONFIDENCE_T_LOW
    assert settings.NO_PRECEDENT_FACTOR < settings.CONFIDENCE_T_HIGH
    assert settings.CONFIDENCE_T_LOW < settings.CONFIDENCE_T_HIGH


def test_tier_boundaries() -> None:
    assert tier_of(settings.CONFIDENCE_T_HIGH) == TIER_AUTO
    assert tier_of(settings.CONFIDENCE_T_HIGH - 0.001) == TIER_REVIEW
    assert tier_of(settings.CONFIDENCE_T_LOW) == TIER_REVIEW
    assert tier_of(settings.CONFIDENCE_T_LOW - 0.001) == TIER_HUMAN
    assert tier_of(0.0) == TIER_HUMAN


# ── Điểm từng câu ─────────────────────────────────────────────────────────

def test_capability_sentence_scores_highest() -> None:
    assert sentence_confidence(_sentence(origin="capability")) == (
        settings.CONFIDENCE_CAPABILITY_BASE
    )


def test_precedent_score_rises_with_rerank() -> None:
    low = sentence_confidence(_sentence(), rerank_by_source={"S1": 0.0})
    high = sentence_confidence(_sentence(), rerank_by_source={"S1": 1.0})
    assert low == settings.CONFIDENCE_PRECEDENT_BASE
    assert high == pytest.approx(
        settings.CONFIDENCE_PRECEDENT_BASE + settings.CONFIDENCE_PRECEDENT_SPAN
    )
    assert low < high


def test_unverifiable_scores_low_and_contradicted_zero() -> None:
    assert sentence_confidence(_sentence(verdict="UNVERIFIABLE")) == (
        settings.CONFIDENCE_UNVERIFIABLE
    )
    assert sentence_confidence(_sentence(verdict="CONTRADICTED")) == 0.0


def test_bridge_and_user_sentences_are_not_scored() -> None:
    """Câu nối không mang sự thật; câu người dùng là trách nhiệm của họ."""
    assert sentence_confidence(_sentence(origin="bridge", source_id=None)) is None
    assert sentence_confidence(_sentence(origin="user", source_id=None)) is None


def test_rerank_out_of_range_is_clamped() -> None:
    assert sentence_confidence(_sentence(), rerank_by_source={"S1": 9.9}) <= 1.0
    assert sentence_confidence(_sentence(), rerank_by_source={"S1": -5}) >= 0.0


# ── Điểm mục: ba tầng ─────────────────────────────────────────────────────

def test_full_coverage_with_precedent_is_auto_tier() -> None:
    state = _state([_sentence()])
    info = section_confidence(state, state["sections"][0])
    assert info["tier"] == TIER_AUTO
    assert info["score"] >= settings.CONFIDENCE_T_HIGH


def test_capability_only_section_is_review_tier() -> None:
    state = _state([_sentence(origin="capability", source_id="capabilities:x")])
    info = section_confidence(state, state["sections"][0])
    assert info["tier"] == TIER_REVIEW
    assert not info["has_precedent"]


def test_missing_requirement_drops_to_human_tier() -> None:
    state = _state([_sentence()], requirements=3)  # chỉ phủ 3.1
    info = section_confidence(state, state["sections"][0])
    assert info["tier"] == TIER_HUMAN
    assert (info["covered_requirements"], info["total_requirements"]) == (1, 3)


def test_fixed_section_without_chapters_is_not_penalised() -> None:
    """会社概要 vốn chỉ dựng từ bảng năng lực — phạt nó là hạ oan một mục đủ
    căn cứ. Lỗi này bắt được khi chạy trên state thật."""
    state = _state([_sentence(origin="capability", source_id="capabilities:x")])
    state["sections"][0]["source_chapters"] = []
    info = section_confidence(state, state["sections"][0])
    assert info["tier"] == TIER_AUTO
    assert info["total_requirements"] == 0


def test_user_sentences_counted_but_not_scored() -> None:
    state = _state([_sentence(), _sentence(origin="user", source_id=None)])
    info = section_confidence(state, state["sections"][0])
    assert info["user_sentences"] == 1
    assert info["scored_sentences"] == 1


def test_empty_section_lands_in_human_tier() -> None:
    state = _state([])
    info = section_confidence(state, state["sections"][0])
    assert info["tier"] == TIER_HUMAN and info["score"] == 0.0


# ── annotate(): gắn vào state, không sửa state cũ ─────────────────────────

def test_annotate_adds_confidence_without_mutating() -> None:
    state = _state([_sentence()])
    result = annotate(state)
    assert "confidence" in result["sections"][0]
    assert "confidence" not in state["sections"][0]


def test_overall_score_is_the_weakest_section() -> None:
    """Hồ sơ chỉ đáng tin bằng mục yếu nhất — trung bình sẽ giấu mất mục hỏng."""
    state = _state([_sentence()])
    state["sections"].append(
        {
            "key": "weak",
            "title_ja": "x",
            "source_chapters": ["3"],
            "status": "INSUFFICIENT_EVIDENCE",
            "sentences": [],
        }
    )
    result = annotate(state)
    assert result["confidence"]["score"] == 0.0
    assert result["confidence"]["tier"] == TIER_HUMAN
    assert result["confidence"]["by_tier"][TIER_AUTO] == 1


def test_annotate_is_deterministic() -> None:
    """Không có lệnh gọi LLM nào -> chạy lại trên cùng state ra cùng số."""
    state = _state([_sentence()])
    assert annotate(state)["confidence"] == annotate(state)["confidence"]
