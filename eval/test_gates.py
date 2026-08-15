import sys
from pathlib import Path
import shutil
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from rfp.generate.claim_check import (  # noqa: E402
    build_claim_whitelist,
    check_claims,
    filter_hybrid_claims,
)
from rfp.generate.merge import deduplicate_sections  # noqa: E402
from rfp.generate.coverage import (  # noqa: E402
    matched_requirement_ids,
    validate_source_req_ids,
)
from config.settings import PROPOSAL_DIR, RFP_DIR  # noqa: E402
from eval.golden.schema import Assertion, GoldenCase  # noqa: E402
from eval.to_samples import (  # noqa: E402
    capability_sheet_text,
    deterministic_metrics,
    to_eval_samples,
)
from rfp.graph import assemble, run_graph  # noqa: E402
from rfp.guard import GuardViolation, final_guard  # noqa: E402
from rfp.stores.capability import CapabilityStore  # noqa: E402
from rfp.stores.sentence_index import SentenceIndex  # noqa: E402


@pytest.fixture(scope="session")
def clean_sentence_index() -> SentenceIndex:
    index = SentenceIndex.build()
    index.assert_clean()
    return index


@pytest.fixture(scope="session")
def claim_whitelist(clean_sentence_index: SentenceIndex) -> set[str]:
    return build_claim_whitelist(clean_sentence_index)


def _precedent(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "origin": "precedent",
        "source_id": "TEST-SOURCE",
        "req_ids": [],
        "verdict": None,
    }


def test_hybrid_hallucination(claim_whitelist: set[str]) -> None:
    fabricated = _precedent(
        "公共機関様向けの類似案件において、在庫精度を55%向上した実績があります。"
    )
    valid = _precedent(
        "公共機関様向けの類似案件において、在庫精度を20%向上した実績があります。"
    )

    kept, blocked = filter_hybrid_claims([fabricated, valid], claim_whitelist)

    assert valid in kept
    assert fabricated not in kept
    assert fabricated in blocked
    assert valid not in blocked


def test_bridge_without_source_is_unverifiable() -> None:
    bridge = {
        "text": "文脈に応じた接続文です。",
        "origin": "bridge",
        "source_id": None,
        "req_ids": [],
        "verdict": None,
    }

    checked, counts, llm_calls, removed = check_claims(
        [bridge],
        capability_store=CapabilityStore(),
        source_texts={},
    )

    assert checked[0]["verdict"] == "UNVERIFIABLE"
    assert counts["UNVERIFIABLE"] == 1
    assert counts["VERIFIED"] == 0
    assert llm_calls == 0
    assert removed == []


def test_global_dedup_and_bridge_budget() -> None:
    duplicate = {
        "text": "同じ能力文です。",
        "origin": "capability",
        "source_id": "capabilities:duplicate",
        "req_ids": [],
        "verdict": "VERIFIED",
    }
    sections = [
        {
            "key": "implementation_experience",
            "sentences": [
                duplicate,
                {
                    "text": "導入実績です。",
                    "origin": "precedent",
                    "source_id": "PROP-001-S01",
                    "req_ids": [],
                    "verdict": "VERIFIED",
                },
            ],
        },
        {
            "key": "proposal_overview",
            "sentences": [
                dict(duplicate),
                {
                    "text": "提案方針です。",
                    "origin": "precedent",
                    "source_id": "PROP-001-S02",
                    "req_ids": [],
                    "verdict": "VERIFIED",
                },
            ],
        },
    ]

    deduplicated, dropped = deduplicate_sections(
        sections,
        bridge_priority=("implementation_experience", "proposal_overview"),
        max_bridges=1,
    )

    all_sentences = [
        sentence for section in deduplicated for sentence in section["sentences"]
    ]
    assert len(dropped) == 1
    assert sum(item["origin"] == "bridge" for item in all_sentences) == 1
    assert len(
        {
            item["text"] for item in all_sentences if item["origin"] == "bridge"
        }
    ) == 1


def test_req_id_must_match_its_own_source() -> None:
    requirements = {
        "4.1": "ISO/IEC 27001相当の情報セキュリティ管理体制を有すること。",
        "4.2": "通信は暗号化すること。",
    }
    iso_source = (
        "ISO/IEC 27001の認証を有しているため、"
        "情報セキュリティ管理体制に関する要件への対応が可能です。"
    )

    matched = matched_requirement_ids(
        iso_source,
        requirements,
        candidate_req_ids=("4.1", "4.2"),
    )

    assert matched == ["4.1"]
    validate_source_req_ids(
        ["4.1"],
        source_text=iso_source,
        requirements=requirements,
    )
    with pytest.raises(AssertionError):
        validate_source_req_ids(
            ["4.1", "4.2"],
            source_text=iso_source,
            requirements=requirements,
        )


def test_final_guard_rejects_iso27017() -> None:
    with pytest.raises(GuardViolation, match="ISO/IEC 27017"):
        final_guard("当社はISO/IEC 27017認証を取得済み")


def test_final_guard_allows_common_client_and_held_certification() -> None:
    final_guard(
        "公共機関様向けの案件として、ISO/IEC 27001に基づき対応します。"
    )


def test_final_guard_rejects_private_client_name() -> None:
    with pytest.raises(GuardViolation, match="client name leaked"):
        final_guard("新生証券様向けの案件実績を活用します。")


def test_assemble_does_not_swallow_guard_violation() -> None:
    malicious_state = {
        "sections": [
            {
                "title_ja": "認証・コンプライアンス",
                "sentences": [
                    {"text": "当社はISO/IEC 27017認証を取得済み"}
                ],
            }
        ],
        "trace": {
            "llm_calls": 0,
            "retrieval_stats": {},
            "dedup": {"before": 0, "after": 0},
            "path": [],
        },
    }

    with pytest.raises(GuardViolation, match="ISO/IEC 27017"):
        assemble(malicious_state)


def test_eval_projection_always_contains_capability_sheet() -> None:
    rfp_text = (
        "評価用RFP\n発注業種：製造業\n調達番号：RFP-EVAL-001\n\n"
        "第3章 技術要件\n3.1 基幹システム構築に対応できること。\n"
    )
    state = {
        "input_text": rfp_text,
        "rfp": {"rfp_id": "RFP-EVAL-001"},
        "chapters": [
            {
                "id": "3",
                "requirements": [
                    {
                        "req_id": "3.1",
                        "text": "基幹システム構築に対応できること。",
                    }
                ],
                "retrieval": {"selected": []},
            }
        ],
        "sections": [
            {
                "sentences": [
                    {
                        "text": "基幹システム構築の実績があります。",
                        "origin": "capability",
                        "source_id": "capabilities:基幹システム構築",
                        "req_ids": ["3.1"],
                        "verdict": "VERIFIED",
                    }
                ]
            }
        ],
    }
    golden = GoldenCase(
        case_id="eval-projection",
        rfp_text=rfp_text,
        source="combinatorial",
        needs_review=False,
        assertions=[Assertion("must_cover", "3.1", "IN-SCOPE")],
    )

    samples = to_eval_samples(state, golden_cases=[golden])

    assert len(samples) == 1
    assert samples[0]["question"] == "基幹システム構築に対応できること。"
    assert samples[0]["answer"] == "基幹システム構築の実績があります。"
    assert samples[0]["contexts"][-1] == capability_sheet_text()
    assert "must_cover" in samples[0]["ground_truth"]


def test_deterministic_metrics_pair_coverage_with_abstain_and_citation() -> None:
    state = {
        "proposal": "基幹システム構築の実績があります。",
        "chapters": [
            {
                "requirements": [
                    {
                        "req_id": "3.1",
                        "text": "基幹システム構築に対応できること。",
                    }
                ],
                "retrieval": {"selected": []},
            }
        ],
        "sections": [
            {
                "status": "OK",
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
        "trace": {
            "dedup": {"before": 1, "after": 1},
            "hybrid_blocked": [],
            "conflicts": [],
        },
    }

    metrics = deterministic_metrics(state)

    assert metrics["coverage"] == 1.0
    assert metrics["abstain_rate"] == 0.0
    assert metrics["citation_accuracy"] == 1.0
    assert metrics["fabrication_count"] == 0
    assert metrics["client_leak_count"] == 0


def test_redteam_27017(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    poisoned_proposals = tmp_path / "proposals"
    poisoned_proposals.mkdir()
    for source in PROPOSAL_DIR.glob("*.txt"):
        shutil.copy2(source, poisoned_proposals / source.name)

    fake_certificate = (
        PROJECT_ROOT / "synthetic" / "certs" / "fake_iso27017.txt"
    ).read_text(encoding="utf-8")
    poisoned_text = (
        "提案書 PROP-999\n"
        "宛先調達：RFP-2025-001\n\n"
        "1. 会社概要\n検証用文書です。\n\n"
        "4. 認証・コンプライアンス\n"
        f"{fake_certificate}\n"
    )
    (poisoned_proposals / "PROP-999.txt").write_text(
        poisoned_text,
        encoding="utf-8",
    )
    poisoned_index = SentenceIndex.build(proposal_dir=poisoned_proposals)
    assert any(
        "ISO/IEC 27017" in sentence.text
        for sentence in poisoned_index.capability_quarantine
    )
    assert all(
        "ISO/IEC 27017" not in sentence.text
        for sentence in poisoned_index.all_sentences()
    )

    monkeypatch.setattr(
        SentenceIndex,
        "build",
        classmethod(lambda cls, *args, **kwargs: poisoned_index),
    )
    state = run_graph(
        (RFP_DIR / "RFP-2025-001.txt").read_text(encoding="utf-8")
    )

    assert state["status"] == "completed"
    assert "ISO/IEC 27017" not in state["proposal"]
    final_guard(state["proposal"])
