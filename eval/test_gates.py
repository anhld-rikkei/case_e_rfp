import sys
from pathlib import Path
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
