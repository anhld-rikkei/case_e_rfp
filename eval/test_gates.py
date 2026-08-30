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
from config.settings import (  # noqa: E402
    EVAL_PROPOSAL_DIR as PROPOSAL_DIR,
    EVAL_RFP_DIR as RFP_DIR,
)
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


@pytest.mark.parametrize(
    "payload",
    [
        "当社はISO 27017認証を取得済みです。",
        "当社はISO27017認証を取得済みです。",
        "ISO-27017の認証を保有しています。",
        "IEC 27017に準拠した運用体制です。",
        "ISO/IEC27017:2015に準拠しています。",
        "ＩＳＯ２７０１７認証を取得しています。",
        "27017認証を取得済みです。",
        "クラウドセキュリティ規格（27017）に準拠します。",
        "iso/iec 27018の認証があります。",
    ],
)
def test_final_guard_rejects_cert_variants(payload: str) -> None:
    with pytest.raises(GuardViolation, match="blocklist"):
        final_guard(payload)


def test_ingest_blocklist_uses_same_variant_patterns() -> None:
    from rfp.sanitize.blocklist import CapabilityBlocklist

    blocklist = CapabilityBlocklist()
    assert blocklist.contradicts("ISO27017認証を取得しています。")
    assert blocklist.contradicts("ＩＳＯ２７０１７認証を取得しています。")
    assert not blocklist.contradicts("ISO/IEC 27001認証を取得しています。")


def test_final_guard_allows_common_client_and_held_certification() -> None:
    final_guard(
        "公共機関様向けの案件として、ISO/IEC 27001に基づき対応します。"
    )


def _search_result(sent_id: str, text: str, *, bm25: float, dense: float):
    from rfp.retrieve.hybrid import SearchResult
    from rfp.schema import Sentence

    return SearchResult(
        sentence=Sentence(
            sent_id=sent_id,
            proposal_id="PROP-TEST",
            responds_to="RFP-TEST",
            section="技術要件",
            text=text,
            claim_kind="capability",
            flags={},
        ),
        score=0.5 * bm25 + 0.5 * dense,
        bm25_score=bm25,
        dense_score=dense,
        industry="製造業",
    )


def test_rerank_flag_off_preserves_hybrid_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rfp.retrieve.rerank as rerank_module

    # Hybrid 0.5/0.5: a (0.500) đứng trước b (0.425).
    # Rerank 0.4*dense+0.3*bm25: b (0.340) vượt a (0.300).
    a = _search_result("S1", "文A", bm25=1.0, dense=0.0)
    b = _search_result("S2", "文B", bm25=0.0, dense=0.85)

    monkeypatch.setattr(rerank_module, "RETRIEVAL_USE_RERANK", False)
    off = rerank_module.rerank_candidates(
        [a, b], target_industry="金融", target_section="調達概要"
    )
    assert [item.sentence.sent_id for item in off] == ["S1", "S2"]
    assert off[0].score == a.score  # điểm giữ nguyên điểm hybrid

    monkeypatch.setattr(rerank_module, "RETRIEVAL_USE_RERANK", True)
    on = rerank_module.rerank_candidates(
        [a, b], target_industry="金融", target_section="調達概要"
    )
    assert [item.sentence.sent_id for item in on] == ["S2", "S1"]


def test_bm25_flag_off_builds_dense_only_retriever(
    clean_sentence_index: SentenceIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config.settings as settings

    saved = clean_sentence_index._retriever
    try:
        monkeypatch.setattr(settings, "RETRIEVAL_USE_BM25", False)
        clean_sentence_index._retriever = None
        retriever = clean_sentence_index._get_retriever()
        assert retriever.bm25_weight == 0.0
        assert retriever.dense_weight == 1.0
    finally:
        clean_sentence_index._retriever = saved


def test_final_guard_allows_held_certification_variants() -> None:
    final_guard("ISO27001:2013に基づく運用体制です。ISO 9001も取得済みです。")


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
