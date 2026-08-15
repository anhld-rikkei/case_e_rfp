import re
from dataclasses import dataclass

from config.settings import (
    RERANK_BM25_WEIGHT,
    RERANK_DENSE_WEIGHT,
    RERANK_INDUSTRY_WEIGHT,
    RERANK_SECTION_WEIGHT,
)

from ..schema import Sentence
from .hybrid import SearchResult


CLAIM_RE = re.compile(
    r"(?P<client>[^\s・]{2,10})様向け.*?"
    r"(?P<metric>処理時間|在庫精度|年間運用コスト|障害件数)を"
    r"(?P<value>\d+)%"
)


@dataclass(frozen=True)
class RerankedResult:
    result: SearchResult
    score: float
    industry_match: float
    same_section_prior: float

    @property
    def sentence(self) -> Sentence:
        return self.result.sentence


def rerank_candidates(
    results: list[SearchResult],
    *,
    target_industry: str,
    target_section: str,
    top_k: int | None = None,
) -> list[RerankedResult]:
    reranked = []
    for result in results:
        industry_match = float(result.industry == target_industry)
        same_section_prior = float(result.sentence.section == target_section)
        score = (
            RERANK_DENSE_WEIGHT * result.dense_score
            + RERANK_BM25_WEIGHT * result.bm25_score
            + RERANK_INDUSTRY_WEIGHT * industry_match
            + RERANK_SECTION_WEIGHT * same_section_prior
        )
        reranked.append(
            RerankedResult(
                result=result,
                score=score,
                industry_match=industry_match,
                same_section_prior=same_section_prior,
            )
        )
    reranked.sort(key=lambda item: (-item.score, item.sentence.sent_id))
    return reranked if top_k is None else reranked[:top_k]


def claim_signature(sentence: Sentence) -> tuple[tuple[str, str], str] | None:
    match = CLAIM_RE.search(sentence.text)
    if match is None:
        return None
    return (match.group("client"), match.group("metric")), match.group("value")


def resolve_conflicts(
    results: list[RerankedResult],
    *,
    reference_rfp: str,
    industry: str,
    industries: dict[str, str],
) -> tuple[list[RerankedResult], list[RerankedResult]]:
    def source_rank(item: RerankedResult) -> tuple[int, str]:
        sentence = item.sentence
        rank = (
            0
            if sentence.responds_to == reference_rfp
            else 1
            if industries.get(sentence.responds_to) == industry
            else 2
        )
        return rank, sentence.sent_id

    kept: list[RerankedResult] = []
    dropped: list[RerankedResult] = []
    values_by_claim: dict[tuple[str, str], str] = {}
    for item in sorted(results, key=source_rank):
        signature = claim_signature(item.sentence)
        if signature is None:
            kept.append(item)
            continue
        claim_key, value = signature
        selected_value = values_by_claim.get(claim_key)
        if selected_value is None:
            values_by_claim[claim_key] = value
            kept.append(item)
        elif selected_value == value:
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped
