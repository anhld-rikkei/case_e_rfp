from dataclasses import dataclass

from config.settings import MMR_LAMBDA

from .hybrid import tokenize
from .rerank import RerankedResult


@dataclass(frozen=True)
class MMRSelection:
    selected: tuple[RerankedResult, ...]
    before: int
    after: int
    unique_texts: int


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def maximal_marginal_relevance(
    candidates: list[RerankedResult],
    *,
    k: int,
    lambda_mult: float = MMR_LAMBDA,
) -> MMRSelection:
    if k <= 0:
        return MMRSelection(selected=(), before=len(candidates), after=0, unique_texts=0)
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("MMR lambda phải nằm trong [0, 1]")

    unique: list[RerankedResult] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        if candidate.sentence.text not in seen_texts:
            seen_texts.add(candidate.sentence.text)
            unique.append(candidate)
    if len(unique) < k:
        raise ValueError(f"MMR cần {k} văn bản duy nhất nhưng chỉ có {len(unique)}")

    token_sets = {
        item.sentence.sent_id: set(tokenize(item.sentence.text)) for item in unique
    }
    remaining = list(unique)
    selected: list[RerankedResult] = []
    while remaining and len(selected) < k:
        def mmr_key(item: RerankedResult) -> tuple[float, str]:
            similarity = max(
                (
                    _jaccard(
                        token_sets[item.sentence.sent_id],
                        token_sets[chosen.sentence.sent_id],
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            score = lambda_mult * item.score - (1.0 - lambda_mult) * similarity
            return -score, item.sentence.sent_id

        chosen = min(remaining, key=mmr_key)
        selected.append(chosen)
        remaining.remove(chosen)

    unique_count = len({item.sentence.text for item in selected})
    if unique_count != k:
        raise AssertionError(f"MMR diversity fail: unique={unique_count}, k={k}")
    return MMRSelection(
        selected=tuple(selected),
        before=len(candidates),
        after=len(selected),
        unique_texts=unique_count,
    )
