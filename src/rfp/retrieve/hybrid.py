import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from ..schema import Sentence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:/[A-Za-z]+)*|\d+(?:\.\d+)?%?|[ぁ-んァ-ヶ一-龯ー]")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return TOKEN_RE.findall(normalized)


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)


@dataclass(frozen=True)
class SearchResult:
    sentence: Sentence
    score: float
    bm25_score: float
    dense_score: float
    industry: str


class HybridRetriever:
    def __init__(
        self,
        sentences: Iterable[Sentence],
        industries: dict[str, str],
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.sentences = list(sentences)
        if not self.sentences:
            raise ValueError("Không có câu proposal sạch để index")
        if abs((bm25_weight + dense_weight) - 1.0) > 1e-9:
            raise ValueError("Tổng trọng số BM25 và dense phải bằng 1")

        self.industries = dict(industries)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Okapi([tokenize(sentence.text) for sentence in self.sentences])

        # Chỉ encode text của Sentence lấy từ proposal đã sanitize. Capability sheet
        # không được truyền vào retriever và không đi qua model embedding.
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [sentence.text for sentence in self.sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)

    def _candidate_indices(
        self,
        *,
        section: str | None,
        industry: str | None,
        claim_kind: str | None,
    ) -> list[int]:
        candidates: list[int] = []
        for index, sentence in enumerate(self.sentences):
            sentence_industry = self.industries.get(sentence.responds_to, "")
            if section is not None and sentence.section != section:
                continue
            if industry is not None and sentence_industry != industry:
                continue
            if claim_kind is not None and sentence.claim_kind != claim_kind:
                continue
            candidates.append(index)
        return candidates

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        section: str | None = None,
        industry: str | None = None,
        claim_kind: str | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        candidates = self._candidate_indices(
            section=section,
            industry=industry,
            claim_kind=claim_kind,
        )
        if not candidates:
            return []

        bm25_all = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        distances, indices = self.faiss_index.search(query_embedding, len(self.sentences))
        dense_all = np.zeros(len(self.sentences), dtype=np.float32)
        for distance, index in zip(distances[0], indices[0]):
            if index >= 0:
                dense_all[index] = distance

        candidate_array = np.asarray(candidates, dtype=np.int64)
        bm25_normalized = _normalize(bm25_all[candidate_array])
        dense_normalized = _normalize(dense_all[candidate_array])
        combined = (
            self.bm25_weight * bm25_normalized
            + self.dense_weight * dense_normalized
        )

        ranked = sorted(
            range(len(candidates)),
            key=lambda offset: (
                -float(combined[offset]),
                self.sentences[candidates[offset]].sent_id,
            ),
        )[:k]

        results: list[SearchResult] = []
        for offset in ranked:
            sentence = self.sentences[candidates[offset]]
            results.append(
                SearchResult(
                    sentence=sentence,
                    score=float(combined[offset]),
                    bm25_score=float(bm25_normalized[offset]),
                    dense_score=float(dense_normalized[offset]),
                    industry=self.industries.get(sentence.responds_to, ""),
                )
            )
        return results
