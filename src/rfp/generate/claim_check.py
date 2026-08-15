import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel

from ..llm import structured
from ..stores.capability import CapabilityStore
from ..stores.sentence_index import SentenceIndex
from .capability import render_fact


CLAIM_RE = re.compile(
    r"(処理時間|在庫精度|年間運用コスト|障害件数)を"
    r"(\d+)%(向上|短縮|削減|低減)"
)

CLAIM_CHECK_SYSTEM = (
    "あなたは提案書の事実確認担当です。主張と、唯一使用可能な根拠を比較してください。"
    "根拠が主張を直接支持すればVERIFIED、判断材料がなければUNVERIFIABLE、"
    "根拠と矛盾すればCONTRADICTEDです。外部知識や推測を使わず、指定schemaで返してください。"
)


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["VERIFIED", "UNVERIFIABLE", "CONTRADICTED"]
    evidence: str | None


def build_claim_whitelist(index: SentenceIndex) -> set[str]:
    return {
        match.group(0)
        for sentence in index.all_sentences()
        for match in CLAIM_RE.finditer(sentence.text)
    }


def check_hybrid(sentence: dict[str, Any], whitelist: set[str]) -> bool:
    if sentence["origin"] != "precedent":
        return True
    return all(
        match.group(0) in whitelist
        for match in CLAIM_RE.finditer(sentence["text"])
    )


def filter_hybrid_claims(
    sentences: list[dict[str, Any]],
    whitelist: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for sentence in sentences:
        (kept if check_hybrid(sentence, whitelist) else blocked).append(sentence)
    return kept, blocked


def _evidence_text(
    sentence: dict[str, Any],
    *,
    capability_store: CapabilityStore,
    source_texts: dict[str, str],
) -> str | None:
    source_id = sentence["source_id"]
    if sentence["origin"] == "capability" and source_id in capability_store:
        return render_fact(capability_store, source_id)
    if sentence["origin"] == "precedent" and source_id in source_texts:
        return source_texts[source_id]
    return None


def check_claims(
    sentences: list[dict[str, Any]],
    *,
    capability_store: CapabilityStore,
    source_texts: dict[str, str],
) -> tuple[list[dict[str, Any]], Counter[str], int, list[dict[str, Any]]]:
    checked: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    llm_calls = 0

    for sentence in sentences:
        if sentence["origin"] == "bridge" or sentence["source_id"] is None:
            updated = dict(sentence)
            updated["verdict"] = "UNVERIFIABLE"
            checked.append(updated)
            counts["UNVERIFIABLE"] += 1
            continue

        evidence_text = _evidence_text(
            sentence,
            capability_store=capability_store,
            source_texts=source_texts,
        )
        verdict = structured(
            CLAIM_CHECK_SYSTEM,
            "主張：\n"
            f"{sentence['text']}\n\n"
            f"根拠ID：{sentence['source_id']}\n"
            f"根拠：\n{evidence_text or '根拠なし'}",
            ClaimVerdict,
        )
        llm_calls += 1
        counts[verdict.verdict] += 1
        updated = dict(sentence)
        updated["verdict"] = verdict.verdict
        if verdict.verdict == "CONTRADICTED":
            removed.append(updated)
        else:
            checked.append(updated)

    for verdict_name in ("VERIFIED", "UNVERIFIABLE", "CONTRADICTED"):
        counts.setdefault(verdict_name, 0)
    return checked, counts, llm_calls, removed
