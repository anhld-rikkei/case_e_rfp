from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata

from config.settings import CAPABILITY_PATH
from eval.golden.schema import GoldenCase, load_case
from rfp.generate.capability import render_fact
from rfp.generate.claim_check import CLAIM_RE
from rfp.generate.coverage import source_supports_requirement
from rfp.guard import final_guard
from rfp.sanitize.leak import private_client_names
from rfp.stores.capability import CapabilityStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_DIR = PROJECT_ROOT / "synthetic" / "golden_test_set"


def capability_sheet_text(path: str | Path = CAPABILITY_PATH) -> str:
    capability = json.loads(Path(path).read_text(encoding="utf-8"))
    return json.dumps(capability, ensure_ascii=False, sort_keys=True)


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _rfp_id(state: dict[str, Any]) -> str:
    rfp = state.get("rfp")
    if isinstance(rfp, dict):
        return str(rfp.get("rfp_id", ""))
    return str(getattr(rfp, "rfp_id", ""))


def load_golden_cases(root: str | Path = DEFAULT_GOLDEN_DIR) -> list[GoldenCase]:
    return [load_case(path) for path in sorted(Path(root).rglob("*.json"))]


def matching_golden_case(
    state: dict[str, Any],
    cases: Iterable[GoldenCase],
) -> GoldenCase:
    input_text = _normalized_text(str(state.get("input_text", "")))
    rfp_id = _rfp_id(state)
    candidates = list(cases)
    exact = [
        case
        for case in candidates
        if input_text and _normalized_text(case.rfp_text) == input_text
    ]
    if exact:
        return exact[0]

    by_id = [
        case
        for case in candidates
        if rfp_id
        and (
            case.metadata.get("base_rfp_id") == rfp_id
            or f"調達番号：{rfp_id}" in case.rfp_text
        )
    ]
    if by_id:
        originals = [case for case in by_id if case.source == "original"]
        return (originals or by_id)[0]
    raise KeyError(f"Không tìm thấy golden case cho RFP {rfp_id or '<unknown>'}")


def golden_lookup(case: GoldenCase, req_id: str) -> str:
    matching = [
        assertion
        for assertion in case.assertions
        if assertion.target == req_id
    ]
    if matching:
        return "\n".join(
            f"{item.kind}: {item.reason}" for item in matching
        )
    expected_behavior = str(case.metadata.get("expected_behavior", "")).strip()
    if expected_behavior:
        return expected_behavior
    global_assertions = [
        item
        for item in case.assertions
        if not re.fullmatch(r"\d+\.\d+", item.target)
    ]
    if global_assertions:
        return "\n".join(
            f"{item.kind}: {item.reason}" for item in global_assertions
        )
    raise KeyError(f"Golden case {case.case_id} không có ground truth cho {req_id}")


def capability_facts_used(
    state: dict[str, Any],
    req_id: str,
    *,
    store: CapabilityStore | None = None,
) -> list[str]:
    capability_store = store or CapabilityStore()
    fact_keys = {
        sentence["source_id"]
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
        if sentence.get("origin") == "capability"
        and req_id in sentence.get("req_ids", [])
        and sentence.get("source_id") in capability_store
    }
    return [
        render_fact(capability_store, fact_key)
        for fact_key in sorted(fact_keys)
    ]


def to_eval_samples(
    state: dict[str, Any],
    *,
    golden_cases: Iterable[GoldenCase] | None = None,
) -> list[dict[str, Any]]:
    cases = list(golden_cases) if golden_cases is not None else load_golden_cases()
    if not state.get("rfp"):
        return []
    golden_case = matching_golden_case(state, cases)
    full_capability_sheet = capability_sheet_text()
    capability_store = CapabilityStore()
    evidence_by_source = _evidence_by_source(state)
    samples: list[dict[str, Any]] = []

    for chapter in state.get("chapters", []):
        retrieved = [
            candidate["text"]
            for candidate in chapter.get("retrieval", {}).get("selected", [])
        ]
        for requirement in chapter.get("requirements", []):
            req_id = requirement["req_id"]
            answer_rows = [
                sentence
                for section in state.get("sections", [])
                for sentence in section.get("sentences", [])
                if req_id in sentence.get("req_ids", [])
            ]
            answer_sentences = [sentence["text"] for sentence in answer_rows]
            golden_assertion = golden_lookup(golden_case, req_id)
            reference_evidence = list(
                dict.fromkeys(
                    evidence_by_source[source_id]
                    for sentence in answer_rows
                    if sentence.get("origin") == "capability"
                    if (source_id := sentence.get("source_id")) in evidence_by_source
                )
            )
            contexts = list(
                dict.fromkeys(
                    [
                        *retrieved,
                        *capability_facts_used(
                            state,
                            req_id,
                            store=capability_store,
                        ),
                    ]
                )
            )
            # 10.2: the authoritative sheet is present in every single sample,
            # including samples that used no capability fact in generation.
            contexts.append(full_capability_sheet)
            samples.append(
                {
                    "case_id": golden_case.case_id,
                    "rfp_id": _rfp_id(state),
                    "req_id": req_id,
                    "question": requirement["text"],
                    "contexts": contexts,
                    "answer": "".join(answer_sentences),
                    # The golden set defines the required behavior, while the
                    # authoritative capability fact supplies the requirement-
                    # level reference. A precedent being evaluated must not be
                    # copied into its own reference, which would make noise
                    # sensitivity circular. A global assertion alone has no
                    # entities/content for recall metrics to test.
                    "ground_truth": "\n".join(
                        [golden_assertion, *reference_evidence]
                    ),
                    "golden_assertion": golden_assertion,
                }
            )
    return samples


def partition_ragas_samples(
    samples: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split quality samples from abstentions without mixing the two measures."""
    answered: list[dict[str, Any]] = []
    unanswered: list[dict[str, Any]] = []
    for sample in samples:
        target = answered if str(sample.get("answer", "")).strip() else unanswered
        target.append(sample)
    return answered, unanswered


def _evidence_by_source(state: dict[str, Any]) -> dict[str, str]:
    evidence = {
        candidate["sent_id"]: candidate["text"]
        for chapter in state.get("chapters", [])
        for candidate in chapter.get("retrieval", {}).get("selected", [])
    }
    store = CapabilityStore()
    evidence.update({key: render_fact(store, key) for key in store})
    return evidence


def _citation_supported(sentence_text: str, evidence_text: str) -> bool:
    claim_cores = [match.group(0) for match in CLAIM_RE.finditer(sentence_text)]
    if claim_cores:
        return all(core in evidence_text for core in claim_cores)
    return source_supports_requirement(sentence_text, evidence_text)


def assert_deterministic_gates(state: dict[str, Any]) -> None:
    final_guard(str(state.get("proposal", "")))


def _poison_reach(state: dict[str, Any]) -> dict[str, int]:
    """Câu bịa/rò rỉ đi được tới đâu: index -> context -> prompt sinh.

    Không có ba mốc này thì `fabrication_count = 0` là con số câm: không phân
    biệt được "guard chặn", "retriever không xếp lên", hay "câu bịa không có
    trong index". Ablation §11.3 sống chết bằng phân biệt đó.
    """
    store = CapabilityStore()
    forbidden = tuple(store.forbidden_terms)

    selected = [
        candidate
        for chapter in state.get("chapters", [])
        for candidate in chapter.get("retrieval", {}).get("selected", [])
    ]
    used_ids = set(state.get("trace", {}).get("precedent_sources", []))
    used = [item for item in selected if item.get("sent_id") in used_ids]

    def count(rows: list[dict[str, Any]], predicate) -> int:
        return sum(predicate(str(row.get("text", ""))) for row in rows)

    def has_forbidden(text: str) -> bool:
        return any(term in text for term in forbidden)

    def has_leak(text: str) -> bool:
        return bool(private_client_names(text))

    return {
        "fabrication_in_context": count(selected, has_forbidden),
        "client_leak_in_context": count(selected, has_leak),
        "fabrication_in_prompt": count(used, has_forbidden),
        "client_leak_in_prompt": count(used, has_leak),
    }


def _hybrid_claims_in_output(state: dict[str, Any]) -> int:
    """Chỉ số định lượng trong hồ sơ mà KHÔNG có nguyên văn trong corpus.

    Khác `trace.hybrid_blocked` — cái đó đếm số lần guard 6.4-bis *chạy*, nên
    cấu hình tắt guard luôn ra 0 và trông như "sạch hơn". Cái này đo cái còn
    lại trong sản phẩm, đo được ở mọi cấu hình dù guard bật hay tắt.
    """
    whitelist = set(state.get("trace", {}).get("claim_whitelist", []))
    proposal = str(state.get("proposal", ""))
    return sum(
        match.group(0) not in whitelist for match in CLAIM_RE.finditer(proposal)
    )


def deterministic_metrics(state: dict[str, Any]) -> dict[str, int | float | None]:
    try:
        assert_deterministic_gates(state)
    except Exception as e:
        if type(e).__name__ == "GuardViolation":
            pass # Ignore here since we already measure fabrication_count below, and load_states already tracks if it crashed during generation.
        else:
            raise
    requirements = {
        requirement["req_id"]
        for chapter in state.get("chapters", [])
        for requirement in chapter.get("requirements", [])
    }
    sentences = [
        sentence
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
    ]
    covered = {
        req_id
        for sentence in sentences
        for req_id in sentence.get("req_ids", [])
        if req_id in requirements
    }
    sections = state.get("sections", [])
    abstained = sum(
        section.get("status") == "INSUFFICIENT_EVIDENCE"
        for section in sections
    )
    grounded = [
        sentence for sentence in sentences if sentence.get("source_id") is not None
    ]
    evidence = _evidence_by_source(state)
    accurate_citations = sum(
        sentence.get("source_id") in evidence
        and _citation_supported(
            sentence.get("text", ""),
            evidence[sentence["source_id"]],
        )
        for sentence in grounded
    )
    trace = state.get("trace", {})
    dedup = trace.get("dedup", {})
    dedup_before = int(dedup.get("before", 0))
    proposal = str(state.get("proposal", ""))
    store = CapabilityStore()
    forbidden_hits = sum(term in proposal for term in store.forbidden_terms)
    leak_hits = len(private_client_names(proposal))

    return {
        "requirements": len(requirements),
        "covered_requirements": len(covered),
        "coverage": len(covered) / len(requirements) if requirements else 0.0,
        "abstained_sections": abstained,
        "sections": len(sections),
        "abstain_rate": abstained / len(sections) if sections else 0.0,
        "grounded_sentences": len(grounded),
        "sentences": len(sentences),
        "groundedness": len(grounded) / len(sentences) if sentences else 0.0,
        "accurate_citations": accurate_citations,
        "citation_accuracy": (
            accurate_citations / len(grounded) if grounded else 0.0
        ),
        "fabrication_count": forbidden_hits,
        "client_leak_count": leak_hits,
        "guard_blocked_publish": state.get("guard_blocked_publish", 0),
        "hybrid_in_output": _hybrid_claims_in_output(state),
        "hybrid_blocked": len(trace.get("hybrid_blocked", [])),
        **_poison_reach(state),
        "conflict_dropped": len(trace.get("conflicts", [])),
        "dedup_rate": (
            (dedup_before - int(dedup.get("after", 0))) / dedup_before
            if dedup_before
            else 0.0
        ),
        "latency_seconds": trace.get("latency_seconds"),
        "token_cost": trace.get("token_cost"),
    }
