import argparse
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from config.section_map import (
    SECTION_DEFINITIONS,
    map_source_chapters,
    proposal_section_for_chapter,
    section_key_for_chapter,
)
from config.settings import (
    BRIDGE_SECTION_PRIORITY,
    CAPABILITY_FACTS_PER_SECTION,
    COMPANY_FACTS_PER_SECTION,
    MAX_BRIDGES_PER_PROPOSAL,
    MMR_TOP_K,
    PRECEDENTS_PER_CHAPTER,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    ROUTE_EMBEDDING_THRESHOLD,
)
from langgraph.graph import END, START, StateGraph

from .generate.capability import (
    default_fact_keys,
    generate_capabilities,
    render_fact,
)
from .generate.claim_check import (
    build_claim_whitelist,
    check_claims,
    filter_hybrid_claims,
)
from .generate.coverage import assess_coverage, matched_requirement_ids
from .generate.merge import deduplicate_sections, merge_channels
from .generate.precedent import generate_precedents
from .parsers.rfp_parser import (
    CHAPTER_RE,
    DEFAULT_RFP_DIR,
    INDUSTRY_RE,
    RFPParser,
)
from .retrieve.attribute import AttributeCoverage, AttributeRetriever
from .retrieve.hybrid import DEFAULT_EMBEDDING_MODEL, SearchResult
from .retrieve.mmr import MMRSelection, maximal_marginal_relevance
from .retrieve.rerank import RerankedResult, rerank_candidates, resolve_conflicts
from .schema import RFP
from .stores.capability import CapabilityStore
from .stores.sentence_index import SentenceIndex


SECTION_KEYS = {
    "key",
    "title_ja",
    "title_vi",
    "source_chapters",
    "sentences",
    "status",
    "note",
}
TRACE_KEYS = {"llm_calls", "retrieval_stats", "dedup"}


class GraphState(TypedDict, total=False):
    input_text: str
    rfp: RFP | None
    industry: str
    chapter_count: int
    parse_error: str | None
    missing: list[str]
    status: str
    message: str
    reference_rfp: dict[str, Any]
    chapters: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    proposal: str
    trace: dict[str, Any]


def _empty_trace() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "retrieval_stats": {},
        "dedup": {"before": 0, "after": 0},
        "path": [],
        "conflicts": [],
        "hybrid_blocked": [],
        "claim_removed": [],
        "claim_verdicts": {
            "VERIFIED": 0,
            "UNVERIFIABLE": 0,
            "CONTRADICTED": 0,
        },
        "generation_channels": {},
        "claim_whitelist": [],
        "global_dedup": [],
    }


def _with_trace(state: GraphState, node: str) -> dict[str, Any]:
    current = state.get("trace", _empty_trace())
    return {
        "llm_calls": current.get("llm_calls", 0),
        "retrieval_stats": dict(current.get("retrieval_stats", {})),
        "dedup": dict(current.get("dedup", {"before": 0, "after": 0})),
        "path": [*current.get("path", []), node],
        "conflicts": list(current.get("conflicts", [])),
        "hybrid_blocked": list(current.get("hybrid_blocked", [])),
        "claim_removed": list(current.get("claim_removed", [])),
        "claim_verdicts": dict(current.get("claim_verdicts", {})),
        "generation_channels": dict(current.get("generation_channels", {})),
        "claim_whitelist": list(current.get("claim_whitelist", [])),
        "global_dedup": list(current.get("global_dedup", [])),
    }


def parse_input(state: GraphState) -> GraphState:
    text = state["input_text"]
    industry_match = INDUSTRY_RE.search(text)
    chapter_count = len(CHAPTER_RE.findall(text))
    parsed_rfp: RFP | None = None
    parse_error: str | None = None

    try:
        # BƯỚC 4 cấm LLM: parser regex thất bại thì giữ trạng thái thiếu dữ liệu.
        parsed_rfp = RFPParser().parse(text, allow_llm_fallback=False)
    except ValueError as error:
        parse_error = str(error)

    return {
        "rfp": parsed_rfp,
        "industry": (
            parsed_rfp.industry
            if parsed_rfp is not None
            else industry_match.group(1).strip()
            if industry_match is not None
            else ""
        ),
        "chapter_count": (
            len(parsed_rfp.chapters) if parsed_rfp is not None else chapter_count
        ),
        "parse_error": parse_error,
        "trace": _with_trace(state, "parse_input"),
    }


def check_complete(state: GraphState) -> GraphState:
    missing: list[str] = []
    if not state.get("industry"):
        missing.append("industry (発注業種)")
    if state.get("chapter_count", 0) == 0:
        missing.append("chapters (第N章)")
    return {
        "missing": missing,
        "status": "needs_input" if missing else "ready",
        "trace": _with_trace(state, "check_complete"),
    }


def route_after_check(state: GraphState) -> str:
    return "ask_user" if state.get("missing") else "continue"


def ask_user(state: GraphState) -> GraphState:
    missing = ", ".join(state.get("missing", []))
    return {
        "status": "ask_user",
        "message": f"Chưa đủ thông tin để sinh nháp: thiếu {missing}.",
        "trace": _with_trace(state, "ask_user"),
    }


def route_reference_rfp(state: GraphState) -> GraphState:
    rfp = state.get("rfp")
    references = [
        RFPParser().parse_file(path, allow_llm_fallback=False)
        for path in sorted(DEFAULT_RFP_DIR.glob("*.txt"))
    ]
    industry_matches = [
        reference
        for reference in references
        if rfp is not None and reference.industry == rfp.industry
    ]
    if industry_matches:
        industry_matches.sort(
            key=lambda reference: (
                reference.rfp_id != rfp.rfp_id,
                reference.rfp_id,
            )
        )
        selected = industry_matches[0]
        reference_rfp = {
            "rfp_id": selected.rfp_id,
            "method": "industry",
            "score": 1.0,
        }
    elif rfp is not None and references:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        embeddings = model.encode(
            [rfp.title, *(reference.title for reference in references)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(embeddings[1:] @ embeddings[0], dtype=np.float32)
        best_index = int(scores.argmax())
        best_score = float(scores[best_index])
        reference_rfp = (
            {
                "rfp_id": references[best_index].rfp_id,
                "method": "embedding",
                "score": best_score,
            }
            if best_score >= ROUTE_EMBEDDING_THRESHOLD
            else {"rfp_id": "", "method": "none", "score": best_score}
        )
    else:
        reference_rfp = {"rfp_id": "", "method": "none", "score": None}
    return {
        "reference_rfp": reference_rfp,
        "trace": _with_trace(state, "route_reference_rfp"),
    }


def plan_sections(state: GraphState) -> GraphState:
    rfp = state.get("rfp")
    source_chapters = map_source_chapters(rfp.chapters if rfp is not None else [])
    return {
        "sections": [
            {
                "key": key,
                "title_ja": title_ja,
                "title_vi": title_vi,
                "source_chapters": source_chapters[key],
                "sentences": [],
                "status": None,
                "note": None,
            }
            for key, title_ja, title_vi in SECTION_DEFINITIONS
        ],
        "trace": _with_trace(state, "plan_sections"),
    }


def retrieve_per_chapter(state: GraphState) -> GraphState:
    rfp = state.get("rfp")
    if rfp is None:
        return {
            "chapters": [],
            "trace": _with_trace(state, "retrieve_per_chapter"),
        }

    sentence_index = SentenceIndex.build()
    attribute_retriever = AttributeRetriever()
    reference_rfp = state.get("reference_rfp", {}).get("rfp_id", "")
    coverages = {
        chapter.id: attribute_retriever.cover_chapter(chapter)
        for chapter in rfp.chapters
    }
    skip_decisions = _chapter_skip_decisions(
        rfp,
        coverages=coverages,
    )
    chapters: list[dict[str, Any]] = []
    retrieval_stats: dict[str, Any] = {}
    conflicts: list[dict[str, str]] = []
    aggregate_before = 0
    aggregate_after = 0

    for chapter in rfp.chapters:
        coverage = coverages[chapter.id]
        attribute_stage = _attribute_stage(coverage)

        if skip_decisions[chapter.id]:
            fact_keys = [
                fact_key
                for match in coverage.matches
                for fact_key in match.exact_fact_keys
            ]
            dedup = {"before": len(fact_keys), "after": len(set(fact_keys))}
            stages = {
                "attribute": attribute_stage,
                "query_embed": {"skipped": True},
                "rerank": {"skipped": True},
                "mmr": {"skipped": True},
            }
            candidates: list[dict[str, Any]] = []
            selected: list[dict[str, Any]] = []
            unique_texts = 0
            skipped = True
            conflict_dropped = 0
        else:
            query_text = " ".join(
                requirement.text for requirement in chapter.requirements
            )
            initial_results = sentence_index.search(
                query_text,
                k=RETRIEVAL_TOP_K,
            )
            reranked, selection, kept, dropped, expanded = _select_precedents(
                initial_results,
                sentence_index=sentence_index,
                query=query_text,
                target_industry=rfp.industry,
                target_section=proposal_section_for_chapter(chapter.title),
                reference_rfp=reference_rfp,
            )
            dedup = {"before": selection.before, "after": selection.after}
            stages = {
                "attribute": attribute_stage,
                "query_embed": {
                    "skipped": False,
                    "query": query_text,
                    "candidates": len(initial_results),
                    "expanded": expanded,
                },
                "rerank": {
                    "skipped": False,
                    "before": len(initial_results),
                    "after": min(RERANK_TOP_K, len(reranked)),
                },
                "mmr": {
                    "skipped": False,
                    "before": selection.before,
                    "after": selection.after,
                    "unique_texts": selection.unique_texts,
                    "k": MMR_TOP_K,
                },
            }
            candidates = [_search_result_dict(item.result) for item in reranked[:RETRIEVAL_TOP_K]]
            selected = [_reranked_result_dict(item) for item in kept]
            unique_texts = selection.unique_texts
            skipped = False
            conflict_dropped = len(dropped)
            conflicts.extend(
                {
                    "chapter_id": chapter.id,
                    "sent_id": item.sentence.sent_id,
                    "reason": "conflicting metric value",
                }
                for item in dropped
            )

        aggregate_before += dedup["before"]
        aggregate_after += dedup["after"]
        retrieval_stats[chapter.id] = {
            "skipped": skipped,
            "dedup": dedup,
            "unique_texts": unique_texts,
            "selected": len(selected),
            "conflict_dropped": conflict_dropped,
        }
        chapters.append(
            {
                "id": chapter.id,
                "title": chapter.title,
                "requirements": [
                    {"req_id": requirement.req_id, "text": requirement.text}
                    for requirement in chapter.requirements
                ],
                "retrieval": {
                    "stages": stages,
                    "candidates": candidates,
                    "selected": selected,
                    "dedup": dedup,
                },
            }
        )

    trace = _with_trace(state, "retrieve_per_chapter")
    trace["retrieval_stats"] = retrieval_stats
    trace["dedup"] = {"before": aggregate_before, "after": aggregate_after}
    trace["conflicts"] = conflicts
    trace["claim_whitelist"] = sorted(build_claim_whitelist(sentence_index))
    return {
        "chapters": chapters,
        "trace": trace,
    }


def _attribute_stage(coverage: AttributeCoverage) -> dict[str, Any]:
    return {
        "skipped": False,
        "fully_covered": coverage.fully_covered,
        "covered_req_ids": list(coverage.covered_req_ids),
        "missing_req_ids": list(coverage.missing_req_ids),
        "exactly_covered": coverage.exactly_covered,
        "exact_covered_req_ids": list(coverage.exact_covered_req_ids),
        "exact_missing_req_ids": list(coverage.exact_missing_req_ids),
        "matches": [
            {
                "req_id": match.req_id,
                "fact_keys": list(match.fact_keys),
                "exact_fact_keys": list(match.exact_fact_keys),
            }
            for match in coverage.matches
        ],
    }


def _chapter_skip_decisions(
    rfp: RFP,
    *,
    coverages: dict[str, AttributeCoverage],
) -> dict[str, bool]:
    return {
        chapter.id: (
            coverages[chapter.id].fully_covered
            and section_key_for_chapter(chapter.title) != "implementation_experience"
        )
        for chapter in rfp.chapters
    }


def _select_precedents(
    initial_results: list[SearchResult],
    *,
    sentence_index: SentenceIndex,
    query: str,
    target_industry: str,
    target_section: str,
    reference_rfp: str,
) -> tuple[
    list[RerankedResult],
    MMRSelection,
    list[RerankedResult],
    list[RerankedResult],
    bool,
]:
    initial_reranked = rerank_candidates(
        initial_results,
        target_industry=target_industry,
        target_section=target_section,
        top_k=None,
    )
    results = initial_results
    expanded = False
    while True:
        reranked = rerank_candidates(
            results,
            target_industry=target_industry,
            target_section=target_section,
            top_k=None,
        )
        unique_reranked: list[RerankedResult] = []
        seen_texts: set[str] = set()
        for item in reranked:
            if item.sentence.text not in seen_texts:
                seen_texts.add(item.sentence.text)
                unique_reranked.append(item)
            if len(unique_reranked) == RERANK_TOP_K:
                break
        try:
            selection = maximal_marginal_relevance(unique_reranked, k=MMR_TOP_K)
            kept, dropped = resolve_conflicts(
                list(selection.selected),
                reference_rfp=reference_rfp,
                industry=target_industry,
                industries=sentence_index.industries,
            )
            return initial_reranked, selection, kept, dropped, expanded
        except ValueError:
            if len(results) == len(sentence_index.all_sentences()):
                raise
            results = sentence_index.search(
                query,
                k=len(sentence_index.all_sentences()),
            )
            expanded = True


def _search_result_dict(result: SearchResult) -> dict[str, Any]:
    sentence = result.sentence
    return {
        "sent_id": sentence.sent_id,
        "proposal_id": sentence.proposal_id,
        "responds_to": sentence.responds_to,
        "section": sentence.section,
        "text": sentence.text,
        "claim_kind": sentence.claim_kind,
        "flags": dict(sentence.flags),
        "scores": {
            "hybrid": result.score,
            "bm25": result.bm25_score,
            "dense": result.dense_score,
        },
    }


def _reranked_result_dict(item: RerankedResult) -> dict[str, Any]:
    result = _search_result_dict(item.result)
    result["scores"].update(
        {
            "rerank": item.score,
            "industry_match": item.industry_match,
            "same_section_prior": item.same_section_prior,
        }
    )
    return result


def generate_per_section(state: GraphState) -> GraphState:
    capability_store = CapabilityStore()
    source_texts = {
        sentence["sent_id"]: sentence["text"]
        for chapter in state.get("chapters", [])
        for sentence in chapter["retrieval"]["selected"]
    }
    claim_whitelist = set(state.get("trace", {}).get("claim_whitelist", []))
    chapters_by_id = {
        chapter["id"]: chapter for chapter in state.get("chapters", [])
    }
    generated_sections: list[dict[str, Any]] = []
    generation_channels: dict[str, dict[str, int]] = {}
    coverage_inputs: dict[str, dict[str, Any]] = {}
    hybrid_blocked: list[dict[str, Any]] = []
    claim_removed: list[dict[str, Any]] = []
    contradicted_count = 0
    used_fact_keys: set[str] = set()
    llm_calls = 0

    for section in state.get("sections", []):
        source_chapters = [
            chapters_by_id[chapter_id]
            for chapter_id in section["source_chapters"]
        ]
        requirements = {
            requirement["req_id"]: requirement["text"]
            for chapter in source_chapters
            for requirement in chapter["requirements"]
        }

        precedent_sentences: list[dict[str, Any]] = []
        precedent_calls = 0
        for chapter in source_chapters:
            all_selected = chapter["retrieval"]["selected"]
            same_section = [
                item
                for item in all_selected
                if item["section"] == section["title_ja"]
            ]
            selected = (same_section or all_selected)[:PRECEDENTS_PER_CHAPTER]
            chapter_requirements = {
                item["req_id"]: item["text"] for item in chapter["requirements"]
            }
            generated, calls = generate_precedents(
                selected,
                req_ids_by_source={
                    item["sent_id"]: matched_requirement_ids(
                        item["text"],
                        chapter_requirements,
                    )
                    for item in selected
                },
            )
            precedent_sentences.extend(generated)
            precedent_calls += calls

        fact_keys: list[str] = []
        req_ids_by_fact: dict[str, list[str]] = {}
        for chapter in source_chapters:
            attribute_stage = chapter["retrieval"]["stages"]["attribute"]
            for match in attribute_stage["matches"]:
                for fact_key in match["fact_keys"]:
                    fact_keys.append(fact_key)
                    req_ids_by_fact.setdefault(fact_key, []).append(match["req_id"])
        if section["key"] == "company_overview":
            fact_keys = default_fact_keys(section["key"])
        elif not fact_keys:
            fact_keys = default_fact_keys(section["key"])
        fact_keys = [
            key for key in dict.fromkeys(fact_keys) if key not in used_fact_keys
        ]
        if not fact_keys:
            fact_keys = [
                key
                for key in default_fact_keys(section["key"])
                if key not in used_fact_keys
            ]
        if not fact_keys:
            fact_keys = [key for key in capability_store if key not in used_fact_keys]
        fact_limit = (
            COMPANY_FACTS_PER_SECTION
            if section["key"] == "company_overview"
            else CAPABILITY_FACTS_PER_SECTION
        )
        fact_keys = fact_keys[:fact_limit]
        used_fact_keys.update(fact_keys)
        capability_sentences, capability_calls = generate_capabilities(
            fact_keys,
            req_ids_by_fact={
                key: matched_requirement_ids(
                    render_fact(capability_store, key),
                    requirements,
                    candidate_req_ids=req_ids_by_fact.get(key, []),
                )
                for key in fact_keys
            },
            store=capability_store,
        )

        merged = merge_channels(
            capability_sentences,
            precedent_sentences,
            section_key=section["key"],
            add_bridge=False,
        )
        hybrid_clean, blocked = filter_hybrid_claims(merged, claim_whitelist)
        checked, verdicts, check_calls, removed = check_claims(
            hybrid_clean,
            capability_store=capability_store,
            source_texts=source_texts,
        )
        checked = merge_channels(
            [item for item in checked if item["origin"] == "capability"],
            [item for item in checked if item["origin"] == "precedent"],
            section_key=section["key"],
            add_bridge=False,
        )
        llm_calls += precedent_calls + capability_calls + check_calls
        contradicted_count += verdicts["CONTRADICTED"]
        hybrid_blocked.extend(
            {
                "section_key": section["key"],
                "source_id": sentence["source_id"],
                "text": sentence["text"],
            }
            for sentence in blocked
        )
        claim_removed.extend(
            {
                "section_key": section["key"],
                "source_id": sentence["source_id"],
                "text": sentence["text"],
                "verdict": sentence["verdict"],
            }
            for sentence in removed
        )
        skipped = any(
            chapter["retrieval"]["stages"]["query_embed"]["skipped"]
            for chapter in source_chapters
        )
        coverage_inputs[section["key"]] = {
            "source_chapters": list(section["source_chapters"]),
            "requirements": requirements,
            "skipped": skipped,
        }
        generated_sections.append(
            {
                "key": section["key"],
                "title_ja": section["title_ja"],
                "title_vi": section["title_vi"],
                "source_chapters": list(section["source_chapters"]),
                "sentences": checked,
                "status": None,
                "note": None,
            }
        )

    generated_sections, global_dedup = deduplicate_sections(
        generated_sections,
        bridge_priority=BRIDGE_SECTION_PRIORITY,
        max_bridges=MAX_BRIDGES_PER_PROPOSAL,
    )
    for section in generated_sections:
        coverage_input = coverage_inputs[section["key"]]
        status, note = assess_coverage(
            section_key=section["key"],
            source_chapters=coverage_input["source_chapters"],
            requirements=coverage_input["requirements"],
            sentences=section["sentences"],
            skipped=coverage_input["skipped"],
        )
        section["status"] = status
        section["note"] = note
        generation_channels[section["key"]] = {
            origin: sum(
                item["origin"] == origin for item in section["sentences"]
            )
            for origin in ("capability", "precedent", "bridge")
        }

    claim_verdicts = Counter(
        sentence["verdict"]
        for section in generated_sections
        for sentence in section["sentences"]
    )
    claim_verdicts["CONTRADICTED"] += contradicted_count

    trace = _with_trace(state, "generate_per_section")
    trace["llm_calls"] += llm_calls
    trace["generation_channels"] = generation_channels
    trace["hybrid_blocked"] = hybrid_blocked
    trace["claim_removed"] = claim_removed
    trace["global_dedup"] = global_dedup
    trace["claim_verdicts"] = {
        name: claim_verdicts[name]
        for name in ("VERIFIED", "UNVERIFIABLE", "CONTRADICTED")
    }
    return {
        "sections": generated_sections,
        "trace": trace,
    }


def assemble(state: GraphState) -> GraphState:
    proposal = "\n\n".join(
        f"{index}. {section['title_ja']}\n"
        + "".join(sentence.get("text", "") for sentence in section["sentences"])
        for index, section in enumerate(state.get("sections", []), start=1)
    )
    return {
        "proposal": proposal,
        "status": "completed",
        "trace": _with_trace(state, "assemble"),
    }


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("parse_input", parse_input)
    builder.add_node("check_complete", check_complete)
    builder.add_node("ask_user", ask_user)
    builder.add_node("route_reference_rfp", route_reference_rfp)
    builder.add_node("plan_sections", plan_sections)
    builder.add_node("retrieve_per_chapter", retrieve_per_chapter)
    builder.add_node("generate_per_section", generate_per_section)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "parse_input")
    builder.add_edge("parse_input", "check_complete")
    builder.add_conditional_edges(
        "check_complete",
        route_after_check,
        {
            "ask_user": "ask_user",
            "continue": "route_reference_rfp",
        },
    )
    builder.add_edge("ask_user", END)
    builder.add_edge("route_reference_rfp", "plan_sections")
    builder.add_edge("plan_sections", "retrieve_per_chapter")
    builder.add_edge("retrieve_per_chapter", "generate_per_section")
    builder.add_edge("generate_per_section", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile()


GRAPH = build_graph()


def run_graph(text: str) -> GraphState:
    return GRAPH.invoke(
        {
            "input_text": text,
            "reference_rfp": {"rfp_id": "", "method": "none", "score": None},
            "chapters": [],
            "sections": [],
            "trace": _empty_trace(),
        }
    )


def check_schema(state: GraphState) -> None:
    sections = state.get("sections", [])
    if len(sections) != len(SECTION_DEFINITIONS):
        raise AssertionError(f"sections phải có 5 phần tử, hiện có {len(sections)}")
    for index, section in enumerate(sections):
        if set(section) != SECTION_KEYS:
            missing = sorted(SECTION_KEYS - set(section))
            extra = sorted(set(section) - SECTION_KEYS)
            raise AssertionError(
                f"sections[{index}] sai schema: missing={missing}, extra={extra}"
            )

    trace = state.get("trace")
    if not isinstance(trace, dict):
        raise AssertionError("trace phải là dict")
    missing_trace = TRACE_KEYS - set(trace)
    if missing_trace:
        raise AssertionError(f"trace thiếu khoá: {sorted(missing_trace)}")
    if not isinstance(trace["dedup"], dict):
        raise AssertionError("trace.dedup phải là dict")
    if set(trace["dedup"]) != {"before", "after"}:
        raise AssertionError("trace.dedup phải có before và after")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Không thể serialize {type(value).__name__}")


def print_trace(state: GraphState) -> None:
    reference = state["reference_rfp"]
    trace = state["trace"]
    print(
        f"route method={reference['method']}, "
        f"rfp_id={reference['rfp_id']}, llm_calls={trace['llm_calls']}"
    )
    for chapter in state.get("chapters", []):
        retrieval = chapter["retrieval"]
        stats = trace["retrieval_stats"][chapter["id"]]
        dedup = retrieval["dedup"]
        mmr_stage = retrieval["stages"]["mmr"]
        unique = mmr_stage.get("unique_texts", 0)
        k = mmr_stage.get("k", 0)
        print(
            f"chapter {chapter['id']} {chapter['title']}: "
            f"skipped={stats['skipped']}, "
            f"dedup: {{before: {dedup['before']}, after: {dedup['after']}}}, "
            f"unique={unique}/{k}, "
            f"selected={stats['selected']}, "
            f"conflict_dropped={stats['conflict_dropped']}"
        )
    selected_by_chapter = {
        chapter["id"]: len(chapter["retrieval"]["selected"])
        for chapter in state.get("chapters", [])
    }
    for section in state.get("sections", []):
        selected = sum(
            selected_by_chapter[chapter_id]
            for chapter_id in section["source_chapters"]
        )
        print(
            f"section {section['key']}: "
            f"source_chapters={section['source_chapters']}, "
            f"selected={selected}"
        )
    print(f"total_selected={sum(selected_by_chapter.values())}")
    print(f"conflicts={len(trace.get('conflicts', []))}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="RFP proposal graph")
    source = argument_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rfp", type=Path)
    source.add_argument("--text")
    argument_parser.add_argument("--check-schema", action="store_true")
    argument_parser.add_argument("--trace", action="store_true")
    argument_parser.add_argument("--json", type=Path)
    args = argument_parser.parse_args()

    text = (
        args.rfp.read_text(encoding="utf-8")
        if args.rfp is not None
        else args.text
    )
    result = run_graph(text)

    if args.json is not None:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"json={args.json} · status={result['status']}")
        return

    if args.check_schema:
        check_schema(result)
        section_key_text = " · ".join(
            (
                "key",
                "title_ja",
                "title_vi",
                "source_chapters",
                "sentences",
                "status",
                "note",
            )
        )
        print(f"schema=ok · sections=5 · keys={section_key_text}")
        print("trace=dict · llm_calls · retrieval_stats · dedup · path")
        return

    if args.trace:
        print(f"status={result['status']}")
        print_trace(result)
        return

    if result["status"] == "ask_user":
        print("route=ask_user")
        print(result["message"])
        return

    print(f"status={result['status']}")
    print(result["proposal"])


if __name__ == "__main__":
    main()
