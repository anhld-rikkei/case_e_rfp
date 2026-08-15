from typing import Any, Iterable


BRIDGE_TEXTS = {
    "proposal_overview": "提案方針には、過去案件で得た知見も反映しています。",
    "implementation_experience": "これらの保有能力は、関連する導入実績によって裏付けられています。",
    "certification_compliance": "保有認証と過去案件の管理実績を併せて適用します。",
    "delivery_structure": "推進方針には、過去案件で培った運営知見も反映しています。",
}


def _capability_order(sentence: dict[str, Any]) -> tuple[int, str]:
    source_id = str(sentence["source_id"])
    return (0 if source_id.startswith("company_facts:") else 1, source_id)


def merge_channels(
    capability_sentences: list[dict[str, Any]],
    precedent_sentences: list[dict[str, Any]],
    *,
    section_key: str,
    add_bridge: bool,
) -> list[dict[str, Any]]:
    merged = sorted(capability_sentences, key=_capability_order)
    if add_bridge and merged and precedent_sentences:
        merged.append(
            {
                "text": BRIDGE_TEXTS[section_key],
                "origin": "bridge",
                "source_id": None,
                "req_ids": [],
                "verdict": "UNVERIFIABLE",
            }
        )
    merged.extend(precedent_sentences)
    return merged


def deduplicate_sections(
    sections: list[dict[str, Any]],
    *,
    bridge_priority: Iterable[str],
    max_bridges: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_source_ids: set[str] = set()
    seen_texts: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for section in sections:
        kept_sentences: list[dict[str, Any]] = []
        for sentence in section["sentences"]:
            if sentence["origin"] == "bridge":
                continue
            source_id = sentence["source_id"]
            normalized_text = "".join(sentence["text"].split())
            if source_id in seen_source_ids or normalized_text in seen_texts:
                dropped.append(
                    {
                        "section_key": section["key"],
                        "source_id": source_id,
                        "text": sentence["text"],
                    }
                )
                continue
            seen_source_ids.add(source_id)
            seen_texts.add(normalized_text)
            kept_sentences.append(sentence)
        updated = dict(section)
        updated["sentences"] = kept_sentences
        deduplicated.append(updated)

    eligible = {
        section["key"]
        for section in deduplicated
        if any(item["origin"] == "capability" for item in section["sentences"])
        and any(item["origin"] == "precedent" for item in section["sentences"])
    }
    bridge_sections = set(
        [key for key in bridge_priority if key in eligible][:max_bridges]
    )
    for section in deduplicated:
        capability = [
            item for item in section["sentences"] if item["origin"] == "capability"
        ]
        precedent = [
            item for item in section["sentences"] if item["origin"] == "precedent"
        ]
        section["sentences"] = merge_channels(
            capability,
            precedent,
            section_key=section["key"],
            add_bridge=section["key"] in bridge_sections,
        )
    return deduplicated, dropped
