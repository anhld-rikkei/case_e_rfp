import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .generate.capability import render_fact
from .generate.coverage import validate_source_req_ids
from .generate.precedent import numeric_expressions
from .sanitize.blocklist import CapabilityBlocklist
from .stores.capability import CapabilityStore
from .stores.sentence_index import SentenceIndex


VALID_ORIGINS = {"capability", "precedent", "bridge"}
VALID_STATUSES = {"OK", "ATTRIBUTE_ONLY", "INSUFFICIENT_EVIDENCE"}
VALID_VERDICTS = {"VERIFIED", "UNVERIFIABLE", "CONTRADICTED"}
SENTENCE_KEYS = {"text", "origin", "source_id", "req_ids", "verdict"}


def validate_output(state: dict[str, Any]) -> dict[str, Any]:
    sections = state.get("sections", [])
    if len(sections) != 5:
        raise AssertionError(f"sections phải có 5 mục, hiện có {len(sections)}")

    blocklist = CapabilityBlocklist()
    capability_store = CapabilityStore()
    source_sentences = {
        sentence.sent_id: sentence for sentence in SentenceIndex.build().all_sentences()
    }
    requirements = {
        requirement["req_id"]: requirement["text"]
        for chapter in state.get("chapters", [])
        for requirement in chapter["requirements"]
    }
    chapters_by_id = {
        chapter["id"]: chapter for chapter in state.get("chapters", [])
    }
    origin_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    blocklist_hits: list[str] = []
    seen_source_ids: set[str] = set()
    seen_texts: set[str] = set()
    bridge_texts: list[str] = []
    covered_req_ids: set[str] = set()

    for section in sections:
        status = section.get("status")
        if status not in VALID_STATUSES:
            raise AssertionError(f"status không hợp lệ: {status}")
        status_counts[status] += 1
        section_covered_req_ids: set[str] = set()
        for sentence in section.get("sentences", []):
            if set(sentence) != SENTENCE_KEYS:
                raise AssertionError(f"Sentence sai schema: {sentence}")
            origin = sentence["origin"]
            source_id = sentence["source_id"]
            verdict = sentence["verdict"]
            if origin not in VALID_ORIGINS:
                raise AssertionError(f"origin không hợp lệ: {origin}")
            if (source_id is None) != (origin == "bridge"):
                raise AssertionError(
                    f"source_id không khớp origin: {origin=} {source_id=}"
                )
            if verdict not in VALID_VERDICTS:
                raise AssertionError(f"verdict không hợp lệ: {verdict}")
            if source_id is None and verdict != "UNVERIFIABLE":
                raise AssertionError(
                    "Câu không có source_id bắt buộc phải UNVERIFIABLE"
                )
            origin_counts[origin] += 1
            blocklist_hits.extend(blocklist.find(sentence["text"]))

            if origin == "bridge":
                bridge_texts.append(sentence["text"])
            else:
                normalized_text = "".join(sentence["text"].split())
                if source_id in seen_source_ids or normalized_text in seen_texts:
                    raise AssertionError(
                        f"Câu trùng ở tầng toàn hồ sơ: {sentence['text']}"
                    )
                seen_source_ids.add(source_id)
                seen_texts.add(normalized_text)

            if origin == "precedent":
                source = source_sentences.get(source_id)
                if source is None:
                    raise AssertionError(f"Không tìm thấy precedent source_id={source_id}")
                expected_numbers = numeric_expressions(source.text)
                source_text = source.text
            elif origin == "capability":
                if source_id not in capability_store:
                    raise AssertionError(f"Không tìm thấy capability source_id={source_id}")
                expected_numbers = numeric_expressions(
                    render_fact(capability_store, source_id)
                )
                source_text = render_fact(capability_store, source_id)
            else:
                expected_numbers = Counter()
                source_text = ""
            if numeric_expressions(sentence["text"]) != expected_numbers:
                raise AssertionError(f"Số không truy được về nguồn: {sentence['text']}")
            if origin == "bridge" and sentence["req_ids"]:
                raise AssertionError("Bridge không được gắn req_ids")
            if origin != "bridge":
                validate_source_req_ids(
                    sentence["req_ids"],
                    source_text=source_text,
                    requirements=requirements,
                )
                section_covered_req_ids.update(sentence["req_ids"])

        section_requirement_ids = {
            requirement["req_id"]
            for chapter_id in section["source_chapters"]
            for requirement in chapters_by_id[chapter_id]["requirements"]
        }
        section_missing = sorted(section_requirement_ids - section_covered_req_ids)
        if section_missing and status != "INSUFFICIENT_EVIDENCE":
            raise AssertionError(
                f"Section thiếu {section_missing} nhưng status={status}"
            )
        if section_missing and not all(
            req_id in (section.get("note") or "") for req_id in section_missing
        ):
            raise AssertionError(
                f"section.note không nêu đủ requirement thiếu: {section_missing}"
            )
        covered_req_ids.update(section_covered_req_ids)

    if blocklist_hits:
        raise AssertionError(f"Có chuỗi blocklist trong output: {blocklist_hits}")

    verdict_counts = Counter(state.get("trace", {}).get("claim_verdicts", {}))
    for verdict in VALID_VERDICTS:
        verdict_counts.setdefault(verdict, 0)
    bridge_count = origin_counts["bridge"]
    sentence_count = sum(origin_counts.values())
    bridge_rate = bridge_count / sentence_count if sentence_count else 0.0
    if bridge_rate > 0.15:
        raise AssertionError(f"bridge rate vượt 15%: {bridge_count}/{sentence_count}")
    if len(bridge_texts) != len(set(bridge_texts)):
        raise AssertionError("Câu bridge bị lặp nguyên văn giữa các mục")
    total_requirements = len(requirements)
    covered_requirements = len(covered_req_ids & set(requirements))
    return {
        "sentences": sentence_count,
        "origin_counts": origin_counts,
        "status_counts": status_counts,
        "blocklist_hits": len(blocklist_hits),
        "verdict_counts": verdict_counts,
        "bridge_rate": bridge_rate,
        "covered_requirements": covered_requirements,
        "total_requirements": total_requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm contract proposal JSON")
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    with args.json_path.open(encoding="utf-8") as input_file:
        state = json.load(input_file)
    result = validate_output(state)
    origins = result["origin_counts"]
    statuses = result["status_counts"]
    verdicts = result["verdict_counts"]
    print(
        f"sentences={result['sentences']} · origins=ok "
        f"(capability={origins['capability']}, precedent={origins['precedent']}, "
        f"bridge={origins['bridge']}) · source_ids=ok"
    )
    print(f"bridge_rate={result['bridge_rate']:.1%} · <=15%=ok")
    print("blocklist=0/6")
    print(
        f"coverage={result['covered_requirements']}/{result['total_requirements']} "
        "· req_ids=source-matched"
    )
    print(
        "sections=5 · statuses=ok "
        f"(OK={statuses['OK']}, ATTRIBUTE_ONLY={statuses['ATTRIBUTE_ONLY']}, "
        f"INSUFFICIENT_EVIDENCE={statuses['INSUFFICIENT_EVIDENCE']})"
    )
    print(
        "claim verdicts: "
        f"VERIFIED={verdicts['VERIFIED']} · "
        f"UNVERIFIABLE={verdicts['UNVERIFIABLE']} · "
        f"CONTRADICTED={verdicts['CONTRADICTED']}"
    )


if __name__ == "__main__":
    main()
