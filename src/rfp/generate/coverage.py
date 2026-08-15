import re
import unicodedata
from collections.abc import Iterable
from typing import Any, Literal

from config.settings import COVERAGE_MIN_SHARED_ANCHORS


SectionStatus = Literal["OK", "ATTRIBUTE_ONLY", "INSUFFICIENT_EVIDENCE"]

ANCHOR_RE = re.compile(
    r"[一-龯ァ-ヶー]{2,}"
    r"|[A-Za-z]+(?:/[A-Za-z]+)*(?:\s+\d+(?:\.\d+)?)?"
    r"|\d+(?:\.\d+)?%?(?:か月|ヶ月|名|人|件|年)?"
)
IGNORED_ANCHORS = {
    "要件",
    "対応",
    "可能",
    "業務",
    "本調",
    "調達",
    "当社",
    "各要",
}


def source_anchors(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    anchors: set[str] = set()
    for chunk in ANCHOR_RE.findall(normalized):
        compact = chunk.replace(" ", "")
        if re.fullmatch(r"[一-龯ァ-ヶー]+", compact):
            if len(compact) == 2:
                anchors.add(compact)
            else:
                anchors.update(
                    compact[index : index + 2]
                    for index in range(len(compact) - 1)
                )
        else:
            anchors.add(compact)
    return anchors - IGNORED_ANCHORS


def source_supports_requirement(source_text: str, requirement_text: str) -> bool:
    shared = source_anchors(source_text) & source_anchors(requirement_text)
    return len(shared) >= COVERAGE_MIN_SHARED_ANCHORS


def matched_requirement_ids(
    source_text: str,
    requirements: dict[str, str],
    *,
    candidate_req_ids: Iterable[str] | None = None,
) -> list[str]:
    candidates = (
        set(candidate_req_ids)
        if candidate_req_ids is not None
        else set(requirements)
    )
    return [
        req_id
        for req_id, requirement_text in requirements.items()
        if req_id in candidates
        and source_supports_requirement(source_text, requirement_text)
    ]


def validate_source_req_ids(
    req_ids: Iterable[str],
    *,
    source_text: str,
    requirements: dict[str, str],
) -> None:
    invalid = [
        req_id
        for req_id in req_ids
        if req_id not in requirements
        or not source_supports_requirement(source_text, requirements[req_id])
    ]
    if invalid:
        raise AssertionError(
            f"req_ids không truy được về nội dung nguồn: {sorted(invalid)}"
        )


def assess_coverage(
    *,
    section_key: str,
    source_chapters: list[str],
    requirements: dict[str, str],
    sentences: list[dict[str, Any]],
    skipped: bool,
) -> tuple[SectionStatus, str | None]:
    precedent_req_ids = {
        req_id
        for sentence in sentences
        if sentence["origin"] == "precedent"
        for req_id in sentence["req_ids"]
    }
    capability_req_ids = {
        req_id
        for sentence in sentences
        if sentence["origin"] == "capability"
        for req_id in sentence["req_ids"]
    }
    covered = precedent_req_ids | capability_req_ids
    missing = sorted(set(requirements) - covered)

    if missing:
        missing_evidence = "; ".join(
            f"{req_id} {requirements[req_id]} — 対応する能力・先行事例の根拠がありません"
            for req_id in missing
        )
        return "INSUFFICIENT_EVIDENCE", f"INSUFFICIENT_EVIDENCE: {missing_evidence}"
    if not source_chapters:
        if section_key == "company_overview":
            return "OK", None
        return "ATTRIBUTE_ONLY", "RFPに対応する原章がないため、能力表のみで構成しました。"
    if skipped and not precedent_req_ids:
        return "ATTRIBUTE_ONLY", "属性の完全一致により検索を省略し、能力表のみで回答しました。"
    if not precedent_req_ids:
        return "ATTRIBUTE_ONLY", "参照可能な先行事例がないため、能力表のみで回答しました。"
    return "OK", None
