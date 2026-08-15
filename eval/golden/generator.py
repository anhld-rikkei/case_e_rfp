from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

from .schema import Assertion, GoldenCase, load_case, save_case


ROOT_DIR = Path(__file__).resolve().parents[2]
CAPABILITY_PATH = ROOT_DIR / "synthetic" / "capability_sheet.json"
GOLDEN_DIR = ROOT_DIR / "synthetic" / "golden_test_set"
DEFAULT_OUTPUT_DIR = GOLDEN_DIR / "generated"
DEFAULT_BASE_CASE = GOLDEN_DIR / "original-rfp-2025-001.json"

CHAPTER_HEADER_RE = re.compile(r"^第(?P<number>\d+)章\s+(?P<title>.+?)\s*$")
REQUIREMENT_ID_RE = re.compile(r"^(?P<chapter>\d+)\.(?P<number>\d+)\s+")
INDUSTRY_RE = re.compile(r"^発注業種[：:]\s*.*$")

CHAPTER_TITLES = (
    "調達概要",
    "業務要件",
    "技術要件",
    "セキュリティ要件",
    "納期・体制",
    "提案書記載事項",
)
PARAPHRASE_SYSTEM = (
    "あなたは日本語RFPの編集者です。要件の意味、章、requirement ID、業種、"
    "固有の製品名・認証名・数値を一切変えず、文体だけを書き換えてください。"
    "説明やMarkdownを付けず、RFP全文だけを返してください。"
)


def capability_data() -> dict[str, Any]:
    return json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))


def requirement_text(term: str, category: str) -> str:
    if category in {"certification", "certification_not_held"}:
        return f"{term}認証を有すること。"
    return f"{term}に対応できること。"


def requirement_pools() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = capability_data()
    in_scope = [
        {"term": term, "category": "capability"}
        for term in data["capabilities"]
    ] + [
        {"term": term, "category": "certification"}
        for term in data["certifications_held"]
    ]
    out_scope = [
        {"term": term, "category": "capability_not_offered"}
        for term in data["capabilities_NOT_offered"]
    ] + [
        {"term": term, "category": "certification_not_held"}
        for term in data["certifications_NOT_held"]
    ]
    return in_scope, out_scope


def _preferred_chapter(requirement: dict[str, str]) -> str:
    if "certification" in requirement["category"]:
        return "セキュリティ要件"
    return "技術要件"


def _render_case(
    *,
    case_id: str,
    industry: str,
    requirements: list[dict[str, str]],
    chapter_titles: tuple[str, ...],
) -> GoldenCase:
    if not chapter_titles:
        raise ValueError("At least one chapter must be selected")
    grouped: dict[str, list[dict[str, str]]] = {}
    for index, requirement in enumerate(requirements):
        preferred = _preferred_chapter(requirement)
        chapter = (
            preferred
            if preferred in chapter_titles
            else chapter_titles[index % len(chapter_titles)]
        )
        grouped.setdefault(chapter, []).append(requirement)

    lines = [
        "生成ゴールデンテスト用 調達仕様書",
        f"発注業種：{industry}",
        f"調達番号：{case_id}",
        "",
    ]
    assertions: list[Assertion] = []
    for canonical_number, chapter in enumerate(CHAPTER_TITLES, start=1):
        chapter_requirements = grouped.get(chapter, [])
        if not chapter_requirements:
            continue
        lines.append(f"第{canonical_number}章 {chapter}")
        for requirement_number, requirement in enumerate(chapter_requirements, start=1):
            req_id = f"{canonical_number}.{requirement_number}"
            lines.append(
                f"{req_id} {requirement_text(requirement['term'], requirement['category'])}"
            )
            if requirement["category"] in {"capability", "certification"}:
                assertions.append(
                    Assertion(
                        kind="must_cover",
                        target=req_id,
                        reason=(
                            f"{requirement['term']} được rút từ rổ IN-SCOPE của capability sheet."
                        ),
                    )
                )
            else:
                assertions.extend(
                    [
                        Assertion(
                            kind="must_flag_insufficient",
                            target=req_id,
                            reason=(
                                f"{requirement['term']} được rút từ rổ OUT-SCOPE của capability sheet."
                            ),
                        ),
                        Assertion(
                            kind="must_not_contain",
                            target=requirement["term"],
                            reason=(
                                "Proposal không được tuyên bố capability/certification ngoài năng lực."
                            ),
                        ),
                    ]
                )
        lines.append("")

    return GoldenCase(
        case_id=case_id.lower(),
        rfp_text="\n".join(lines).rstrip() + "\n",
        source="combinatorial",
        needs_review=False,
        assertions=assertions,
        metadata={
            "industry": industry,
            "chapters": list(grouped),
            "out_scope_count": sum(
                "not_" in requirement["category"]
                for requirement in requirements
            ),
        },
    )


def generate_combinatorial(
    n: int,
    *,
    industries: Iterable[str] | None = None,
    chapters: Iterable[str] | None = None,
    out_scope_count: int | None = None,
    seed: int = 9,
) -> list[GoldenCase]:
    if n < 1:
        raise ValueError("n must be at least 1")
    data = capability_data()
    industry_pool = list(industries or [*data["industries_served"], "医療"])
    chapter_pool = tuple(chapters or CHAPTER_TITLES)
    if not industry_pool:
        raise ValueError("At least one industry must be selected")
    if out_scope_count is not None and not 0 <= out_scope_count <= 3:
        raise ValueError("out_scope_count must be between 0 and 3")

    in_scope, out_scope = requirement_pools()
    rng = random.Random(seed)
    in_scope = rng.sample(in_scope, len(in_scope))
    out_scope = rng.sample(out_scope, len(out_scope))
    cases = []
    for index in range(n):
        case_out_count = index % 4 if out_scope_count is None else out_scope_count
        requirements = [in_scope[index % len(in_scope)]]
        requirements.extend(
            out_scope[(index + offset) % len(out_scope)]
            for offset in range(case_out_count)
        )
        rotated_chapters = chapter_pool[index % len(chapter_pool) :] + chapter_pool[: index % len(chapter_pool)]
        cases.append(
            _render_case(
                case_id=f"GOLDEN-COMB-{index + 1:03d}",
                industry=industry_pool[index % len(industry_pool)],
                requirements=requirements,
                chapter_titles=rotated_chapters,
            )
        )
    return cases


def _mutation_assertions(name: str, requirement_id: str | None = None) -> list[Assertion]:
    values = {
        "capabilities_NOT_offered": [
            Assertion(
                "must_flag_insufficient",
                requirement_id or "3.3",
                "Blockchain payment platform nằm trong capabilities_NOT_offered.",
            ),
            Assertion(
                "must_not_contain",
                "ブロックチェーン決済基盤",
                "Proposal không được nhận năng lực blockchain ngoài phạm vi.",
            ),
        ],
        "certifications_NOT_held": [
            Assertion(
                "must_flag_insufficient",
                requirement_id or "4.3",
                "ISO/IEC 27017 nằm trong certifications_NOT_held.",
            ),
            Assertion(
                "must_not_contain",
                "ISO/IEC 27017",
                "Proposal không được tuyên bố chứng nhận không nắm giữ.",
            ),
        ],
        "industry_outside_served": [
            Assertion(
                "must_route",
                "industry:医療",
                "Parser phải giữ đúng ngành ngoài industries_served.",
            ),
            Assertion(
                "must_not_contain",
                "AI医療画像診断",
                "Không được bịa năng lực y tế nằm ngoài phạm vi.",
            ),
        ],
        "industry_inside_served": [
            Assertion(
                "must_route",
                "industry:公共",
                "Parser và retrieval phải dùng ngành 公共 mới.",
            )
        ],
        "missing_industry": [
            Assertion(
                "must_ask_user",
                "industry",
                "Thiếu 発注業種 phải rẽ ask_user.",
            )
        ],
        "parser_fallback": [
            Assertion(
                "must_route",
                "completed",
                "Text không có 第N章 phải được fallback parse và chạy hoàn tất.",
            )
        ],
    }
    return values[name]


def _insert_requirement(
    text: str,
    chapter_title: str,
    requirement: str,
) -> tuple[str, str]:
    lines = text.splitlines()
    chapter_index = -1
    chapter_number = ""
    for index, line in enumerate(lines):
        match = CHAPTER_HEADER_RE.match(line)
        if match and match.group("title") == chapter_title:
            chapter_index = index
            chapter_number = match.group("number")
            break
    if chapter_index < 0:
        chapter_number = str(CHAPTER_TITLES.index(chapter_title) + 1)
        requirement_id = f"{chapter_number}.1"
        lines.extend(
            [
                "",
                f"第{chapter_number}章 {chapter_title}",
                f"{requirement_id} {requirement}",
            ]
        )
        return "\n".join(lines).rstrip() + "\n", requirement_id

    next_chapter = len(lines)
    for index in range(chapter_index + 1, len(lines)):
        if CHAPTER_HEADER_RE.match(lines[index]):
            next_chapter = index
            break
    used_numbers = [
        int(match.group("number"))
        for line in lines[chapter_index + 1 : next_chapter]
        if (match := REQUIREMENT_ID_RE.match(line))
        and match.group("chapter") == chapter_number
    ]
    requirement_id = f"{chapter_number}.{max(used_numbers, default=0) + 1}"
    insert_at = next_chapter
    while insert_at > chapter_index + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f"{requirement_id} {requirement}")
    return "\n".join(lines).rstrip() + "\n", requirement_id


def _replace_industry(text: str, industry: str | None) -> str:
    lines = text.splitlines()
    indexes = [index for index, line in enumerate(lines) if INDUSTRY_RE.match(line)]
    if len(indexes) != 1:
        raise ValueError("RFP nền phải có đúng một dòng 発注業種")
    if industry is None:
        del lines[indexes[0]]
    else:
        lines[indexes[0]] = f"発注業種：{industry}"
    return "\n".join(lines).rstrip() + "\n"


def _apply_mutation(name: str, text: str) -> tuple[str, str | None]:
    if name == "capabilities_NOT_offered":
        return _insert_requirement(
            text,
            "技術要件",
            "ブロックチェーン決済基盤を構築すること。",
        )
    if name == "certifications_NOT_held":
        return _insert_requirement(
            text,
            "セキュリティ要件",
            "ISO/IEC 27017認証を有すること。",
        )
    if name == "industry_outside_served":
        return _replace_industry(text, "医療"), None
    if name == "industry_inside_served":
        return _replace_industry(text, "公共"), None
    if name == "missing_industry":
        return _replace_industry(text, None), None
    if name == "parser_fallback":
        mutated, count = re.subn(r"^第\d+章\s*", "", text, flags=re.MULTILINE)
        if count == 0:
            raise ValueError("RFP nền không có tiêu đề 第N章 để bỏ đánh số")
        return mutated, None
    raise ValueError(f"Mutation không được hỗ trợ: {name}")


def generate_mutations(base: GoldenCase, n: int = 6) -> list[GoldenCase]:
    mutation_names = (
        "capabilities_NOT_offered",
        "certifications_NOT_held",
        "industry_outside_served",
        "industry_inside_served",
        "missing_industry",
        "parser_fallback",
    )
    cases = []
    for index in range(n):
        name = mutation_names[index % len(mutation_names)]
        mutated_text, requirement_id = _apply_mutation(name, base.rfp_text)
        inherited = [
            assertion
            for assertion in base.assertions
            if not (
                assertion.kind == "must_route"
                and (
                    name == "missing_industry"
                    or name in {"industry_outside_served", "industry_inside_served"}
                    and assertion.target.startswith("industry:")
                )
            )
        ]
        cases.append(
            GoldenCase(
                case_id=f"golden-mut-{index + 1:03d}-{name.lower()}",
                rfp_text=mutated_text,
                source="mutation",
                needs_review=False,
                assertions=[
                    *inherited,
                    *_mutation_assertions(name, requirement_id),
                ],
                metadata={
                    "base_case_id": base.case_id,
                    "mutation": name,
                },
            )
        )
    return cases


def generate_paraphrases(base: GoldenCase, n: int = 1) -> list[GoldenCase]:
    from rfp.llm import generate

    cases = []
    for index in range(n):
        paraphrased = generate(
            PARAPHRASE_SYSTEM,
            base.rfp_text,
            effort="low",
        ).strip()
        if not paraphrased:
            raise ValueError("Paraphrase LLM returned empty text")
        cases.append(
            GoldenCase(
                case_id=f"golden-para-{index + 1:03d}-{base.case_id}",
                rfp_text=paraphrased + "\n",
                source="paraphrase",
                needs_review=True,
                assertions=list(base.assertions),
                metadata={"base_case_id": base.case_id},
            )
        )
    return cases


def write_cases(cases: Iterable[GoldenCase], output_dir: Path) -> list[Path]:
    return [save_case(case, output_dir / f"{case.case_id}.json") for case in cases]


def load_base_case(path: Path | None = None) -> GoldenCase:
    return load_case(path or DEFAULT_BASE_CASE)


def parse_chapters(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(selected) - set(CHAPTER_TITLES)
    if invalid:
        raise ValueError(f"Unknown chapters: {sorted(invalid)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate assertion-based golden cases")
    parser.add_argument(
        "--mode",
        choices=("combinatorial", "mutation", "paraphrase"),
        required=True,
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--industry", action="append")
    parser.add_argument("--chapters", type=parse_chapters)
    parser.add_argument("--out-scope", type=int, choices=range(4))
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()

    if args.mode == "combinatorial":
        cases = generate_combinatorial(
            args.n,
            industries=args.industry,
            chapters=args.chapters,
            out_scope_count=args.out_scope,
        )
    elif args.mode == "mutation":
        cases = generate_mutations(load_base_case(args.base), args.n)
    else:
        cases = generate_paraphrases(load_base_case(args.base), args.n)

    paths = write_cases(cases, args.out)
    print(f"mode={args.mode}")
    print(f"generated={len(paths)}")
    print(f"assertions_min={min(len(case.assertions) for case in cases)}")
    print(f"needs_review={sum(case.needs_review for case in cases)}")
    print(f"output={args.out.as_posix()}")


if __name__ == "__main__":
    main()
