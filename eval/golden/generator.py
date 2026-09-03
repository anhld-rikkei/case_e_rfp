from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

from .coverage import (
    CHAPTER_TITLES as COVERAGE_CHAPTERS,
    EDGE_NO_CHAPTER_NUMBER,
    EDGE_NO_INDUSTRY,
    EDGE_NO_SECURITY,
    TIER_COMMON,
    TIER_EDGE,
    TIER_FORBIDDEN,
    coverage_report,
)
from .schema import Assertion, GoldenCase, load_case, save_case


ROOT_DIR = Path(__file__).resolve().parents[2]
CAPABILITY_PATH = ROOT_DIR / "synthetic" / "capability_sheet.json"
GOLDEN_DIR = ROOT_DIR / "synthetic" / "golden_test_set"
DEFAULT_OUTPUT_DIR = GOLDEN_DIR / "generated"
DEFAULT_BASE_CASE = GOLDEN_DIR / "original-rfp-2025-001.json"

CHAPTER_HEADER_RE = re.compile(r"^第(?P<number>\d+)章\s+(?P<title>.+?)\s*$")
REQUIREMENT_ID_RE = re.compile(r"^(?P<chapter>\d+)\.(?P<number>\d+)\s+")
INDUSTRY_RE = re.compile(r"^発注業種[：:]\s*.*$")

# Từ vựng chương định nghĩa ở `coverage.py` và tái xuất ở đây: hai bản sao rồi
# lệch nhau thì bảng độ phủ báo đủ trong khi bộ sinh bỏ sót một chương.
CHAPTER_TITLES = COVERAGE_CHAPTERS
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
    # Chương chỉ định đích danh thắng suy luận theo loại. Cần cho trục phủ
    # "6 chương": nếu để mặc định thì certification luôn rơi vào セキュリティ要件
    # và capability luôn vào 技術要件, bốn chương còn lại không bao giờ có ca.
    explicit = requirement.get("chapter")
    if explicit:
        return explicit
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


MUTATION_NAMES = (
    "capabilities_NOT_offered",
    "certifications_NOT_held",
    "industry_outside_served",
    "industry_inside_served",
    "missing_industry",
    "parser_fallback",
)


def build_mutation(
    base: GoldenCase,
    name: str,
    case_id: str,
    *,
    source: str = "mutation",
    metadata: dict[str, Any] | None = None,
) -> GoldenCase:
    """Một ca mutation từ ca nền. Dùng chung cho sinh theo lô và sinh theo trục phủ."""
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
    return GoldenCase(
        case_id=case_id,
        rfp_text=mutated_text,
        source=source,
        needs_review=False,
        assertions=[*inherited, *_mutation_assertions(name, requirement_id)],
        metadata={
            "base_case_id": base.case_id,
            "mutation": name,
            **(metadata or {}),
        },
    )


def generate_mutations(base: GoldenCase, n: int = 6) -> list[GoldenCase]:
    return [
        build_mutation(
            base,
            MUTATION_NAMES[index % len(MUTATION_NAMES)],
            f"golden-mut-{index + 1:03d}-"
            f"{MUTATION_NAMES[index % len(MUTATION_NAMES)].lower()}",
        )
        for index in range(n)
    ]


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


# ─────────────────────────────────────────────────────────────────────────────
# Sinh THEO TIÊU CHÍ PHỦ
#
# Khác hẳn `generate_combinatorial`: chỗ đó bảo "sinh cho tôi 30 ca" rồi hy vọng
# 30 ca đó chạm hết mọi thứ cần chạm. Ở đây người test chọn TRỤC cần phủ, hàm tự
# tính còn thiếu gì so với bộ đang có và chỉ sinh đúng phần thiếu.
# ─────────────────────────────────────────────────────────────────────────────

COVERAGE_AXES = (
    "out_scope",
    "blocklist",
    "in_scope",
    "chapters",
    "client_leak",
    "industries",
    "edge_cases",
    "assertion_kinds",
)


def _requirement_of(term: str, *, chapter: str | None = None) -> dict[str, str]:
    data = capability_data()
    if term in data["certifications_held"]:
        category = "certification"
    elif term in data["capabilities"]:
        category = "capability"
    elif term in data["certifications_NOT_held"]:
        category = "certification_not_held"
    elif term in data["capabilities_NOT_offered"]:
        category = "capability_not_offered"
    else:
        raise ValueError(f"Từ vựng không có trong capability sheet: {term}")
    requirement = {"term": term, "category": category}
    if chapter:
        requirement["chapter"] = chapter
    return requirement


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _tag(case: GoldenCase, axis: str, tier: str) -> GoldenCase:
    case.source = "coverage"
    case.metadata = {**case.metadata, "axis": axis, "tier": tier}
    return case


def _fill_axis(
    axis: str,
    missing: tuple[str, ...],
    next_id,
    base: GoldenCase,
) -> list[GoldenCase]:
    data = capability_data()
    any_capability = data["capabilities"][0]

    if axis in {"out_scope", "blocklist"}:
        # Mỗi ca: 1 yêu cầu trong năng lực + tối đa 2 yêu cầu ngoài năng lực.
        # Kèm yêu cầu làm được để ca không chỉ kiểm nhánh từ chối.
        cases = []
        for group in _chunk(list(missing), 2):
            case = _render_case(
                case_id=next_id(),
                industry=data["industries_served"][0],
                requirements=[
                    _requirement_of(any_capability),
                    *(_requirement_of(term) for term in group),
                ],
                chapter_titles=CHAPTER_TITLES,
            )
            cases.append(_tag(case, axis, TIER_FORBIDDEN))
        return cases

    if axis == "in_scope":
        return [
            _tag(
                _render_case(
                    case_id=next_id(),
                    industry=data["industries_served"][0],
                    requirements=[_requirement_of(term) for term in group],
                    chapter_titles=CHAPTER_TITLES,
                ),
                axis,
                TIER_COMMON,
            )
            for group in _chunk(list(missing), 3)
        ]

    if axis == "chapters":
        # Một yêu cầu làm được đặt vào ĐÚNG chương còn thiếu.
        return [
            _tag(
                _render_case(
                    case_id=next_id(),
                    industry=data["industries_served"][0],
                    requirements=[
                        _requirement_of(any_capability, chapter=chapter)
                        for chapter in group
                    ],
                    chapter_titles=CHAPTER_TITLES,
                ),
                axis,
                TIER_COMMON,
            )
            for group in _chunk(list(missing), 3)
        ]

    if axis == "client_leak":
        # Tên khách hàng cũ không nằm trong RFP — nó nằm trong kho hồ sơ cũ mà
        # retrieval sẽ lôi ra. Nên đây là điều kiện canh ĐẦU RA, gắn vào một ca
        # yêu cầu bình thường.
        cases = []
        for index, group in enumerate(_chunk(list(missing), 4)):
            case = _render_case(
                case_id=next_id(),
                industry=data["industries_served"][
                    index % len(data["industries_served"])
                ],
                requirements=[_requirement_of(any_capability)],
                chapter_titles=CHAPTER_TITLES,
            )
            case.assertions = [
                *case.assertions,
                *(
                    Assertion(
                        "must_not_contain",
                        name,
                        f"{name} là tên khách hàng trong hồ sơ cũ — "
                        "không được lộ sang hồ sơ của khách khác.",
                    )
                    for name in group
                ),
            ]
            cases.append(_tag(case, axis, TIER_FORBIDDEN))
        return cases

    if axis == "industries":
        return [
            _tag(
                _render_case(
                    case_id=next_id(),
                    industry=industry,
                    requirements=[_requirement_of(any_capability)],
                    chapter_titles=CHAPTER_TITLES,
                ),
                axis,
                TIER_COMMON,
            )
            for industry in missing
        ]

    if axis == "edge_cases":
        cases = []
        for item in missing:
            if item == EDGE_NO_SECURITY:
                case = _render_case(
                    case_id=next_id(),
                    industry=data["industries_served"][0],
                    requirements=[_requirement_of(any_capability)],
                    chapter_titles=tuple(
                        chapter
                        for chapter in CHAPTER_TITLES
                        if chapter != "セキュリティ要件"
                    ),
                )
                cases.append(_tag(case, axis, TIER_EDGE))
            elif item == EDGE_NO_INDUSTRY:
                cases.append(
                    _tag(
                        build_mutation(base, "missing_industry", next_id().lower()),
                        axis,
                        TIER_EDGE,
                    )
                )
            elif item == EDGE_NO_CHAPTER_NUMBER:
                cases.append(
                    _tag(
                        build_mutation(base, "parser_fallback", next_id().lower()),
                        axis,
                        TIER_EDGE,
                    )
                )
        return cases

    if axis == "assertion_kinds":
        wanted = {
            "must_ask_user": "missing_industry",
            "must_route": "industry_inside_served",
        }
        cases = [
            _tag(
                build_mutation(base, wanted[kind], next_id().lower()),
                axis,
                TIER_EDGE,
            )
            for kind in missing
            if kind in wanted
        ]
        rest = [kind for kind in missing if kind not in wanted]
        if rest:
            # must_cover / must_flag_insufficient / must_not_contain đều ra từ
            # một ca có cả yêu cầu làm được lẫn yêu cầu ngoài năng lực.
            case = _render_case(
                case_id=next_id(),
                industry=data["industries_served"][0],
                requirements=[
                    _requirement_of(any_capability),
                    _requirement_of(data["capabilities_NOT_offered"][0]),
                ],
                chapter_titles=CHAPTER_TITLES,
            )
            cases.append(_tag(case, axis, TIER_EDGE))
        return cases

    raise ValueError(f"Trục phủ không được hỗ trợ: {axis}")


def generate_for_coverage(
    axes: Iterable[str] | None = None,
    *,
    existing: Iterable[GoldenCase] = (),
    base: GoldenCase | None = None,
) -> list[GoldenCase]:
    """Sinh đúng phần còn thiếu để các trục được chọn phủ đủ.

    Tất định, 0 lệnh gọi LLM. Bộ đang có phủ rồi thì trả về danh sách rỗng —
    bấm hai lần không đẻ ra ca trùng.
    """
    selected = tuple(axes) if axes is not None else COVERAGE_AXES
    unknown = [axis for axis in selected if axis not in COVERAGE_AXES]
    if unknown:
        raise ValueError(f"Trục phủ không có: {unknown}")

    base_case = base or load_base_case()
    pool = list(existing)
    generated: list[GoldenCase] = []
    counter = iter(range(1, 1000))

    def next_id() -> str:
        return f"GOLDEN-COV-{next(counter):03d}"

    # Theo thứ tự COVERAGE_AXES: trục phủ rộng đi trước để trục sau khỏi sinh
    # thừa (ví dụ ca out_scope đã kèm sẵn must_not_contain cho blocklist).
    for axis_key in COVERAGE_AXES:
        if axis_key not in selected:
            continue
        axis = next(
            item
            for item in coverage_report([*pool, *generated])
            if item.key == axis_key
        )
        if axis.is_full:
            continue
        fresh = _fill_axis(axis_key, axis.missing, next_id, base_case)
        generated.extend(fresh)
    return generated


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
        choices=("combinatorial", "mutation", "paraphrase", "coverage"),
        required=True,
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--industry", action="append")
    parser.add_argument("--chapters", type=parse_chapters)
    parser.add_argument("--out-scope", type=int, choices=range(4))
    parser.add_argument("--base", type=Path)
    parser.add_argument(
        "--axes",
        type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()),
        default=None,
        help="Danh sách trục phủ, phân tách bằng dấu phẩy. Bỏ trống = tất cả.",
    )
    args = parser.parse_args()

    if args.mode == "coverage":
        from .runner import load_cases

        cases = generate_for_coverage(args.axes, existing=load_cases())
        if not cases:
            print("mode=coverage")
            print("generated=0")
            print("Bộ golden hiện tại đã phủ đủ các trục được chọn.")
            return
    elif args.mode == "combinatorial":
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
