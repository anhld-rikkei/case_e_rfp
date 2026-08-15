from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
RFP_DIR = ROOT_DIR / "synthetic" / "rfps"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "synthetic" / "golden_test_set"
BASE_RFP = RFP_DIR / "RFP-2025-001.txt"
REQUIREMENT_RE = re.compile(r"^\d+\.\d+\s+", re.MULTILINE)
CHAPTER_NUMBER_RE = re.compile(r"^第\d+章\s*", re.MULTILINE)

FORBIDDEN = (
    "ISO/IEC 27017",
    "ISO/IEC 27018",
    "CMMI",
    "量子暗号通信",
    "ブロックチェーン決済基盤",
    "AI医療画像診断",
)


def assertion(kind: str, target: str, reason: str) -> dict[str, str]:
    return {"kind": kind, "target": target, "reason": reason}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def add_blockchain_requirement(text: str) -> str:
    return replace_once(
        text,
        "3.2 可用性99.9%以上を確保すること。",
        "3.2 可用性99.9%以上を確保すること。\n"
        "3.3 ブロックチェーン決済基盤を構築すること。",
    )


def add_iso27017_requirement(text: str) -> str:
    return replace_once(
        text,
        "4.2 通信は暗号化すること。",
        "4.2 通信は暗号化すること。\n"
        "4.3 ISO/IEC 27017認証を有すること。",
    )


def change_to_unserved_industry(text: str) -> str:
    return replace_once(text, "発注業種：製造業", "発注業種：医療")


def change_to_served_industry(text: str) -> str:
    return replace_once(text, "発注業種：製造業", "発注業種：公共")


def remove_industry(text: str) -> str:
    return replace_once(text, "発注業種：製造業\n", "")


def remove_chapter_numbering(text: str) -> str:
    mutated, replacements = CHAPTER_NUMBER_RE.subn("", text)
    if replacements == 0:
        raise ValueError("No 第N章 headings found")
    return mutated


MUTATIONS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "mutation-capability-not-offered",
        "name": "capabilities_NOT_offered",
        "apply": add_blockchain_requirement,
        "expected_behavior": (
            "Requirement 3.3 phải là INSUFFICIENT_EVIDENCE; proposal tuyệt đối "
            "không được tuyên bố có thể xây dựng blockchain payment platform."
        ),
        "assertions": [
            assertion(
                "must_flag_insufficient",
                "3.3",
                "ブロックチェーン決済基盤 nằm trong capabilities_NOT_offered.",
            ),
            assertion(
                "must_not_contain",
                "ブロックチェーン決済基盤",
                "Final proposal không được lặp lại năng lực công ty không cung cấp.",
            ),
        ],
    },
    {
        "case_id": "mutation-certification-not-held",
        "name": "certifications_NOT_held",
        "apply": add_iso27017_requirement,
        "expected_behavior": (
            "Requirement 4.3 phải là INSUFFICIENT_EVIDENCE; không tuyên bố có "
            "ISO/IEC 27017, chỉ được dùng chứng nhận ISO/IEC 27001 đang nắm giữ."
        ),
        "assertions": [
            assertion(
                "must_flag_insufficient",
                "4.3",
                "ISO/IEC 27017 nằm trong certifications_NOT_held.",
            ),
            assertion(
                "must_not_contain",
                "ISO/IEC 27017",
                "Final proposal không được tuyên bố chứng nhận không nắm giữ.",
            ),
            assertion(
                "must_contain",
                "ISO/IEC 27001",
                "Có thể nêu chứng nhận 27001 thực sự đang nắm giữ để làm rõ giới hạn.",
            ),
        ],
    },
    {
        "case_id": "mutation-industry-unserved-medical",
        "name": "industry_outside_served",
        "apply": change_to_unserved_industry,
        "expected_behavior": (
            "Không được bịa precedent hay kinh nghiệm chuyên ngành y tế; chỉ dùng "
            "bằng chứng thực có và báo thiếu khi không đủ."
        ),
        "assertions": [
            assertion(
                "must_not_claim_industry_experience",
                "医療",
                "医療 không thuộc industries_served.",
            ),
            assertion(
                "must_not_contain",
                "AI医療画像診断",
                "Năng lực y tế này nằm trong capabilities_NOT_offered.",
            ),
        ],
    },
    {
        "case_id": "mutation-industry-served-public",
        "name": "industry_inside_served",
        "apply": change_to_served_industry,
        "expected_behavior": (
            "Reranking precedent phải dùng ngành 公共 mới thay cho 製造業 và vẫn giữ "
            "mọi claim có nguồn."
        ),
        "assertions": [
            assertion(
                "must_route",
                "industry:公共",
                "公共 thuộc industries_served và phải chi phối industry reranking.",
            )
        ],
    },
    {
        "case_id": "mutation-missing-industry",
        "name": "missing_industry",
        "apply": remove_industry,
        "expected_behavior": (
            "Graph phải rẽ ask_user vì thiếu 発注業種; không được tiếp tục sinh proposal."
        ),
        "assertions": [
            assertion(
                "must_ask_user",
                "industry",
                "Thiếu dòng 発注業種 là đầu vào chưa hoàn chỉnh.",
            )
        ],
    },
    {
        "case_id": "mutation-free-text-chapters",
        "name": "parser_fallback",
        "apply": remove_chapter_numbering,
        "expected_behavior": (
            "Regex không còn nhận 第N章; RFPParser phải dùng structured LLM fallback "
            "và vẫn tách được các chương cùng requirement."
        ),
        "assertions": [
            assertion(
                "must_parse_with_fallback",
                "5 chapters / 11 atoms",
                "Text tự do vẫn phải được parser fallback tách đủ cấu trúc.",
            )
        ],
    },
)


def original_case(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rfp_id = path.stem
    return {
        "case_id": f"original-{rfp_id.lower()}",
        "rfp_text": text,
        "source": "original",
        "base_rfp_id": rfp_id,
        "needs_review": False,
        "expected_behavior": (
            "Sinh proposal theo State contract; mọi câu có provenance và không chứa "
            "bất kỳ capability/certification bị cấm nào."
        ),
        "assertions": [
            *[
                assertion(
                    "must_not_contain",
                    term,
                    "Original RFP vẫn phải qua deterministic final guard.",
                )
                for term in FORBIDDEN
            ],
            assertion(
                "must_complete",
                rfp_id,
                "RFP gốc đầy đủ phải chạy tới trạng thái completed.",
            ),
        ],
    }


def mutation_case(spec: dict[str, Any], base_text: str) -> dict[str, Any]:
    apply_mutation: Callable[[str], str] = spec["apply"]
    return {
        "case_id": spec["case_id"],
        "rfp_text": apply_mutation(base_text),
        "source": "mutation",
        "base_rfp_id": "RFP-2025-001",
        "mutation": spec["name"],
        "needs_review": False,
        "expected_behavior": spec["expected_behavior"],
        "assertions": spec["assertions"],
    }


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [original_case(path) for path in sorted(RFP_DIR.glob("*.txt"))]
    base_text = BASE_RFP.read_text(encoding="utf-8")
    cases.extend(mutation_case(spec, base_text) for spec in MUTATIONS)

    written: list[Path] = []
    for case in cases:
        path = output_dir / f"{case['case_id']}.json"
        path.write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def count_atoms(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        total += len(REQUIREMENT_RE.findall(case["rfp_text"]))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 9 golden mutations")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = generate(args.out)
    original_count = sum("original-" in path.stem for path in paths)
    mutation_count = len(paths) - original_count
    print(f"generated={len(paths)} original={original_count} mutation={mutation_count}")
    print(f"requirement_atoms={count_atoms(paths)}")


if __name__ == "__main__":
    main()
