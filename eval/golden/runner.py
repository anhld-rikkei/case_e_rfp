from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schema import Assertion, GoldenCase, load_case


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_DIR = ROOT_DIR / "synthetic" / "golden_test_set"


@dataclass(frozen=True)
class AssertionResult:
    kind: str
    target: str
    expected: str
    actual: str
    passed: bool
    reason: str


@dataclass
class CaseRunResult:
    case_id: str
    source: str
    needs_review: bool
    assertions: list[AssertionResult]
    graph_status: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(item.passed for item in self.assertions)

    @property
    def eligible_for_metrics(self) -> bool:
        return not self.needs_review


def _grounded_req_ids(state: dict[str, Any]) -> set[str]:
    return {
        req_id
        for section in state.get("sections", [])
        for sentence in section.get("sentences", [])
        if sentence.get("source_id") is not None
        and sentence.get("verdict") == "VERIFIED"
        for req_id in sentence.get("req_ids", [])
    }


def evaluate_assertion(
    assertion: Assertion,
    state: dict[str, Any],
) -> AssertionResult:
    proposal = state.get("proposal", "")
    if assertion.kind == "must_not_contain":
        passed = assertion.target not in proposal
        expected = f"proposal không chứa {assertion.target}"
        actual = "không chứa" if passed else "đã xuất hiện trong proposal"
    elif assertion.kind == "must_flag_insufficient":
        matching_sections = [
            section["key"]
            for section in state.get("sections", [])
            if section.get("status") == "INSUFFICIENT_EVIDENCE"
            and assertion.target in (section.get("note") or "")
        ]
        passed = bool(matching_sections)
        expected = f"req {assertion.target} là INSUFFICIENT_EVIDENCE"
        actual = ",".join(matching_sections) if matching_sections else "không được flag"
    elif assertion.kind == "must_cover":
        grounded = _grounded_req_ids(state)
        passed = assertion.target in grounded
        expected = f"req {assertion.target} có câu VERIFIED với source_id"
        actual = "đã phủ" if passed else "chưa phủ"
    elif assertion.kind == "must_route":
        target = assertion.target
        reference = state.get("reference_rfp", {})
        if target == "completed":
            passed = state.get("status") == "completed"
            actual = str(state.get("status"))
        elif target.startswith("industry:"):
            expected_industry = target.split(":", 1)[1]
            passed = state.get("industry") == expected_industry
            actual = f"industry:{state.get('industry', '')}"
        elif target.startswith("method:"):
            expected_method = target.split(":", 1)[1]
            passed = reference.get("method") == expected_method
            actual = f"method:{reference.get('method', '')}"
        else:
            passed = reference.get("rfp_id") == target
            actual = str(reference.get("rfp_id", ""))
        expected = target
    elif assertion.kind == "must_ask_user":
        missing = " ".join(state.get("missing", []))
        passed = state.get("status") == "ask_user" and assertion.target in missing
        expected = f"ask_user vì thiếu {assertion.target}"
        actual = f"status={state.get('status')} missing={missing or '—'}"
    else:
        raise AssertionError(f"Unhandled assertion kind: {assertion.kind}")

    return AssertionResult(
        kind=assertion.kind,
        target=assertion.target,
        expected=expected,
        actual=actual,
        passed=passed,
        reason=assertion.reason,
    )


def run_case(case: GoldenCase) -> CaseRunResult:
    from rfp.graph import run_graph

    try:
        state = run_graph(case.rfp_text)
        results = [evaluate_assertion(item, state) for item in case.assertions]
        return CaseRunResult(
            case_id=case.case_id,
            source=case.source,
            needs_review=case.needs_review,
            assertions=results,
            graph_status=str(state.get("status", "")),
        )
    except Exception as error:
        return CaseRunResult(
            case_id=case.case_id,
            source=case.source,
            needs_review=case.needs_review,
            assertions=[
                AssertionResult(
                    kind=item.kind,
                    target=item.target,
                    expected=item.reason,
                    actual=f"graph error: {type(error).__name__}: {error}",
                    passed=False,
                    reason=item.reason,
                )
                for item in case.assertions
            ],
            graph_status="error",
            error=f"{type(error).__name__}: {error}",
        )


def discover_case_files(root: Path = DEFAULT_GOLDEN_DIR) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if not path.name.startswith(".")
    )


def load_cases(root: Path = DEFAULT_GOLDEN_DIR) -> list[GoldenCase]:
    return [load_case(path) for path in discover_case_files(root)]


def run_cases(
    cases: Iterable[GoldenCase],
    *,
    workers: int = 4,
) -> list[CaseRunResult]:
    case_list = list(cases)
    if workers <= 1 or len(case_list) <= 1:
        return [run_case(case) for case in case_list]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_case, case_list))


def assertion_rows(result: CaseRunResult) -> list[dict[str, Any]]:
    return [
        {
            "assertion": f"{item.kind} · {item.target}",
            "kỳ vọng": item.expected,
            "thực tế": item.actual,
            "PASS/FAIL": "PASS" if item.passed else "FAIL",
            "reason": item.reason if not item.passed else "",
        }
        for item in result.assertions
    ]


def case_rows(results: Iterable[CaseRunResult]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": result.case_id,
            "source": result.source,
            "needs_review": result.needs_review,
            "assertions": len(result.assertions),
            "pass": sum(item.passed for item in result.assertions),
            "fail": sum(not item.passed for item in result.assertions),
            "result": (
                "EXCLUDED_REVIEW"
                if result.needs_review
                else "PASS"
                if result.passed
                else "FAIL"
            ),
        }
        for result in results
    ]


def source_summary(results: Iterable[CaseRunResult]) -> list[dict[str, Any]]:
    buckets: dict[str, list[CaseRunResult]] = defaultdict(list)
    for result in results:
        if result.eligible_for_metrics:
            buckets[result.source].append(result)
    rows = []
    for source, source_results in sorted(buckets.items()):
        passed = sum(result.passed for result in source_results)
        rows.append(
            {
                "source": source,
                "eligible": len(source_results),
                "passed": passed,
                "failed": len(source_results) - passed,
                "pass_rate": passed / len(source_results) if source_results else 0.0,
            }
        )
    return rows


def acceptance_summary(results: Iterable[CaseRunResult]) -> dict[str, Any]:
    eligible = [result for result in results if result.eligible_for_metrics]
    out_scope = [
        result
        for result in eligible
        if any(item.kind == "must_flag_insufficient" for item in result.assertions)
    ]
    out_scope_safe = sum(
        all(
            item.passed
            for item in result.assertions
            if item.kind in {"must_flag_insufficient", "must_not_contain"}
        )
        for result in out_scope
    )
    all_in_scope = [
        result
        for result in eligible
        if result.source == "combinatorial"
        and not any(
            item.kind == "must_flag_insufficient"
            for item in result.assertions
        )
    ]
    cover_assertions = [
        item
        for result in all_in_scope
        for item in result.assertions
        if item.kind == "must_cover"
    ]
    covered = sum(item.passed for item in cover_assertions)
    return {
        "out_scope_cases": len(out_scope),
        "out_scope_safe": out_scope_safe,
        "all_in_scope_cases": len(all_in_scope),
        "all_in_scope_covered": covered,
        "all_in_scope_cover_total": len(cover_assertions),
        "all_in_scope_abstain_rate": (
            1.0 - covered / len(cover_assertions)
            if cover_assertions
            else 0.0
        ),
    }


def print_results(results: list[CaseRunResult]) -> None:
    print("case_id | source | review | assertions | pass | fail | result")
    print("-" * 96)
    for row in case_rows(results):
        print(
            f"{row['case_id']} | {row['source']} | {row['needs_review']} | "
            f"{row['assertions']} | {row['pass']} | {row['fail']} | {row['result']}"
        )
    print()
    print("source | eligible | passed | failed | pass_rate")
    print("-" * 64)
    for row in source_summary(results):
        print(
            f"{row['source']} | {row['eligible']} | {row['passed']} | "
            f"{row['failed']} | {row['pass_rate']:.3f}"
        )
    excluded = sum(result.needs_review for result in results)
    eligible = sum(result.eligible_for_metrics for result in results)
    failures = [
        result.case_id
        for result in results
        if result.eligible_for_metrics and not result.passed
    ]
    print()
    print(f"eligible_for_metrics={eligible}")
    print(f"excluded_needs_review={excluded}")
    print(f"failed_cases={','.join(failures) if failures else 'none'}")
    acceptance = acceptance_summary(results)
    print(
        "out_scope_insufficient_and_no_fabrication="
        f"{acceptance['out_scope_safe']}/{acceptance['out_scope_cases']}"
    )
    print(
        "all_in_scope_coverage="
        f"{acceptance['all_in_scope_covered']}/"
        f"{acceptance['all_in_scope_cover_total']}"
    )
    print(
        "all_in_scope_abstain_rate="
        f"{acceptance['all_in_scope_abstain_rate']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run assertion-based golden cases")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.all and args.case is None:
        parser.error("choose --all or --case PATH")
    cases = load_cases(args.path) if args.all else [load_case(args.case)]
    results = run_cases(cases, workers=args.workers)
    print_results(results)


if __name__ == "__main__":
    main()
