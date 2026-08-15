import json
from pathlib import Path
from typing import Any

from config.templates_ja import COMMON_CLIENT_NAMES

from ..parsers.proposal_parser import ProposalParser
from .blocklist import CapabilityBlocklist
from .leak import extract_client_names, has_client_leak, is_private_client_name


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROPOSAL_DIR = PROJECT_ROOT / "synthetic" / "proposals"
LABEL_PATH = PROJECT_ROOT / "synthetic" / "proposals_index.jsonl"


def _load_labels(path: Path = LABEL_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as label_file:
        return [json.loads(line) for line in label_file if line.strip()]


def _compact_ids(proposal_ids: set[str]) -> str:
    ordered = sorted(proposal_ids)
    if not ordered:
        return "—"
    return ordered[0] + "".join(f",{proposal_id.removeprefix('PROP-')}" for proposal_id in ordered[1:])


def _assert_equal(label: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AssertionError(
            f"{label} lệch nhãn: missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    labels = _load_labels()
    parser = ProposalParser()
    blocklist = CapabilityBlocklist()

    detected_leak_files: set[str] = set()
    contradiction_files: set[str] = set()
    observed_common_names: set[str] = set()
    false_positive_names: set[str] = set()

    for path in sorted(PROPOSAL_DIR.glob("*.txt")):
        sentences = parser.parse_file(path)
        proposal_id = sentences[0].proposal_id
        if any(has_client_leak(sentence.text) for sentence in sentences):
            detected_leak_files.add(proposal_id)
        if any(blocklist.contradicts(sentence.text) for sentence in sentences):
            contradiction_files.add(proposal_id)

        for sentence in sentences:
            for name in extract_client_names(sentence.text):
                if name in COMMON_CLIENT_NAMES:
                    observed_common_names.add(name)
                    if is_private_client_name(name):
                        false_positive_names.add(name)

    expected_leak_files = {
        label["proposal_id"] for label in labels if label["contains_client_leak"]
    }
    expected_contradiction_files = {
        label["proposal_id"] for label in labels if label["fabricated_claim"] is not None
    }

    _assert_equal("leak", detected_leak_files, expected_leak_files)
    _assert_equal("contradiction", contradiction_files, expected_contradiction_files)
    if observed_common_names != set(COMMON_CLIENT_NAMES):
        raise AssertionError(
            "Corpus không chứa đủ whitelist tên chung: "
            f"{sorted(observed_common_names)}"
        )
    if false_positive_names:
        raise AssertionError(f"Leak false positive: {sorted(false_positive_names)}")

    print(
        f"leak detected : {len(detected_leak_files)} files  → "
        f"{_compact_ids(detected_leak_files)}     "
        f"(khớp {len(detected_leak_files)}/{len(expected_leak_files)} nhãn)"
    )
    print(
        f"leak FP       : {len(false_positive_names)}        → "
        f"{len(observed_common_names)} tên chung không bị bắt"
    )
    print(
        f"contradiction : {len(contradiction_files)} files  → "
        f"{_compact_ids(contradiction_files)}                  "
        f"(khớp {len(contradiction_files)}/{len(expected_contradiction_files)} nhãn)"
    )


if __name__ == "__main__":
    main()
