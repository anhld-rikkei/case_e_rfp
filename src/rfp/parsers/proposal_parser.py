import argparse
import hashlib
import re
from pathlib import Path

from ..schema import Sentence


PROPOSAL_ID_RE = re.compile(r"^提案書\s+(PROP-\d+)\s*$", re.MULTILINE)
RESPONDS_TO_RE = re.compile(r"^宛先調達：(RFP-\d{4}-\d{3})\b", re.MULTILINE)
SECTION_RE = re.compile(r"^([1-5])\.\s*(\S.*)$")
SENTENCE_RE = re.compile(r".+?(?:。|[！？]|$)")

SECTIONS = (
    "会社概要",
    "提案の概要",
    "導入実績",
    "認証・コンプライアンス",
    "推進体制",
)

METRIC_RE = re.compile(r"(?:処理時間|在庫精度|年間運用コスト|障害件数)を\d+%")
CERTIFICATION_RE = re.compile(r"ISO(?:/IEC)?|プライバシーマーク|認証")
COMPANY_FACT_RE = re.compile(r"設立\d+年|従業員\d+名")
CAPABILITY_RE = re.compile(r"実績")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROPOSAL_DIR = PROJECT_ROOT / "synthetic" / "proposals"


def _claim_kind(text: str) -> str:
    if METRIC_RE.search(text):
        return "metric"
    if CERTIFICATION_RE.search(text):
        return "certification"
    if COMPANY_FACT_RE.search(text):
        return "company_fact"
    if CAPABILITY_RE.search(text):
        return "capability"
    return "boilerplate"


def _split_sentences(line: str) -> list[str]:
    normalized = line.strip().lstrip("・").strip()
    return [
        match.group(0).strip()
        for match in SENTENCE_RE.finditer(normalized)
        if match.group(0).strip()
    ]


class ProposalParser:
    def parse(self, text: str) -> list[Sentence]:
        proposal_match = PROPOSAL_ID_RE.search(text)
        responds_to_match = RESPONDS_TO_RE.search(text)
        if proposal_match is None or responds_to_match is None:
            raise ValueError("Proposal thiếu proposal_id hoặc responds_to")

        proposal_id = proposal_match.group(1)
        responds_to = responds_to_match.group(1)
        current_section: str | None = None
        parsed: list[Sentence] = []

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            section_match = SECTION_RE.match(stripped)
            if section_match is not None:
                section = section_match.group(2).strip()
                if section not in SECTIONS:
                    raise ValueError(f"Mục proposal không hợp lệ: {section}")
                current_section = section
                continue

            if current_section is None or not stripped:
                continue

            for sentence_text in _split_sentences(stripped):
                sentence_index = len(parsed) + 1
                digest = hashlib.sha1(sentence_text.encode("utf-8")).hexdigest()[:8]
                parsed.append(
                    Sentence(
                        sent_id=f"{proposal_id}-S{sentence_index:02d}-{digest}",
                        proposal_id=proposal_id,
                        responds_to=responds_to,
                        section=current_section,
                        text=sentence_text,
                        claim_kind=_claim_kind(sentence_text),
                        flags={
                            "client_leak": False,
                            "contradicts_capability": False,
                        },
                    )
                )

        return parsed

    def parse_file(self, path: str | Path) -> list[Sentence]:
        return self.parse(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def sections(text: str) -> list[str]:
        return [
            match.group(2).strip()
            for line in text.splitlines()
            if (match := SECTION_RE.match(line.strip())) is not None
        ]


def parse_proposal(text: str) -> list[Sentence]:
    return ProposalParser().parse(text)


def parse_proposal_file(path: str | Path) -> list[Sentence]:
    return ProposalParser().parse_file(path)


def print_stats(proposal_dir: Path = DEFAULT_PROPOSAL_DIR) -> None:
    parser = ProposalParser()
    paths = sorted(proposal_dir.glob("*.txt"))
    sentence_count = 0
    section_counts: list[int] = []

    for path in paths:
        text = path.read_text(encoding="utf-8")
        section_counts.append(len(parser.sections(text)))
        sentence_count += len(parser.parse(text))

    uniform = bool(paths) and all(count == len(SECTIONS) for count in section_counts)
    uniform_text = "đồng nhất" if uniform else "không đồng nhất"
    print(
        f"{len(paths)} file · {len(SECTIONS)} section/file "
        f"{uniform_text} · {sentence_count} câu"
    )


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Parse Japanese proposals")
    argument_parser.add_argument("--stats", action="store_true")
    args = argument_parser.parse_args()
    if args.stats:
        print_stats()


if __name__ == "__main__":
    main()
