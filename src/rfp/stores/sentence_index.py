import argparse
from pathlib import Path

from ..parsers.proposal_parser import ProposalParser
from ..parsers.rfp_parser import RFPParser
from ..retrieve.hybrid import HybridRetriever, SearchResult
from ..sanitize.blocklist import CapabilityBlocklist
from ..sanitize.leak import filter_client_leaks, has_client_leak
from ..schema import Sentence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROPOSAL_DIR = PROJECT_ROOT / "synthetic" / "proposals"
DEFAULT_RFP_DIR = PROJECT_ROOT / "synthetic" / "rfps"


class SentenceIndex:
    def __init__(
        self,
        sentences: list[Sentence],
        industries: dict[str, str],
        *,
        leak_quarantine: list[Sentence] | None = None,
        capability_quarantine: list[Sentence] | None = None,
    ) -> None:
        self.sentences = list(sentences)
        self.industries = dict(industries)
        self.leak_quarantine = list(leak_quarantine or [])
        self.capability_quarantine = list(capability_quarantine or [])
        self._retriever: HybridRetriever | None = None

    @classmethod
    def build(
        cls,
        proposal_dir: str | Path = DEFAULT_PROPOSAL_DIR,
        rfp_dir: str | Path = DEFAULT_RFP_DIR,
    ) -> "SentenceIndex":
        proposal_parser = ProposalParser()
        parsed_sentences = [
            sentence
            for path in sorted(Path(proposal_dir).glob("*.txt"))
            for sentence in proposal_parser.parse_file(path)
        ]

        leak_clean, leak_quarantine = filter_client_leaks(parsed_sentences)
        blocklist = CapabilityBlocklist()
        clean, capability_quarantine = blocklist.partition(leak_clean)

        rfp_parser = RFPParser()
        industries = {
            rfp.rfp_id: rfp.industry
            for path in sorted(Path(rfp_dir).glob("*.txt"))
            for rfp in [rfp_parser.parse_file(path)]
        }
        missing_industries = sorted(
            {
                sentence.responds_to
                for sentence in clean
                if sentence.responds_to not in industries
            }
        )
        if missing_industries:
            raise ValueError(f"Không tìm thấy industry cho: {missing_industries}")

        return cls(
            clean,
            industries,
            leak_quarantine=leak_quarantine,
            capability_quarantine=capability_quarantine,
        )

    def all_sentences(self) -> list[Sentence]:
        return list(self.sentences)

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            from config.settings import RETRIEVAL_USE_BM25

            # Cờ ablation V0: dense-only. BM25 vẫn được dựng (điểm ghi vào trace)
            # nhưng trọng số 0 nên không tham gia xếp hạng.
            bm25_weight = 0.5 if RETRIEVAL_USE_BM25 else 0.0
            self._retriever = HybridRetriever(
                self.sentences,
                self.industries,
                bm25_weight=bm25_weight,
                dense_weight=1.0 - bm25_weight,
            )
        return self._retriever

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        section: str | None = None,
        industry: str | None = None,
        claim_kind: str | None = None,
    ) -> list[SearchResult]:
        return self._get_retriever().search(
            query,
            k=k,
            section=section,
            industry=industry,
            claim_kind=claim_kind,
        )

    def cleanliness(self) -> dict[str, int]:
        blocklist = CapabilityBlocklist()
        return {
            "indexed": len(self.sentences),
            "blocklist_hits": sum(
                blocklist.contradicts(sentence.text) for sentence in self.sentences
            ),
            "client_leaks": sum(
                has_client_leak(sentence.text) for sentence in self.sentences
            ),
            "leak_quarantine": len(self.leak_quarantine),
            "capability_quarantine": len(self.capability_quarantine),
        }

    def assert_clean(self) -> dict[str, int]:
        counts = self.cleanliness()
        if counts["blocklist_hits"] != 0 or counts["client_leaks"] != 0:
            raise AssertionError(f"Sentence index không sạch: {counts}")
        return counts


def _print_results(results: list[SearchResult]) -> None:
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result.score:.4f} "
            f"bm25={result.bm25_score:.4f} dense={result.dense_score:.4f} "
            f"[{result.sentence.sent_id}] {result.sentence.text}"
        )


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Clean hybrid sentence index")
    mode = argument_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query")
    mode.add_argument("--assert-clean", action="store_true")
    argument_parser.add_argument("--k", type=int, default=5)
    argument_parser.add_argument("--section")
    argument_parser.add_argument("--industry")
    argument_parser.add_argument("--claim-kind")
    args = argument_parser.parse_args()

    sentence_index = SentenceIndex.build()
    if args.assert_clean:
        counts = sentence_index.assert_clean()
        print(
            f"indexed={counts['indexed']} · blocklist=0 · client_leak=0 · "
            f"quarantine={counts['capability_quarantine']} · "
            f"leak_dropped={counts['leak_quarantine']}"
        )
        return

    results = sentence_index.search(
        args.query,
        k=args.k,
        section=args.section,
        industry=args.industry,
        claim_kind=args.claim_kind,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
