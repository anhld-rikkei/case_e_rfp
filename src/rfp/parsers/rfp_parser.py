import argparse
import re
from pathlib import Path

from ..schema import Chapter, Requirement, RFP


CHAPTER_RE = re.compile(r"^第(\d+)章\s*(\S+)", re.MULTILINE)
REQUIREMENT_RE = re.compile(r"^(\d+\.\d+)\s*(.+)", re.MULTILINE)
INDUSTRY_RE = re.compile(r"^発注業種：(.+)$", re.MULTILINE)
RFP_ID_RE = re.compile(r"^調達番号：(.+)$", re.MULTILINE)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RFP_DIR = PROJECT_ROOT / "synthetic" / "rfps"


class RFPParser:
    def parse(self, text: str, *, allow_llm_fallback: bool = True) -> RFP:
        parsed = self._parse_regex(text)
        if parsed is not None:
            return parsed
        if not allow_llm_fallback:
            raise ValueError("RFP text không khớp format chương/requirement")
        return self._parse_llm(text)

    def parse_file(
        self,
        path: str | Path,
        *,
        allow_llm_fallback: bool = True,
    ) -> RFP:
        text = Path(path).read_text(encoding="utf-8")
        return self.parse(text, allow_llm_fallback=allow_llm_fallback)

    @staticmethod
    def _parse_regex(text: str) -> RFP | None:
        chapter_matches = list(CHAPTER_RE.finditer(text))
        requirement_matches = list(REQUIREMENT_RE.finditer(text))
        if not chapter_matches or not requirement_matches:
            return None

        industry_match = INDUSTRY_RE.search(text)
        rfp_id_match = RFP_ID_RE.search(text)
        if industry_match is None or rfp_id_match is None:
            return None

        first_nonblank = next(
            (line.strip() for line in text.splitlines() if line.strip()),
            "",
        )
        chapters: list[Chapter] = []

        for index, chapter_match in enumerate(chapter_matches):
            segment_end = (
                chapter_matches[index + 1].start()
                if index + 1 < len(chapter_matches)
                else len(text)
            )
            segment = text[chapter_match.end() : segment_end]
            requirements = [
                Requirement(
                    req_id=requirement_match.group(1).strip(),
                    text=requirement_match.group(2).strip(),
                )
                for requirement_match in REQUIREMENT_RE.finditer(segment)
            ]
            chapters.append(
                Chapter(
                    id=chapter_match.group(1),
                    title=chapter_match.group(2).strip(),
                    requirements=requirements,
                )
            )

        if sum(len(chapter.requirements) for chapter in chapters) == 0:
            return None

        return RFP(
            rfp_id=rfp_id_match.group(1).strip(),
            title=first_nonblank,
            industry=industry_match.group(1).strip(),
            chapters=chapters,
        )

    @staticmethod
    def _parse_llm(text: str) -> RFP:
        # Import lười để đường regex không nạp SDK, không khởi tạo client và không gọi LLM.
        from pydantic import BaseModel

        from ..llm import structured

        class FallbackRequirement(BaseModel):
            req_id: str
            text: str

        class FallbackChapter(BaseModel):
            id: str
            title: str
            requirements: list[FallbackRequirement]

        class FallbackRFP(BaseModel):
            rfp_id: str
            title: str
            industry: str
            chapters: list[FallbackChapter]

        result = structured(
            system=(
                "Bạn là parser RFP tiếng Nhật. Hãy giữ nguyên văn nội dung và tách "
                "rfp_id, title, industry, chapters, requirements theo schema."
            ),
            user=text,
            model_cls=FallbackRFP,
        )
        return RFP(
            rfp_id=result.rfp_id,
            title=result.title,
            industry=result.industry,
            chapters=[
                Chapter(
                    id=chapter.id,
                    title=chapter.title,
                    requirements=[
                        Requirement(req_id=req.req_id, text=req.text)
                        for req in chapter.requirements
                    ],
                )
                for chapter in result.chapters
            ],
        )


def parse_rfp(text: str, *, allow_llm_fallback: bool = True) -> RFP:
    return RFPParser().parse(text, allow_llm_fallback=allow_llm_fallback)


def parse_rfp_file(
    path: str | Path,
    *,
    allow_llm_fallback: bool = True,
) -> RFP:
    return RFPParser().parse_file(path, allow_llm_fallback=allow_llm_fallback)


def print_stats(rfp_dir: Path = DEFAULT_RFP_DIR) -> None:
    parser = RFPParser()
    total_atoms = 0
    for path in sorted(rfp_dir.glob("*.txt")):
        rfp = parser.parse_file(path)
        atom_count = sum(len(chapter.requirements) for chapter in rfp.chapters)
        total_atoms += atom_count
        print(f"{rfp.rfp_id}: {len(rfp.chapters)} chương / {atom_count} atom")
    print(f"Tổng: {total_atoms} atom")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Parse Japanese RFP text")
    argument_parser.add_argument("--stats", action="store_true")
    args = argument_parser.parse_args()
    if args.stats:
        print_stats()


if __name__ == "__main__":
    main()
