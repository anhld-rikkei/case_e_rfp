import logging
from collections.abc import Iterable
from typing import Any


logger = logging.getLogger(__name__)


SECTION_DEFINITIONS = (
    ("company_overview", "会社概要", "Tổng quan công ty"),
    ("proposal_overview", "提案の概要", "Tổng quan đề xuất"),
    ("implementation_experience", "導入実績", "Kinh nghiệm triển khai"),
    (
        "certification_compliance",
        "認証・コンプライアンス",
        "Chứng nhận và tuân thủ",
    ),
    ("delivery_structure", "推進体制", "Cơ cấu triển khai"),
)

SECTION_CHAPTER_TITLES = {
    "company_overview": (),
    "proposal_overview": ("調達概要",),
    "implementation_experience": ("業務要件", "技術要件"),
    "certification_compliance": ("セキュリティ要件",),
    "delivery_structure": ("納期・体制", "提案書記載事項"),
}

DEFAULT_SECTION_KEY = "implementation_experience"
SECTION_BY_KEY = {
    key: {"key": key, "title_ja": title_ja, "title_vi": title_vi}
    for key, title_ja, title_vi in SECTION_DEFINITIONS
}


def section_key_for_chapter(chapter_title: str) -> str:
    for section_key, chapter_titles in SECTION_CHAPTER_TITLES.items():
        if chapter_title in chapter_titles:
            return section_key
    logger.warning(
        "Chương lạ %s được đẩy vào section %s",
        chapter_title,
        DEFAULT_SECTION_KEY,
    )
    return DEFAULT_SECTION_KEY


def proposal_section_for_chapter(chapter_title: str) -> str:
    section_key = section_key_for_chapter(chapter_title)
    return SECTION_BY_KEY[section_key]["title_ja"]


def map_source_chapters(chapters: Iterable[Any]) -> dict[str, list[str]]:
    mapped = {key: [] for key in SECTION_BY_KEY}
    for chapter in chapters:
        section_key = section_key_for_chapter(chapter.title)
        mapped[section_key].append(chapter.id)
    return mapped
