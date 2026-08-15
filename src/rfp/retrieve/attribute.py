import re
from dataclasses import dataclass

from ..schema import Chapter, Requirement
from ..stores.capability import CapabilityStore


ATTRIBUTE_PATTERNS = {
    "capabilities:基幹システム構築": re.compile(
        r"基幹システム|生産管理|受注|在庫|出荷|CRM|顧客情報|取引履歴|問い合わせ履歴"
    ),
    "capabilities:クラウド移行（AWS・Azure）": re.compile(
        r"クラウド|IaaS|AWS|Azure|データ移行"
    ),
    "capabilities:Webアプリケーション開発": re.compile(
        r"Webブラウザ|Webアプリケーション|ブラウザ"
    ),
    "capabilities:データ分析基盤構築": re.compile(
        r"データ分析|販売データ|需要予測|データウェアハウス|BIツール"
    ),
    "capabilities:RPA導入": re.compile(r"RPA|業務自動化"),
    "capabilities:ネットワーク構築・保守": re.compile(r"ネットワーク構築|通信基盤"),
    "certifications:ISO 9001": re.compile(r"ISO 9001|品質管理"),
    "certifications:ISO/IEC 27001": re.compile(
        r"ISO/IEC 27001|情報セキュリティ|暗号化|情報管理基準|アクセス権限"
    ),
    "certifications:プライバシーマーク": re.compile(r"個人情報|プライバシー"),
}


@dataclass(frozen=True)
class AttributeMatch:
    req_id: str
    fact_keys: tuple[str, ...]
    exact_fact_keys: tuple[str, ...]


@dataclass(frozen=True)
class AttributeCoverage:
    matches: tuple[AttributeMatch, ...]
    covered_req_ids: tuple[str, ...]
    missing_req_ids: tuple[str, ...]
    fully_covered: bool
    exact_covered_req_ids: tuple[str, ...]
    exact_missing_req_ids: tuple[str, ...]
    exactly_covered: bool


class AttributeRetriever:
    def __init__(self, store: CapabilityStore | None = None) -> None:
        self.store = store or CapabilityStore()
        missing_keys = sorted(key for key in ATTRIBUTE_PATTERNS if key not in self.store)
        if missing_keys:
            raise ValueError(f"Attribute rule trỏ fact không tồn tại: {missing_keys}")
        self.exact_fact_names = {
            key: key.split(":", 1)[1]
            for key in self.store
            if key.startswith(("capabilities:", "certifications:"))
        }

    def match_requirement(self, requirement: Requirement) -> AttributeMatch:
        fact_keys = tuple(
            key
            for key, pattern in ATTRIBUTE_PATTERNS.items()
            if pattern.search(requirement.text)
        )
        exact_fact_keys = tuple(
            key
            for key, fact_name in self.exact_fact_names.items()
            if fact_name in requirement.text
        )
        return AttributeMatch(
            req_id=requirement.req_id,
            fact_keys=fact_keys,
            exact_fact_keys=exact_fact_keys,
        )

    def cover_chapter(self, chapter: Chapter) -> AttributeCoverage:
        matches = tuple(self.match_requirement(req) for req in chapter.requirements)
        covered = tuple(match.req_id for match in matches if match.fact_keys)
        missing = tuple(match.req_id for match in matches if not match.fact_keys)
        exact_covered = tuple(
            match.req_id for match in matches if match.exact_fact_keys
        )
        exact_missing = tuple(
            match.req_id for match in matches if not match.exact_fact_keys
        )
        return AttributeCoverage(
            matches=matches,
            covered_req_ids=covered,
            missing_req_ids=missing,
            fully_covered=bool(matches) and not missing,
            exact_covered_req_ids=exact_covered,
            exact_missing_req_ids=exact_missing,
            exactly_covered=bool(matches) and not exact_missing,
        )
