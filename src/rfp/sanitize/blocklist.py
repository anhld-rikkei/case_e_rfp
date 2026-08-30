import json
import re
import unicodedata
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from ..schema import Sentence
from ..stores.capability import CapabilityStore


# Chứng chỉ dạng "ISO/IEC NNNNN" cần bắt cả biến thể viết tắt/dính liền/số trần,
# các mục cấm khác (CMMI, 量子暗号通信...) chỉ cần khớp literal sau NFKC.
_ISO_CERT_RE = re.compile(r"^ISO\s*/\s*IEC\s+(\d{4,5})$", re.IGNORECASE)

# Số hiệu trần (vd "27017") chỉ bị chặn khi đứng cạnh ngữ cảnh chứng chỉ,
# để không chặn nhầm số vô can; số của chứng chỉ ĐANG có (27001, 9001) không nằm
# trong danh sách này nên không bao giờ bị quét.
_CERT_CONTEXT = r"認証|取得|準拠|認定|規格|証明"


def _variant_pattern(term: str) -> str:
    normalized = unicodedata.normalize("NFKC", term).strip()
    iso_match = _ISO_CERT_RE.match(normalized)
    if iso_match:
        number = iso_match.group(1)
        bare = rf"(?<!\d){number}(?!\d)"
        # ISO/IEC 27017 · ISO 27017 · ISO27017 · ISO-27017 · IEC 27017 · hậu tố :2015
        core = rf"(?:ISO(?:\s*[/\-]\s*IEC)?|IEC)[\s\-]*{bare}(?:\s*:\s*\d{{4}})?"
        near = (
            rf"(?:(?:{_CERT_CONTEXT})[^\n。]{{0,20}}{bare}"
            rf"|{bare}[^\n。]{{0,20}}(?:{_CERT_CONTEXT}))"
        )
        return rf"(?:{core}|{near})"
    escaped = re.escape(normalized)
    return escaped.replace(" ", r"\s*").replace("/", r"\s*/\s*")


def build_blocklist_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    return re.compile(
        "|".join(_variant_pattern(term) for term in terms), re.IGNORECASE
    )


class CapabilityBlocklist:
    def __init__(self, capability_store: CapabilityStore | None = None) -> None:
        self.capability_store = capability_store or CapabilityStore()
        self.terms = self.capability_store.forbidden_terms
        self.pattern = build_blocklist_pattern(self.terms)

    def find(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text)
        return list(
            dict.fromkeys(match.group(0) for match in self.pattern.finditer(normalized))
        )

    def contradicts(self, text: str) -> bool:
        return self.pattern.search(unicodedata.normalize("NFKC", text)) is not None

    def mark(self, sentence: Sentence) -> Sentence:
        flags = dict(sentence.flags)
        flags["contradicts_capability"] = True
        return replace(sentence, flags=flags)

    def partition(
        self,
        sentences: Iterable[Sentence],
    ) -> tuple[list[Sentence], list[Sentence]]:
        clean: list[Sentence] = []
        quarantined: list[Sentence] = []
        for sentence in sentences:
            if self.contradicts(sentence.text):
                quarantined.append(self.mark(sentence))
            else:
                clean.append(sentence)
        return clean, quarantined


def write_quarantine(sentences: Iterable[Sentence], path: str | Path) -> None:
    quarantine_path = Path(path)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    with quarantine_path.open("w", encoding="utf-8", newline="\n") as output:
        for sentence in sentences:
            output.write(json.dumps(asdict(sentence), ensure_ascii=False) + "\n")
