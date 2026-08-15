import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from ..schema import Sentence
from ..stores.capability import CapabilityStore


class CapabilityBlocklist:
    def __init__(self, capability_store: CapabilityStore | None = None) -> None:
        self.capability_store = capability_store or CapabilityStore()
        self.terms = self.capability_store.forbidden_terms
        self.pattern = re.compile("|".join(re.escape(term) for term in self.terms))

    def find(self, text: str) -> list[str]:
        return list(dict.fromkeys(match.group(0) for match in self.pattern.finditer(text)))

    def contradicts(self, text: str) -> bool:
        return self.pattern.search(text) is not None

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
