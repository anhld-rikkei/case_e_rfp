import json
from pathlib import Path
from typing import Any, Iterator

from config.templates_ja import (
    CAPABILITY_TEMPLATES,
    CERTIFICATION_TEMPLATES,
    COMPANY_FACT_TEMPLATES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPABILITY_PATH = PROJECT_ROOT / "synthetic" / "capability_sheet.json"


class CapabilityStore:
    def __init__(self, path: str | Path = DEFAULT_CAPABILITY_PATH) -> None:
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as capability_file:
            self.capability: dict[str, Any] = json.load(capability_file)
        self._facts = self._build_facts()

    def _build_facts(self) -> dict[str, dict[str, Any]]:
        facts: dict[str, dict[str, Any]] = {}

        for field in ("company", "established", "headcount"):
            facts[f"company_facts:{field}"] = {
                "value": self.capability[field],
                "template": COMPANY_FACT_TEMPLATES[field],
            }

        for capability in self.capability["capabilities"]:
            facts[f"capabilities:{capability}"] = {
                "template": CAPABILITY_TEMPLATES[capability]
            }

        for certification in self.capability["certifications_held"]:
            facts[f"certifications:{certification}"] = {
                "template": CERTIFICATION_TEMPLATES[certification]
            }

        return facts

    @property
    def forbidden_terms(self) -> tuple[str, ...]:
        return tuple(
            self.capability["certifications_NOT_held"]
            + self.capability["capabilities_NOT_offered"]
        )

    @property
    def industries_served(self) -> tuple[str, ...]:
        return tuple(self.capability["industries_served"])

    def get(self, key: str, default: Any = None) -> dict[str, Any] | Any:
        return self._facts.get(key, default)

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self._facts[key]

    def __contains__(self, key: object) -> bool:
        return key in self._facts

    def __iter__(self) -> Iterator[str]:
        return iter(self._facts)

    def items(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(self._facts.items())

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._facts.items()}


def load_capability_store(
    path: str | Path = DEFAULT_CAPABILITY_PATH,
) -> CapabilityStore:
    return CapabilityStore(path)
