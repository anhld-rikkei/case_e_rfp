from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


ASSERTION_KINDS = frozenset(
    {
        "must_not_contain",
        "must_flag_insufficient",
        "must_cover",
        "must_route",
        "must_ask_user",
    }
)


@dataclass(frozen=True)
class Assertion:
    kind: str
    target: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in ASSERTION_KINDS:
            raise ValueError(f"Unsupported assertion kind: {self.kind}")
        if not self.target.strip():
            raise ValueError("Assertion target must not be empty")
        if not self.reason.strip():
            raise ValueError("Assertion reason must not be empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Assertion":
        return cls(
            kind=str(value["kind"]),
            target=str(value["target"]),
            reason=str(value["reason"]),
        )


@dataclass
class GoldenCase:
    case_id: str
    rfp_text: str
    source: str
    needs_review: bool
    assertions: list[Assertion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.rfp_text.strip():
            raise ValueError("rfp_text must not be empty")
        if not self.assertions:
            raise ValueError(f"Golden case {self.case_id} must have at least one assertion")

    @property
    def eligible_for_metrics(self) -> bool:
        return not self.needs_review

    def to_dict(self) -> dict[str, Any]:
        value = {
            "case_id": self.case_id,
            "rfp_text": self.rfp_text,
            "source": self.source,
            "needs_review": self.needs_review,
            "assertions": [asdict(item) for item in self.assertions],
        }
        value.update(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldenCase":
        known = {"case_id", "rfp_text", "source", "needs_review", "assertions"}
        assertions = [
            normalize_legacy_assertion(item, value)
            for item in value.get("assertions", [])
        ]
        return cls(
            case_id=str(value["case_id"]),
            rfp_text=str(value["rfp_text"]),
            source=str(value["source"]),
            needs_review=bool(value.get("needs_review", False)),
            assertions=assertions,
            metadata={key: item for key, item in value.items() if key not in known},
        )


def normalize_legacy_assertion(
    value: dict[str, Any],
    case_data: dict[str, Any],
) -> Assertion:
    kind = str(value["kind"])
    target = str(value["target"])
    reason = str(value["reason"])
    if kind in ASSERTION_KINDS:
        return Assertion(kind=kind, target=target, reason=reason)
    if kind in {"must_complete", "must_parse_with_fallback"}:
        return Assertion(
            kind="must_route",
            target="completed",
            reason=reason,
        )
    if kind == "must_not_claim_industry_experience":
        return Assertion(
            kind="must_not_contain",
            target=f"{target}分野での実績",
            reason=reason,
        )
    if kind == "must_contain":
        req_id = next(
            (
                str(item["target"])
                for item in case_data.get("assertions", [])
                if item.get("kind") == "must_flag_insufficient"
            ),
            "completed",
        )
        return Assertion(
            kind="must_flag_insufficient" if req_id != "completed" else "must_route",
            target=req_id,
            reason=f"Legacy must_contain normalized: {reason}",
        )
    raise ValueError(f"Unsupported legacy assertion kind: {kind}")


def load_case(path: str | Path) -> GoldenCase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldenCase.from_dict(data)


def save_case(case: GoldenCase, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(case.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
