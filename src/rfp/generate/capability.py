from typing import Any

from config.settings import GENERATION_EFFORT

from ..llm import generate
from ..stores.capability import CapabilityStore
from .precedent import DECORATION_TOKENS, numeric_expressions


CAPABILITY_SYSTEM = (
    "あなたは日本語の提案書編集者です。提示された能力表の文を自然な一文に整えてください。"
    "能力表にない事実・数値・認証・実績を追加せず、数値と単位を変更しないでください。"
    "説明や箇条書き記号を付けず、完成した一文だけを返してください。"
)


DEFAULT_FACT_KEYS = {
    "company_overview": (
        "company_facts:company",
        "company_facts:established",
    ),
    "proposal_overview": ("company_facts:company",),
    "implementation_experience": ("capabilities:基幹システム構築",),
    "certification_compliance": ("certifications:ISO/IEC 27001",),
    "delivery_structure": ("company_facts:headcount",),
}


def default_fact_keys(section_key: str) -> list[str]:
    return list(DEFAULT_FACT_KEYS[section_key])


def render_fact(store: CapabilityStore, fact_key: str) -> str:
    fact = store[fact_key]
    values: dict[str, Any] = {"v": fact.get("value")}
    rendered = fact["template"].format(**values)
    if fact_key == "company_facts:headcount":
        rendered = rendered.replace(str(fact["value"]), f"{fact['value']}名", 1)
    return rendered


def _safe_smooth(template_text: str) -> tuple[str, int]:
    smoothed = generate(
        CAPABILITY_SYSTEM,
        f"能力表の文：\n{template_text}",
        effort=GENERATION_EFFORT,
    ).strip()
    if (
        not smoothed
        or numeric_expressions(smoothed) != numeric_expressions(template_text)
        or any(
            token in smoothed and token not in template_text
            for token in DECORATION_TOKENS
        )
    ):
        return template_text, 1
    return smoothed, 1


def generate_capabilities(
    fact_keys: list[str],
    *,
    req_ids_by_fact: dict[str, list[str]],
    store: CapabilityStore | None = None,
) -> tuple[list[dict[str, Any]], int]:
    capability_store = store or CapabilityStore()
    sentences: list[dict[str, Any]] = []
    llm_calls = 0
    for fact_key in dict.fromkeys(fact_keys):
        template_text = render_fact(capability_store, fact_key)
        text, calls = _safe_smooth(template_text)
        llm_calls += calls
        sentences.append(
            {
                "text": text,
                "origin": "capability",
                "source_id": fact_key,
                "req_ids": list(req_ids_by_fact.get(fact_key, [])),
                "verdict": None,
            }
        )
    return sentences, llm_calls
