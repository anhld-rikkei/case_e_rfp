import re
from collections import Counter
from typing import Any

from config.settings import GENERATION_EFFORT

from ..llm import generate


NUMBER_RE = re.compile(
    r"\d+(?:[.,]\d+)?(?:%|年|名|人|件|か月|ヶ月)?"
    r"|[〇一二三四五六七八九十百千万億兆]+(?:%|年|名|人|件|か月|ヶ月)"
)
METRIC_CORE_RE = re.compile(
    r"(?:処理時間|在庫精度|年間運用コスト|障害件数)を"
    r"\d+%(?:向上|短縮|削減|低減)"
)
DECORATION_TOKENS = ("【", "】", "```")

PRECEDENT_SYSTEM = (
    "あなたは日本語の提案書編集者です。提示された根拠文を一文だけに軽く整えてください。"
    "事実を追加せず、数値・単位・認証名・顧客区分を変更しないでください。"
    "説明や箇条書き記号を付けず、完成した一文だけを返してください。"
)


def numeric_expressions(text: str) -> Counter[str]:
    return Counter(NUMBER_RE.findall(text))


def metric_cores(text: str) -> Counter[str]:
    return Counter(METRIC_CORE_RE.findall(text))


def _safe_rewrite(source_text: str) -> tuple[str, int]:
    rewritten = generate(
        PRECEDENT_SYSTEM,
        f"根拠文：\n{source_text}",
        effort=GENERATION_EFFORT,
    ).strip()
    if (
        not rewritten
        or numeric_expressions(rewritten) != numeric_expressions(source_text)
        or metric_cores(rewritten) != metric_cores(source_text)
        or any(
            token in rewritten and token not in source_text
            for token in DECORATION_TOKENS
        )
    ):
        return source_text, 1
    return rewritten, 1


def generate_precedents(
    selected: list[dict[str, Any]],
    *,
    req_ids_by_source: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], int]:
    sentences: list[dict[str, Any]] = []
    llm_calls = 0
    seen_source_ids: set[str] = set()
    for source in selected:
        source_id = source["sent_id"]
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        text, calls = _safe_rewrite(source["text"])
        llm_calls += calls
        sentences.append(
            {
                "text": text,
                "origin": "precedent",
                "source_id": source_id,
                "req_ids": list(req_ids_by_source.get(source_id, [])),
                "verdict": None,
            }
        )
    return sentences, llm_calls
