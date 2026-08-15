import re
from dataclasses import replace
from typing import Iterable

from config.templates_ja import COMMON_CLIENT_NAMES

from ..schema import Sentence


CLIENT_NAME_RE = re.compile(r"([^\s・]{2,10})様向け")


def extract_client_names(text: str) -> list[str]:
    return [match.group(1) for match in CLIENT_NAME_RE.finditer(text)]


def is_private_client_name(name: str) -> bool:
    return name not in COMMON_CLIENT_NAMES


def private_client_names(text: str) -> list[str]:
    return [name for name in extract_client_names(text) if is_private_client_name(name)]


def has_client_leak(text: str) -> bool:
    return bool(private_client_names(text))


def mark_client_leak(sentence: Sentence) -> Sentence:
    flags = dict(sentence.flags)
    flags["client_leak"] = True
    return replace(sentence, flags=flags)


def sanitize_sentence(sentence: Sentence) -> Sentence | None:
    if has_client_leak(sentence.text):
        return None
    return sentence


def filter_client_leaks(
    sentences: Iterable[Sentence],
) -> tuple[list[Sentence], list[Sentence]]:
    clean: list[Sentence] = []
    rejected: list[Sentence] = []
    for sentence in sentences:
        if has_client_leak(sentence.text):
            rejected.append(mark_client_leak(sentence))
        else:
            clean.append(sentence)
    return clean, rejected
