import re
import unicodedata

from .sanitize.leak import private_client_names
from .stores.capability import CapabilityStore


class GuardViolation(RuntimeError):
    pass


_CAPABILITY_STORE = CapabilityStore()
FORBIDDEN = _CAPABILITY_STORE.forbidden_terms
BLOCKLIST_RE = re.compile("|".join(re.escape(term) for term in FORBIDDEN))


def final_guard(text: str) -> None:
    normalized = unicodedata.normalize("NFKC", text)
    blocklist_hits = list(
        dict.fromkeys(match.group(0) for match in BLOCKLIST_RE.finditer(normalized))
    )
    if blocklist_hits:
        raise GuardViolation(f"blocklist: {blocklist_hits}")

    leaked_clients = list(dict.fromkeys(private_client_names(normalized)))
    if leaked_clients:
        raise GuardViolation(f"client name leaked: {leaked_clients}")
