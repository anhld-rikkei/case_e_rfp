import unicodedata

from .sanitize.blocklist import CapabilityBlocklist
from .sanitize.leak import private_client_names


class GuardViolation(RuntimeError):
    pass


# Dùng chung đúng một bộ regex biến thể với lớp quarantine lúc ingest (BB-2).
_BLOCKLIST = CapabilityBlocklist()
FORBIDDEN = _BLOCKLIST.terms


def final_guard(text: str) -> None:
    normalized = unicodedata.normalize("NFKC", text)
    blocklist_hits = _BLOCKLIST.find(normalized)
    if blocklist_hits:
        raise GuardViolation(f"blocklist: {blocklist_hits}")

    leaked_clients = list(dict.fromkeys(private_client_names(normalized)))
    if leaked_clients:
        raise GuardViolation(f"client name leaked: {leaked_clients}")
