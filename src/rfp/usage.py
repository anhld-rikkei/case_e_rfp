from dataclasses import dataclass, field
import threading

@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_stage: dict[str, int] = field(default_factory=dict)
    tokens_by_stage: dict[str, int] = field(default_factory=dict)

_LOCK = threading.Lock()
_USAGE = Usage()

def record(stage: str, prompt_tokens: int, completion_tokens: int, elapsed: float) -> None:
    with _LOCK:
        _USAGE.calls += 1
        _USAGE.seconds += elapsed
        _USAGE.prompt_tokens += prompt_tokens
        _USAGE.completion_tokens += completion_tokens
        _USAGE.by_stage[stage] = _USAGE.by_stage.get(stage, 0) + 1
        
        total_tokens = prompt_tokens + completion_tokens
        _USAGE.tokens_by_stage[stage] = _USAGE.tokens_by_stage.get(stage, 0) + total_tokens

def snapshot() -> Usage:
    with _LOCK:
        return Usage(
            calls=_USAGE.calls,
            prompt_tokens=_USAGE.prompt_tokens,
            completion_tokens=_USAGE.completion_tokens,
            seconds=_USAGE.seconds,
            by_stage=dict(_USAGE.by_stage),
            tokens_by_stage=dict(_USAGE.tokens_by_stage),
        )

def reset() -> None:
    global _USAGE
    with _LOCK:
        _USAGE = Usage()
