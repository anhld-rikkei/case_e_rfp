import functools
import os
import time

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel


PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
MODEL = os.getenv("LLM_MODEL")
MODEL_EVAL = os.getenv("LLM_MODEL_EVAL", MODEL)

RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))


class LLMUnavailable(RuntimeError):
    """Lỗi tạm thời phía provider, đã retry hết mà vẫn không phản hồi.

    Tầng graph bắt lỗi này để trả state `partial` thay vì crash mất kết quả
    các bước trước. Lỗi 4xx khác (sai key, sai schema, sai tham số) KHÔNG
    thuộc loại này: retry vô ích nên ném thẳng cho người sửa cấu hình."""

    def __init__(self, kind: str, attempts: int, cause: Exception) -> None:
        self.kind = kind
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"LLM không phản hồi sau {attempts} lần gọi (lỗi {kind}): {cause}"
        )


def _error_kind(exc: Exception) -> str | None:
    """Phân loại lỗi retry được; None nghĩa là không nên retry."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout"
    if "Connection" in name:
        return "connection"
    return None


def _with_retries(call):
    delay = RETRY_BASE_DELAY_SECONDS
    last_kind: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 2):  # 1 lần gọi + RETRY_ATTEMPTS retry
        try:
            return call()
        except Exception as exc:  # phân loại ngay bên dưới, không nuốt lỗi lạ
            kind = _error_kind(exc)
            if kind is None:
                raise
            last_kind, last_exc = kind, exc
            if attempt <= RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    raise LLMUnavailable(last_kind, RETRY_ATTEMPTS + 1, last_exc)


def _require_model() -> str:
    # Kiểm lúc GỌI, không phải lúc import. Dựng client hay bắt lỗi thiếu biến
    # ngay khi import nghĩa là mọi file lỡ import module này đều đòi API key —
    # kể cả tầng eval regex vốn không gọi LLM một lần nào (BUILD_GUIDE §11.5).
    if not MODEL:
        raise ValueError("Thiếu biến môi trường LLM_MODEL")
    return MODEL


if PROVIDER == "anthropic":
    import anthropic
    from rfp.usage import record

    @functools.lru_cache(maxsize=1)
    def _client() -> "anthropic.Anthropic":
        return anthropic.Anthropic()

    def generate(system: str, user: str, effort: str = "medium") -> str:
        model = _require_model()
        client = _client()
        t0 = time.perf_counter()
        response = _with_retries(
            lambda: client.messages.create(
                model=model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        )
        elapsed = time.perf_counter() - t0
        u = getattr(response, "usage", None)
        pt = getattr(u, "input_tokens", 0) if u else 0
        ct = getattr(u, "output_tokens", 0) if u else 0
        record("generate", pt, ct, elapsed)
        return next(
            block.text for block in response.content if block.type == "text"
        )

    def structured(
        system: str,
        user: str,
        model_cls: type[BaseModel],
    ) -> BaseModel:
        model = _require_model()
        client = _client()
        t0 = time.perf_counter()
        response = _with_retries(
            lambda: client.messages.parse(
                model=model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=model_cls,
            )
        )
        elapsed = time.perf_counter() - t0
        u = getattr(response, "usage", None)
        pt = getattr(u, "input_tokens", 0) if u else 0
        ct = getattr(u, "output_tokens", 0) if u else 0
        record("structured", pt, ct, elapsed)
        return response.parsed_output

elif PROVIDER == "openai":
    from openai import OpenAI
    from rfp.usage import record

    TOKEN_ARG = os.getenv("LLM_TOKEN_ARG")
    USE_TEMP = os.getenv("LLM_USE_TEMP") == "1"
    USE_EFFORT = os.getenv("LLM_USE_EFFORT") == "1"

    @functools.lru_cache(maxsize=1)
    def _client() -> "OpenAI":
        return OpenAI()

    def _openai_kwargs(
        limit: int,
        effort: str,
        *,
        structured_output: bool = False,
    ) -> dict[str, object]:
        if not TOKEN_ARG:
            raise ValueError("Thiếu biến môi trường LLM_TOKEN_ARG")
        kwargs: dict[str, object] = {TOKEN_ARG: limit}
        # gpt-5.4-mini accepts these options separately but rejects their
        # combination. Generation honors its public effort argument; structured
        # output favors deterministic temperature when both flags are enabled.
        if USE_TEMP and (structured_output or not USE_EFFORT):
            kwargs["temperature"] = 0
        if USE_EFFORT and not (structured_output and USE_TEMP):
            kwargs["reasoning_effort"] = effort
        return kwargs

    def generate(system: str, user: str, effort: str = "medium") -> str:
        model = _require_model()
        client = _client()
        t0 = time.perf_counter()
        response = _with_retries(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **_openai_kwargs(16000, effort),
            )
        )
        elapsed = time.perf_counter() - t0
        u = getattr(response, "usage", None)
        pt = getattr(u, "prompt_tokens", 0) if u else 0
        ct = getattr(u, "completion_tokens", 0) if u else 0
        record("generate", pt, ct, elapsed)
        return response.choices[0].message.content

    def structured(
        system: str,
        user: str,
        model_cls: type[BaseModel],
    ) -> BaseModel:
        model = _require_model()
        client = _client()
        t0 = time.perf_counter()
        response = _with_retries(
            lambda: client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=model_cls,
                **_openai_kwargs(4000, "low", structured_output=True),
            )
        )
        elapsed = time.perf_counter() - t0
        u = getattr(response, "usage", None)
        pt = getattr(u, "prompt_tokens", 0) if u else 0
        ct = getattr(u, "completion_tokens", 0) if u else 0
        record("structured", pt, ct, elapsed)
        return response.choices[0].message.parsed

else:
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {PROVIDER}")
