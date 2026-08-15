import os
import time

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel


PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
MODEL = os.getenv("LLM_MODEL")
MODEL_EVAL = os.getenv("LLM_MODEL_EVAL", MODEL)

if not MODEL:
    raise ValueError("Thiếu biến môi trường LLM_MODEL")

if PROVIDER == "anthropic":
    import anthropic
    from rfp.usage import record

    client = anthropic.Anthropic()

    def generate(system: str, user: str, effort: str = "medium") -> str:
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
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
        t0 = time.perf_counter()
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=model_cls,
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

    client = OpenAI()
    TOKEN_ARG = os.getenv("LLM_TOKEN_ARG")
    USE_TEMP = os.getenv("LLM_USE_TEMP") == "1"
    USE_EFFORT = os.getenv("LLM_USE_EFFORT") == "1"

    if not TOKEN_ARG:
        raise ValueError("Thiếu biến môi trường LLM_TOKEN_ARG")

    def _openai_kwargs(
        limit: int,
        effort: str,
        *,
        structured_output: bool = False,
    ) -> dict[str, object]:
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
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **_openai_kwargs(16000, effort),
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
        t0 = time.perf_counter()
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=model_cls,
            **_openai_kwargs(4000, "low", structured_output=True),
        )
        elapsed = time.perf_counter() - t0
        u = getattr(response, "usage", None)
        pt = getattr(u, "prompt_tokens", 0) if u else 0
        ct = getattr(u, "completion_tokens", 0) if u else 0
        record("structured", pt, ct, elapsed)
        return response.choices[0].message.parsed

else:
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {PROVIDER}")
