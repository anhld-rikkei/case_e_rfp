import os

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

    client = anthropic.Anthropic()

    def generate(system: str, user: str, effort: str = "medium") -> str:
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
        return next(
            block.text for block in response.content if block.type == "text"
        )

    def structured(
        system: str,
        user: str,
        model_cls: type[BaseModel],
    ) -> BaseModel:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=model_cls,
        )
        return response.parsed_output

elif PROVIDER == "openai":
    from openai import OpenAI

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
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **_openai_kwargs(16000, effort),
        )
        return response.choices[0].message.content

    def structured(
        system: str,
        user: str,
        model_cls: type[BaseModel],
    ) -> BaseModel:
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=model_cls,
            **_openai_kwargs(4000, "low", structured_output=True),
        )
        return response.choices[0].message.parsed

else:
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {PROVIDER}")
