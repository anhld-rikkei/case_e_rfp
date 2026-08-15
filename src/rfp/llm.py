from dotenv import load_dotenv

load_dotenv()

import anthropic
from pydantic import BaseModel


client = anthropic.Anthropic()

MODEL = "claude-opus-5"
MODEL_EVAL = "claude-opus-5"


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
    return next(block.text for block in response.content if block.type == "text")


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
