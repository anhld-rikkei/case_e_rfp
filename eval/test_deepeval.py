from __future__ import annotations

import asyncio
from pathlib import Path
import statistics
import sys
from typing import Any

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from config.settings import (  # noqa: E402
    EVAL_JUDGE_EFFORT,
    EVAL_JUDGE_MODEL,
    EVAL_JUDGE_RUNS,
)
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import GEval  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402
from eval.to_samples import capability_sheet_text  # noqa: E402
from rfp.llm import MODEL, structured  # noqa: E402


COMPLIANCE_CRITERIA = (
    "Đối chiếu actual_output với capability sheet trong context. "
    "Phạt nặng mọi tuyên bố về chứng chỉ hoặc năng lực KHÔNG có trong sheet, "
    "kể cả khi tuyên bố đó xuất hiện trong tài liệu tham chiếu. "
    "Câu 'không có năng lực này' hoặc bỏ trống KHÔNG bị phạt."
)
DEEPEVAL_SYSTEM = (
    "Bạn là judge đánh giá tuân thủ năng lực cho hồ sơ thầu tiếng Nhật. "
    "Chỉ dùng dữ liệu trong prompt, không dùng kiến thức ngoài. "
    "Trả kết quả đúng structured schema được yêu cầu."
)


class _TextResponse(BaseModel):
    text: str


class StructuredJudgeModel(DeepEvalBaseLLM):
    """DeepEval adapter that keeps all SDK access inside rfp.llm.structured."""

    def __init__(self) -> None:
        if not EVAL_JUDGE_MODEL:
            raise ValueError("Thiếu LLM_MODEL_EVAL hoặc LLM_MODEL trong .env")
        if EVAL_JUDGE_MODEL != MODEL:
            raise ValueError(
                "structured() dùng LLM_MODEL; để pin judge mà không gọi SDK trực tiếp, "
                "LLM_MODEL_EVAL phải bằng LLM_MODEL"
            )
        super().__init__(model=EVAL_JUDGE_MODEL)

    def load_model(self, *_: Any, **__: Any) -> "StructuredJudgeModel":
        return self

    def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        **_: Any,
    ) -> BaseModel | str:
        response_model = schema or _TextResponse
        result = structured(DEEPEVAL_SYSTEM, str(prompt), response_model)
        return result if schema is not None else result.text

    async def a_generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> BaseModel | str:
        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema=schema,
            **kwargs,
        )

    def get_model_name(self) -> str:
        return f"{EVAL_JUDGE_MODEL} (structured, effort={EVAL_JUDGE_EFFORT})"

    def supports_structured_outputs(self) -> bool:
        return True


def capability_compliance_metric() -> GEval:
    return GEval(
        name="CapabilityCompliance",
        criteria=COMPLIANCE_CRITERIA,
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        model=StructuredJudgeModel(),
        threshold=0.9,
        async_mode=False,
    )


def assert_capability_compliance(
    test_case: LLMTestCase,
    *,
    runs: int = EVAL_JUDGE_RUNS,
) -> tuple[float, float]:
    if runs != EVAL_JUDGE_RUNS:
        raise ValueError(f"Judge runs được pin ở n={EVAL_JUDGE_RUNS}")
    scores: list[float] = []
    for _ in range(runs):
        metric = capability_compliance_metric()
        assert_test(test_case, [metric], run_async=False)
        scores.append(float(metric.score))
    return statistics.fmean(scores), statistics.pstdev(scores)


def test_capability_compliance_with_held_certification() -> None:
    case = LLMTestCase(
        input="ISO/IEC 27001相当の管理体制を有すること。",
        actual_output="当社はISO/IEC 27001認証を有しています。",
        context=[capability_sheet_text()],
    )
    mean, std = assert_capability_compliance(case)
    print(
        f"CapabilityCompliance: {mean:.4f} ± {std:.4f} "
        f"· judge={EVAL_JUDGE_MODEL} · n={EVAL_JUDGE_RUNS}"
    )
