from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import types
from typing import Any, Iterable

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks.base import BaseCallbackHandler
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from rfp.usage import record, snapshot

from config.settings import (  # noqa: E402
    EVAL_JUDGE_MAX_COMPLETION_TOKENS,
    EVAL_JUDGE_MODEL,
    EVAL_JUDGE_RUNS,
    EVAL_JUDGE_TEMPERATURE,
)
from eval.to_samples import (  # noqa: E402
    deterministic_metrics,
    partition_ragas_samples,
    to_eval_samples,
)
from rfp.graph import run_graph  # noqa: E402
from rfp.retrieve.hybrid import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    get_embedding_model,
)


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "noise_sensitivity",
    "context_entity_recall",
)


def _install_ragas_vertex_compatibility() -> None:
    """Bridge a removed optional LangChain module imported by ragas 0.4.3."""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


class LocalSentenceEmbeddings(Embeddings):
    """LangChain-compatible adapter over the retrieval embedding model."""

    def __init__(self) -> None:
        # RAGAS records ``model`` in EmbeddingUsageEvent, whose schema requires
        # a string. Keep the actual local encoder separate from that metadata.
        self.model = DEFAULT_EMBEDDING_MODEL
        self._encoder = get_embedding_model(self.model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        values = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class JudgeUsageCallback(BaseCallbackHandler):
    # RAGAS chạy nhiều lệnh gọi judge song song trên cùng MỘT instance handler.
    # Một biến self._t0 dùng chung sẽ bị lệnh gọi sau ghi đè, nên elapsed của
    # lệnh gọi trước bị rút ngắn về "khoảng cách giữa hai lần start". LangChain
    # truyền run_id riêng cho từng lệnh gọi — mốc thời gian phải khoá theo nó.
    # LangChain nuốt exception trong handler và chỉ log warning nếu cờ này False.
    # Nuốt ở đây nghĩa là mất luôn bản ghi token của lệnh gọi đó -> số đo thấp
    # hơn thực tế mà không ai biết. Thà nổ.
    raise_error = True

    def __init__(self) -> None:
        super().__init__()
        self._starts: dict[Any, float] = {}
        self._lock = threading.Lock()

    def on_llm_error(self, error, *, run_id: Any = None, **kwargs) -> None:
        with self._lock:
            self._starts.pop(run_id, None)

    def on_llm_start(self, *args, run_id: Any = None, **kwargs) -> None:
        with self._lock:
            self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id: Any = None, **kwargs) -> None:
        now = time.perf_counter()
        with self._lock:
            started = self._starts.pop(run_id, None)
        if started is None:
            raise RuntimeError(
                f"JudgeUsageCallback nhận on_llm_end không có on_llm_start khớp "
                f"(run_id={run_id!r}) — thời gian judge sẽ sai"
            )
        elapsed = now - started
        prompt_tokens = 0
        completion_tokens = 0
        found = False

        llm_output = getattr(response, "llm_output", None) or {}
        if "token_usage" in llm_output:
            u = llm_output["token_usage"]
            prompt_tokens = u.get("prompt_tokens", 0)
            completion_tokens = u.get("completion_tokens", 0)
            found = True
        else:
            try:
                msg = response.generations[0][0].message
                usage_meta = getattr(msg, "usage_metadata", None)
                if usage_meta:
                    prompt_tokens = usage_meta.get("input_tokens", 0)
                    completion_tokens = usage_meta.get("output_tokens", 0)
                    found = True
            except (IndexError, AttributeError):
                pass

        if not found:
            import warnings
            warnings.warn("JudgeUsageCallback không tìm thấy thông tin token usage trong response")

        record("judge", prompt_tokens, completion_tokens, elapsed)


def _ragas_components():
    _install_ragas_vertex_compatibility()
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_entities_recall import ContextEntityRecall
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._noise_sensitivity import NoiseSensitivity

    if not EVAL_JUDGE_MODEL:
        raise ValueError("Thiếu LLM_MODEL_EVAL hoặc LLM_MODEL trong .env")
    judge = ChatOpenAI(
        model=EVAL_JUDGE_MODEL,
        max_completion_tokens=EVAL_JUDGE_MAX_COMPLETION_TOKENS,
        callbacks=[JudgeUsageCallback()],
    )
    # langchain-openai 1.5 assumes every non-chat gpt-5 rejects temperature
    # and silently removes it during validation. The repository's model probe
    # is the source of truth: gpt-5.4-mini accepts temperature when no
    # reasoning_effort is sent. Set it after construction and intentionally do
    # not configure reasoning_effort, matching rfp.llm.structured().
    judge.temperature = EVAL_JUDGE_TEMPERATURE
    # gpt-5.4-mini returns one generation per request. RAGAS' historical
    # answer-relevancy default is strictness=3, which is an internal ensemble,
    # unrelated to this harness' three independent outer runs.
    ragas_judge = LangchainLLMWrapper(judge, bypass_n=True)
    embeddings = LangchainEmbeddingsWrapper(LocalSentenceEmbeddings())
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        LLMContextPrecisionWithReference(name="context_precision"),
        LLMContextRecall(),
        NoiseSensitivity(mode="irrelevant"),
        ContextEntityRecall(),
    ]
    return evaluate, EvaluationDataset, ragas_judge, embeddings, metrics


def _evaluation_dataset(samples: list[dict[str, Any]]):
    _, EvaluationDataset, _, _, _ = _ragas_components()
    return EvaluationDataset.from_list(
        [
            {
                "user_input": sample["question"],
                "retrieved_contexts": sample["contexts"],
                "response": sample["answer"],
                "reference": sample["ground_truth"],
            }
            for sample in samples
        ],
        name="case_e_rfp_requirement_atoms",
    )


def _result_means(result: Any) -> dict[str, float]:
    frame = result.to_pandas()
    means: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        # RAGAS 0.4.x appends the selected mode to ModeMetric column names.
        result_name = (
            "noise_sensitivity(mode=irrelevant)"
            if metric_name == "noise_sensitivity"
            else metric_name
        )
        if result_name not in frame:
            raise RuntimeError(f"RAGAS không trả metric {metric_name}")
        values = [
            float(value)
            for value in frame[result_name].tolist()
            if value is not None and math.isfinite(float(value))
        ]
        if not values:
            raise RuntimeError(f"RAGAS không tính được metric {metric_name}")
        means[metric_name] = statistics.fmean(values)
    return means


def run_ragas(
    samples: list[dict[str, Any]],
    *,
    runs: int = EVAL_JUDGE_RUNS,
) -> dict[str, dict[str, Any]]:
    if runs != EVAL_JUDGE_RUNS:
        raise ValueError(
            f"Judge runs được pin ở n={EVAL_JUDGE_RUNS}, nhận n={runs}"
        )
    answered, unanswered = partition_ragas_samples(samples)
    if unanswered:
        raise ValueError(
            "run_ragas chỉ nhận requirement có answer; "
            f"nhận {len(unanswered)} sample không có answer"
        )
    if not answered:
        raise ValueError("Không có requirement có answer để chạy RAGAS")
    evaluate, EvaluationDataset, judge, embeddings, metric_templates = (
        _ragas_components()
    )
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": sample["question"],
                "retrieved_contexts": sample["contexts"],
                "response": sample["answer"],
                "reference": sample["ground_truth"],
            }
            for sample in answered
        ],
        name="case_e_rfp_requirement_atoms",
    )
    per_run: list[dict[str, float]] = []
    start_time = time.perf_counter()
    rfp_ids = list(dict.fromkeys(s.get("rfp_id", "unknown") for s in answered))
    rfp_id_str = rfp_ids[0] if len(rfp_ids) == 1 else f"{len(rfp_ids)} RFPs"

    for run_idx in range(runs):
        result = evaluate(
            dataset=dataset,
            metrics=metric_templates,
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=True,
            show_progress=False,
        )
        per_run.append(_result_means(result))
        
        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / (run_idx + 1)
        remaining = avg_time * (runs - 1 - run_idx)
        usage = snapshot()
        
        m_sec, s_sec = divmod(elapsed, 60)
        rem_m, rem_s = divmod(remaining, 60)
        judge_toks = usage.tokens_by_stage.get("judge", 0)
        
        print(f"{rfp_id_str} · lượt {run_idx + 1}/{runs} · đã: {int(m_sec)}m{int(s_sec)}s · còn ~{int(rem_m)}m{int(rem_s)}s · judge {judge_toks/1000:.1f}k tok")

    summary: dict[str, dict[str, Any]] = {}
    for metric_name in METRIC_NAMES:
        values = [row[metric_name] for row in per_run]
        if len(values) != runs or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Metric {metric_name} thiếu kết quả hữu hạn: {values}")
        summary[metric_name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "runs": values,
        }
    return summary


def load_states(
    *,
    state_paths: Iterable[Path],
    rfp_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for path in state_paths:
        states.append(json.loads(path.read_text(encoding="utf-8")))

    from rfp.graph import run_graph_eval
    
    for path in rfp_paths:
        state = run_graph_eval(path.read_text(encoding="utf-8"))
        if state.get("status") == "guard_blocked":
            print(f"Guard chặn không cho xuất bản {path}")
        if state.get("status") != "completed":
            print(f"Graph không completed cho {path}: {state.get('status')}. Sẽ tính là fail coverage/abstain.")
        states.append(state)
        
    return states


def print_deterministic(state: dict[str, Any], metrics: dict[str, Any]) -> None:
    rfp = state.get("rfp")
    rfp_id = rfp.get("rfp_id", "") if isinstance(rfp, dict) else getattr(rfp, "rfp_id", "")
    print(
        f"deterministic {rfp_id}: "
        f"coverage={metrics['covered_requirements']}/{metrics['requirements']} "
        f"({metrics['coverage']:.3f}) · "
        f"abstain_rate={metrics['abstained_sections']}/{metrics['sections']} "
        f"({metrics['abstain_rate']:.3f}) · "
        f"groundedness={metrics['groundedness']:.3f} · "
        f"citation_accuracy={metrics['citation_accuracy']:.3f} · "
        f"fabrication={metrics['fabrication_count']} · "
        f"leak={metrics['client_leak_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-layer RFP evaluation harness")
    parser.add_argument("--state", action="append", type=Path, default=[])
    parser.add_argument("--rfp", action="append", type=Path, default=[])
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--disable-guards", action="store_true", help="Vô hiệu hóa guard 6.4 và 7 (dùng cho ablation)")
    parser.add_argument("--disable-quarantine", action="store_true", help="Vô hiệu hóa quarantine lúc ingest (dùng cho ablation)")
    args = parser.parse_args()
    if not args.state and not args.rfp:
        parser.error("provide at least one --state or --rfp")

    patchers = []
    if args.disable_guards:
        from unittest.mock import patch
        from collections import Counter

        def mock_check_claims(sentences, **kwargs):
            checked = []
            for s in sentences:
                sc = dict(s)
                sc["verdict"] = "VERIFIED"
                checked.append(sc)
            return checked, Counter({"VERIFIED": len(sentences)}), 0, []

        def mock_filter_hybrid_claims(sentences, whitelist):
            return sentences, []

        def mock_final_guard(proposal):
            pass

        patchers.extend([
            patch("rfp.graph.check_claims", mock_check_claims),
            patch("rfp.graph.filter_hybrid_claims", mock_filter_hybrid_claims),
            patch("rfp.graph.final_guard", mock_final_guard),
            patch("eval.to_samples.final_guard", mock_final_guard),
        ])

    if args.disable_quarantine:
        from unittest.mock import patch
        
        def mock_filter_client_leaks(sentences):
            return sentences, []
        
        def mock_partition(self, sentences):
            return sentences, []
            
        patchers.extend([
            patch("rfp.stores.sentence_index.filter_client_leaks", mock_filter_client_leaks),
            patch("rfp.stores.sentence_index.CapabilityBlocklist.partition", mock_partition),
        ])

    for p in patchers:
        p.start()

    try:
        states = load_states(
            state_paths=args.state, 
            rfp_paths=args.rfp
        )
        samples: list[dict[str, Any]] = []
        deterministic_rows = []
        for state in states:
            row = deterministic_metrics(state)
            deterministic_rows.append(row)
            print_deterministic(state, row)
            samples.extend(to_eval_samples(state))
    finally:
        for p in patchers:
            p.stop()
    ragas_samples, unanswered_samples = partition_ragas_samples(samples)
    print(
        f"ragas_samples={len(ragas_samples)}/{len(samples)} "
        f"(loại {len(unanswered_samples)} sample không có answer)"
    )
    print("capability_sheet_in_every_context=True")

    report: dict[str, Any] = {
        "judge": {
            "model": EVAL_JUDGE_MODEL,
            "temperature": EVAL_JUDGE_TEMPERATURE,
            "runs": EVAL_JUDGE_RUNS,
        },
        "deterministic": deterministic_rows,
        "sample_count": len(samples),
        "ragas_sample_count": len(ragas_samples),
        "unanswered_sample_count": len(unanswered_samples),
    }
    if not args.deterministic_only:
        summary = run_ragas(ragas_samples)
        report["ragas"] = summary
        print(
            f"judge={EVAL_JUDGE_MODEL} · temperature={EVAL_JUDGE_TEMPERATURE:g} "
            f"· n={EVAL_JUDGE_RUNS}"
        )
        for metric_name, values in summary.items():
            print(
                f"{metric_name}: runs="
                f"[{', '.join(f'{value:.4f}' for value in values['runs'])}] "
                f"· {values['mean']:.4f} ± {values['std']:.4f}"
            )
    if args.out is not None:
        from rfp.usage import snapshot
        usage = snapshot()
        report["usage"] = {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "seconds": usage.seconds,
            "by_stage": usage.by_stage,
            "tokens_by_stage": usage.tokens_by_stage,
            "seconds_by_stage": usage.seconds_by_stage,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output={args.out}")


if __name__ == "__main__":
    main()
