"""Unit test cho Bước 4 (v1.1): retry LLM + checkpoint/resume theo job_id.

Toàn bộ test cách ly: không gọi mạng, không cần API key, không build index,
không đọc config/settings.py ngoài import mặc định. Node của graph được thay
bằng stub qua monkeypatch — build_graph() tra hàm node theo tên trong module
lúc gọi, nên bản compile mới (đường job_id) nhận stub, còn GRAPH mặc định
compile sẵn lúc import thì không bị ảnh hưởng.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.llm as llm  # noqa: E402
from rfp.llm import LLMUnavailable, _error_kind, _with_retries  # noqa: E402
import rfp.graph as graph_module  # noqa: E402
from rfp.graph import run_graph, run_graph_eval  # noqa: E402


class _FakeAPIError(Exception):
    def __init__(self, status_code: int | None = None) -> None:
        super().__init__(f"fake status={status_code}")
        self.status_code = status_code


class _FakeTimeoutError(Exception):
    pass


class _FakeConnectionError(Exception):
    pass


# ── Phân loại lỗi ──────────────────────────────────────────────────────────

def test_error_kind_classification() -> None:
    assert _error_kind(_FakeAPIError(429)) == "rate_limit"
    assert _error_kind(_FakeAPIError(500)) == "server_error"
    assert _error_kind(_FakeAPIError(503)) == "server_error"
    assert _error_kind(_FakeTimeoutError()) == "timeout"
    assert _error_kind(_FakeConnectionError()) == "connection"
    # 4xx khác và lỗi lập trình: không retry
    assert _error_kind(_FakeAPIError(400)) is None
    assert _error_kind(_FakeAPIError(401)) is None
    assert _error_kind(ValueError("sai schema")) is None


# ── Retry + backoff ────────────────────────────────────────────────────────

def test_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeAPIError(500)
        return "ok"

    assert _with_retries(flaky) == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]  # backoff nhân đôi


def test_retry_exhausted_raises_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def always_rate_limited() -> None:
        raise _FakeAPIError(429)

    with pytest.raises(LLMUnavailable) as excinfo:
        _with_retries(always_rate_limited)

    assert excinfo.value.kind == "rate_limit"
    assert excinfo.value.attempts == 4  # 1 lần gọi + 3 retry
    assert sleeps == [1.0, 2.0, 4.0]
    assert isinstance(excinfo.value.cause, _FakeAPIError)


def test_non_retryable_error_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def bad_request() -> None:
        raise _FakeAPIError(401)

    with pytest.raises(_FakeAPIError):
        _with_retries(bad_request)
    assert sleeps == []  # không tốn một lần retry nào


# ── Checkpoint + resume theo job_id ────────────────────────────────────────

@pytest.fixture()
def stub_nodes(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Thay node thật bằng stub đếm số lần chạy.

    generate_per_section hỏng đúng 1 lần đầu (LLMUnavailable) rồi chạy được —
    mô phỏng job bị ngắt giữa chừng vì provider sập.
    """
    counters = {"parse_input": 0, "generate_per_section": 0}

    def parse_input(state):
        counters["parse_input"] += 1
        return {"chapters": [{"id": "1"}]}

    def check_complete(state):
        return {"missing": []}

    def route_reference_rfp(state):
        return {"reference_rfp": {"rfp_id": "R", "method": "stub", "score": None}}

    def plan_sections(state):
        return {"sections": []}

    def retrieve_per_chapter(state):
        return {}

    def generate_per_section(state):
        counters["generate_per_section"] += 1
        if counters["generate_per_section"] == 1:
            raise LLMUnavailable("rate_limit", 4, _FakeAPIError(429))
        return {"sections": [{"key": "stub"}]}

    def assemble(state):
        return {"status": "completed", "proposal": "stub proposal"}

    for name, fn in [
        ("parse_input", parse_input),
        ("check_complete", check_complete),
        ("route_reference_rfp", route_reference_rfp),
        ("plan_sections", plan_sections),
        ("retrieve_per_chapter", retrieve_per_chapter),
        ("generate_per_section", generate_per_section),
        ("assemble", assemble),
    ]:
        monkeypatch.setattr(graph_module, name, fn)
    return counters


def test_checkpoint_resume_after_llm_outage(
    tmp_path: Path, stub_nodes: dict[str, int]
) -> None:
    db = tmp_path / "checkpoints.sqlite"

    first = run_graph("dummy rfp", job_id="job-1", checkpoint_db=db)
    assert first["status"] == "partial"
    assert first["error"]["kind"] == "rate_limit"
    assert first["job_id"] == "job-1"
    # State các node trước điểm hỏng được checkpoint lại
    assert first["chapters"] == [{"id": "1"}]
    assert stub_nodes["parse_input"] == 1

    second = run_graph("dummy rfp", job_id="job-1", resume=True, checkpoint_db=db)
    assert second["status"] == "completed"
    assert second["proposal"] == "stub proposal"
    # Resume KHÔNG chạy lại node đã xong, chỉ chạy lại node hỏng
    assert stub_nodes["parse_input"] == 1
    assert stub_nodes["generate_per_section"] == 2

    # Resume job đã hoàn tất: trả state đã lưu, không chạy lại gì
    third = run_graph("dummy rfp", job_id="job-1", resume=True, checkpoint_db=db)
    assert third["status"] == "completed"
    assert stub_nodes["generate_per_section"] == 2


def test_fresh_job_id_runs_from_start(
    tmp_path: Path, stub_nodes: dict[str, int]
) -> None:
    db = tmp_path / "checkpoints.sqlite"
    stub_nodes["generate_per_section"] = 5  # qua ngưỡng hỏng, chạy thẳng

    result = run_graph("dummy rfp", job_id="job-moi", resume=True, checkpoint_db=db)
    # resume=True nhưng job chưa có checkpoint -> chạy mới từ đầu, không lỗi
    assert result["status"] == "completed"
    assert stub_nodes["parse_input"] == 1


# ── Trả partial thay vì crash (đường không checkpoint) ─────────────────────

class _StubStreamGraph:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def stream(self, state, stream_mode="values"):
        yield {"input_text": state["input_text"], "chapters": [{"id": "1"}]}
        raise self._error


def test_run_graph_returns_partial_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outage = LLMUnavailable("timeout", 4, _FakeTimeoutError())
    monkeypatch.setattr(graph_module, "GRAPH", _StubStreamGraph(outage))

    result = run_graph("dummy rfp")

    assert result["status"] == "partial"
    assert result["error"]["kind"] == "timeout"
    assert result["error"]["attempts"] == 4
    # Kết quả các node đã xong không bị mất
    assert result["chapters"] == [{"id": "1"}]


def test_run_graph_eval_returns_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    outage = LLMUnavailable("server_error", 4, _FakeAPIError(502))
    monkeypatch.setattr(graph_module, "GRAPH", _StubStreamGraph(outage))

    result = run_graph_eval("dummy rfp")

    assert result["status"] == "partial"
    assert result["error"]["kind"] == "server_error"


def test_run_graph_does_not_swallow_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_module, "GRAPH", _StubStreamGraph(ValueError("bug thật"))
    )
    with pytest.raises(ValueError, match="bug thật"):
        run_graph("dummy rfp")
