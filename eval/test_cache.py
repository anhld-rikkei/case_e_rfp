"""Test cache + invalidation (Bước 5, v1.2).

Cách ly hoàn toàn: không gọi mạng, không build FAISS. Node của graph được thay
bằng stub đếm số lệnh gọi LLM giả lập, nên "lần 2 không gọi LLM" là khẳng định
đo được chứ không phải suy đoán.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.cache as cache_module  # noqa: E402
import rfp.graph as graph_module  # noqa: E402
from rfp.cache import (  # noqa: E402
    Cache,
    capability_fingerprint,
    corpus_fingerprint,
    make_key,
)
from rfp.schema import Sentence  # noqa: E402


def _sentence(sent_id: str, text: str) -> Sentence:
    return Sentence(
        sent_id=sent_id,
        proposal_id="PROP-001",
        responds_to="RFP-2025-001",
        section="技術要件",
        text=text,
        claim_kind="capability",
        flags={},
    )


def _fake_index(sentences, *, leak=0, quarantine=0):
    return SimpleNamespace(
        sentences=sentences,
        leak_quarantine=[None] * leak,
        capability_quarantine=[None] * quarantine,
    )


def _key(**overrides):
    base = dict(
        rfp_text="RFP nội dung",
        model="gpt-5.4-mini",
        corpus="corpus-fp",
        capability="cap-fp",
        scope="generate_per_section",
    )
    base.update(overrides)
    return make_key(**base)


# ── Cache key đổi khi và chỉ khi có thứ thật sự đổi ────────────────────────

def test_same_inputs_give_same_key() -> None:
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("rfp_text", "RFP khác"),
        ("model", "gpt-4o-mini"),
        ("corpus", "corpus-khac"),
        ("capability", "cap-khac"),
        ("scope", "retrieve_per_chapter"),
    ],
)
def test_key_changes_when_component_changes(field: str, value: str) -> None:
    assert _key(**{field: value}) != _key()


def test_key_changes_when_prompt_version_bumped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _key()
    # Bump TƯƠNG ĐỐI, không hardcode: giá trị thật đổi theo thời gian và một
    # test patch trúng đúng giá trị hiện tại sẽ thành no-op mà vẫn xanh.
    monkeypatch.setattr(
        cache_module, "PROMPT_VERSION", cache_module.PROMPT_VERSION + "-bumped"
    )
    assert _key() != before


def test_key_changes_when_template_version_bumped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _key()
    monkeypatch.setattr(
        cache_module, "TEMPLATE_VERSION", cache_module.TEMPLATE_VERSION + "-bumped"
    )
    assert _key() != before


@pytest.mark.parametrize(
    "flag", ["RETRIEVAL_USE_BM25", "RETRIEVAL_USE_RERANK", "RETRIEVAL_USE_MMR"]
)
def test_key_changes_when_retrieval_flag_flipped(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _key()
    monkeypatch.setattr(cache_module, flag, False)
    assert _key() != before


def test_key_changes_when_precedents_per_chapter_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _key()
    monkeypatch.setattr(cache_module, "PRECEDENTS_PER_CHAPTER", 5)
    assert _key() != before


# ── Vân tay corpus: băm nội dung, không băm mtime ──────────────────────────

def test_corpus_fingerprint_stable_for_same_content() -> None:
    left = _fake_index([_sentence("S1", "文A"), _sentence("S2", "文B")])
    # Cùng nội dung, khác THỨ TỰ -> vân tay không đổi (manifest đã sort).
    right = _fake_index([_sentence("S2", "文B"), _sentence("S1", "文A")])
    assert corpus_fingerprint(left) == corpus_fingerprint(right)


def test_corpus_fingerprint_changes_when_document_added() -> None:
    before = _fake_index([_sentence("S1", "文A")])
    after = _fake_index([_sentence("S1", "文A"), _sentence("S2", "文B")])
    assert corpus_fingerprint(before) != corpus_fingerprint(after)


def test_corpus_fingerprint_changes_when_sentence_edited() -> None:
    before = _fake_index([_sentence("S1", "文A")])
    after = _fake_index([_sentence("S1", "文A（修正）")])
    assert corpus_fingerprint(before) != corpus_fingerprint(after)


def test_corpus_fingerprint_changes_when_quarantine_count_changes() -> None:
    """Nới/siết blocklist đổi vân tay, dù tập câu sạch tình cờ không đổi."""
    before = _fake_index([_sentence("S1", "文A")], quarantine=4)
    after = _fake_index([_sentence("S1", "文A")], quarantine=3)
    assert corpus_fingerprint(before) != corpus_fingerprint(after)


def test_capability_fingerprint_changes_when_sheet_edited(tmp_path: Path) -> None:
    sheet = tmp_path / "capability_sheet.json"
    sheet.write_text(json.dumps({"headcount": 1800}), encoding="utf-8")
    before = capability_fingerprint(sheet)
    sheet.write_text(json.dumps({"headcount": 1900}), encoding="utf-8")
    assert capability_fingerprint(sheet) != before


def test_capability_fingerprint_ignores_key_order(tmp_path: Path) -> None:
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    right.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
    assert capability_fingerprint(left) == capability_fingerprint(right)


# ── Hành vi Cache: TTL, force_regen, file hỏng ─────────────────────────────

def test_put_then_get_round_trip(tmp_path: Path) -> None:
    cache = Cache(tmp_path, enabled=True)
    cache.put("k1", {"sections": ["x"]})
    assert cache.get("k1") == {"sections": ["x"]}


def test_get_misses_after_ttl_expires(tmp_path: Path) -> None:
    cache = Cache(tmp_path, ttl_seconds=0, enabled=True)
    cache.put("k1", {"v": 1})
    assert cache.get("k1") is None


def test_force_regen_skips_read_but_still_writes(tmp_path: Path) -> None:
    cache = Cache(tmp_path, enabled=True)
    cache.put("k1", {"v": "cũ"})
    assert cache.get("k1", force_regen=True) is None
    cache.put("k1", {"v": "mới"})
    assert cache.get("k1") == {"v": "mới"}


def test_disabled_cache_neither_reads_nor_writes(tmp_path: Path) -> None:
    cache = Cache(tmp_path, enabled=False)
    cache.put("k1", {"v": 1})
    assert cache.get("k1") is None
    assert list(tmp_path.glob("*.json")) == []


def test_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = Cache(tmp_path, enabled=True)
    cache.put("k1", {"v": 1})
    (tmp_path / "k1.json").write_text("{ khong phai json", encoding="utf-8")
    assert cache.get("k1") is None


# ── Hit trọn gói qua graph: lần 2 không gọi LLM ────────────────────────────

@pytest.fixture()
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Thay node thật bằng stub đếm 'lệnh gọi LLM'."""
    calls = {"retrieve": 0, "generate": 0}
    index = _fake_index([_sentence("S1", "文A")])

    monkeypatch.setattr(
        graph_module.SentenceIndex,
        "build",
        classmethod(lambda cls, *a, **k: index),
    )
    monkeypatch.setattr(
        graph_module, "capability_fingerprint", lambda *a, **k: "cap-fp"
    )
    monkeypatch.setattr(graph_module.llm, "MODEL", "gpt-5.4-mini")

    real_retrieve_body = {"chapters": [{"id": "1"}], "trace": {"llm_calls": 0}}
    real_generate_body = {"sections": [{"key": "s"}], "trace": {"llm_calls": 3}}

    def fake_retrieve_work():
        calls["retrieve"] += 1
        return dict(real_retrieve_body)

    def fake_generate_work():
        calls["generate"] += 1
        return dict(real_generate_body)

    def retrieve(state):
        sentence_index = graph_module.SentenceIndex.build()
        fingerprints = {
            "corpus": graph_module.corpus_fingerprint(sentence_index),
            "capability": graph_module.capability_fingerprint(),
        }
        state = {**state, "fingerprints": fingerprints}
        cache = graph_module._node_cache()
        key = graph_module._cache_key_for(state, "retrieve_per_chapter")
        cached = cache.get(key, force_regen=bool(state.get("force_regen")))
        if cached is not None:
            return {**cached, "fingerprints": fingerprints}
        result = fake_retrieve_work()
        cache.put(key, result)
        return {**result, "fingerprints": fingerprints}

    def generate(state):
        cache = graph_module._node_cache()
        key = graph_module._cache_key_for(state, "generate_per_section")
        cached = cache.get(key, force_regen=bool(state.get("force_regen")))
        if cached is not None:
            return cached
        result = fake_generate_work()
        cache.put(key, result)
        return result

    return {"calls": calls, "retrieve": retrieve, "generate": generate}


def _run(stub, cache, text="RFP thử", force_regen=False):
    with graph_module.use_cache(cache):
        state = {"input_text": text, "force_regen": force_regen}
        state.update(stub["retrieve"](state))
        state.update(stub["generate"](state))
    return state


def test_second_run_is_full_cache_hit_with_zero_llm_calls(
    tmp_path: Path, stub_pipeline: dict
) -> None:
    cache = Cache(tmp_path, enabled=True)
    calls = stub_pipeline["calls"]

    first = _run(stub_pipeline, cache)
    assert calls == {"retrieve": 1, "generate": 1}

    second = _run(stub_pipeline, cache)
    # Lần 2: không một lệnh gọi nào chạy lại, kết quả y hệt.
    assert calls == {"retrieve": 1, "generate": 1}
    assert second["sections"] == first["sections"]
    assert second["chapters"] == first["chapters"]


def test_changed_rfp_text_misses_cache(tmp_path: Path, stub_pipeline: dict) -> None:
    cache = Cache(tmp_path, enabled=True)
    _run(stub_pipeline, cache, text="RFP A")
    _run(stub_pipeline, cache, text="RFP B")
    assert stub_pipeline["calls"] == {"retrieve": 2, "generate": 2}


def test_force_regen_reruns_both_nodes(tmp_path: Path, stub_pipeline: dict) -> None:
    cache = Cache(tmp_path, enabled=True)
    _run(stub_pipeline, cache)
    _run(stub_pipeline, cache, force_regen=True)
    assert stub_pipeline["calls"] == {"retrieve": 2, "generate": 2}


def test_added_document_invalidates_cache(
    tmp_path: Path, stub_pipeline: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingest thêm tài liệu -> vân tay corpus đổi -> cache miss."""
    cache = Cache(tmp_path, enabled=True)
    _run(stub_pipeline, cache)
    assert stub_pipeline["calls"] == {"retrieve": 1, "generate": 1}

    bigger = _fake_index([_sentence("S1", "文A"), _sentence("S2", "文B mới")])
    monkeypatch.setattr(
        graph_module.SentenceIndex,
        "build",
        classmethod(lambda cls, *a, **k: bigger),
    )
    _run(stub_pipeline, cache)
    assert stub_pipeline["calls"] == {"retrieve": 2, "generate": 2}


def test_changed_capability_sheet_invalidates_cache(
    tmp_path: Path, stub_pipeline: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(tmp_path, enabled=True)
    _run(stub_pipeline, cache)

    monkeypatch.setattr(
        graph_module, "capability_fingerprint", lambda *a, **k: "cap-fp-moi"
    )
    _run(stub_pipeline, cache)
    assert stub_pipeline["calls"] == {"retrieve": 2, "generate": 2}


# ── Đường eval không được đụng cache ───────────────────────────────────────

def test_eval_path_neither_reads_nor_writes_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_graph_eval ép Cache(enabled=False) bất kể settings bật cache."""
    seen: list[str] = []

    class SpyCache(Cache):
        def get(self, key, *, force_regen=False):
            seen.append("get")
            return super().get(key, force_regen=force_regen)

        def put(self, key, value):
            seen.append("put")
            return super().put(key, value)

    # Nếu run_graph_eval quên ép tắt, nó sẽ rơi vào cache mặc định này.
    monkeypatch.setattr(
        graph_module, "_ACTIVE_CACHE", SpyCache(tmp_path, enabled=True)
    )

    captured: list[Cache] = []

    class StubGraph:
        def stream(self, state, stream_mode="values"):
            captured.append(graph_module._node_cache())
            yield {"status": "completed", "proposal": "x"}

    monkeypatch.setattr(graph_module, "GRAPH", StubGraph())
    graph_module.run_graph_eval("RFP thử")

    assert captured and captured[0].enabled is False
    assert seen == []
    assert list(tmp_path.glob("*.json")) == []
