"""Test chat-refine (v1.6).

Cách ly: mock `structured`, không mạng, không dựng index. Trọng tâm là các chốt
cứng — lưới an toàn không có ngoại lệ cho chat, chat không thêm được câu, không
đụng cache, và eval không có đường đi qua chat.
"""
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.refine as refine_module  # noqa: E402
from rfp.refine import (  # noqa: E402
    RefinePlan,
    SentenceEdit,
    refine_all,
    refine_section,
    section_sources,
)


def _sentence(text: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "text": text,
        "origin": "precedent",
        "source_id": "PROP-001-S01",
        "req_ids": ["3.1"],
        "verdict": "VERIFIED",
    }
    base.update(overrides)
    return base


def _state(sentences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sentences = sentences or [
        _sentence("基幹システム構築の実績があります。"),
        _sentence("在庫精度を20%向上した実績があります。", source_id="PROP-001-S02"),
    ]
    return {
        "chapters": [
            {
                "id": "3",
                "title": "技術要件",
                "requirements": [{"req_id": "3.1", "text": "…"}],
                "retrieval": {
                    "selected": [
                        {"sent_id": "PROP-001-S01", "text": "基幹システム構築の実績。"},
                        {
                            "sent_id": "PROP-001-S02",
                            "text": "在庫精度を20%向上した実績があります。",
                        },
                    ]
                },
            }
        ],
        "sections": [
            {
                "key": "technical",
                "title_ja": "技術要件への対応",
                "title_vi": "Đáp ứng yêu cầu kỹ thuật",
                "source_chapters": ["3"],
                "status": "OK",
                "note": "",
                "sentences": sentences,
            }
        ],
        "proposal": "cũ",
    }


def _mock_plan(monkeypatch: pytest.MonkeyPatch, edits: list[SentenceEdit]) -> None:
    monkeypatch.setattr(
        refine_module, "structured", lambda *a, **k: RefinePlan(edits=edits)
    )
    # Claim-check giữ nguyên mọi câu, để test tách bạch phần chat.
    monkeypatch.setattr(
        refine_module,
        "check_claims",
        lambda sentences, **k: (
            [{**s, "verdict": "VERIFIED"} for s in sentences],
            {},
            len(sentences),
            [],
        ),
    )


# ── Cơ chế cơ bản ─────────────────────────────────────────────────────────

def test_rewrite_updates_text_and_reassembles_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="短くしました。")])
    result = refine_section(_state(), section_key="technical", instruction="ngắn hơn")

    assert result.changed == 1
    assert result.state["sections"][0]["sentences"][0]["text"] == "短くしました。"
    assert "短くしました。" in result.state["proposal"]


def test_keep_leaves_sentence_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="keep")])
    result = refine_section(state, section_key="technical", instruction="x")

    assert not result.touched
    assert result.state is state  # không dựng state mới khi không đổi gì


def test_drop_removes_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=1, action="drop")])
    result = refine_section(_state(), section_key="technical", instruction="bỏ câu 2")

    assert result.dropped == 1
    assert len(result.state["sections"][0]["sentences"]) == 1


def test_unknown_section_is_a_no_op() -> None:
    state = _state()
    result = refine_section(state, section_key="khong-ton-tai", instruction="x")
    assert result.state is state and result.llm_calls == 0


def test_original_state_is_never_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    before = state["sections"][0]["sentences"][0]["text"]
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="別の文。")])
    refine_section(state, section_key="technical", instruction="x")
    assert state["sections"][0]["sentences"][0]["text"] == before


# ── RED-TEAM: chỉ thị người dùng là input không tin cậy ───────────────────

def test_chat_cannot_inject_forbidden_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat đòi thêm ISO 27017 -> bị chặn, hồ sơ giữ nguyên, KHÔNG crash."""
    _mock_plan(
        monkeypatch,
        [
            SentenceEdit(
                index=0,
                action="rewrite",
                text="当社はISO/IEC 27017認証を取得しています。",
            )
        ],
    )
    # Chỉ thị VÔ HẠI để đi qua tiền kiểm — ca này kiểm nhánh model tự chèn
    # chuỗi cấm vào bản viết lại (tiền kiểm chỉ thị có test riêng bên dưới).
    state = _state()
    result = refine_section(
        state, section_key="technical", instruction="viết trang trọng hơn"
    )

    assert result.changed == 0
    assert result.state["sections"][0]["sentences"][0]["text"] == (
        state["sections"][0]["sentences"][0]["text"]
    )
    assert any(
        "không có trong bảng năng lực" in item["reason"] for item in result.rejected
    )


def test_chat_cannot_inject_forbidden_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Biến thể 'ISO 27017' (không có IEC) cũng phải bị chặn — v1.1."""
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=0, action="rewrite", text="ISO 27017認証を取得済みです。")],
    )
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.changed == 0 and result.rejected


def test_chat_cannot_change_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(
        monkeypatch,
        [
            SentenceEdit(
                index=1,
                action="rewrite",
                text="在庫精度を55%向上した実績があります。",
            )
        ],
    )
    result = refine_section(_state(), section_key="technical", instruction="nói mạnh hơn")

    assert result.changed == 0
    assert "số liệu" in " ".join(item["reason"] for item in result.rejected)


def test_chat_cannot_add_new_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Câu mới cần source_id mà không ai cấp được -> từ chối (BB-4)."""
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=99, action="rewrite", text="全く新しい主張です。")],
    )
    result = refine_section(_state(), section_key="technical", instruction="thêm câu")

    assert result.changed == 0
    assert len(result.state["sections"][0]["sentences"]) == 2
    assert any("không thêm được câu mới" in item["reason"] for item in result.rejected)


@pytest.mark.parametrize("bad", ["", "   ", "【装飾】文です。"])
def test_empty_or_decorated_rewrite_is_rejected(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text=bad)])
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.changed == 0 and result.rejected


# ── BB-4: sửa câu chữ không được làm mất dấu vết nguồn ───────────────────

def test_rewrite_preserves_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="整えた文です。")])
    result = refine_section(_state(), section_key="technical", instruction="x")

    sentence = result.state["sections"][0]["sentences"][0]
    assert sentence["origin"] == "precedent"
    assert sentence["source_id"] == "PROP-001-S01"
    assert sentence["req_ids"] == ["3.1"]


def test_rewritten_sentence_goes_back_through_claim_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[dict[str, Any]]] = []

    monkeypatch.setattr(
        refine_module,
        "structured",
        lambda *a, **k: RefinePlan(
            edits=[SentenceEdit(index=0, action="rewrite", text="整えた文です。")]
        ),
    )

    def spy_check(sentences, **kwargs):
        seen.append(sentences)
        return ([{**s, "verdict": "VERIFIED"} for s in sentences], {}, 1, [])

    monkeypatch.setattr(refine_module, "check_claims", spy_check)
    result = refine_section(_state(), section_key="technical", instruction="x")

    assert seen and seen[0][0]["text"] == "整えた文です。"
    assert result.state["sections"][0]["sentences"][0]["verdict"] == "VERIFIED"


def test_contradicted_sentence_is_removed_after_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        refine_module,
        "structured",
        lambda *a, **k: RefinePlan(
            edits=[SentenceEdit(index=0, action="rewrite", text="怪しい文です。")]
        ),
    )
    monkeypatch.setattr(
        refine_module,
        "check_claims",
        lambda sentences, **k: ([], {}, 1, list(sentences)),  # tất cả CONTRADICTED
    )
    result = refine_section(_state(), section_key="technical", instruction="x")

    texts = [s["text"] for s in result.state["sections"][0]["sentences"]]
    assert "怪しい文です。" not in texts
    assert any("mâu thuẫn" in item["reason"] for item in result.rejected)


# ── Kho nguồn: đúng những gì đã truy xuất, không thêm ────────────────────

def test_section_sources_come_from_existing_retrieval_only() -> None:
    state = _state()
    sources = section_sources(state, state["sections"][0])
    assert set(sources) == {"PROP-001-S01", "PROP-001-S02"}


def test_section_sources_ignore_other_chapters() -> None:
    state = _state()
    state["chapters"].append(
        {
            "id": "9",
            "title": "別章",
            "requirements": [],
            "retrieval": {"selected": [{"sent_id": "PROP-999-S01", "text": "khác"}]},
        }
    )
    sources = section_sources(state, state["sections"][0])
    assert "PROP-999-S01" not in sources


# ── Cache: đường chat không đọc và không ghi ─────────────────────────────

def test_refine_never_touches_the_node_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rfp.cache import Cache
    import rfp.graph as graph_module

    spy = Cache(tmp_path, enabled=True)
    monkeypatch.setattr(graph_module, "_ACTIVE_CACHE", spy)
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="整えた文。")])

    refine_section(_state(), section_key="technical", instruction="x")

    assert list(tmp_path.glob("*.json")) == []


def test_refine_module_does_not_import_cache() -> None:
    """Ngăn cách cấu trúc, không phải kỷ luật con người."""
    source = (PROJECT_ROOT / "src" / "rfp" / "refine.py").read_text(encoding="utf-8")
    assert "from .cache import" not in source
    assert "import cache" not in source


# ── Eval không có đường đi qua chat ──────────────────────────────────────

def test_graph_does_not_import_refine() -> None:
    source = (PROJECT_ROOT / "src" / "rfp" / "graph.py").read_text(encoding="utf-8")
    assert "refine" not in source


def test_eval_modules_do_not_import_refine() -> None:
    for name in ("run_ragas.py", "run_ablation.py", "to_samples.py"):
        source = (PROJECT_ROOT / "eval" / name).read_text(encoding="utf-8")
        assert "refine" not in source, name


def test_refine_is_not_a_graph_node() -> None:
    from rfp.graph import PIPELINE_STAGES

    assert not any("refine" in stage for stage in PIPELINE_STAGES)


# ── Ba kết cục phải phân biệt được ───────────────────────────────────────

def test_outcome_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="整えた文です。")])
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.outcome == "changed"
    assert (result.changed, result.kept, result.blocked) == (1, 1, 0)


def test_outcome_no_change_when_model_keeps_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ca (a): chỉ thị không khớp nội dung mục, model trả keep hết."""
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=0, action="keep"), SentenceEdit(index=1, action="keep")],
    )
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.outcome == "no_change"
    assert result.kept == 2 and result.blocked == 0


def test_outcome_blocked_when_safety_net_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ca (b): có edit nhưng bị lưới an toàn chặn — KHÁC ca (a)."""
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=0, action="rewrite", text="ISO 27017認証があります。")],
    )
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.outcome == "blocked"
    assert result.blocked == 1 and result.changed == 0


def test_counts_add_up_to_sentence_total(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(
        monkeypatch,
        [
            SentenceEdit(index=0, action="rewrite", text="整えた文です。"),
            SentenceEdit(index=1, action="drop"),
        ],
    )
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.changed + result.dropped + result.kept == 2


# ── Tiền kiểm chỉ thị: chặn trước khi tốn lệnh gọi LLM ───────────────────

def test_instruction_asking_for_forbidden_cert_is_blocked_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("chỉ thị đòi chuỗi cấm mà vẫn gọi LLM")

    monkeypatch.setattr(refine_module, "structured", explode)
    result = refine_section(
        _state(), section_key="technical", instruction="thêm chứng chỉ ISO 27017 vào"
    )

    assert result.outcome == "blocked"
    assert result.llm_calls == 0
    assert "công ty không có" in result.rejected[0]["reason"]
    assert "capability_sheet.json" in result.rejected[0]["reason"]


def test_instruction_conflicts_detects_variants() -> None:
    from rfp.refine import instruction_conflicts

    assert instruction_conflicts("thêm ISO 27017")
    assert instruction_conflicts("nói là có ISO27018 đi")
    assert instruction_conflicts("") == []
    assert instruction_conflicts("viết ngắn gọn hơn") == []


def test_reason_text_tells_user_how_to_fix() -> None:
    """Lý do phải nói được bước tiếp theo, không chỉ 'bị từ chối'."""
    from rfp.refine import REASON_VI

    for key in ("forbidden", "added", "instruction_forbidden"):
        assert "capability_sheet.json" in REASON_VI[key]
        assert "Nạp lại kho" in REASON_VI[key]


def test_refine_all_does_not_repeat_the_same_instruction_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chỉ thị đòi chuỗi cấm bị chặn ở mọi mục — chỉ báo một lần."""
    state = _state()
    state["sections"].append(
        {
            "key": "security",
            "title_ja": "セキュリティ要件",
            "title_vi": "Bảo mật",
            "source_chapters": ["3"],
            "status": "OK",
            "note": "",
            "sentences": [_sentence("セキュリティ体制を整備しています。")],
        }
    )
    result = refine_all(state, instruction="thêm ISO 27017")
    assert result.llm_calls == 0
    assert len(result.rejected) == 1


# ── refine_all: cùng chỉ thị cho mọi mục ─────────────────────────────────

def test_refine_all_visits_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    state["sections"].append(
        {
            "key": "security",
            "title_ja": "セキュリティ要件",
            "title_vi": "Bảo mật",
            "source_chapters": ["3"],
            "status": "OK",
            "note": "",
            "sentences": [_sentence("セキュリティ体制を整備しています。")],
        }
    )
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text="整えた文です。")])

    result = refine_all(state, instruction="ngắn hơn")

    assert result.llm_calls >= 2  # mỗi mục ít nhất một lệnh gọi
    assert result.changed == 2
