"""Test chat-refine (v1.6 · triết lý v1.7).

Cách ly: mock `structured`, không mạng, không dựng index.

Từ v1.7 chat **làm theo ý người dùng rồi dán nhãn** thay vì chặn: thêm câu, đổi
số liệu, viết lại tự do đều được thực hiện, và câu nào không còn trung thành với
nguồn thì mang `origin="user"` / `verdict="USER_PROVIDED"`. Lằn ranh duy nhất
không nhân nhượng là blocklist chứng chỉ/năng lực cấm — các test red-team ở đây
giữ nguyên. Ngoài ra: không đụng cache, và eval không có đường đi qua chat.
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


def _mock_plan(
    monkeypatch: pytest.MonkeyPatch,
    edits: list[SentenceEdit],
    *,
    added: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        refine_module,
        "structured",
        lambda *a, **k: RefinePlan(edits=edits, added=list(added or [])),
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


def test_number_change_is_applied_then_labeled(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1.7: đổi số theo chỉ thị được LÀM THẬT, nhưng câu mang nhãn user.

    Câu đã đổi số không còn là thứ nguồn nói, nên nó không được đội lốt câu có
    căn cứ — nhưng cũng không bị chặn, vì người dùng chịu trách nhiệm.
    """
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

    assert result.changed == 1
    edited = result.state["sections"][0]["sentences"][1]
    assert edited["text"] == "在庫精度を55%向上した実績があります。"
    assert edited["origin"] == "user"
    assert edited["verdict"] == "USER_PROVIDED"
    assert edited["source_id"] is None


def test_added_sentence_is_applied_and_labeled(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1.7: chat THÊM được câu, câu đó mang nhãn user."""
    _mock_plan(monkeypatch, [], added=["全く新しい主張です。"])
    result = refine_section(_state(), section_key="technical", instruction="thêm câu")

    assert result.added == 1
    sentences = result.state["sections"][0]["sentences"]
    assert len(sentences) == 3
    new = sentences[-1]
    assert new["text"] == "全く新しい主張です。"
    assert new["origin"] == "user" and new["verdict"] == "USER_PROVIDED"


def test_out_of_range_index_becomes_an_added_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model dùng chỉ mục ngoài dải để thêm câu — gom vào cùng đường `added`."""
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=99, action="rewrite", text="追加の主張です。")],
    )
    result = refine_section(_state(), section_key="technical", instruction="thêm câu")

    assert result.added == 1
    assert result.state["sections"][0]["sentences"][-1]["origin"] == "user"


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_rewrite_is_skipped(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_plan(monkeypatch, [SentenceEdit(index=0, action="rewrite", text=bad)])
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.changed == 0 and result.rejected


def test_decoration_is_stripped_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ký tự trang trí là lỗi định dạng của model, không phải ý người dùng."""
    _mock_plan(
        monkeypatch, [SentenceEdit(index=0, action="rewrite", text="【装飾】文です。")]
    )
    result = refine_section(_state(), section_key="technical", instruction="x")
    assert result.changed == 1
    assert "【" not in result.state["sections"][0]["sentences"][0]["text"]


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


def test_contradicted_sentence_is_labeled_not_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1.7: claim-check CONTRADICTED chuyển câu sang nhãn user, không gỡ."""
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

    sentences = result.state["sections"][0]["sentences"]
    kept = next(s for s in sentences if s["text"] == "怪しい文です。")
    assert kept["origin"] == "user" and kept["verdict"] == "USER_PROVIDED"


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


# ── Ghim câu người dùng (v1.7) ───────────────────────────────────────────

def _state_with_user(text: str = "BIツールとPL-300資格で対応します。") -> dict[str, Any]:
    from rfp.refine import user_sentence

    state = _state()
    state["sections"][0]["sentences"].append(user_sentence(text, req_ids=["3.1"]))
    return state


def test_user_sentence_is_pinned_by_default() -> None:
    from rfp.refine import user_sentence

    assert user_sentence("x")["pinned"] is True


def test_pinned_sentence_survives_chat_about_another_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lỗi thật: thêm câu ở lượt 2, lượt 3 nói về mục khác thì model xoá mất."""
    state = _state_with_user()
    # Model bảo drop đúng câu người dùng đã ghim (index 2)
    _mock_plan(monkeypatch, [SentenceEdit(index=2, action="drop")])
    result = refine_section(
        state, section_key="technical", instruction="viết lại phần 2.1 và 2.2"
    )

    texts = [item["text"] for item in result.state["sections"][0]["sentences"]]
    assert "BIツールとPL-300資格で対応します。" in texts
    assert result.restored == 1


def test_pinned_sentence_protected_from_rewrite_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state_with_user()
    _mock_plan(
        monkeypatch, [SentenceEdit(index=2, action="rewrite", text="別の文に変えた。")]
    )
    result = refine_section(state, section_key="technical", instruction="ngắn hơn")

    texts = [item["text"] for item in result.state["sections"][0]["sentences"]]
    assert "BIツールとPL-300資格で対応します。" in texts
    assert "別の文に変えた。" not in texts
    assert result.restored == 1


def test_instruction_naming_the_content_can_edit_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nhắm đích danh thì sửa được — ghim không phải khoá vĩnh viễn."""
    state = _state_with_user()
    _mock_plan(
        monkeypatch,
        [SentenceEdit(index=2, action="rewrite", text="BIツールで対応します。")],
    )
    result = refine_section(
        state, section_key="technical", instruction="sửa câu về PL-300 cho gọn"
    )

    texts = [item["text"] for item in result.state["sections"][0]["sentences"]]
    assert "BIツールで対応します。" in texts
    assert result.restored == 0


def test_unpinned_sentence_can_be_edited(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state_with_user()
    state["sections"][0]["sentences"][2]["pinned"] = False
    _mock_plan(monkeypatch, [SentenceEdit(index=2, action="drop")])
    result = refine_section(state, section_key="technical", instruction="bỏ bớt")

    assert result.dropped == 1 and result.restored == 0


def test_instruction_targets_matches_distinctive_tokens() -> None:
    from rfp.refine import instruction_targets

    text = "BIツールとPL-300資格で対応します。"
    assert instruction_targets(text, "sửa câu về PL-300")
    assert instruction_targets(text, "câu BI viết lại đi")
    # Từ tiếng Nhật 2 chữ quá phổ biến -> KHÔNG được coi là nhắm đích danh
    assert not instruction_targets(text, "viết lại phần 2.1 và 2.2")
    assert not instruction_targets(text, "")


def test_restored_count_is_reported_not_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Không bao giờ mất im lặng: có khôi phục thì phải đếm được."""
    state = _state_with_user()
    _mock_plan(
        monkeypatch,
        [
            SentenceEdit(index=0, action="rewrite", text="短くしました。"),
            SentenceEdit(index=2, action="drop"),
        ],
    )
    result = refine_section(state, section_key="technical", instruction="ngắn hơn")
    assert result.restored == 1
    assert result.changed == 1  # thay đổi hợp lệ vẫn được áp dụng


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

    for key in ("forbidden", "instruction_forbidden"):
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
