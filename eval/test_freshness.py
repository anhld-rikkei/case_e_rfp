"""Test đối chiếu dữ liệu nguồn với lần nạp gần nhất (v1.5).

Cách ly: thư mục tạm, không mạng, không dựng FAISS.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

from rfp.freshness import (  # noqa: E402
    CAPABILITY_KEY,
    SourceDiff,
    compare,
    current_diff,
    load_manifest,
    quarantine_report,
    scan_sources,
    state_is_stale,
    write_manifest,
)
from rfp.schema import Sentence  # noqa: E402


def _corpus(tmp_path: Path, proposals: dict[str, str], rfps: dict[str, str]):
    proposal_dir = tmp_path / "proposals"
    rfp_dir = tmp_path / "rfps"
    proposal_dir.mkdir(exist_ok=True)
    rfp_dir.mkdir(exist_ok=True)
    for name, text in proposals.items():
        (proposal_dir / name).write_text(text, encoding="utf-8")
    for name, text in rfps.items():
        (rfp_dir / name).write_text(text, encoding="utf-8")
    capability = tmp_path / "capability_sheet.json"
    capability.write_text(json.dumps({"headcount": 1800}), encoding="utf-8")
    return {
        "proposal_dir": proposal_dir,
        "rfp_dir": rfp_dir,
        "capability_path": capability,
    }


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


def _fake_index(sentences, *, quarantine=(), leak=()):
    return SimpleNamespace(
        sentences=list(sentences),
        capability_quarantine=list(quarantine),
        leak_quarantine=list(leak),
        cleanliness=lambda: {
            "indexed": len(sentences),
            "capability_quarantine": len(quarantine),
            "leak_quarantine": len(leak),
        },
    )


# ── scan_sources: băm nội dung, không dùng mtime ──────────────────────────

def test_scan_lists_every_source_file(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a", "P2.txt": "b"}, {"R1.txt": "c"})
    files = scan_sources(**kwargs)
    assert set(files) == {"proposals/P1.txt", "proposals/P2.txt", "rfps/R1.txt", CAPABILITY_KEY}


def test_scan_hash_follows_content_not_mtime(tmp_path: Path) -> None:
    """Chạm mtime mà không đổi nội dung thì KHÔNG được sinh cảnh báo giả."""
    kwargs = _corpus(tmp_path, {"P1.txt": "nội dung"}, {})
    before = scan_sources(**kwargs)
    path = kwargs["proposal_dir"] / "P1.txt"
    path.touch()  # chỉ đổi mtime
    assert scan_sources(**kwargs) == before

    path.write_text("nội dung khác", encoding="utf-8")
    assert scan_sources(**kwargs) != before


def test_scan_survives_missing_directories(tmp_path: Path) -> None:
    files = scan_sources(
        proposal_dir=tmp_path / "khong-ton-tai",
        rfp_dir=tmp_path / "cung-khong",
        capability_path=tmp_path / "thieu.json",
    )
    assert files == {}


def test_scan_ignores_non_txt_files(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    (kwargs["proposal_dir"] / "ghi-chu.md").write_text("bỏ qua", encoding="utf-8")
    assert set(scan_sources(**kwargs)) == {"proposals/P1.txt", CAPABILITY_KEY}


# ── compare: thêm / sửa / xoá / không đổi ─────────────────────────────────

def test_compare_detects_nothing_when_unchanged() -> None:
    files = {"proposals/P1.txt": "h1"}
    diff = compare({"files": files}, dict(files))
    assert not diff.changed and diff.total == 0


def test_compare_detects_added_modified_removed() -> None:
    previous = {"proposals/P1.txt": "h1", "proposals/P2.txt": "h2"}
    current = {"proposals/P1.txt": "h1-doi", "proposals/P3.txt": "h3"}
    diff = compare({"files": previous}, current)
    assert diff.added == ["proposals/P3.txt"]
    assert diff.modified == ["proposals/P1.txt"]
    assert diff.removed == ["proposals/P2.txt"]
    assert diff.total == 3


def test_compare_flags_capability_sheet_separately() -> None:
    diff = compare({"files": {CAPABILITY_KEY: "cu"}}, {CAPABILITY_KEY: "moi"})
    assert diff.capability_changed is True
    assert diff.modified == [CAPABILITY_KEY]


def test_compare_without_capability_change() -> None:
    diff = compare(
        {"files": {"proposals/P1.txt": "h1", CAPABILITY_KEY: "c"}},
        {"proposals/P1.txt": "h2", CAPABILITY_KEY: "c"},
    )
    assert diff.capability_changed is False


def test_compare_without_manifest_means_never_ingested() -> None:
    diff = compare(None, {"proposals/P1.txt": "h1"})
    assert diff.has_manifest is False
    assert diff.added == ["proposals/P1.txt"]


# ── manifest: ghi, đọc, và ca file hỏng ──────────────────────────────────

def test_write_then_load_manifest(tmp_path: Path) -> None:
    index = _fake_index([_sentence("S1", "文A")], quarantine=[_sentence("S2", "x")])
    path = tmp_path / "manifest.json"
    written = write_manifest(index, path=path, files={"proposals/P1.txt": "h"}, now=123.0)
    loaded = load_manifest(path)
    assert loaded == written
    assert loaded["counts"] == {"indexed": 1, "quarantine": 1, "leak_dropped": 0}
    assert loaded["ingested_at"] == 123.0
    assert loaded["corpus_fingerprint"] and loaded["capability_fingerprint"]


def test_load_manifest_missing_returns_none(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "chua-co.json") is None


def test_corrupt_manifest_is_treated_as_never_ingested(tmp_path: Path) -> None:
    """Manifest hỏng không được làm chết màn hình."""
    path = tmp_path / "manifest.json"
    path.write_text("{ khong phai json", encoding="utf-8")
    assert load_manifest(path) is None


def test_current_diff_reports_never_ingested(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    diff, manifest = current_diff(manifest_path=tmp_path / "chua-co.json", **kwargs)
    assert manifest is None and diff.has_manifest is False


def test_current_diff_clean_after_write(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    path = tmp_path / "manifest.json"
    write_manifest(_fake_index([]), path=path, files=scan_sources(**kwargs))
    diff, _ = current_diff(manifest_path=path, **kwargs)
    assert not diff.changed


# ── state_is_stale: kết quả đang xem có cũ hơn dữ liệu không ─────────────

def test_state_is_stale_false_when_fingerprint_matches(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    path = tmp_path / "manifest.json"
    manifest = write_manifest(_fake_index([]), path=path, files=scan_sources(**kwargs))
    state = {"fingerprints": {"corpus": manifest["corpus_fingerprint"]}}
    assert state_is_stale(state, manifest_path=path, **kwargs) is False


def test_state_is_stale_true_after_source_changes(tmp_path: Path) -> None:
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    path = tmp_path / "manifest.json"
    manifest = write_manifest(_fake_index([]), path=path, files=scan_sources(**kwargs))
    state = {"fingerprints": {"corpus": manifest["corpus_fingerprint"]}}
    (kwargs["proposal_dir"] / "P2.txt").write_text("mới", encoding="utf-8")
    assert state_is_stale(state, manifest_path=path, **kwargs) is True


def test_state_without_fingerprint_is_not_flagged(tmp_path: Path) -> None:
    """Không có bằng chứng thì không dựng cảnh báo."""
    kwargs = _corpus(tmp_path, {"P1.txt": "a"}, {})
    assert state_is_stale({}, manifest_path=tmp_path / "m.json", **kwargs) is False
    assert state_is_stale(None, manifest_path=tmp_path / "m.json", **kwargs) is False


# ── Báo cáo cách ly: nói được VÌ SAO câu bị loại ─────────────────────────

def test_quarantine_report_explains_each_dropped_sentence() -> None:
    index = _fake_index(
        [],
        quarantine=[_sentence("S1", "当社はISO/IEC 27017認証を取得済み")],
        leak=[_sentence("S2", "新生証券様向けの実績があります。")],
    )
    rows = quarantine_report(index)
    kinds = {row["loai"] for row in rows}
    assert kinds == {"Chuỗi cấm", "Tên khách hàng"}
    forbidden = next(row for row in rows if row["loai"] == "Chuỗi cấm")
    assert "27017" in forbidden["ly_do"]
    leaked = next(row for row in rows if row["loai"] == "Tên khách hàng")
    assert "新生証券" in leaked["ly_do"]


def test_quarantine_report_empty_when_nothing_dropped() -> None:
    assert quarantine_report(_fake_index([_sentence("S1", "普通の文です。")])) == []


# ── Ingest không có đường tắt ────────────────────────────────────────────

def test_reload_uses_the_real_ingest_pipeline() -> None:
    """Nút Nạp lại phải gọi SentenceIndex.build() — nơi quarantine chạy thật."""
    import inspect

    import app

    source = inspect.getsource(app.reload_knowledge_base)
    assert "SentenceIndex.build()" in source
    assert "assert_clean" in source


# ── Corpus eval tách khỏi corpus vận hành (B4) ───────────────────────────

def test_eval_dirs_are_separate_constants() -> None:
    from config import settings

    assert hasattr(settings, "EVAL_PROPOSAL_DIR")
    assert hasattr(settings, "EVAL_RFP_DIR")


def test_eval_modules_read_eval_dirs_not_runtime_dirs() -> None:
    import eval.mutations as mutations
    from config.settings import EVAL_RFP_DIR

    assert mutations.RFP_DIR == EVAL_RFP_DIR


def test_eval_dirs_default_to_current_corpus() -> None:
    """Hôm nay hai bên trỏ cùng chỗ -> không đổi hành vi, số đo giữ nguyên."""
    from config.settings import (
        EVAL_PROPOSAL_DIR,
        EVAL_RFP_DIR,
        PROPOSAL_DIR,
        RFP_DIR,
    )

    assert EVAL_PROPOSAL_DIR == PROPOSAL_DIR
    assert EVAL_RFP_DIR == RFP_DIR
