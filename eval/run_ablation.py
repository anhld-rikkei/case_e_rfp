"""Chạy bảng ablation §11.3 trên 9 RFP golden.

Một script cho mọi cấu hình. Ba bản run_ablation / _4 / _5 trước đây là cùng
một file chép ra, khác đúng hai cờ — sửa một chỗ là ba chỗ lệch nhau.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "config" / "settings.py"


# prec = PRECEDENTS_PER_CHAPTER (0 = tắt kênh precedent 6.3)
# cap  = CAPABILITY/COMPANY_FACTS_PER_SECTION (0 = tắt kênh capability 6.2)
# guards      = claim-check 6.4 · chống ảo giác lai 6.4-bis · final guard 7
# quarantine  = lọc leak + blocklist lúc ingest (Bước 2.2)
# bm25/rerank/mmr = thang retrieval V0→V1→V4 (bảng §11.3); mặc định bật hết
CONFIGS = {
    "V4_V5_A2": dict(prec=1, cap=2, guards=True, quarantine=True),
    # Thang retrieval: chunk câu + graph A2 giữ nguyên, mỗi bậc đổi đúng 1 biến.
    "V0_A2_dense_only": dict(
        prec=1, cap=2, guards=True, quarantine=True,
        bm25=False, rerank=False, mmr=False,
    ),
    "V1_A2_hybrid": dict(
        prec=1, cap=2, guards=True, quarantine=True,
        bm25=True, rerank=False, mmr=False,
    ),
    "V4_A2_mmr": dict(
        prec=1, cap=2, guards=True, quarantine=True,
        bm25=True, rerank=False, mmr=True,
    ),
    "only_capability": dict(prec=0, cap=2, guards=True, quarantine=True),
    "only_precedent": dict(prec=1, cap=0, guards=True, quarantine=True),
    "only_precedent_no_guard": dict(prec=1, cap=0, guards=False, quarantine=True),
    "only_precedent_no_guard_no_quarantine": dict(
        prec=1, cap=0, guards=False, quarantine=False
    ),
    "force_precedent_k5_no_guard": dict(
        prec=5, cap=0, guards=False, quarantine=False
    ),
    "force_precedent_k5_with_guard": dict(
        prec=5, cap=0, guards=True, quarantine=False
    ),
}


def set_config(
    prec: int,
    cap: int,
    *,
    bm25: bool = True,
    rerank: bool = True,
    mmr: bool = True,
) -> None:
    content = SETTINGS_PATH.read_text(encoding="utf-8")
    content = re.sub(r"PRECEDENTS_PER_CHAPTER\s*=\s*\d+", f"PRECEDENTS_PER_CHAPTER = {prec}", content)
    content = re.sub(r"CAPABILITY_FACTS_PER_SECTION\s*=\s*\d+", f"CAPABILITY_FACTS_PER_SECTION = {cap}", content)
    content = re.sub(r"COMPANY_FACTS_PER_SECTION\s*=\s*\d+", f"COMPANY_FACTS_PER_SECTION = {cap}", content)
    content = re.sub(r"RETRIEVAL_USE_BM25\s*=\s*\w+", f"RETRIEVAL_USE_BM25 = {bm25}", content)
    content = re.sub(r"RETRIEVAL_USE_RERANK\s*=\s*\w+", f"RETRIEVAL_USE_RERANK = {rerank}", content)
    content = re.sub(r"RETRIEVAL_USE_MMR\s*=\s*\w+", f"RETRIEVAL_USE_MMR = {mmr}", content)
    SETTINGS_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation runner cho Bước 11")
    parser.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIGS),
        help="chạy đúng cấu hình này (lặp lại được); mặc định chạy cả 5",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="bỏ RAGAS: chỉ token sản phẩm, không tốn token judge",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "results",
        help="thư mục ghi kết quả (mặc định eval/results)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="hậu tố tên file kết quả, để không đè lần chạy trước",
    )
    parser.add_argument(
        "--restrict-samples",
        type=Path,
        help="chuyển thẳng cho run_ragas: chỉ chấm RAGAS trên các atom chỉ định",
    )
    args = parser.parse_args()
    names = args.config or list(CONFIGS)

    golden_files = sorted((ROOT / "synthetic" / "golden_test_set").glob("*.json"))
    if not golden_files:
        raise SystemExit("Không tìm thấy golden case nào")
    print(f"Found {len(golden_files)} golden cases.")

    # Khôi phục nguyên văn settings.py, không ghi lại hằng số mặc định đoán được.
    original_settings = SETTINGS_PATH.read_text(encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="ablation_rfp_") as tmp:
            rfp_paths = []
            for gf in golden_files:
                data = json.loads(gf.read_text(encoding="utf-8"))
                txt_path = Path(tmp) / f"{gf.stem}.txt"
                txt_path.write_text(data["rfp_text"], encoding="utf-8")
                rfp_paths.append(txt_path)

            for name in names:
                config = CONFIGS[name]
                print(f"\n{'=' * 40}\nRunning config: {name}\n{'=' * 40}")
                set_config(
                    config["prec"],
                    config["cap"],
                    bm25=config.get("bm25", True),
                    rerank=config.get("rerank", True),
                    mmr=config.get("mmr", True),
                )

                suffix = ".det.json" if args.deterministic_only else ".json"
                out_file = f"{name}{args.tag}{suffix}"
                cmd = [
                    str(ROOT / ".venv" / "Scripts" / "python.exe"),
                    "-m",
                    "eval.run_ragas",
                    "--out",
                    str(args.out_dir / out_file),
                ]
                if args.restrict_samples is not None:
                    cmd.extend(["--restrict-samples", str(args.restrict_samples)])
                if not config["guards"]:
                    cmd.append("--disable-guards")
                if not config["quarantine"]:
                    cmd.append("--disable-quarantine")
                if args.deterministic_only:
                    cmd.append("--deterministic-only")
                for p in rfp_paths:
                    cmd.extend(["--rfp", str(p)])

                subprocess.run(cmd, check=True, cwd=ROOT)
    finally:
        SETTINGS_PATH.write_text(original_settings, encoding="utf-8")


if __name__ == "__main__":
    main()
