"""In thống kê vận hành từ nhật ký mỗi lượt chạy (v1.8, mục 5).

    python -m eval.ops_stats            # bảng người đọc
    python -m eval.ops_stats --json     # cho máy đọc

Chỉ đọc `cache/metrics.jsonl` — không chạy lại gì, không gọi LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config.settings import METRICS_LOG_PATH  # noqa: E402
from rfp.metrics import read_records, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=METRICS_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize(read_records(args.path))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not summary.get("runs"):
        print(f"Chưa có lượt chạy nào trong {args.path}")
        return
    print(f"Lượt chạy đã ghi : {summary['runs']}")
    print(f"Tự trả lời       : {summary['auto_rate']:.1%}")
    print(f"Chuyển người     : {summary['handover_rate']:.1%}")
    print(f"Không ra hồ sơ   : {summary['failed_rate']:.1%}")
    print(f"Thời gian p50/p95: {summary['p50_seconds']}s / {summary['p95_seconds']}s")
    print(f"Token sản phẩm/lượt: {summary['tokens_product_avg']:,.1f}")
    print(f"Token judge (tổng) : {summary['tokens_judge_total']:,}")
    print("Theo nhánh:")
    for name, count in summary["by_branch"].items():
        print(f"  {name:16} {count}")


if __name__ == "__main__":
    main()
