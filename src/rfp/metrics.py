"""Nhật ký đo lường mỗi lượt chạy, một dòng JSON (v1.8).

Ghi **sau khi** một lượt chạy kết thúc, từ những gì state và `usage` đã có —
không đo thêm gì, không gọi thêm gì. Mỗi dòng đủ để trả lời bốn câu hỏi vận
hành mà không phải mở lại hồ sơ:

  - lượt này tự trả lời được hay phải chuyển người (tier + branch)
  - mất bao lâu (seconds) và tốn bao nhiêu token (product/judge tách riêng)
  - điểm tin cậy bao nhiêu, phân bố theo tầng thế nào
  - kết thúc ra sao: completed / partial / guard_blocked / ask_user

Ghi lỗi **không được làm hỏng lượt chạy**: nhật ký là thứ phụ, hồ sơ là thứ
chính. Mọi lỗi ghi file đều bị nuốt có chủ đích, và điều đó được nói rõ ở đây
để không ai tưởng là quên xử lý.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from config.settings import METRICS_ENABLED, METRICS_LOG_PATH


BRANCH_AUTO = "auto"
BRANCH_REVIEW = "human_review"
BRANCH_HUMAN = "human_takeover"
BRANCH_FAILED = "failed"

_BRANCH_BY_TIER = {
    "T1": BRANCH_AUTO,
    "T2": BRANCH_REVIEW,
    "T3": BRANCH_HUMAN,
}


def branch_of(state: dict[str, Any]) -> str:
    """Lượt này đi nhánh nào — dùng để tính '% tự trả lời vs chuyển người'."""
    status = state.get("status")
    if status in ("partial", "guard_blocked"):
        return BRANCH_FAILED
    if status == "ask_user":
        return BRANCH_HUMAN
    tier = (state.get("confidence") or {}).get("tier")
    return _BRANCH_BY_TIER.get(tier, BRANCH_HUMAN)


def build_record(
    state: dict[str, Any],
    *,
    usage: Any = None,
    seconds: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    rfp = state.get("rfp") or {}
    rfp_id = rfp.get("rfp_id", "") if isinstance(rfp, dict) else getattr(rfp, "rfp_id", "")
    confidence = state.get("confidence") or {}
    tokens = getattr(usage, "tokens_by_stage", {}) or {}
    product = sum(
        value for key, value in tokens.items() if key in ("generate", "structured")
    )
    return {
        "at": now if now is not None else time.time(),
        "rfp_id": rfp_id,
        "status": state.get("status"),
        "branch": branch_of(state),
        "score": confidence.get("score"),
        "tier": confidence.get("tier"),
        "by_tier": confidence.get("by_tier", {}),
        "seconds": round(
            seconds if seconds is not None else getattr(usage, "seconds", 0.0) or 0.0,
            3,
        ),
        "llm_calls": getattr(usage, "calls", 0),
        "tokens_product": product,
        "tokens_judge": tokens.get("judge", 0),
        "sections": len(state.get("sections", [])),
    }


def log_run(
    state: dict[str, Any],
    *,
    usage: Any = None,
    seconds: float | None = None,
    path: str | Path = METRICS_LOG_PATH,
    enabled: bool | None = None,
) -> dict[str, Any] | None:
    if not (METRICS_ENABLED if enabled is None else enabled):
        return None
    record = build_record(state, usage=usage, seconds=seconds)
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Có chủ đích: hồ sơ quan trọng hơn nhật ký. Xem docstring đầu file.
        return record
    return record


def read_records(path: str | Path = METRICS_LOG_PATH) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Dòng hỏng (ghi dở lúc bị ngắt) thì bỏ, không làm chết cả báo cáo.
            continue
    return records


def percentile(values: list[float], fraction: float) -> float | None:
    """p50/p95 kiểu nearest-rank — không nội suy, hợp với mẫu nhỏ."""
    if not values:
        return None
    ordered = sorted(values)
    # ceil, KHÔNG phải round: `round(2.5)` trong Python là 2 (làm tròn về số
    # chẵn), nên p50 của 5 mẫu sẽ trả phần tử thứ 2 thay vì thứ 3.
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Thống kê vận hành. Không có dữ liệu thì trả 0/None, không bịa."""
    total = len(records)
    if not total:
        return {"runs": 0}
    seconds = [float(item.get("seconds") or 0.0) for item in records]
    product = [int(item.get("tokens_product") or 0) for item in records]
    branches = {name: 0 for name in (BRANCH_AUTO, BRANCH_REVIEW, BRANCH_HUMAN, BRANCH_FAILED)}
    for item in records:
        branches[item.get("branch", BRANCH_HUMAN)] = (
            branches.get(item.get("branch", BRANCH_HUMAN), 0) + 1
        )
    auto = branches[BRANCH_AUTO]
    return {
        "runs": total,
        "by_branch": branches,
        "auto_rate": round(auto / total, 4),
        "handover_rate": round(
            (branches[BRANCH_REVIEW] + branches[BRANCH_HUMAN]) / total, 4
        ),
        "failed_rate": round(branches[BRANCH_FAILED] / total, 4),
        "p50_seconds": percentile(seconds, 0.50),
        "p95_seconds": percentile(seconds, 0.95),
        "tokens_product_avg": round(sum(product) / total, 1),
        "tokens_judge_total": sum(int(item.get("tokens_judge") or 0) for item in records),
    }
