"""Cấu hình chung cho mọi test trong eval/.

Cache bị tắt cứng ở đây, không đọc cờ từ `config/settings.py`: một test chạy
trúng cache của lần chạy trước sẽ pass mà không thực thi đường code nó định
kiểm, và bảng đo token thì về gần 0. Muốn kiểm chính cơ chế cache thì dựng
`Cache` riêng trong test (xem `test_cache.py`), không bật lại cache toàn cục.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]

import rfp.metrics as metrics_module  # noqa: E402
from rfp.cache import Cache  # noqa: E402
from rfp.graph import use_cache  # noqa: E402


@pytest.fixture(autouse=True)
def disable_cache_everywhere():
    with use_cache(Cache(enabled=False)):
        yield


@pytest.fixture(autouse=True)
def disable_metrics_log(monkeypatch):
    """Test KHÔNG được ghi vào nhật ký vận hành thật.

    Đã xảy ra: graph giả trong `test_resilience` ném LLMUnavailable, `run_graph`
    ghi một dòng `partial` vào `cache/metrics.jsonl`, và thống kê "% tự trả lời"
    tụt xuống vì những lượt chạy chưa bao giờ tồn tại. Cùng loại lỗi với việc
    eval đụng cache — chặn ở cùng một chỗ.
    """
    monkeypatch.setattr(metrics_module, "METRICS_ENABLED", False)
