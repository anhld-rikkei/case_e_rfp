"""Đối chiếu dữ liệu nguồn với lần nạp gần nhất (v1.5).

## Nói đúng thứ tính năng này làm

Repo **không lưu kho tri thức xuống đĩa**: `SentenceIndex.build()` đọc và parse
lại `synthetic/` mỗi lượt chạy, và cache Bước 5 đã khoá theo `corpus_fingerprint`
nên tự miss khi dữ liệu đổi. Nghĩa là hệ **không** âm thầm chạy dữ liệu cũ, và
một banner kiểu "kho đang cũ, hãy nạp lại" sẽ là banner nói sai.

Thứ *thật sự* cũ được là **kết quả đang hiển thị hoặc đã xuất ra**: hồ sơ sinh
lúc 9h, ai đó sửa `proposals/` lúc 10h, bản trên màn hình vẫn là bản cũ mà
không có dấu hiệu nào. Nên module này làm ba việc:

  1. cho biết dữ liệu nguồn có đổi kể từ lần nạp gần nhất không, đổi file nào;
  2. cảnh báo riêng khi `capability_sheet.json` đổi — nó là trọng tài BB-1,
     quyết định câu nào được phép nói;
  3. đánh dấu kết quả nào đã sinh từ dữ liệu cũ hơn hiện tại.

## Băm nội dung, không dùng mtime

Cùng nguyên tắc với `cache.py`: clone lại repo, checkout nhánh khác hay chép
file qua máy khác đều đổi mtime mà không đổi nội dung. Dùng mtime là sinh cảnh
báo giả, và cảnh báo giả lặp lại vài lần thì người ta ngừng đọc cảnh báo.

Blocklist không có mục riêng trong manifest: nó suy ra từ
`certifications_NOT_held` + `capabilities_NOT_offered`, đã nằm trọn trong
`capability_fingerprint`.

## Tương tác với các cơ chế sẵn có

- **Cache (Bước 5):** không cần can thiệp — `corpus_fingerprint` đã là một phần
  của cache key, nên dữ liệu đổi là cache tự miss.
- **Checkpoint (Bước 4):** job `--resume` chạy tiếp trên state cũ. Nếu dữ liệu
  nguồn đã đổi giữa chừng thì phần chạy trước và phần chạy sau thuộc hai thế hệ
  dữ liệu khác nhau, nên `stale_warning()` được dùng để cảnh báo; muốn sạch thì
  chạy lại với `--force-regen`.
- **Chat-refine (v1.6):** các bản trong lịch sử phiên bản **giữ nguyên** khi nạp
  lại kho — đó là công sức người dùng, xoá đi là mất. Nhưng chúng bị đánh dấu
  "sinh từ dữ liệu cũ" và **lượt chat tiếp theo bị chặn**: nguồn của một lượt
  chat lấy từ `state["chapters"]` cũ, cho chỉnh tiếp là trộn hai thế hệ dữ liệu
  vào cùng một hồ sơ.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import CACHE_DIR, CAPABILITY_PATH, PROPOSAL_DIR, RFP_DIR

from .cache import capability_fingerprint, corpus_fingerprint


DEFAULT_MANIFEST_PATH = Path(CACHE_DIR) / "ingest_manifest.json"
CAPABILITY_KEY = "capability_sheet.json"


@dataclass(frozen=True)
class SourceDiff:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    capability_changed: bool = False
    has_manifest: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_sources(
    *,
    proposal_dir: str | Path = PROPOSAL_DIR,
    rfp_dir: str | Path = RFP_DIR,
    capability_path: str | Path = CAPABILITY_PATH,
) -> dict[str, str]:
    """Đường dẫn tương đối -> sha256 nội dung. Thiếu file thì bỏ qua, không nổ."""
    files: dict[str, str] = {}
    for label, directory in (("proposals", proposal_dir), ("rfps", rfp_dir)):
        directory = Path(directory)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.txt")):
            files[f"{label}/{path.name}"] = _sha256_file(path)
    capability = Path(capability_path)
    if capability.exists():
        files[CAPABILITY_KEY] = _sha256_file(capability)
    return files


def compare(manifest: dict[str, Any] | None, current: dict[str, str]) -> SourceDiff:
    """So dữ liệu hiện tại với manifest. Chưa có manifest = chưa từng nạp."""
    if not manifest:
        return SourceDiff(
            added=sorted(current),
            has_manifest=False,
            capability_changed=CAPABILITY_KEY in current,
        )
    previous: dict[str, str] = manifest.get("files", {})
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    modified = sorted(
        name for name in set(previous) & set(current) if previous[name] != current[name]
    )
    capability_changed = CAPABILITY_KEY in (set(added) | set(removed) | set(modified))
    return SourceDiff(
        added=added,
        modified=modified,
        removed=removed,
        capability_changed=capability_changed,
    )


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any] | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Manifest hỏng = coi như chưa nạp lần nào. Thà báo "chưa nạp" còn hơn
        # làm chết màn hình vì một file phụ trợ.
        return None


def write_manifest(
    sentence_index,
    *,
    path: str | Path = DEFAULT_MANIFEST_PATH,
    files: dict[str, str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    counts = sentence_index.cleanliness()
    manifest = {
        "ingested_at": now if now is not None else time.time(),
        "corpus_fingerprint": corpus_fingerprint(sentence_index),
        "capability_fingerprint": capability_fingerprint(),
        "files": files if files is not None else scan_sources(),
        "counts": {
            "indexed": counts["indexed"],
            "quarantine": counts["capability_quarantine"],
            "leak_dropped": counts["leak_quarantine"],
        },
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def current_diff(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    **scan_kwargs: Any,
) -> tuple[SourceDiff, dict[str, Any] | None]:
    manifest = load_manifest(manifest_path)
    return compare(manifest, scan_sources(**scan_kwargs)), manifest


def state_is_stale(
    state: dict[str, Any] | None,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    **scan_kwargs: Any,
) -> bool:
    """Kết quả này có sinh từ dữ liệu cũ hơn hiện tại không?

    So `fingerprints.corpus` mà node retrieval đã ghi vào state với vân tay của
    dữ liệu **đang có trên đĩa**. State không mang vân tay (bản dump cũ) thì trả
    False: không có bằng chứng thì không dựng cảnh báo.
    """
    if not state:
        return False
    stamped = (state.get("fingerprints") or {}).get("corpus")
    if not stamped:
        return False
    manifest = load_manifest(manifest_path)
    if not manifest:
        # Chưa nạp lần nào = KHÔNG BIẾT, không phải "đã cũ". Coi là cũ ở đây thì
        # mọi kết quả đều bị dán nhãn cảnh báo ngay lần chạy đầu, và cảnh báo
        # luôn bật là cảnh báo không ai đọc.
        return False
    if manifest.get("files") == scan_sources(**scan_kwargs):
        return stamped != manifest.get("corpus_fingerprint")
    # Dữ liệu trên đĩa đã khác manifest -> mọi kết quả cũ đều thuộc thế hệ trước.
    return True


def quarantine_report(sentence_index) -> list[dict[str, str]]:
    """Câu bị cách ly lúc nạp, kèm lý do — để nút Nạp lại nói được vì sao."""
    from .sanitize.blocklist import CapabilityBlocklist
    from .sanitize.leak import private_client_names

    blocklist = CapabilityBlocklist()
    rows: list[dict[str, str]] = []
    for sentence in sentence_index.capability_quarantine:
        hits = blocklist.find(sentence.text)
        rows.append(
            {
                "sent_id": sentence.sent_id,
                "loai": "Chuỗi cấm",
                "ly_do": ", ".join(hits) or "khớp blocklist",
                "text": sentence.text,
            }
        )
    for sentence in sentence_index.leak_quarantine:
        names = private_client_names(sentence.text)
        rows.append(
            {
                "sent_id": sentence.sent_id,
                "loai": "Tên khách hàng",
                "ly_do": ", ".join(dict.fromkeys(names)) or "khớp tên riêng",
                "text": sentence.text,
            }
        )
    return rows
