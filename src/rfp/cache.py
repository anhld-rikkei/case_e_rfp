"""Cache kết quả pipeline ở mức NODE, có invalidation đầy đủ (Bước 5, v1.2).

Cache key phải gồm MỌI thứ có thể đổi kết quả, nếu không thì "đổi cấu hình rồi
mà số không nhúc nhích" biến thành một lần chạy tin được nhầm:

  rfp_text · prompt_version · template_version · model
  · retrieval_config   (top_k, các cờ ablation, số câu mỗi chương/mục)
  · corpus_fingerprint (nội dung index SAU sanitize)
  · capability_fingerprint (capability_sheet.json — trọng tài BB-1)

Hai vân tay cuối băm NỘI DUNG, không băm mtime: clone lại repo hay dựng lại
index trên máy khác không được sinh cache miss giả. Corpus băm sau sanitize nên
đổi blocklist/leak-filter cũng đổi vân tay — đúng thứ Bước 1 (v1.1) vừa sửa.

## Vì sao cache trọn node, KHÔNG cache từng section

Các section trong `generate_per_section` **không độc lập**: `used_fact_keys` tích
luỹ qua vòng lặp và `reserved_fact_owners` phân bổ trước chủ sở hữu fact, nên
output của section thứ 3 phụ thuộc vào section 1–2 đã dùng fact nào. Cache từng
section rồi ghép lại sẽ cho ra hồ sơ **lặp fact** — đúng thứ cơ chế dedup đó sinh
ra để chặn. Muốn cache đúng ở mức section thì key phải gồm cả `used_fact_keys`
tại thời điểm vào section, mà muốn biết nó thì phải chạy lại các section trước —
cache mất ý nghĩa.

Nếu sau này làm **chat-refine (#12)** — sửa lại đúng một mục — thì mới đáng xét
lại đơn vị cache. Khi đó phải tách việc phân bổ fact thành bước tiền xử lý tất
định (không phụ thuộc thứ tự sinh) và **chạy eval riêng** chứng minh hành vi
không đổi trước khi hạ đơn vị cache xuống mức section.

## An toàn

Cache chỉ giữ kết quả của `retrieve_per_chapter` và `generate_per_section`.
Node `assemble` — nơi `final_guard()` chạy — **không bao giờ được cache**, nên
mọi hồ sơ xuất ra đều đi qua guard, kể cả khi toàn bộ phần sinh là cache hit.

Mọi đường chạy trong `eval/` ép `Cache(enabled=False)` ngay trong code eval, không
đọc cờ từ settings: cache hit sẽ làm token sản phẩm đo được về gần 0 và phá bảng
ablation §11.3. Cache chỉ phục vụ app/CLI.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config.settings import (
    CACHE_DIR,
    CACHE_ENABLED,
    CACHE_TTL_SECONDS,
    CAPABILITY_FACTS_PER_SECTION,
    CAPABILITY_PATH,
    COMPANY_FACTS_PER_SECTION,
    MMR_LAMBDA,
    MMR_TOP_K,
    PRECEDENTS_PER_CHAPTER,
    PROMPT_VERSION,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    RETRIEVAL_USE_BM25,
    RETRIEVAL_USE_MMR,
    RETRIEVAL_USE_RERANK,
    TEMPLATE_VERSION,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieval_config() -> dict[str, Any]:
    """Mọi tham số retrieval/sinh có thể đổi nội dung hồ sơ."""
    return {
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "mmr_top_k": MMR_TOP_K,
        "mmr_lambda": MMR_LAMBDA,
        "use_bm25": RETRIEVAL_USE_BM25,
        "use_rerank": RETRIEVAL_USE_RERANK,
        "use_mmr": RETRIEVAL_USE_MMR,
        "precedents_per_chapter": PRECEDENTS_PER_CHAPTER,
        "capability_facts_per_section": CAPABILITY_FACTS_PER_SECTION,
        "company_facts_per_section": COMPANY_FACTS_PER_SECTION,
    }


def corpus_fingerprint(sentence_index) -> str:
    """Vân tay corpus: nội dung câu SAU sanitize + số câu bị loại từng loại.

    Băm text chứ không băm đường dẫn/mtime. Đếm cả quarantine để việc nới hay
    siết blocklist cũng đổi vân tay, dù tập câu sạch tình cờ không đổi.
    """
    manifest = {
        "sentences": sorted(
            f"{sentence.sent_id}:{_sha256(sentence.text)}"
            for sentence in sentence_index.sentences
        ),
        "indexed": len(sentence_index.sentences),
        "leak_quarantine": len(sentence_index.leak_quarantine),
        "capability_quarantine": len(sentence_index.capability_quarantine),
    }
    return _sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def capability_fingerprint(path: str | Path = CAPABILITY_PATH) -> str:
    """Vân tay capability sheet — trọng tài BB-1, sửa nó phải bỏ mọi cache cũ."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True))


def make_key(
    *,
    rfp_text: str,
    model: str,
    corpus: str,
    capability: str,
    scope: str = "",
) -> str:
    payload = {
        "rfp_text": _sha256(rfp_text),
        "prompt_version": PROMPT_VERSION,
        "template_version": TEMPLATE_VERSION,
        "model": model,
        "retrieval_config": retrieval_config(),
        "corpus_fingerprint": corpus,
        "capability_fingerprint": capability,
        "scope": scope,
    }
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


class Cache:
    """Cache file JSON theo key. Không cache thứ gì chưa qua guard."""

    def __init__(
        self,
        directory: str | Path = CACHE_DIR,
        *,
        ttl_seconds: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.enabled = CACHE_ENABLED if enabled is None else enabled

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str, *, force_regen: bool = False) -> Any | None:
        if not self.enabled or force_regen:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # File hỏng thì coi như miss, không làm chết cả lượt chạy.
            return None
        created_at = entry.get("created_at")
        if not isinstance(created_at, (int, float)):
            return None
        # >= chứ không phải >: ttl_seconds=0 phải là "hết hạn ngay". Đồng hồ
        # Windows có độ phân giải ~15ms nên elapsed đo được có thể đúng bằng
        # 0.0, và với dấu > thì entry đó sống sót — TTL=0 hoá ra vô hiệu.
        if self.ttl_seconds is not None and (
            time.time() - created_at >= self.ttl_seconds
        ):
            return None
        return entry.get("value")

    def put(self, key: str, value: Any) -> None:
        """Ghi cả khi force_regen: lần chạy sau vẫn phải dùng lại được."""
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"created_at": time.time(), "value": value}
        self._path(key).write_text(
            json.dumps(entry, ensure_ascii=False), encoding="utf-8"
        )
