"""Điểm tin cậy 0–1 và phân tầng T1/T2/T3 (v1.8).

## Quy từ căn cứ ĐÃ CÓ, không thêm một lệnh gọi LLM nào

Mọi tín hiệu dùng ở đây đều đã nằm sẵn trong state sau khi pipeline chạy:

  - `verdict` của từng câu (claim-check ở Bước 6.4)
  - điểm `rerank` của câu nguồn (khâu truy xuất)
  - `origin` (capability tất định · precedent qua LLM · bridge · user)
  - coverage: requirement nào của mục đã có câu dẫn

Chấm điểm bằng một lệnh gọi LLM nữa sẽ đắt hơn cả khâu sinh, và tệ hơn: nó đưa
một phán đoán xác suất vào đúng chỗ mà hệ đang cố tránh phán đoán. Điểm ở đây là
**hàm tất định** của bằng chứng — chạy lại trên cùng state luôn ra cùng số.

## Ba tầng và vì sao mặc định KHÔNG đổi hành vi

Yêu cầu: ngưỡng mặc định phải giữ nguyên hành vi hiện tại — mục `OK` tự trả lời,
`ATTRIBUTE_ONLY` ở giữa, `INSUFFICIENT_EVIDENCE` chuyển người. Để bảo đảm điều
đó *bằng cấu trúc* chứ không bằng may mắn, công thức nhân ba hệ số:

    score = base(bằng chứng) × complete(phủ đủ requirement?) × support(có precedent?)

với `INCOMPLETE_COVERAGE_FACTOR < T_LOW` và `NO_PRECEDENT_FACTOR < T_HIGH`. Hai
bất đẳng thức đó ép:

    thiếu requirement  -> score < T_LOW          -> T3 (chuyển người)
    đủ, không precedent -> T_LOW ≤ score < T_HIGH -> T2 (người xem lại)
    đủ, có precedent    -> score ≥ T_HIGH         -> T1 (tự trả lời)

Có test khẳng định hai bất đẳng thức này, nên chỉnh ngưỡng mà phá quan hệ đó là
test đỏ ngay chứ không phải phát hiện lúc demo. Trong mỗi tầng, điểm vẫn xê dịch
theo chất lượng bằng chứng để so được hai mục cùng tầng với nhau.

## Câu người dùng bổ sung không được chấm

`origin="user"` là nội dung người dùng tự chịu trách nhiệm (v1.7). Chấm điểm cho
nó là giả vờ hệ thống có ý kiến về thứ nó không kiểm chứng được, nên các câu đó
bị loại khỏi phép tính và đếm riêng.
"""
from __future__ import annotations

from typing import Any

from config.settings import (
    CONFIDENCE_CAPABILITY_BASE,
    CONFIDENCE_PRECEDENT_BASE,
    CONFIDENCE_PRECEDENT_SPAN,
    CONFIDENCE_T_HIGH,
    CONFIDENCE_T_LOW,
    CONFIDENCE_UNVERIFIABLE,
    CONFIDENCE_USER_APPROVED,
    INCOMPLETE_COVERAGE_FACTOR,
    NO_PRECEDENT_FACTOR,
)


TIER_AUTO = "T1"
TIER_REVIEW = "T2"
TIER_HUMAN = "T3"
TIERS = (TIER_AUTO, TIER_REVIEW, TIER_HUMAN)


def sentence_confidence(
    sentence: dict[str, Any],
    *,
    rerank_by_source: dict[str, float] | None = None,
) -> float | None:
    """Điểm của MỘT câu. None = không chấm (câu nối, câu người dùng thêm)."""
    # Người viết đã bấm duyệt câu này -> nó có căn cứ, và căn cứ là chữ ký của
    # người chịu trách nhiệm. Kiểm trước cả nhánh "user" bên dưới, vì đúng
    # những câu đó mới là thứ cần duyệt.
    if sentence.get("approved_by_user"):
        return CONFIDENCE_USER_APPROVED
    origin = sentence.get("origin")
    if origin in ("bridge", "user"):
        return None
    verdict = sentence.get("verdict")
    if verdict == "CONTRADICTED":
        return 0.0
    if origin == "capability":
        # Đến thẳng từ capability sheet qua template — không qua phán đoán nào.
        return CONFIDENCE_CAPABILITY_BASE if verdict == "VERIFIED" else CONFIDENCE_UNVERIFIABLE
    if verdict != "VERIFIED":
        return CONFIDENCE_UNVERIFIABLE
    rerank = (rerank_by_source or {}).get(sentence.get("source_id") or "", 0.0)
    return CONFIDENCE_PRECEDENT_BASE + CONFIDENCE_PRECEDENT_SPAN * _clamp(rerank)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def rerank_scores(state: dict[str, Any]) -> dict[str, float]:
    """Điểm rerank của từng câu nguồn — đã có sẵn trong trace truy xuất."""
    scores: dict[str, float] = {}
    for chapter in state.get("chapters", []):
        for item in chapter.get("retrieval", {}).get("selected", []):
            value = (item.get("scores") or {}).get("rerank")
            if value is not None:
                scores[item["sent_id"]] = float(value)
    return scores


def section_confidence(
    state: dict[str, Any],
    section: dict[str, Any],
    *,
    rerank_by_source: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Điểm + tầng của một mục, kèm các thành phần để giải thích được vì sao."""
    scores = rerank_by_source if rerank_by_source is not None else rerank_scores(state)
    sentences = section.get("sentences", [])
    values = [
        value
        for sentence in sentences
        if (value := sentence_confidence(sentence, rerank_by_source=scores)) is not None
    ]
    base = sum(values) / len(values) if values else 0.0

    covered, total, missing = _coverage(state, section)
    # total == 0 là mục cố định không neo vào chương nào (会社概要): không có
    # yêu cầu nào thì không thiếu yêu cầu nào. Coi nó là "chưa phủ đủ" sẽ đẩy
    # một mục `OK` xuống tầng chuyển người — đúng lỗi đã bắt được khi chạy thật.
    complete = total == 0 or covered == total
    has_precedent = any(item.get("origin") == "precedent" for item in sentences)
    # Người viết đã duyệt cũng là một chỗ dựa: phạt "không có hồ sơ quá khứ
    # chống lưng" ở đây sẽ đẩy mục xuống tầng "cần người xem lại" trong khi
    # người vừa xem xong. T2 nghĩa là CHƯA ai xem — nói vậy là sai sự thật.
    approved = any(item.get("approved_by_user") for item in sentences)
    supported = has_precedent or approved

    score = base
    if not complete:
        score *= INCOMPLETE_COVERAGE_FACTOR
    elif not supported and total > 0:
        # Phạt "không có hồ sơ quá khứ nào chống lưng" chỉ đúng khi mục CÓ yêu
        # cầu phải đáp. Mục cố định (会社概要) vốn chỉ dựng từ bảng năng lực
        # theo thiết kế — phạt nó là hạ tầng một mục vốn đủ căn cứ.
        score *= NO_PRECEDENT_FACTOR
    score = round(_clamp(score), 4)

    return {
        "score": score,
        "tier": tier_of(score),
        "scored_sentences": len(values),
        "user_sentences": sum(1 for s in sentences if s.get("origin") == "user"),
        "covered_requirements": covered,
        "total_requirements": total,
        "missing_requirements": missing,
        "has_precedent": has_precedent,
        "approved_sentences": sum(
            1 for item in sentences if item.get("approved_by_user")
        ),
    }


def missing_requirements(
    state: dict[str, Any], section: dict[str, Any]
) -> list[str]:
    """Yêu cầu của mục chưa có câu nào dẫn.

    Công khai vì `refine` cũng cần đúng con số này: chỗ đó quyết câu người dùng
    vừa thêm gắn mã yêu cầu gì, chỗ này quyết mục còn thiếu hay đủ. Hai phép
    đếm riêng thì có ngày người viết điền xong mà mục vẫn báo thiếu.
    """
    return _coverage(state, section)[2]


def _coverage(
    state: dict[str, Any], section: dict[str, Any]
) -> tuple[int, int, list[str]]:
    covered = {
        req_id
        for sentence in section.get("sentences", [])
        if sentence.get("origin") in ("capability", "precedent")
        or sentence.get("approved_by_user")
        for req_id in sentence.get("req_ids", [])
    }
    chapters = {chapter["id"]: chapter for chapter in state.get("chapters", [])}
    required = [
        requirement["req_id"]
        for chapter_id in section.get("source_chapters", [])
        if chapter_id in chapters
        for requirement in chapters[chapter_id].get("requirements", [])
    ]
    if not required:
        # Mục cố định (会社概要) không neo vào chương nào — coi như đáp ứng đủ,
        # đúng cách `assess_coverage` đang xử lý.
        return 0, 0, []
    missing = [req_id for req_id in required if req_id not in covered]
    return len(required) - len(missing), len(required), missing


def tier_of(score: float) -> str:
    if score >= CONFIDENCE_T_HIGH:
        return TIER_AUTO
    if score >= CONFIDENCE_T_LOW:
        return TIER_REVIEW
    return TIER_HUMAN


def annotate(state: dict[str, Any]) -> dict[str, Any]:
    """Gắn `confidence` vào từng mục + tổng hợp mức hồ sơ, trả state MỚI."""
    scores = rerank_scores(state)
    sections = []
    for section in state.get("sections", []):
        info = section_confidence(state, section, rerank_by_source=scores)
        sections.append({**section, "confidence": info})
    per_section = [item["confidence"]["score"] for item in sections]
    overall = round(min(per_section), 4) if per_section else 0.0
    return {
        **state,
        "sections": sections,
        "confidence": {
            "score": overall,
            "tier": tier_of(overall),
            "by_tier": {
                tier: sum(1 for item in sections if item["confidence"]["tier"] == tier)
                for tier in TIERS
            },
        },
    }
