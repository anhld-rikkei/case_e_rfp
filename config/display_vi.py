"""Nhãn tiếng Việt cho tầng hiển thị (v1.4).

## Chỉ là lớp áo, không phải dữ liệu

Mọi thứ ở đây **chỉ dùng lúc render**. Giá trị thật trong state, trong
`eval/results/*.json` và trong file xuất ra vẫn là enum gốc (`VERIFIED`,
`INSUFFICIENT_EVIDENCE`, `capability`…). Không được dịch ngược nhãn tiếng Việt
thành giá trị rồi đem đi so sánh hay ghi xuống: bộ lọc trên UI phải map **ngược
về giá trị gốc** trước khi lọc (xem `to_raw`).

Lý do cứng: `check.py`, `eval/to_samples.py`, `eval/golden/runner.py` và toàn bộ
bảng ablation đối chiếu bằng chuỗi enum. Đổi giá trị vì lý do hiển thị là làm
hỏng đúng những thứ đó, và hỏng im lặng.

## Nội dung hồ sơ vẫn là tiếng Nhật

Chỉ dịch **nhãn giao diện**. Câu trong hồ sơ, tiêu đề mục (`title_ja`) và văn
bản RFP giữ nguyên tiếng Nhật — đó là sản phẩm giao cho khách, không phải chú
thích cho người vận hành.
"""
from __future__ import annotations


UNKNOWN = "—"


# ── Nguồn của câu (origin) — canonical: rfp.check.VALID_ORIGINS ────────────
# `user` KHÔNG nằm trong `check.py:VALID_ORIGINS` — nó là nhãn hậu-pipeline do
# chat-refine gắn (v1.7), chỉ sống trong session app và file xuất. Pipeline
# không bao giờ sinh ra nó, và contract checker vẫn fail loud nếu gặp.
UI_ONLY_ORIGINS = {"user"}

ORIGIN_VI = {
    "capability": "Năng lực công ty",
    "precedent": "Hồ sơ quá khứ",
    "bridge": "Câu nối",
    "user": "Người dùng bổ sung",
}

ORIGIN_HINT = {
    "capability": "Lấy từ bảng năng lực đã khai báo của công ty",
    "precedent": "Viết lại từ một câu trong hồ sơ thầu cũ, có mã nguồn",
    "bridge": "Câu chuyển ý, không mang thông tin sự thật nên không có nguồn",
    "user": (
        "Người dùng yêu cầu thêm hoặc sửa qua chat — hệ thống chưa kiểm chứng "
        "được, người nộp hồ sơ chịu trách nhiệm nội dung này"
    ),
}


# ── Kết luận kiểm chứng (verdict) — canonical: ClaimVerdict ────────────────
UI_ONLY_VERDICTS = {"USER_PROVIDED"}

VERDICT_VI = {
    "VERIFIED": "Có nguồn xác thực",
    "UNVERIFIABLE": "Không kiểm chứng được",
    "CONTRADICTED": "Mâu thuẫn với năng lực",
    "USER_PROVIDED": "Người dùng bổ sung — chưa kiểm chứng",
}

VERDICT_ICON = {
    "VERIFIED": "🟢",
    "UNVERIFIABLE": "🟡",
    "CONTRADICTED": "🔴",
    "USER_PROVIDED": "✎",
}

VERDICT_HINT = {
    "VERIFIED": "Đối chiếu được với bằng chứng gốc",
    "UNVERIFIABLE": "Không có bằng chứng để đối chiếu — câu nối luôn thuộc nhóm này",
    "CONTRADICTED": "Trái với bảng năng lực; câu loại này đã bị gỡ khỏi hồ sơ",
    "USER_PROVIDED": (
        "Do người dùng yêu cầu qua chat, hệ thống không kiểm chứng — phải rà "
        "bằng tay trước khi nộp"
    ),
}

# Đánh dấu hiển thị theo mức độ can thiệp (v1.7). Hai mức thay vì ba: câu máy
# sinh nguyên bản không đánh dấu gì cả, nên chỉ cần phân biệt "người dùng đưa
# vào" với "chat sửa cách viết nhưng giữ nguồn và số liệu".
USER_BLOCK_LABEL = "✎ Người dùng bổ sung — chưa kiểm chứng"
EDITED_LABEL = "✎ đã chỉnh cách viết"
MARK_LEGEND = (
    "Nền vàng = người dùng bổ sung qua chat, hệ thống chưa kiểm chứng · "
    "viền trái = chat chỉnh cách viết nhưng giữ nguyên nguồn và số liệu · "
    "không đánh dấu = máy sinh từ căn cứ"
)


# ── Trạng thái mục — canonical: coverage.SectionStatus ─────────────────────
SECTION_STATUS_VI = {
    "OK": "Đủ căn cứ",
    "ATTRIBUTE_ONLY": "Chỉ có thông tin công ty",
    "INSUFFICIENT_EVIDENCE": "Thiếu căn cứ — cần người bổ sung",
}

SECTION_STATUS_ICON = {
    "OK": "🟢",
    "ATTRIBUTE_ONLY": "🟡",
    "INSUFFICIENT_EVIDENCE": "🟠",
}


# ── Bước pipeline — canonical: rfp.graph.PIPELINE_STAGES ───────────────────
STAGE_VI = {
    "parse_input": "Đọc RFP",
    "check_complete": "Kiểm tra đủ thông tin",
    "ask_user": "Hỏi thêm người dùng",
    "route_reference_rfp": "Chọn hồ sơ tham chiếu",
    "plan_sections": "Lập dàn ý",
    "retrieve_per_chapter": "Truy xuất nguồn",
    "generate_per_section": "Sinh từng mục",
    "review": "Review chất lượng",
    "assemble": "Ghép + kiểm tra cuối",
}

# Trạng thái một bước khi hiển thị. "blocked" KHÁC "failed": bị guard chặn là
# hành vi ĐÚNG của hệ thống (thà không nộp còn hơn nộp hồ sơ sai chứng chỉ),
# còn "failed" là hỏng kỹ thuật. Gộp hai cái làm một là nói dối người dùng.
STAGE_STATUS_VI = {
    "pending": "chưa tới",
    "running": "đang chạy",
    "completed": "xong",
    "skipped": "bỏ qua",
    "blocked": "bị chặn xuất bản",
    "failed": "lỗi",
}

STAGE_STATUS_ICON = {
    "pending": "⚪",
    "running": "🟡",
    "completed": "✅",
    "skipped": "⏭️",
    "blocked": "🛑",
    "failed": "❌",
}


# ── Kết quả chạy (state["status"]) ─────────────────────────────────────────
RUN_STATUS_VI = {
    "completed": "Sinh xong hồ sơ nháp",
    "ask_user": "Cần bổ sung thông tin đầu vào",
    "partial": "Chạy dở — nhà cung cấp LLM không phản hồi",
    "guard_blocked": "Bị chặn xuất bản do vi phạm quy tắc an toàn",
    "needs_input": "Thiếu thông tin đầu vào",
    "ready": "Đủ thông tin, đang xử lý",
}


# ── Cách chọn hồ sơ tham chiếu ─────────────────────────────────────────────
ROUTE_METHOD_VI = {
    "industry": "khớp đúng ngành",
    "embedding": "gần nhất theo ngữ nghĩa",
    "none": "không có hồ sơ nào khớp",
}


# ── Review — canonical: rfp.review.SEVERITIES / ISSUE_TYPES ────────────────
SEVERITY_VI = {
    "critical": "Nghiêm trọng",
    "major": "Đáng sửa",
    "minor": "Nhỏ",
}

ISSUE_TYPE_VI = {
    "structure": "Bố cục",
    "tone": "Giọng văn",
    "redundancy": "Trùng lặp",
    "unclear_reference": "Tham chiếu không rõ",
    "format": "Định dạng",
}

PERSONA_VI = {
    "coverage": "Bố cục & mạch ý",
    "quality": "Văn phong",
}


# ── Vì sao một bước bị bỏ qua ──────────────────────────────────────────────
# "Bỏ qua" ở đây là tối ưu có chủ đích, không phải hỏng. Không nói lý do thì
# người xem đọc ⏭ thành "chỗ này lỗi/thiếu", và đó là hiểu sai đắt nhất trên
# màn hình này.
SKIP_REASON_VI = {
    "ask_user": "RFP đã đủ thông tin bắt buộc",
    "route_reference_rfp": "cần bổ sung thông tin đầu vào trước",
    "plan_sections": "cần bổ sung thông tin đầu vào trước",
    "retrieve_per_chapter": "cần bổ sung thông tin đầu vào trước",
    "generate_per_section": "cần bổ sung thông tin đầu vào trước",
    "review": "cần bổ sung thông tin đầu vào trước",
    "assemble": "cần bổ sung thông tin đầu vào trước",
}

# Lý do dùng cho một CHƯƠNG và cho các bước con của khâu truy xuất: chương nào
# mọi yêu cầu đã khớp thẳng bảng năng lực thì không cần tìm trong hồ sơ cũ.
ATTRIBUTE_COVERED_REASON = (
    "mọi yêu cầu khớp thẳng bảng năng lực, không cần tìm trong hồ sơ quá khứ"
)

SKIP_LEGEND = "bỏ qua có chủ đích để tiết kiệm — không phải lỗi"


# ── Bước con của khâu truy xuất ────────────────────────────────────────────
RETRIEVAL_STAGE_VI = {
    "attribute": "Đối chiếu bảng năng lực",
    "query_embed": "Tìm câu gần nghĩa",
    "rerank": "Chấm điểm lại",
    "mmr": "Chọn câu đa dạng",
}


# ── Ghi chú mục (coverage.py sinh ra bằng tiếng Nhật) ─────────────────────
# Ba câu ATTRIBUTE_ONLY là chuỗi cố định trong `assess_coverage`, khớp nguyên
# văn được. Ghi chú lạ thì giữ nguyên tiếng Nhật, không đoán.
SECTION_NOTE_VI = {
    "RFPに対応する原章がないため、能力表のみで構成しました。": (
        "RFP không có chương tương ứng, nên mục này chỉ dựng từ bảng năng lực "
        "công ty."
    ),
    "属性の完全一致により検索を省略し、能力表のみで回答しました。": (
        "Mọi yêu cầu khớp thẳng bảng năng lực nên bỏ qua bước tìm trong hồ sơ "
        "cũ."
    ),
    "参照可能な先行事例がないため、能力表のみで回答しました。": (
        "Không có hồ sơ quá khứ nào dùng được, nên chỉ trả lời bằng bảng năng "
        "lực công ty."
    ),
}

# Lý do một yêu cầu không có căn cứ — bản dịch của
# 「対応する能力・先行事例の根拠がありません」.
NO_EVIDENCE_REASON = "Chưa có năng lực công ty hay hồ sơ quá khứ nào làm căn cứ"


# ── Tên cấu hình trong eval/results ───────────────────────────────────────
RUN_LABEL_VI = {
    "V4_V5_A2": "Cấu hình đề xuất (V4+V5+A2)",
    "only_capability": "Chỉ dùng bảng năng lực",
    "only_precedent": "Chỉ dùng hồ sơ quá khứ",
    "only_precedent_no_guard": "Chỉ hồ sơ quá khứ, tắt lưới an toàn",
    "only_precedent_no_guard_no_quarantine": "Tắt cả lọc lúc nạp dữ liệu",
    "force_precedent_k5_no_guard": "Ép k=5, TẮT lưới an toàn",
    "force_precedent_k5_with_guard": "Ép k=5, BẬT lưới an toàn",
    "V0_A2_dense_only": "Thang V0 — chỉ tìm theo ngữ nghĩa",
    "V1_A2_hybrid": "Thang V1 — thêm tìm theo từ khoá (BM25)",
    "V4_A2_mmr": "Thang V4 — thêm chọn câu đa dạng (MMR)",
    "rfp001": "Lượt chạy thử một RFP",
    "review_cost": "Đo chi phí vòng review (không phải lượt đánh giá)",
}

RUN_SUFFIX_VI = {
    "_intersect": " · chấm trên cùng tập mẫu",
    ".det": " · chỉ đo tất định",
}


def run_label(name: str) -> str:
    """Nhãn dễ đọc cho một file kết quả; giữ được cả hậu tố."""
    suffix_text = ""
    base = name
    for suffix, text in RUN_SUFFIX_VI.items():
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            suffix_text = text
            break
    return f"{RUN_LABEL_VI.get(base, base)}{suffix_text}"


def label(mapping: dict[str, str], key, *, unknown: str = UNKNOWN) -> str:
    """Nhãn hiển thị; giá trị lạ thì trả về NGUYÊN VĂN, không nuốt.

    Nuốt giá trị lạ thành "—" sẽ giấu mất một enum mới chưa được dịch. Hiện
    nguyên văn thì người vận hành thấy ngay là thiếu bản dịch.
    """
    if key is None or key == "":
        return unknown
    return mapping.get(key, str(key))


def to_raw(mapping: dict[str, str], labels) -> list[str]:
    """Nhãn tiếng Việt -> giá trị gốc, dùng cho bộ lọc trên UI.

    Nhãn không nằm trong mapping được giữ nguyên: đó là giá trị gốc chưa có bản
    dịch, lọc theo nó vẫn phải đúng.
    """
    reverse = {vi: raw for raw, vi in mapping.items()}
    return [reverse.get(item, item) for item in labels]
