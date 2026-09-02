"""Dịch một ca golden sang lời người chấm đọc được.

`must_flag_insufficient · 4.1` không nói gì với người test. Cái họ cần biết là:
RFP giả này đưa vào cái gì, và máy sẽ chấm đúng/sai dựa trên điều gì.

Để riêng khỏi `config/display_vi.py`: file kia là nhãn cho màn hồ sơ (người đọc
là người làm thầu), còn ở đây phải tra `capability_sheet.json` mới nói được vì
sao một điều kiện là đúng — đó là logic, không phải bảng nhãn.
"""
from __future__ import annotations

from dataclasses import dataclass

from .coverage import (
    capability_data,
    chapters_of,
    industry_of,
    leaked_client_names,
    requirements_of,
)
from .schema import Assertion, GoldenCase


AXIS_PURPOSE = {
    "in_scope": "Đường bình thường — RFP hỏi đúng việc công ty làm được.",
    "out_scope": "Bẫy năng lực — RFP hỏi việc công ty KHÔNG làm được.",
    "blocklist": "Bẫy chứng chỉ — chuỗi cấm không được lọt ra hồ sơ.",
    "client_leak": (
        "Bẫy rò rỉ — tên khách hàng trong hồ sơ cũ không được sang hồ sơ khách mới."
    ),
    "chapters": "Chương ít gặp — mọi chương RFP đều phải có đường xử lý.",
    "industries": "Theo 業種 — gồm cả ngành công ty không phục vụ.",
    "edge_cases": "Đầu vào méo — thiếu chương, thiếu ngành, hoặc không đánh số.",
    "assertion_kinds": "Nhánh rẽ — dừng lại hỏi người dùng, hoặc đổi ngành.",
}

MUTATION_PURPOSE = {
    "capabilities_NOT_offered": "Chèn một yêu cầu ngoài năng lực vào RFP thật.",
    "certifications_NOT_held": "Chèn một chứng chỉ công ty không có vào RFP thật.",
    "industry_outside_served": "Đổi sang ngành công ty KHÔNG phục vụ.",
    "industry_inside_served": "Đổi sang một ngành khác vẫn trong phạm vi phục vụ.",
    "missing_industry": "Xoá dòng 業種 để xem hệ thống có dừng lại hỏi không.",
    "parser_fallback": "Bỏ đánh số 第N章 để xem còn tách được yêu cầu không.",
}


# Tên trường trong assertion là tên kỹ thuật; người chấm cần thấy đúng cái ô
# họ nhìn trong RFP.
ROUTE_TARGET_VI = {
    "completed": "chạy xong và ra được hồ sơ",
    "ask_user": "dừng lại hỏi người dùng",
}

MISSING_FIELD_VI = {
    "industry": "業種 (ngành đặt hàng)",
    "chapter": "第N章 (tiêu đề chương)",
    "requirement": "yêu cầu cần đáp",
}


@dataclass(frozen=True)
class Check:
    icon: str
    what: str
    why: str


@dataclass(frozen=True)
class CaseBrief:
    purpose: str
    tier: str
    inputs: list[str]
    checks: list[Check]


def _tier_of(case: GoldenCase) -> str:
    tier = case.metadata.get("tier")
    if tier:
        return str(tier)
    if any(item.kind == "must_flag_insufficient" for item in case.assertions):
        return "cấm-sai"
    if any(item.kind in {"must_route", "must_ask_user"} for item in case.assertions):
        return "biên"
    return "phổ biến"


def purpose_of(case: GoldenCase) -> str:
    axis = case.metadata.get("axis")
    if axis in AXIS_PURPOSE:
        return AXIS_PURPOSE[axis]
    mutation = case.metadata.get("mutation")
    if mutation in MUTATION_PURPOSE:
        # Ca viết tay ghi `base_rfp_id`, ca sinh máy ghi `base_case_id`. Không
        # có thì bỏ luôn phần trong ngoặc, đừng in "nền: RFP nền".
        base = case.metadata.get("base_case_id") or case.metadata.get("base_rfp_id")
        return (
            f"{MUTATION_PURPOSE[mutation]} (nền: {base})"
            if base
            else MUTATION_PURPOSE[mutation]
        )
    if case.source == "original":
        return "RFP thật, chưa bị đục sửa — đường chạy bình thường."
    if any(item.kind == "must_flag_insufficient" for item in case.assertions):
        return "Bẫy năng lực — RFP hỏi việc công ty KHÔNG làm được."
    return "Đường bình thường — RFP hỏi đúng việc công ty làm được."


def input_lines(case: GoldenCase) -> list[str]:
    industry = industry_of(case)
    chapters = chapters_of(case)
    requirements = requirements_of(case)
    lines = [
        f"業種 (ngành): **{industry}**"
        if industry
        else "**Không ghi 業種** — cố ý bỏ để xem hệ thống có dừng lại hỏi không"
    ]
    if chapters:
        lines.append(f"{len(chapters)} chương: {' · '.join(chapters)}")
    else:
        lines.append("**Không có tiêu đề 第N章** — cố ý bỏ đánh số chương")
    lines.append(f"{len(requirements)} yêu cầu cần đáp ứng")
    return lines


def _quote(text: str, limit: int = 46) -> str:
    trimmed = text if len(text) <= limit else text[: limit - 1] + "…"
    return f"「{trimmed}」"


def _requirement_label(case: GoldenCase, req_id: str) -> str:
    text = requirements_of(case).get(req_id)
    return f"Mục {req_id} {_quote(text)}" if text else f"Mục {req_id}"


def _forbidden_why(term: str) -> str:
    data = capability_data()
    if term in data["certifications_NOT_held"]:
        return f"Công ty KHÔNG có chứng chỉ {term} — viết vào là tuyên bố sai."
    if term in data["capabilities_NOT_offered"]:
        return f"Công ty KHÔNG làm {term} — nhận là hứa quá năng lực."
    if term in leaked_client_names():
        return (
            f"{term} là tên khách hàng trong hồ sơ cũ — "
            "lộ sang hồ sơ khách khác là rò rỉ."
        )
    return "Chuỗi này không được xuất hiện trong hồ sơ."


def describe_assertion(case: GoldenCase, assertion: Assertion) -> Check:
    if assertion.kind == "must_cover":
        return Check(
            "✅",
            f"{_requirement_label(case, assertion.target)} → phải có câu trả lời",
            "Yêu cầu này nằm trong năng lực thật, bỏ trống là sót.",
        )
    if assertion.kind == "must_flag_insufficient":
        return Check(
            "🚫",
            f"{_requirement_label(case, assertion.target)} → phải ghi "
            "“không đủ căn cứ”",
            "Không có thật thì nói không có. Viết bừa cho đẹp hồ sơ là sai.",
        )
    if assertion.kind == "must_not_contain":
        return Check(
            "🚫",
            f"Cả hồ sơ không được xuất hiện {_quote(assertion.target)}",
            _forbidden_why(assertion.target),
        )
    if assertion.kind == "must_route":
        if assertion.target.startswith("industry:"):
            industry = assertion.target.split(":", 1)[1]
            return Check(
                "↪",
                f"Phải đọc đúng ngành **{industry}** rồi mới đi tìm hồ sơ cũ",
                "Đọc nhầm ngành là lấy nhầm hồ sơ tham chiếu.",
            )
        return Check(
            "↪",
            "Phải "
            f"**{ROUTE_TARGET_VI.get(assertion.target, assertion.target)}**",
            "RFP viết khác thường vẫn phải ra được hồ sơ, không được chết giữa chừng.",
        )
    if assertion.kind == "must_ask_user":
        return Check(
            "❓",
            f"Thiếu **{MISSING_FIELD_VI.get(assertion.target, assertion.target)}** "
            "→ phải dừng lại hỏi người dùng",
            "Thiếu thông tin thì hỏi, không được tự đoán rồi chạy tiếp.",
        )
    return Check("•", f"{assertion.kind} · {assertion.target}", assertion.reason)


def describe_case(case: GoldenCase) -> CaseBrief:
    return CaseBrief(
        purpose=purpose_of(case),
        tier=_tier_of(case),
        inputs=input_lines(case),
        checks=[describe_assertion(case, item) for item in case.assertions],
    )
