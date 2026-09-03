"""Đo golden set phủ được những trục nào — và hở ở đâu.

Đếm số ca test không nói lên điều gì: 39 ca đều đánh vào một góc thì vẫn là
một góc. Cái đo được là **trục phủ**: mỗi thứ mà đề bài bắt hệ thống phải làm
đúng có ít nhất một ca canh nó không.

Vũ trụ của mỗi trục lấy từ chính nguồn chân lý của sản phẩm (capability sheet,
kho proposal), không chép tay — thêm một chứng chỉ vào capability sheet là bảng
độ phủ tự động báo hở, không cần ai nhớ sửa file này.

Ba tầng theo đề bài: **phổ biến** (việc thường ngày) · **biên** (đầu vào thiếu,
méo) · **cấm-sai** (thứ sai một lần là hỏng cả hồ sơ).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable

from .schema import ASSERTION_KINDS, GoldenCase


ROOT_DIR = Path(__file__).resolve().parents[2]
CAPABILITY_PATH = ROOT_DIR / "synthetic" / "capability_sheet.json"
PROPOSAL_DIR = ROOT_DIR / "synthetic" / "proposals"

TIER_COMMON = "phổ biến"
TIER_EDGE = "biên"
TIER_FORBIDDEN = "cấm-sai"

CHAPTER_TITLES = (
    "調達概要",
    "業務要件",
    "技術要件",
    "セキュリティ要件",
    "納期・体制",
    "提案書記載事項",
)
UNSERVED_INDUSTRY = "医療"

CHAPTER_HEADER_RE = re.compile(r"^第(?P<number>\d+)章\s+(?P<title>.+?)\s*$")
REQUIREMENT_RE = re.compile(r"^(?P<req_id>\d+\.\d+)\s+(?P<text>.+)$")
INDUSTRY_RE = re.compile(r"^発注業種[：:]\s*(?P<industry>.+?)\s*$")

# Ba ca biên mà đề bài nêu đích danh: RFP-003 không có chương セキュリティ要件,
# RFP thiếu 発注業種, RFP viết liền không đánh số chương.
EDGE_NO_SECURITY = "thiếu chương セキュリティ要件"
EDGE_NO_INDUSTRY = "thiếu dòng 発注業種"
EDGE_NO_CHAPTER_NUMBER = "không đánh số 第N章"
EDGE_CASES = (EDGE_NO_SECURITY, EDGE_NO_INDUSTRY, EDGE_NO_CHAPTER_NUMBER)


@lru_cache(maxsize=1)
def capability_data() -> dict:
    return json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))


def in_scope_terms() -> tuple[str, ...]:
    data = capability_data()
    return tuple(data["capabilities"]) + tuple(data["certifications_held"])


def out_scope_terms() -> tuple[str, ...]:
    """Cũng chính là 6 chuỗi blocklist — cùng một danh sách, kiểm hai đầu khác nhau.

    Trục `out_scope` hỏi "RFP giả có hỏi tới nó chưa"; trục `blocklist` hỏi "có ca
    nào canh nó KHÔNG lọt ra hồ sơ chưa". Hở một trong hai là hở thật.
    """
    data = capability_data()
    return tuple(data["capabilities_NOT_offered"]) + tuple(
        data["certifications_NOT_held"]
    )


def industries() -> tuple[str, ...]:
    return tuple(capability_data()["industries_served"]) + (UNSERVED_INDUSTRY,)


@lru_cache(maxsize=1)
def leaked_client_names() -> tuple[str, ...]:
    """7 tên khách hàng riêng có thật trong kho proposal.

    Quét từ kho chứ không chép tay: thêm một hồ sơ cũ có tên khách mới là bảng
    độ phủ báo hở ngay.
    """
    from rfp.sanitize.leak import private_client_names

    names: set[str] = set()
    for path in sorted(PROPOSAL_DIR.glob("*.txt")):
        names.update(private_client_names(path.read_text(encoding="utf-8")))
    return tuple(sorted(names))


@dataclass(frozen=True)
class Axis:
    key: str
    label: str
    tier: str
    universe: tuple[str, ...]
    covered: tuple[str, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(item for item in self.universe if item not in set(self.covered))

    @property
    def total(self) -> int:
        return len(self.universe)

    @property
    def hit(self) -> int:
        return self.total - len(self.missing)

    @property
    def is_full(self) -> bool:
        return not self.missing


def requirements_of(case: GoldenCase) -> dict[str, str]:
    return {
        match.group("req_id"): match.group("text")
        for line in case.rfp_text.splitlines()
        if (match := REQUIREMENT_RE.match(line))
    }


def chapters_of(case: GoldenCase) -> list[str]:
    return [
        match.group("title")
        for line in case.rfp_text.splitlines()
        if (match := CHAPTER_HEADER_RE.match(line))
    ]


def industry_of(case: GoldenCase) -> str | None:
    for line in case.rfp_text.splitlines():
        match = INDUSTRY_RE.match(line)
        if match:
            return match.group("industry")
    return None


def _targets(cases: Iterable[GoldenCase], kind: str) -> set[str]:
    return {
        assertion.target
        for case in cases
        for assertion in case.assertions
        if assertion.kind == kind
    }


def _terms_bound_to(cases: Iterable[GoldenCase], kind: str, terms: Iterable[str]) -> set[str]:
    """Từ vựng nằm trong requirement mà requirement đó ĐƯỢC canh bởi `kind`.

    Chỉ kiểm "từ có xuất hiện trong RFP giả" là không đủ: xuất hiện mà không ca
    nào canh kết quả thì nó chưa được test.
    """
    found: set[str] = set()
    for case in cases:
        guarded = {
            assertion.target
            for assertion in case.assertions
            if assertion.kind == kind
        }
        for req_id, text in requirements_of(case).items():
            if req_id not in guarded:
                continue
            found.update(term for term in terms if term in text)
    return found


def _edge_cases_covered(cases: Iterable[GoldenCase]) -> set[str]:
    found: set[str] = set()
    for case in cases:
        chapters = chapters_of(case)
        if chapters and "セキュリティ要件" not in chapters:
            found.add(EDGE_NO_SECURITY)
        if industry_of(case) is None:
            found.add(EDGE_NO_INDUSTRY)
        if not chapters and requirements_of(case):
            found.add(EDGE_NO_CHAPTER_NUMBER)
    return found


def coverage_report(cases: Iterable[GoldenCase]) -> list[Axis]:
    """Bảng độ phủ. Thứ tự cố định để so hai lần chạy với nhau được."""
    case_list = list(cases)
    in_terms = in_scope_terms()
    out_terms = out_scope_terms()
    leaks = leaked_client_names()
    blocked = _targets(case_list, "must_not_contain")
    kinds_used = {
        assertion.kind for case in case_list for assertion in case.assertions
    }
    return [
        Axis(
            "in_scope",
            "Năng lực CÓ — phải trả lời được",
            TIER_COMMON,
            in_terms,
            tuple(sorted(_terms_bound_to(case_list, "must_cover", in_terms))),
        ),
        Axis(
            "chapters",
            "6 chương RFP đều có ca",
            TIER_COMMON,
            CHAPTER_TITLES,
            tuple(
                sorted(
                    {
                        chapter
                        for case in case_list
                        for chapter in chapters_of(case)
                        if chapter in CHAPTER_TITLES
                    }
                )
            ),
        ),
        Axis(
            "industries",
            f"業種 — gồm {UNSERVED_INDUSTRY} là ngành KHÔNG phục vụ",
            TIER_COMMON,
            industries(),
            tuple(
                sorted(
                    {
                        industry
                        for case in case_list
                        if (industry := industry_of(case)) is not None
                    }
                )
            ),
        ),
        Axis(
            "out_scope",
            "Năng lực KHÔNG có — phải ghi INSUFFICIENT_EVIDENCE",
            TIER_FORBIDDEN,
            out_terms,
            tuple(
                sorted(_terms_bound_to(case_list, "must_flag_insufficient", out_terms))
            ),
        ),
        Axis(
            "blocklist",
            "6 chuỗi cấm — phải có ca canh không lọt ra hồ sơ",
            TIER_FORBIDDEN,
            out_terms,
            tuple(sorted(term for term in out_terms if term in blocked)),
        ),
        Axis(
            "client_leak",
            "Tên khách hàng cũ — không được lộ sang hồ sơ mới",
            TIER_FORBIDDEN,
            leaks,
            tuple(sorted(name for name in leaks if name in blocked)),
        ),
        Axis(
            "assertion_kinds",
            "5 loại điều kiện đều được dùng",
            TIER_EDGE,
            tuple(sorted(ASSERTION_KINDS)),
            tuple(sorted(kinds_used & ASSERTION_KINDS)),
        ),
        Axis(
            "edge_cases",
            "Ca biên: RFP thiếu chương / thiếu ngành / không đánh số",
            TIER_EDGE,
            EDGE_CASES,
            tuple(
                item for item in EDGE_CASES if item in _edge_cases_covered(case_list)
            ),
        ),
    ]


def report_rows(axes: Iterable[Axis]) -> list[dict]:
    """Bảng cho UI. Ô 'thiếu' liệt kê ĐÍCH DANH, không chỉ nói còn mấy cái."""
    return [
        {
            "tầng": axis.tier,
            "trục": axis.label,
            "phủ": f"{axis.hit}/{axis.total}",
            "đạt": "✅" if axis.is_full else "⚠",
            "còn thiếu": "—" if axis.is_full else " · ".join(axis.missing),
        }
        for axis in axes
    ]


def missing_axes(axes: Iterable[Axis]) -> list[Axis]:
    return [axis for axis in axes if not axis.is_full]


def summary_line(axes: Iterable[Axis]) -> str:
    axis_list = list(axes)
    full = sum(axis.is_full for axis in axis_list)
    hit = sum(axis.hit for axis in axis_list)
    total = sum(axis.total for axis in axis_list)
    return f"{full}/{len(axis_list)} trục phủ đủ · {hit}/{total} mục"
