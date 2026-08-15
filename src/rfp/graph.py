import argparse
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .parsers.rfp_parser import CHAPTER_RE, INDUSTRY_RE, RFPParser
from .schema import RFP


SECTION_DEFINITIONS = (
    ("company_overview", "会社概要", "Tổng quan công ty"),
    ("proposal_overview", "提案の概要", "Tổng quan đề xuất"),
    ("implementation_experience", "導入実績", "Kinh nghiệm triển khai"),
    (
        "certification_compliance",
        "認証・コンプライアンス",
        "Chứng nhận và tuân thủ",
    ),
    ("delivery_structure", "推進体制", "Cơ cấu triển khai"),
)

SECTION_KEYS = {
    "key",
    "title_ja",
    "title_vi",
    "source_chapters",
    "sentences",
    "status",
    "note",
}
TRACE_KEYS = {"llm_calls", "retrieval_stats", "dedup"}


class GraphState(TypedDict, total=False):
    input_text: str
    rfp: RFP | None
    industry: str
    chapter_count: int
    parse_error: str | None
    missing: list[str]
    status: str
    message: str
    reference_rfp: dict[str, Any]
    chapters: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    proposal: str
    trace: dict[str, Any]


def _empty_trace() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "retrieval_stats": {},
        "dedup": {"before": 0, "after": 0},
        "path": [],
    }


def _with_trace(state: GraphState, node: str) -> dict[str, Any]:
    current = state.get("trace", _empty_trace())
    return {
        "llm_calls": current.get("llm_calls", 0),
        "retrieval_stats": dict(current.get("retrieval_stats", {})),
        "dedup": dict(current.get("dedup", {"before": 0, "after": 0})),
        "path": [*current.get("path", []), node],
    }


def parse_input(state: GraphState) -> GraphState:
    text = state["input_text"]
    industry_match = INDUSTRY_RE.search(text)
    chapter_count = len(CHAPTER_RE.findall(text))
    parsed_rfp: RFP | None = None
    parse_error: str | None = None

    try:
        # BƯỚC 4 cấm LLM: parser regex thất bại thì giữ trạng thái thiếu dữ liệu.
        parsed_rfp = RFPParser().parse(text, allow_llm_fallback=False)
    except ValueError as error:
        parse_error = str(error)

    return {
        "rfp": parsed_rfp,
        "industry": (
            parsed_rfp.industry
            if parsed_rfp is not None
            else industry_match.group(1).strip()
            if industry_match is not None
            else ""
        ),
        "chapter_count": (
            len(parsed_rfp.chapters) if parsed_rfp is not None else chapter_count
        ),
        "parse_error": parse_error,
        "trace": _with_trace(state, "parse_input"),
    }


def check_complete(state: GraphState) -> GraphState:
    missing: list[str] = []
    if not state.get("industry"):
        missing.append("industry (発注業種)")
    if state.get("chapter_count", 0) == 0:
        missing.append("chapters (第N章)")
    return {
        "missing": missing,
        "status": "needs_input" if missing else "ready",
        "trace": _with_trace(state, "check_complete"),
    }


def route_after_check(state: GraphState) -> str:
    return "ask_user" if state.get("missing") else "continue"


def ask_user(state: GraphState) -> GraphState:
    missing = ", ".join(state.get("missing", []))
    return {
        "status": "ask_user",
        "message": f"Chưa đủ thông tin để sinh nháp: thiếu {missing}.",
        "trace": _with_trace(state, "ask_user"),
    }


def route_reference_rfp(state: GraphState) -> GraphState:
    rfp = state.get("rfp")
    return {
        "reference_rfp": {
            "rfp_id": rfp.rfp_id if rfp is not None else "",
            "method": "placeholder",
            "score": None,
        },
        "trace": _with_trace(state, "route_reference_rfp"),
    }


def plan_sections(state: GraphState) -> GraphState:
    return {
        "sections": [
            {
                "key": key,
                "title_ja": title_ja,
                "title_vi": title_vi,
                "source_chapters": [],
                "sentences": [],
                "status": None,
                "note": None,
            }
            for key, title_ja, title_vi in SECTION_DEFINITIONS
        ],
        "trace": _with_trace(state, "plan_sections"),
    }


def retrieve_per_chapter(state: GraphState) -> GraphState:
    rfp = state.get("rfp")
    chapters = (
        [
            {
                "id": chapter.id,
                "title": chapter.title,
                "requirements": [
                    {"req_id": requirement.req_id, "text": requirement.text}
                    for requirement in chapter.requirements
                ],
                "retrieval": {
                    "stages": {},
                    "candidates": [],
                    "selected": [],
                },
            }
            for chapter in rfp.chapters
        ]
        if rfp is not None
        else []
    )
    return {
        "chapters": chapters,
        "trace": _with_trace(state, "retrieve_per_chapter"),
    }


def generate_per_section(state: GraphState) -> GraphState:
    return {
        "sections": [
            {
                "key": section["key"],
                "title_ja": section["title_ja"],
                "title_vi": section["title_vi"],
                "source_chapters": list(section["source_chapters"]),
                "sentences": list(section["sentences"]),
                "status": section["status"],
                "note": section["note"],
            }
            for section in state.get("sections", [])
        ],
        "trace": _with_trace(state, "generate_per_section"),
    }


def assemble(state: GraphState) -> GraphState:
    proposal = "\n\n".join(
        f"{index}. {section['title_ja']}\n"
        + "".join(sentence.get("text", "") for sentence in section["sentences"])
        for index, section in enumerate(state.get("sections", []), start=1)
    )
    return {
        "proposal": proposal,
        "status": "completed",
        "trace": _with_trace(state, "assemble"),
    }


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("parse_input", parse_input)
    builder.add_node("check_complete", check_complete)
    builder.add_node("ask_user", ask_user)
    builder.add_node("route_reference_rfp", route_reference_rfp)
    builder.add_node("plan_sections", plan_sections)
    builder.add_node("retrieve_per_chapter", retrieve_per_chapter)
    builder.add_node("generate_per_section", generate_per_section)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "parse_input")
    builder.add_edge("parse_input", "check_complete")
    builder.add_conditional_edges(
        "check_complete",
        route_after_check,
        {
            "ask_user": "ask_user",
            "continue": "route_reference_rfp",
        },
    )
    builder.add_edge("ask_user", END)
    builder.add_edge("route_reference_rfp", "plan_sections")
    builder.add_edge("plan_sections", "retrieve_per_chapter")
    builder.add_edge("retrieve_per_chapter", "generate_per_section")
    builder.add_edge("generate_per_section", "assemble")
    builder.add_edge("assemble", END)
    return builder.compile()


GRAPH = build_graph()


def run_graph(text: str) -> GraphState:
    return GRAPH.invoke(
        {
            "input_text": text,
            "reference_rfp": {"rfp_id": "", "method": "none", "score": None},
            "chapters": [],
            "sections": [],
            "trace": _empty_trace(),
        }
    )


def check_schema(state: GraphState) -> None:
    sections = state.get("sections", [])
    if len(sections) != len(SECTION_DEFINITIONS):
        raise AssertionError(f"sections phải có 5 phần tử, hiện có {len(sections)}")
    for index, section in enumerate(sections):
        if set(section) != SECTION_KEYS:
            missing = sorted(SECTION_KEYS - set(section))
            extra = sorted(set(section) - SECTION_KEYS)
            raise AssertionError(
                f"sections[{index}] sai schema: missing={missing}, extra={extra}"
            )

    trace = state.get("trace")
    if not isinstance(trace, dict):
        raise AssertionError("trace phải là dict")
    missing_trace = TRACE_KEYS - set(trace)
    if missing_trace:
        raise AssertionError(f"trace thiếu khoá: {sorted(missing_trace)}")
    if not isinstance(trace["dedup"], dict):
        raise AssertionError("trace.dedup phải là dict")
    if set(trace["dedup"]) != {"before", "after"}:
        raise AssertionError("trace.dedup phải có before và after")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="RFP proposal graph")
    source = argument_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rfp", type=Path)
    source.add_argument("--text")
    argument_parser.add_argument("--check-schema", action="store_true")
    args = argument_parser.parse_args()

    text = (
        args.rfp.read_text(encoding="utf-8")
        if args.rfp is not None
        else args.text
    )
    result = run_graph(text)

    if args.check_schema:
        check_schema(result)
        section_key_text = " · ".join(
            (
                "key",
                "title_ja",
                "title_vi",
                "source_chapters",
                "sentences",
                "status",
                "note",
            )
        )
        print(f"schema=ok · sections=5 · keys={section_key_text}")
        print("trace=dict · llm_calls · retrieval_stats · dedup · path")
        return

    if result["status"] == "ask_user":
        print("route=ask_user")
        print(result["message"])
        return

    print(f"status={result['status']}")
    print(result["proposal"])


if __name__ == "__main__":
    main()
