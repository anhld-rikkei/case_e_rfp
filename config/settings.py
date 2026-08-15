from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT_DIR / "synthetic"

RFP_DIR = SYNTHETIC_DIR / "rfps"
PROPOSAL_DIR = SYNTHETIC_DIR / "proposals"
CAPABILITY_PATH = SYNTHETIC_DIR / "capability_sheet.json"

# Retrieval defaults. Business logic is implemented in later build steps.
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 8
MMR_TOP_K = 5

RERANK_DENSE_WEIGHT = 0.40
RERANK_BM25_WEIGHT = 0.30
RERANK_INDUSTRY_WEIGHT = 0.20
RERANK_SECTION_WEIGHT = 0.10

MMR_LAMBDA = 0.70
ROUTE_EMBEDDING_THRESHOLD = 0.35

# Generation defaults.
GENERATION_EFFORT = "low"
PRECEDENTS_PER_CHAPTER = 1
CAPABILITY_FACTS_PER_SECTION = 2
COMPANY_FACTS_PER_SECTION = 2
MAX_BRIDGES_PER_PROPOSAL = 1
BRIDGE_SECTION_PRIORITY = (
    "implementation_experience",
    "proposal_overview",
    "delivery_structure",
    "certification_compliance",
)
COVERAGE_MIN_SHARED_ANCHORS = 2

# Streamlit translation runs once after proposal assembly, outside the graph.
TRANSLATION_EFFORT = "low"
TRANSLATION_SYSTEM_PROMPT = (
    "Bạn là biên dịch viên Nhật-Việt. Dịch đầy đủ hai phần RFP và proposal sang "
    "tiếng Việt, giữ nguyên tiêu đề, mã requirement, số liệu và nhãn [RFP]/[PROPOSAL]. "
    "Chỉ trả bản dịch, không bình luận thêm."
)
