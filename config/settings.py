import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
SYNTHETIC_DIR = ROOT_DIR / "synthetic"

RFP_DIR = SYNTHETIC_DIR / "rfps"
PROPOSAL_DIR = SYNTHETIC_DIR / "proposals"
CAPABILITY_PATH = SYNTHETIC_DIR / "capability_sheet.json"

# Corpus dùng cho ĐO LƯỜNG, tách khỏi corpus vận hành (v1.5).
#
# Hôm nay hai bên trỏ cùng chỗ nên không đổi hành vi gì. Tách ra vì bảng ablation
# §11.3, golden set và một test cổng (`test_hybrid_hallucination` cần câu
# 「在庫精度を20%向上」 có thật trong corpus) đều neo theo dữ liệu hiện tại —
# người vận hành thêm hồ sơ mới vào `proposals/` là số liệu đo trôi và test đỏ,
# mà nguyên nhân thì rất khó truy.
#
# Khi bắt đầu thêm dữ liệu vận hành: chạy `python scripts/freeze_eval_corpus.py`
# để đóng băng corpus hiện tại rồi trỏ hai hằng số này sang bản đóng băng. Chưa
# chép sẵn vì nhân đôi 40 file khi chưa có nhu cầu là rác.
EVAL_PROPOSAL_DIR = PROPOSAL_DIR
EVAL_RFP_DIR = RFP_DIR

# Retrieval defaults. Business logic is implemented in later build steps.
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 8
MMR_TOP_K = 5

# Cờ ablation retrieval (bảng §11.3, thang V0→V1→V4→đề xuất — mỗi bậc một biến).
# run_ablation.py ghi đè tạm các cờ này; mặc định là cấu hình đề xuất (bật hết).
RETRIEVAL_USE_BM25 = True    # False -> dense-only (V0)
RETRIEVAL_USE_RERANK = True  # False -> giữ nguyên thứ tự hybrid, không cộng prior
RETRIEVAL_USE_MMR = True     # False -> lấy thẳng top-k văn bản duy nhất sau rerank

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

# Evaluation judge: model comes only from environment; parameters are pinned so
# Step 11 comparisons do not silently change between runs.
EVAL_JUDGE_MODEL = os.getenv("LLM_MODEL_EVAL", os.getenv("LLM_MODEL", ""))
EVAL_JUDGE_RUNS = 3
EVAL_JUDGE_TEMPERATURE = 0.0
EVAL_JUDGE_MAX_COMPLETION_TOKENS = 4000
EVAL_JUDGE_EFFORT = "low"

COST_PER_1M_INPUT = None
COST_PER_1M_OUTPUT = None

# Cache (Bước 5). PROMPT_VERSION/TEMPLATE_VERSION bump TAY mỗi khi sửa prompt
# hoặc template sinh câu: cache key gồm chúng, nên quên bump là dùng lại kết quả
# của prompt cũ mà không ai biết.
PROMPT_VERSION = "1"
TEMPLATE_VERSION = "1"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_TTL_SECONDS = 7 * 24 * 3600
CACHE_ENABLED = True

# Review loop (Bước 6). Chỉ soi chất lượng văn bản — compliance là việc của
# guard deterministic (BB-1/BB-2), xem docstring src/rfp/review.py.
REVIEW_ENABLED = True
MAX_REVIEW_ROUNDS = 3
# Số lệnh gọi persona chạy song song. Chỉ có một model nên persona phân hoá
# bằng prompt, không bằng model size.
MAX_REVIEW_WORKERS = 4

# Streamlit translation runs once after proposal assembly, outside the graph.
TRANSLATION_EFFORT = "low"
TRANSLATION_SYSTEM_PROMPT = (
    "Bạn là biên dịch viên Nhật-Việt. Dịch đầy đủ hai phần RFP và proposal sang "
    "tiếng Việt, giữ nguyên tiêu đề, mã requirement, số liệu và nhãn [RFP]/[PROPOSAL]. "
    "Chỉ trả bản dịch, không bình luận thêm."
)
