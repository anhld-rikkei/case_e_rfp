from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT_DIR / "synthetic"

RFP_DIR = SYNTHETIC_DIR / "rfps"
PROPOSAL_DIR = SYNTHETIC_DIR / "proposals"
CAPABILITY_PATH = SYNTHETIC_DIR / "capability_sheet.json"

# Retrieval defaults. Business logic is implemented in later build steps.
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 8
MMR_TOP_K = 12
