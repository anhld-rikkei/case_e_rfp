import json
import subprocess
import time
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def set_config(prec: int, cap: int):
    settings_path = ROOT / "config" / "settings.py"
    content = settings_path.read_text(encoding="utf-8")
    content = re.sub(r"PRECEDENTS_PER_CHAPTER\s*=\s*\d+", f"PRECEDENTS_PER_CHAPTER = {prec}", content)
    content = re.sub(r"CAPABILITY_FACTS_PER_SECTION\s*=\s*\d+", f"CAPABILITY_FACTS_PER_SECTION = {cap}", content)
    content = re.sub(r"COMPANY_FACTS_PER_SECTION\s*=\s*\d+", f"COMPANY_FACTS_PER_SECTION = {cap}", content)
    settings_path.write_text(content, encoding="utf-8")

def main():
    golden_dir = ROOT / "synthetic" / "golden_test_set"
    golden_files = [f for f in golden_dir.glob("*.json")]
    print(f"Found {len(golden_files)} golden cases.")
    
    rfp_paths = []
    for gf in golden_files:
        data = json.loads(gf.read_text(encoding="utf-8"))
        txt_path = ROOT / "eval" / "results" / f"temp5_{gf.stem}.txt"
        txt_path.write_text(data["rfp_text"], encoding="utf-8")
        rfp_paths.append(txt_path)
        
    try:
        print(f"\n{'='*40}\nRunning config: only_precedent_no_guard_no_quarantine\n{'='*40}")
        set_config(1, 0)
        
        out_file = ROOT / "eval" / "results" / "only_precedent_no_guard_no_quarantine.json"
        
        cmd = [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m", "eval.run_ragas",
            "--out", str(out_file),
            "--disable-guards",
            "--disable-quarantine"
        ]
        for p in rfp_paths:
            cmd.extend(["--rfp", str(p)])
            
        subprocess.run(cmd, check=True)
    finally:
        for p in rfp_paths:
            if p.exists():
                p.unlink()
        set_config(1, 2)

if __name__ == "__main__":
    main()
