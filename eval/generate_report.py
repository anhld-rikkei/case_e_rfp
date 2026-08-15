import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "eval" / "results"

def load_json(name):
    path = results_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def format_mean_std(data, key):
    if key not in data:
        return ""
    return f"{data[key]['mean']:.3f}±{data[key]['std']:.3f}"

configs = [
    ("V4_V5_A2", "**V4+V5+A2**", "**đề xuất**"),
    ("only_capability", "— chỉ capability", "(ablation nguồn)"),
    ("only_precedent", "— chỉ precedent (kênh sinh)", "(ablation nguồn)"),
    ("only_precedent_no_guard", "— không có capability sheet làm trọng tài", "(ablation nguồn & guard)"),
]

def main():
    lines = []
    
    base_data = None
    for c in configs:
        data = load_json(c[0])
        if data:
            base_data = data
            break
            
    if not base_data:
        print("No data found")
        return

    judge_model = base_data.get("judge", {}).get("model", "unknown")
    n_runs = base_data.get("judge", {}).get("runs", 3)
    
    total_prod = 0
    total_judge = 0
    for c in configs:
        d = load_json(c[0])
        if d and "usage" in d:
            t = d["usage"].get("tokens_by_stage", {})
            prod = sum(v for k,v in t.items() if k in ("generate", "structured"))
            jud = t.get("judge", 0)
            total_prod += prod
            total_judge += jud

    lines.append(f"Judge: {judge_model}, effort=low, n={n_runs} (mean±std)")
    lines.append("Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~94 requirement atom")
    lines.append(f"Tổng token Sản phẩm: {total_prod:,} | Tổng token Đo lường (Judge): {total_judge:,}")
    lines.append("")
    lines.append("| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    lines.append("| V0+A0 | Naive baseline | | | | / | | | | | | | | | |")
    lines.append("| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |")
    lines.append("| V4+A1 | +MMR | | | | / | | | | | | | | | |")
    lines.append("| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |")
    lines.append("| V6+A2 | +HyDE | | | | / | | | | | | | | | |")

    def format_row(id_col, conf_col, d):
        if not d:
            return f"| {id_col} | {conf_col} | | | | / | | | | | | | | | |"
            
        det_rows = d.get("deterministic", [])
        
        def mean_det(key):
            vals = [float(r.get(key, 0)) for r in det_rows if r.get("sections", 0) > 0]
            if not vals: return 0.0
            return sum(vals)/len(vals)
            
        def sum_det(key):
            return sum(int(r.get(key, 0)) for r in det_rows)

        fabric = sum_det("fabrication_count")
        leak = sum_det("client_leak_count")
        hybrid = sum_det("hybrid_blocked")
        
        cov = mean_det("coverage")
        abst = mean_det("abstain_rate")
        cite = mean_det("citation_accuracy")
        
        ragas = d.get("ragas", {})
        
        row = [
            id_col,
            conf_col,
            f"**{fabric}**" if fabric > 0 else "0",
            f"**{leak}**" if leak > 0 else "0",
            f"**{hybrid}**" if hybrid > 0 else "0",
            f"{cov:.3f} / {abst:.3f}",
            f"{cite:.3f}",
            format_mean_std(ragas, "context_precision"),
            format_mean_std(ragas, "context_recall"),
            format_mean_std(ragas, "noise_sensitivity"),
            format_mean_std(ragas, "faithfulness"),
            format_mean_std(ragas, "answer_relevancy"),
            "—",
            f"{d.get('usage', {}).get('seconds', 0):.1f}s",
            "N/A"
        ]
        return "| " + " | ".join(row) + " |"

    # Print V4_V5_A2
    lines.append(format_row(configs[0][1], configs[0][2], load_json(configs[0][0])))
    
    # Print ReAct placeholder
    lines.append("| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |")
    
    # Print only_capability
    lines.append(format_row(configs[1][1], configs[1][2], load_json(configs[1][0])))
    
    # Print only_precedent
    lines.append(format_row(configs[2][1], configs[2][2], load_json(configs[2][0])))
    
    # Print only_precedent_no_guard
    lines.append(format_row(configs[3][1], configs[3][2], load_json(configs[3][0])))
    
    out = "\n".join(lines)
    (results_dir / "report.md").write_text(out, encoding="utf-8")
    
if __name__ == "__main__":
    main()
