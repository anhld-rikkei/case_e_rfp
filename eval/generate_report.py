import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "eval" / "results"

# Ô trống không phải số 0. File kết quả chạy trước khi có một số đo thì thiếu
# khoá đó, và in ra "0" là biến "chưa đo" thành "đã đo, sạch".
NOT_MEASURED = "—"


def load_json(name):
    path_det = results_dir / f"{name}.det.json"
    path_ragas = results_dir / f"{name}.json"
    
    data = {}
    is_merged = False
    
    if path_ragas.exists():
        data = json.loads(path_ragas.read_text(encoding="utf-8"))
        
    if path_det.exists():
        det_data = json.loads(path_det.read_text(encoding="utf-8"))
        data["deterministic"] = det_data.get("deterministic", [])
        data["usage"] = det_data.get("usage", {})
        if path_ragas.exists():
            is_merged = True
            
    if not data:
        return None, False
        
    return data, is_merged


def format_mean_std(data, key):
    if key not in data:
        return ""
    return f"{data[key]['mean']:.3f}±{data[key]['std']:.3f}"


configs = [
    ("V4_V5_A2", "**V4+V5+A2**", "**đề xuất**"),
    ("only_capability", "— chỉ capability", "(ablation nguồn)"),
    ("only_precedent", "— chỉ precedent (kênh sinh)", "(ablation nguồn)"),
    ("only_precedent_no_guard", "— không có capability sheet làm trọng tài", "(ablation nguồn & guard)"),
    ("only_precedent_no_guard_no_quarantine", "— tắt cả quarantine lúc ingest", "(ablation ingest & guard)"),
    ("force_precedent_k5_no_guard", "— ép k=5, tắt guard", "(minh chứng BB-1/BB-2)"),
    ("force_precedent_k5_with_guard", "— ép k=5, BẬT guard", "(minh chứng BB-1/BB-2)"),
]


def sum_det(det_rows, key):
    """Tổng một cột đếm; None nếu lần chạy đó chưa đo cột này."""
    if not det_rows or any(key not in row for row in det_rows):
        return None
    return sum(int(row.get(key, 0)) for row in det_rows)


def fmt_count(value):
    if value is None:
        return NOT_MEASURED
    return f"**{value}**" if value > 0 else "0"


def format_row(id_col, conf_col, d_info):
    d, is_merged = d_info if d_info else ({}, False)
    if not d:
        return f"| {id_col} | {conf_col} | | | | / | | | | | | | | | |"

    if is_merged:
        id_col = f"{id_col}†"

    det_rows = d.get("deterministic", [])

    def mean_det(key):
        vals = [float(r.get(key, 0)) for r in det_rows if r.get("sections", 0) > 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    fabric = sum_det(det_rows, "fabrication_count")
    leak = sum_det(det_rows, "client_leak_count")
    # Cột này đo ảo giác lai CÒN LẠI trong hồ sơ, không đo số lần guard chạy.
    # trace.hybrid_blocked là guard hoạt động: tắt guard thì nó về 0 và đọc
    # thành "sạch hơn" — ngược hẳn nghĩa mũi tên ↓.
    hybrid = sum_det(det_rows, "hybrid_in_output")

    cov = mean_det("coverage")
    abst = mean_det("abstain_rate")
    cite = mean_det("citation_accuracy")

    ragas = d.get("ragas", {})
    usage = d.get("usage", {})
    seconds_by_stage = usage.get("seconds_by_stage")
    if seconds_by_stage:
        product_seconds = sum(v for k, v in seconds_by_stage.items() if k != "judge")
        if "judge" in seconds_by_stage:
            latency = f"{product_seconds:.1f}s SP / {seconds_by_stage['judge']:.1f}s judge"
        else:
            latency = f"{product_seconds:.1f}s SP / — judge"
    else:
        # Lần chạy cũ chỉ có tổng gộp, và tổng đó còn tính hụt thời gian judge
        # (một mốc t0 dùng chung cho các lệnh gọi song song).
        latency = f"{usage.get('seconds', 0):.1f}s gộp*"

    row = [
        id_col,
        conf_col,
        fmt_count(fabric),
        fmt_count(leak),
        fmt_count(hybrid),
        f"{cov:.3f} / {abst:.3f}",
        f"{cite:.3f}",
        format_mean_std(ragas, "context_precision"),
        format_mean_std(ragas, "context_recall"),
        format_mean_std(ragas, "noise_sensitivity"),
        format_mean_std(ragas, "faithfulness"),
        format_mean_std(ragas, "answer_relevancy"),
        "—",
        latency,
        "N/A",
    ]
    return "| " + " | ".join(row) + " |"


def poison_reach_rows(lines):
    """Bảng chẩn đoán: câu bịa/rò rỉ đi được tới mốc nào."""
    lines.append("### Chất độc đi tới đâu (giải thích cột fabric./leak ở trên)")
    lines.append("")
    lines.append("| Cấu hình | trong context retrieval | vào prompt sinh | guard chặn xuất bản | còn trong hồ sơ |")
    lines.append("|---|---|---|---|---|")

    reach = {}
    for key, id_col, _ in configs:
        d, _ = load_json(key) or ({}, False)
        det_rows = d.get("deterministic", []) if d else []
        entry = {
            name: sum_det(det_rows, name)
            for name in (
                "fabrication_in_context",
                "client_leak_in_context",
                "fabrication_in_prompt",
                "client_leak_in_prompt",
                "guard_blocked_publish",
                "fabrication_count",
                "client_leak_count",
            )
        }
        reach[key] = entry

        def pair(fab_key, leak_key):
            fab = entry[fab_key]
            leak = entry[leak_key]
            if fab is None or leak is None:
                return NOT_MEASURED
            return f"fab {fab} · leak {leak}"

        lines.append(
            f"| {id_col} | {pair('fabrication_in_context', 'client_leak_in_context')} "
            f"| {pair('fabrication_in_prompt', 'client_leak_in_prompt')} "
            f"| {fmt_count(entry.get('guard_blocked_publish'))} "
            f"| {pair('fabrication_count', 'client_leak_count')} |"
        )
    return reach


def interpretation(reach):
    no_guard = reach.get("force_precedent_k5_no_guard", {})
    with_guard = reach.get("force_precedent_k5_with_guard", {})
    
    ctx = no_guard.get("fabrication_in_context", 0)
    leak_ctx = no_guard.get("client_leak_in_context", 0)
    prompt = no_guard.get("fabrication_in_prompt", 0)
    
    fab_off = no_guard.get("fabrication_count", 0)
    leak_off = no_guard.get("client_leak_count", 0)
    
    blocked = with_guard.get("guard_blocked_publish", 0)
    
    return [
        "> **Cặp dòng ép k=5 — bằng chứng cho BB-2.**",
        f"> Hai dòng khác nhau đúng một biến. Cùng k=5, cùng tắt quarantine, cùng {ctx} câu bịa và",
        f"> {leak_ctx} câu rò rỉ lọt vào context, cùng {prompt} câu bịa vào prompt sinh. Chỉ khác",
        "> guard bật hay tắt.",
        f"> - **Tắt guard:** hồ sơ xuất ra chứa {fab_off} lần chuỗi cấm và {leak_off} lần tên khách",
        ">   hàng riêng. Ép được ảo giác — đây là ô khác 0 đầu tiên của bảng.",
        f"> - **Bật guard:** final guard Bước 7 ném GuardViolation ở {blocked}/9 RFP. Ô `fabric = 0`",
        ">   ở dòng này **không phải hồ sơ sạch mà là KHÔNG CÓ hồ sơ** — hệ thống trả về lỗi chứ",
        ">   không trả về tài liệu. Đọc ô đó phải đọc kèm cột \"guard chặn xuất bản\".",
        ">",
        "> Guard là **cầu dao, không phải bộ lọc**: nó không cứu được hồ sơ, nó chặn hồ sơ hỏng ra",
        "> khỏi cửa. Với hồ sơ thầu, tuyên bố sai chứng chỉ thì thà không nộp — nên đây là hành vi",
        "> đúng. Nhưng thứ khiến 5 dòng trên vừa `fabrication = 0` vừa **xuất bản được** là",
        "> quarantine tầng ingest, không phải guard.",
        ">",
        "> Claim-check 6.4 không đóng vai trò gì ở kịch bản này: bằng chứng nó đối chiếu chính là",
        "> câu nguồn bịa đó (`_evidence_text` trả `source_texts[source_id]`), nên nó phán VERIFIED.",
        "> Lưới duy nhất chặn được chuỗi cấm là regex ở Bước 7 — đúng như BB-2 quy định."
    ]


def main():
    lines = []

    base_data, _ = load_json(configs[0][0])
    if not base_data:
        print("No data found")
        return

    judge_model = base_data.get("judge", {}).get("model", "unknown")
    n_runs = base_data.get("judge", {}).get("runs", 3)

    total_prod = 0
    total_judge = 0
    prod_runs = 0
    judge_runs = 0
    
    for c in configs:
        path_det = results_dir / f"{c[0]}.det.json"
        path_ragas = results_dir / f"{c[0]}.json"
        
        src_path = path_det if path_det.exists() else path_ragas
        if src_path.exists():
            d = json.loads(src_path.read_text(encoding="utf-8"))
            t = d.get("usage", {}).get("tokens_by_stage", {})
            prod = sum(v for k, v in t.items() if k in ("generate", "structured"))
            total_prod += prod
            prod_runs += 1
            
        if path_ragas.exists():
            d = json.loads(path_ragas.read_text(encoding="utf-8"))
            jud = d.get("usage", {}).get("tokens_by_stage", {}).get("judge", 0)
            total_judge += jud
            if jud > 0:
                judge_runs += 1

    lines.append(f"Judge: {judge_model}, effort=low, n={n_runs} (mean±std)")
    lines.append("Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~94 requirement atom")
    lines.append(f"Tổng token Sản phẩm: {total_prod:,} (từ {prod_runs} lần chạy) | Tổng token Đo lường (Judge): {total_judge:,} (từ {judge_runs} lần chạy có RAGAS)")
    lines.append("")
    lines.append("| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    lines.append("| V0+A0 | Naive baseline | | | | / | | | | | | | | | |")
    lines.append("| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |")
    lines.append("| V4+A1 | +MMR | | | | / | | | | | | | | | |")
    lines.append("| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |")
    lines.append("| V6+A2 | +HyDE | | | | / | | | | | | | | | |")

    lines.append(format_row(configs[0][1], configs[0][2], load_json(configs[0][0])))
    lines.append("| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |")
    for key, id_col, conf_col in configs[1:]:
        lines.append(format_row(id_col, conf_col, load_json(key)))

    lines.append("")
    lines.append(f"`{NOT_MEASURED}` = lần chạy đó chưa đo cột này, **không phải** đo ra 0.")
    lines.append("`†` = Cột deterministic lấy từ lần chạy mới (.det.json), cột RAGAS lấy từ lần chạy cũ (.json).")
    lines.append(
        "`*` = tổng gộp sản phẩm + judge, từ lần chạy trước khi tách `seconds_by_stage`; "
        "phần judge trong đó thấp hơn thực tế."
    )
    lines.append("")

    reach = poison_reach_rows(lines)
    lines.append("")
    lines.extend(interpretation(reach))

    out = "\n".join(lines)
    (results_dir / "report.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
