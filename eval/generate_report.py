import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "eval" / "results"

# Ô trống không phải số 0. File kết quả chạy trước khi có một số đo thì thiếu
# khoá đó, và in ra "0" là biến "chưa đo" thành "đã đo, sạch".
NOT_MEASURED = "—"


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


def format_row(id_col, conf_col, d):
    if not d:
        return f"| {id_col} | {conf_col} | | | | / | | | | | | | | | |"

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
        judge_seconds = seconds_by_stage.get("judge", 0.0)
        latency = f"{product_seconds:.1f}s SP / {judge_seconds:.1f}s judge"
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
        d = load_json(key)
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
    """Câu kết luận suy ra từ số trong bảng, không viết sẵn.

    Bản trước hardcode 'cấu hình cuối cùng ... bộc lộ fabrication > 0' trong khi
    chính bảng nó vừa sinh ra ghi 0 ở mọi dòng.
    """
    last = reach.get("only_precedent_no_guard_no_quarantine", {})
    no_guard = reach.get("only_precedent_no_guard", {})
    lines = ["> **Đọc bảng ablation (guard & ingest):**"]

    ctx = last.get("fabrication_in_context")
    prompt = last.get("fabrication_in_prompt")
    output = last.get("fabrication_count")
    leak_ctx = last.get("client_leak_in_context")
    leak_out = last.get("client_leak_count")

    if ctx is None or prompt is None or output is None:
        lines.append(
            "> Cấu hình `tắt cả quarantine lúc ingest` chưa có số đo đường đi của câu bịa "
            "(`fabrication_in_context` / `fabrication_in_prompt`). Thiếu hai số đó thì "
            "`fabrication = 0` **không diễn giải được**: không phân biệt nổi guard chặn, "
            "retriever không xếp lên, hay câu bịa vắng mặt trong index. Chạy lại eval bằng "
            "code hiện tại rồi sinh lại báo cáo."
        )
        return lines

    if no_guard.get("fabrication_count") == 0:
        lines.append(
            "> Dòng `không có capability sheet làm trọng tài` (tắt 6.2 + 6.4 + 7) ra "
            "`fabrication = 0` **không** chứng minh lưới an toàn lúc sinh là thừa: 4 câu bịa "
            "đã bị quarantine từ Bước 2 nên không có mặt trong index để retriever lấy ra."
        )

    if output > 0:
        lines.append(
            f"> Tắt cả quarantine: câu bịa vào index, lọt vào context retrieval {ctx} lần, "
            f"vào prompt sinh {prompt} lần, còn lại trong hồ sơ {output} lần. "
            f"Rò rỉ tên khách hàng: {leak_ctx} lần trong context, {leak_out} lần trong hồ sơ. "
            "Đây là dòng chứng minh BB-1/BB-2."
        )
        return lines

    if ctx > 0 and prompt == 0:
        lines.append(
            f"> Tắt cả quarantine: câu bịa **có** vào index và lọt vào context retrieval "
            f"{ctx} lần (rò rỉ: {leak_ctx} lần), nhưng **không câu nào tới được prompt sinh** "
            f"({prompt}), nên hồ sơ vẫn ra `fabrication = 0` · `leak = {leak_out}`. Chặn nằm ở "
            "phép chọn precedent — mỗi chương chỉ lấy `PRECEDENTS_PER_CHAPTER` câu sau khi ưu "
            "tiên câu cùng mục — chứ **không phải** ở guard, và cũng **không phải** vì câu bịa "
            "vắng mặt trong index."
        )
        lines.append(
            "> Kết luận trung thực: **bảng này chưa ép được hệ sinh ra ảo giác**, kể cả khi gỡ "
            "cả hai tầng lưới; nó **chưa** chứng minh BB-1/BB-2. Muốn có dòng chứng minh thì "
            "phải để câu bịa tới được prompt sinh — nâng `PRECEDENTS_PER_CHAPTER`, bỏ ưu tiên "
            "cùng mục, hoặc dựng RFP hỏi thẳng vào 認証・コンプライアンス. Đừng đọc ô 0 này "
            "thành 'lưới an toàn thừa': nó chỉ nói lưới an toàn **chưa có việc để làm**."
        )
        return lines

    lines.append(
        f"> Tắt cả quarantine: câu bịa vào context {ctx} lần, vào prompt sinh {prompt} lần, "
        f"còn trong hồ sơ {output} lần (rò rỉ: context {leak_ctx}, hồ sơ {leak_out}). "
        "Đọc đủ ba mốc này trước khi kết luận về lưới an toàn."
    )
    return lines


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
            prod = sum(v for k, v in t.items() if k in ("generate", "structured"))
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

    lines.append(format_row(configs[0][1], configs[0][2], load_json(configs[0][0])))
    lines.append("| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |")
    for key, id_col, conf_col in configs[1:]:
        lines.append(format_row(id_col, conf_col, load_json(key)))

    lines.append("")
    lines.append(f"`{NOT_MEASURED}` = lần chạy đó chưa đo cột này, **không phải** đo ra 0.")
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
