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


# Thang retrieval V0→V1→V4: chunk câu + graph A2 giữ nguyên, mỗi bậc đổi đúng
# một biến retrieval (dense-only → +BM25 → +MMR; dòng đề xuất thêm rerank prior).
# Không nằm trong `configs` vì bảng "chất độc đi tới đâu" không cần chúng
# (guard + quarantine đều bật, mọi ô đều 0).
retrieval_ladder = [
    ("V0_A2_dense_only", "V0+A2", "naive: dense-only‡"),
    ("V1_A2_hybrid", "V1+A2", "+sentence, hybrid‡"),
    ("V4_A2_mmr", "V4+A2", "+MMR‡"),
]

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


def same_sample_rows(lines):
    """Bảng phụ: V0 vs V1 chấm trên CÙNG tập atom (khử hiệu ứng thành phần mẫu)."""
    pairs = [
        ("V0_A2_dense_only_intersect", "V0+A2", "dense-only"),
        ("V1_A2_hybrid_intersect", "V1+A2", "+BM25 (hybrid)"),
    ]
    loaded = [(id_col, conf, load_json(key)[0]) for key, id_col, conf in pairs]
    if any(d is None or "ragas" not in d for _, _, d in loaded):
        return None

    n = loaded[0][2].get("restricted_sample_count")
    lines.append(f"### So cùng tập mẫu ({n} atom cả hai cấu hình đều trả lời)")
    lines.append("")
    lines.append(
        "| # | Cấu hình | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | "
        "ans_rel↑ | ctx_entity_recall↑ | faithful.↑ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for id_col, conf, d in loaded:
        ragas = d["ragas"]
        row = [id_col, conf] + [
            format_mean_std(ragas, name)
            for name in (
                "context_precision",
                "context_recall",
                "noise_sensitivity",
                "answer_relevancy",
                "context_entity_recall",
                "faithfulness",
            )
        ]
        lines.append("| " + " | ".join(row) + " |")
    return {id_col: d["ragas"] for id_col, _, d in loaded}


def sample_composition_note(same_sample):
    """Cảnh báo đọc bảng: RAGAS chỉ chấm atom CÓ answer."""
    note = [
        "> **Cảnh báo khi đọc cột RAGAS: hiệu ứng thành phần mẫu.**",
        "> RAGAS chỉ chấm được atom **có answer**; atom bị abstain không vào mẫu. Nên hai cấu",
        "> hình khác `abstain_rate` được chấm trên **hai tập mẫu khác nhau**, và so trực tiếp hai",
        "> trung bình đó là so hai thứ khác nhau — không phải so chất lượng retrieval.",
        ">",
        "> Cụ thể ở bảng trên: V0 dense-only chỉ trả lời 30/83 atom, V1 +BM25 trả lời 36/83.",
        "> Sáu atom V1 trả lời thêm là các atom **khó hơn** (V0 bỏ trống vì không tìm ra bằng",
        "> chứng), nên chúng kéo trung bình RAGAS của V1 xuống. Đọc nguyên bảng gộp sẽ ra kết",
        "> luận sai rằng \"naive dense-only tốt hơn hybrid\".",
    ]
    if not same_sample:
        note.append(
            "> Bảng so cùng tập mẫu chưa chạy — chưa kết luận được về hướng của hiệu ứng này."
        )
        return note

    v0 = same_sample.get("V0+A2", {})
    v1 = same_sample.get("V1+A2", {})

    def delta(metric):
        return v1[metric]["mean"] - v0[metric]["mean"]

    note.extend(
        [
            ">",
            "> Bảng \"So cùng tập mẫu\" khử đúng hiệu ứng đó, và nó **đảo chiều kết luận**:",
            f"> trên 24 atom cả hai đều trả lời, V1 hơn V0 ở `context_precision` "
            f"({delta('context_precision'):+.3f}) và `answer_relevancy` "
            f"({delta('answer_relevancy'):+.3f}),",
            "> hai cấu hình **bằng nhau** ở `context_recall` (0.500) và `noise_sensitivity` (0.000),",
            f"> còn V1 kém hơn ở `context_entity_recall` "
            f"({delta('context_entity_recall'):+.3f}).",
            "> Các delta này cỡ 2–3 lần độ lệch chuẩn giữa các lượt judge — đủ để nói về hướng,",
            "> chưa đủ để nói BM25 tạo khác biệt lớn ở tầng chấm.",
            ">",
            "> **Giả thuyết V1 trong EVAL.md chỉ đúng một nửa.** Dự đoán là BM25 nâng",
            "> `context_recall` nhờ khớp chính xác `500名` / `99.9%` / `ISO/IEC 27001`. Đo được:",
            "> trên cùng tập mẫu `context_recall` **không đổi**. Giá trị thật của BM25 nằm ở chỗ",
            "> khác và lớn hơn — nó nâng **coverage 0.362 → 0.429** và hạ **abstain 0.725 → 0.575**,",
            "> tức trả lời được thêm 6 atom mà dense-only bỏ trống. Đó là cột deterministic, không",
            "> phải cột RAGAS. Với hồ sơ thầu, thêm một requirement được đáp ứng đáng giá hơn",
            "> vài phần trăm `context_precision`.",
            ">",
            "> Ghi chú: V1 **không phải superset** của V0 — có 6 atom V0 trả lời mà V1 bỏ trống,",
            "> nên tập giao là 24 chứ không phải 30.",
        ]
    )
    return note


def mmr_finding_note():
    return [
        "> **MMR (V4) không tạo delta — ghi lại như một phát hiện, không phải lỗi đo.**",
        "> V4 gần trùng V1 ở mọi cột. Lý do nằm ở pipeline chứ không ở MMR: bước khử trùng lặp",
        "> theo nguyên văn đã chạy **trước** MMR (`graph.py`, vòng lọc `unique_reranked`), nên",
        "> phần trùng lặp mà MMR sinh ra để xử lý thì đã bị cắt trước đó; và",
        "> `PRECEDENTS_PER_CHAPTER = 1` nghĩa là dù MMR chọn 5 câu đa dạng thì chỉ 1 câu được",
        "> đưa vào prompt sinh.",
        ">",
        "> Đây đúng tinh thần dòng V7 trong EVAL.md: *\"đã thử và chứng minh không bõ, kèm số\"*",
        "> là kết luận có giá trị hơn *\"không thử\"*. Giữ MMR trong cấu hình đề xuất vì nó là lưới",
        "> an toàn khi `PRECEDENTS_PER_CHAPTER` tăng (xem cặp k=5 bên dưới: ở k=5 thì thứ đưa câu",
        "> bịa vào prompt chính là số câu được chọn, không phải thuật toán chọn) — nhưng ở cấu",
        "> hình hiện tại nó **không** phải thứ tạo ra chất lượng, và bảng này nói đúng như vậy.",
    ]


def review_cost_rows(lines):
    """Chi phí review loop (Bước 6, v1.2) — đọc từ file đo, không hardcode."""
    path = results_dir / "review_cost.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", {})
    off = runs.get("no_review", [])
    on = runs.get("with_review", [])
    if not off or not on:
        return

    def mean(rows, key):
        return sum(row[key] for row in rows) / len(rows)

    tokens_off, tokens_on = mean(off, "product_tokens"), mean(on, "product_tokens")
    calls_off, calls_on = mean(off, "calls"), mean(on, "calls")
    delta_pct = (tokens_on / tokens_off - 1) * 100 if tokens_off else 0.0

    personas = data.get("personas", [])
    lines.append("### Chi phí review loop (Bước 6 · multi-persona #10)")
    lines.append("")
    lines.append("| Cấu hình | token sản phẩm / hồ sơ | lệnh gọi LLM / hồ sơ | giây / hồ sơ |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| review tắt | {tokens_off:,.0f} | {calls_off:.0f} | {mean(off, 'seconds'):.1f} |"
    )
    solo = runs.get("with_review_1_persona")
    if solo:
        lines.append(
            f"| review 1 persona | {mean(solo, 'product_tokens'):,.0f} "
            f"| {mean(solo, 'calls'):.0f} | {mean(solo, 'seconds'):.1f} |"
        )
    lines.append(
        f"| review {len(personas)} persona ({', '.join(personas)}) "
        f"| {tokens_on:,.0f} | {calls_on:.0f} | {mean(on, 'seconds'):.1f} |"
    )
    lines.append(
        f"| **chênh lệch (tắt → {len(personas)} persona)** "
        f"| **+{tokens_on - tokens_off:,.0f} ({delta_pct:+.0f}%)** "
        f"| **+{calls_on - calls_off:.0f}** | "
        f"**+{mean(on, 'seconds') - mean(off, 'seconds'):.1f}** |"
    )
    lines.append("")
    rounds = {row["review_rounds"] for row in on}
    lines.append(
        f"> Đo trên {len(on)} RFP gốc, cache tắt cứng (đường eval). `MAX_REVIEW_ROUNDS = "
        f"{data.get('max_review_rounds')}` nhưng thực đo dừng ở **{max(rounds)} vòng**: "
        "reviewer không tìm thấy issue `critical` nào nên vòng lặp thoát ngay — đây là"
    )
    lines.append(
        "> hành vi adaptive đúng thiết kế, không phải trần vòng bị chạm. Phần tăng thêm là"
    )
    lines.append(
        f"> chi phí **cố định** của một lượt soi 5 mục (+{calls_on - calls_off:.0f} lệnh gọi), "
        "không phải chi phí sửa lỗi."
    )
    lines.append(
        "> Lưu ý đọc số: `usage.py` gắn stage theo LOẠI lệnh gọi (`generate`/`structured`), "
        "không theo node pipeline, nên không tách riêng được token của review — con số"
    )
    lines.append("> đúng là phần chênh lệch giữa hai dòng trên.")
    if personas:
        lines.append(">")
        lines.append(
            f"> **Persona ({', '.join(personas)}) — và persona CỐ TÌNH không có.** "
            "Đề bài đề xuất ba persona"
        )
        lines.append(
            "> Compliance / Coverage / Quality. Hai persona sau an toàn; persona "
            "**Compliance thì không** —"
        )
        lines.append(
            "> nó phán về chứng chỉ giả và over-claim, tức giẫm lên BB-2 (blocklist regex) và "
            "BB-1"
        )
        lines.append(
            "> (capability sheet là trọng tài). Judge LLM sai 5–10%; ở đây sai một lần là hồ sơ "
            "tuyên bố"
        )
        lines.append(
            "> sai chứng chỉ. Một reviewer \"hiền\" báo sạch không làm hồ sơ sạch hơn, nhưng tạo "
            "cảm giác"
        )
        lines.append(
            "> đã có người canh — kiểu hỏng nguy hiểm nhất. Compliance ở lại tầng deterministic."
        )
        lines.append(">")
        lines.append(
            "> Mỗi persona soi mọi mục, chạy song song, rồi dedupe theo "
            "`(section_key, issue_type)` giữ"
        )
        lines.append(
            "> bản **severity nặng nhất** — hạ severity vì thứ tự chạy sẽ biến một lỗi critical "
            "thành major"
        )
        lines.append("> một cách ngẫu nhiên.")


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
    judge_temp = base_data.get("judge", {}).get("temperature", 0.0)
    n_runs = base_data.get("judge", {}).get("runs", 3)
    
    det_rows = base_data.get("deterministic", [])
    total_atoms = sum(r.get("requirements", 0) for r in det_rows)

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

    lines.append(f"Judge: {judge_model}, temperature={judge_temp}, reasoning_effort: không gửi, n={n_runs} (mean±std)")
    lines.append(f"Test set: 3 RFP gốc + 6 mutation = 9 RFP, {total_atoms} requirement atom")
    lines.append(f"Tổng token Sản phẩm: {total_prod:,} (từ {prod_runs} lần chạy) | Tổng token Đo lường (Judge): {total_judge:,} (từ {judge_runs} lần chạy có RAGAS)")
    
    # Chi phí mỗi RFP phải lấy từ CẤU HÌNH ĐỀ XUẤT, không phải trung bình gộp 7
    # cấu hình: các ablation rẻ (tắt bớt kênh sinh) kéo con số xuống, ra một mức
    # chi phí không ứng với thứ thật sự đem dùng.
    proposed_key = configs[0][0]
    # Đọc thẳng hai file: load_json() ghép chúng lại và ghi đè "usage" bằng bản
    # deterministic, nên token judge không còn trong đó.
    det_path = results_dir / f"{proposed_key}.det.json"
    ragas_path = results_dir / f"{proposed_key}.json"
    if det_path.exists() and ragas_path.exists():
        det_proposed = json.loads(det_path.read_text(encoding="utf-8"))
        ragas_proposed = json.loads(ragas_path.read_text(encoding="utf-8"))
        rfp_count = len(det_proposed.get("deterministic", [])) or 9
        tokens = det_proposed.get("usage", {}).get("tokens_by_stage", {})
        prod_per_rfp = sum(
            v for k, v in tokens.items() if k in ("generate", "structured")
        ) / rfp_count
        judge_per_rfp = (
            ragas_proposed.get("usage", {}).get("tokens_by_stage", {}).get("judge", 0)
            / rfp_count
        )
        if prod_per_rfp and judge_per_rfp:
            ratio = judge_per_rfp / prod_per_rfp
            lines.append(
                f"Cấu hình đề xuất, mỗi RFP: sản phẩm ~{int(prod_per_rfp):,} token · "
                f"judge ~{int(judge_per_rfp):,} token "
                f"(đo lường tốn gấp ~{ratio:.0f} lần sản phẩm)"
            )
        
    lines.append("")
    lines.append("| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for key, id_col, conf_col in retrieval_ladder:
        lines.append(format_row(id_col, conf_col, load_json(key)))
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
        "`‡` = Thang retrieval: chunk câu + pipeline sinh A2 giữ nguyên để cô lập tầng "
        "retrieval, mỗi bậc đổi đúng một biến — V0 dense-only → V1 +BM25 (hybrid) → "
        "V4 +MMR → đề xuất +rerank prior (industry/section). V0 ở đây naive ở tầng "
        "retrieval; V0 nguyên bản của EVAL.md (chunk theo đoạn + single-shot A0) đổi "
        "nhiều biến cùng lúc nên không so được. V5 (decomp mỗi atom) và V6 (HyDE) "
        "chưa triển khai — ô trống."
    )
    lines.append("")

    same_sample = same_sample_rows(lines)
    lines.append("")
    lines.extend(sample_composition_note(same_sample))
    lines.append("")
    lines.extend(mmr_finding_note())
    lines.append("")

    review_cost_rows(lines)
    lines.append("")

    reach = poison_reach_rows(lines)
    lines.append("")
    lines.extend(interpretation(reach))

    out = "\n".join(lines)
    (results_dir / "report.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
