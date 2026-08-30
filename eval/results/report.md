Judge: gpt-5.4-mini, temperature=0.0, reasoning_effort: không gửi, n=3 (mean±std)
Test set: 3 RFP gốc + 6 mutation = 9 RFP, 83 requirement atom
Tổng token Sản phẩm: 157,655 (từ 7 lần chạy) | Tổng token Đo lường (Judge): 11,487,457 (từ 7 lần chạy có RAGAS)
Cấu hình đề xuất, mỗi RFP: sản phẩm ~4,723 token · judge ~299,072 token (đo lường tốn gấp ~63 lần sản phẩm)

| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A2† | naive: dense-only‡ | 0 | 0 | 0 | 0.362 / 0.725 | 1.000 | 0.455±0.009 | 0.400±0.000 | 0.133±0.000 | 0.994±0.005 | 0.576±0.015 | — | 147.4s SP / — judge | N/A |
| V1+A2† | +sentence, hybrid‡ | 0 | 0 | 0 | 0.429 / 0.575 | 1.000 | 0.406±0.003 | 0.333±0.000 | 0.333±0.000 | 0.997±0.004 | 0.625±0.000 | — | 146.3s SP / — judge | N/A |
| V4+A2† | +MMR‡ | 0 | 0 | 0 | 0.429 / 0.575 | 1.000 | 0.395±0.010 | 0.333±0.000 | 0.345±0.005 | 0.995±0.007 | 0.606±0.008 | — | 153.4s SP / — judge | N/A |
| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |
| V6+A2 | +HyDE | | | | / | | | | | | | | | |
| **V4+V5+A2**† | **đề xuất** | 0 | 0 | 0 | 0.429 / 0.575 | 1.000 | 0.388±0.005 | 0.333±0.000 | 0.333±0.000 | 0.994±0.004 | 0.606±0.009 | — | 138.2s SP / — judge | N/A |
| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |
| — chỉ capability† | (ablation nguồn) | 0 | 0 | 0 | 0.296 / 0.725 | 1.000 | 0.557±0.006 | 0.500±0.000 | 0.000±0.000 | 1.000±0.000 | 0.631±0.001 | — | 80.9s SP / — judge | N/A |
| — chỉ precedent (kênh sinh)† | (ablation nguồn) | 0 | 0 | 0 | 0.143 / 0.625 | 1.000 | 0.085±0.006 | 0.000±0.000 | 0.929±0.009 | 1.000±0.000 | 0.535±0.021 | — | 52.8s SP / — judge | N/A |
| — không có capability sheet làm trọng tài† | (ablation nguồn & guard) | 0 | 0 | 0 | 0.143 / 0.625 | 1.000 | 0.094±0.012 | 0.000±0.000 | 0.942±0.000 | 1.000±0.000 | 0.537±0.004 | — | 27.4s SP / — judge | N/A |
| — tắt cả quarantine lúc ingest† | (ablation ingest & guard) | 0 | 0 | 0 | 0.209 / 0.625 | 1.000 | 0.092±0.013 | 0.000±0.000 | 0.939±0.022 | 1.000±0.000 | 0.534±0.001 | — | 27.1s SP / — judge | N/A |
| — ép k=5, tắt guard† | (minh chứng BB-1/BB-2) | **16** | **6** | 0 | 0.265 / 0.625 | 1.000 | 0.109±0.013 | 0.058±0.020 | 0.960±0.005 | 0.969±0.008 | 0.542±0.013 | — | 65.2s SP / — judge | N/A |
| — ép k=5, BẬT guard† | (minh chứng BB-1/BB-2) | 0 | 0 | 0 | 0.265 / 0.625 | 1.000 | 0.103±0.007 | 0.014±0.020 | 0.953±0.011 | 0.964±0.009 | 0.558±0.007 | — | 133.8s SP / — judge | N/A |

`—` = lần chạy đó chưa đo cột này, **không phải** đo ra 0.
`†` = Cột deterministic lấy từ lần chạy mới (.det.json), cột RAGAS lấy từ lần chạy cũ (.json).
`‡` = Thang retrieval: chunk câu + pipeline sinh A2 giữ nguyên để cô lập tầng retrieval, mỗi bậc đổi đúng một biến — V0 dense-only → V1 +BM25 (hybrid) → V4 +MMR → đề xuất +rerank prior (industry/section). V0 ở đây naive ở tầng retrieval; V0 nguyên bản của EVAL.md (chunk theo đoạn + single-shot A0) đổi nhiều biến cùng lúc nên không so được. V5 (decomp mỗi atom) và V6 (HyDE) chưa triển khai — ô trống.

### So cùng tập mẫu (24 atom cả hai cấu hình đều trả lời)

| # | Cấu hình | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | ans_rel↑ | ctx_entity_recall↑ | faithful.↑ |
|---|---|---|---|---|---|---|---|
| V0+A2 | dense-only | 0.543±0.012 | 0.500±0.000 | 0.000±0.000 | 0.620±0.005 | 0.534±0.013 | 1.000±0.000 |
| V1+A2 | +BM25 (hybrid) | 0.571±0.010 | 0.500±0.000 | 0.000±0.000 | 0.630±0.003 | 0.504±0.006 | 1.000±0.000 |

> **Cảnh báo khi đọc cột RAGAS: hiệu ứng thành phần mẫu.**
> RAGAS chỉ chấm được atom **có answer**; atom bị abstain không vào mẫu. Nên hai cấu
> hình khác `abstain_rate` được chấm trên **hai tập mẫu khác nhau**, và so trực tiếp hai
> trung bình đó là so hai thứ khác nhau — không phải so chất lượng retrieval.
>
> Cụ thể ở bảng trên: V0 dense-only chỉ trả lời 30/83 atom, V1 +BM25 trả lời 36/83.
> Sáu atom V1 trả lời thêm là các atom **khó hơn** (V0 bỏ trống vì không tìm ra bằng
> chứng), nên chúng kéo trung bình RAGAS của V1 xuống. Đọc nguyên bảng gộp sẽ ra kết
> luận sai rằng "naive dense-only tốt hơn hybrid".
>
> Bảng "So cùng tập mẫu" khử đúng hiệu ứng đó, và nó **đảo chiều kết luận**:
> trên 24 atom cả hai đều trả lời, V1 hơn V0 ở `context_precision` (+0.028) và `answer_relevancy` (+0.010),
> hai cấu hình **bằng nhau** ở `context_recall` (0.500) và `noise_sensitivity` (0.000),
> còn V1 kém hơn ở `context_entity_recall` (-0.030).
> Các delta này cỡ 2–3 lần độ lệch chuẩn giữa các lượt judge — đủ để nói về hướng,
> chưa đủ để nói BM25 tạo khác biệt lớn ở tầng chấm.
>
> **Giả thuyết V1 trong EVAL.md chỉ đúng một nửa.** Dự đoán là BM25 nâng
> `context_recall` nhờ khớp chính xác `500名` / `99.9%` / `ISO/IEC 27001`. Đo được:
> trên cùng tập mẫu `context_recall` **không đổi**. Giá trị thật của BM25 nằm ở chỗ
> khác và lớn hơn — nó nâng **coverage 0.362 → 0.429** và hạ **abstain 0.725 → 0.575**,
> tức trả lời được thêm 6 atom mà dense-only bỏ trống. Đó là cột deterministic, không
> phải cột RAGAS. Với hồ sơ thầu, thêm một requirement được đáp ứng đáng giá hơn
> vài phần trăm `context_precision`.
>
> Ghi chú: V1 **không phải superset** của V0 — có 6 atom V0 trả lời mà V1 bỏ trống,
> nên tập giao là 24 chứ không phải 30.

> **MMR (V4) không tạo delta — ghi lại như một phát hiện, không phải lỗi đo.**
> V4 gần trùng V1 ở mọi cột. Lý do nằm ở pipeline chứ không ở MMR: bước khử trùng lặp
> theo nguyên văn đã chạy **trước** MMR (`graph.py`, vòng lọc `unique_reranked`), nên
> phần trùng lặp mà MMR sinh ra để xử lý thì đã bị cắt trước đó; và
> `PRECEDENTS_PER_CHAPTER = 1` nghĩa là dù MMR chọn 5 câu đa dạng thì chỉ 1 câu được
> đưa vào prompt sinh.
>
> Đây đúng tinh thần dòng V7 trong EVAL.md: *"đã thử và chứng minh không bõ, kèm số"*
> là kết luận có giá trị hơn *"không thử"*. Giữ MMR trong cấu hình đề xuất vì nó là lưới
> an toàn khi `PRECEDENTS_PER_CHAPTER` tăng (xem cặp k=5 bên dưới: ở k=5 thì thứ đưa câu
> bịa vào prompt chính là số câu được chọn, không phải thuật toán chọn) — nhưng ở cấu
> hình hiện tại nó **không** phải thứ tạo ra chất lượng, và bảng này nói đúng như vậy.

### Chi phí review loop (Bước 6)

| Cấu hình | token sản phẩm / hồ sơ | lệnh gọi LLM / hồ sơ | giây / hồ sơ |
|---|---|---|---|
| review tắt | 4,994 | 19 | 18.0 |
| review bật | 7,635 | 24 | 24.9 |
| **chênh lệch** | **+2,640 (+53%)** | **+5** | **+6.9** |

> Đo trên 3 RFP gốc, cache tắt cứng (đường eval). `MAX_REVIEW_ROUNDS = 3` nhưng thực đo dừng ở **1 vòng**: reviewer không tìm thấy issue `critical` nào nên vòng lặp thoát ngay — đây là
> hành vi adaptive đúng thiết kế, không phải trần vòng bị chạm. Phần tăng thêm là
> chi phí **cố định** của một lượt soi 5 mục (+5 lệnh gọi), không phải chi phí sửa lỗi.
> Lưu ý đọc số: `usage.py` gắn stage theo LOẠI lệnh gọi (`generate`/`structured`), không theo node pipeline, nên không tách riêng được token của review — con số
> đúng là phần chênh lệch giữa hai dòng trên.

### Chất độc đi tới đâu (giải thích cột fabric./leak ở trên)

| Cấu hình | trong context retrieval | vào prompt sinh | guard chặn xuất bản | còn trong hồ sơ |
|---|---|---|---|---|
| **V4+V5+A2** | fab 0 · leak 0 | fab 0 · leak 0 | 0 | fab 0 · leak 0 |
| — chỉ capability | fab 0 · leak 0 | fab 0 · leak 0 | 0 | fab 0 · leak 0 |
| — chỉ precedent (kênh sinh) | fab 0 · leak 0 | fab 0 · leak 0 | 0 | fab 0 · leak 0 |
| — không có capability sheet làm trọng tài | fab 0 · leak 0 | fab 0 · leak 0 | 0 | fab 0 · leak 0 |
| — tắt cả quarantine lúc ingest | fab 18 · leak 14 | fab 0 · leak 0 | 0 | fab 0 · leak 0 |
| — ép k=5, tắt guard | fab 18 · leak 14 | fab 18 · leak 6 | 0 | fab 16 · leak 6 |
| — ép k=5, BẬT guard | fab 18 · leak 14 | fab 18 · leak 6 | **8** | fab 0 · leak 0 |

> **Cặp dòng ép k=5 — bằng chứng cho BB-2.**
> Hai dòng khác nhau đúng một biến. Cùng k=5, cùng tắt quarantine, cùng 18 câu bịa và
> 14 câu rò rỉ lọt vào context, cùng 18 câu bịa vào prompt sinh. Chỉ khác
> guard bật hay tắt.
> - **Tắt guard:** hồ sơ xuất ra chứa 16 lần chuỗi cấm và 6 lần tên khách
>   hàng riêng. Ép được ảo giác — đây là ô khác 0 đầu tiên của bảng.
> - **Bật guard:** final guard Bước 7 ném GuardViolation ở 8/9 RFP. Ô `fabric = 0`
>   ở dòng này **không phải hồ sơ sạch mà là KHÔNG CÓ hồ sơ** — hệ thống trả về lỗi chứ
>   không trả về tài liệu. Đọc ô đó phải đọc kèm cột "guard chặn xuất bản".
>
> Guard là **cầu dao, không phải bộ lọc**: nó không cứu được hồ sơ, nó chặn hồ sơ hỏng ra
> khỏi cửa. Với hồ sơ thầu, tuyên bố sai chứng chỉ thì thà không nộp — nên đây là hành vi
> đúng. Nhưng thứ khiến 5 dòng trên vừa `fabrication = 0` vừa **xuất bản được** là
> quarantine tầng ingest, không phải guard.
>
> Claim-check 6.4 không đóng vai trò gì ở kịch bản này: bằng chứng nó đối chiếu chính là
> câu nguồn bịa đó (`_evidence_text` trả `source_texts[source_id]`), nên nó phán VERIFIED.
> Lưới duy nhất chặn được chuỗi cấm là regex ở Bước 7 — đúng như BB-2 quy định.