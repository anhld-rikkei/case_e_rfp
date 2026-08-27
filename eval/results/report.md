Judge: gpt-5.4-mini, temperature=0.0, reasoning_effort: không gửi, n=3 (mean±std)
Test set: 3 RFP gốc + 6 mutation = 9 RFP, 83 requirement atom
Tổng token Sản phẩm: 157,655 (từ 7 lần chạy) | Tổng token Đo lường (Judge): 11,487,457 (từ 7 lần chạy có RAGAS)
Cấu hình đề xuất, mỗi RFP: sản phẩm ~4,723 token · judge ~299,072 token (đo lường tốn gấp ~63 lần sản phẩm)

| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A0 | Naive baseline | | | | / | | | | | | | | | |
| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |
| V4+A1 | +MMR | | | | / | | | | | | | | | |
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