Judge: gpt-5.4-mini, effort=low, n=3 (mean±std)
Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~94 requirement atom
Tổng token Sản phẩm: 99,181 | Tổng token Đo lường (Judge): 7,887,951

| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A0 | Naive baseline | | | | / | | | | | | | | | |
| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |
| V4+A1 | +MMR | | | | / | | | | | | | | | |
| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |
| V6+A2 | +HyDE | | | | / | | | | | | | | | |
| **V4+V5+A2** | **đề xuất** | 0 | 0 | — | 0.429 / 0.575 | 1.000 | 0.377±0.003 | 0.333±0.000 | 0.338±0.007 | 0.997±0.004 | 0.601±0.003 | — | 380.0s gộp* | N/A |
| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |
| — chỉ capability | (ablation nguồn) | 0 | 0 | — | 0.296 / 0.725 | 1.000 | 0.563±0.006 | 0.500±0.000 | 0.007±0.010 | 1.000±0.000 | 0.631±0.018 | — | 294.6s gộp* | N/A |
| — chỉ precedent (kênh sinh) | (ablation nguồn) | 0 | 0 | — | 0.143 / 0.625 | 1.000 | 0.085±0.006 | 0.000±0.000 | 0.936±0.009 | 1.000±0.000 | 0.541±0.009 | — | 159.6s gộp* | N/A |
| — không có capability sheet làm trọng tài | (ablation nguồn & guard) | 0 | 0 | — | 0.143 / 0.625 | 1.000 | 0.090±0.010 | 0.000±0.000 | 0.949±0.024 | 1.000±0.000 | 0.550±0.001 | — | 140.3s gộp* | N/A |
| — tắt cả quarantine lúc ingest | (ablation ingest & guard) | 0 | 0 | — | 0.209 / 0.625 | 1.000 | 0.073±0.004 | 0.000±0.000 | 0.943±0.016 | 1.000±0.000 | 0.577±0.001 | — | 151.8s gộp* | N/A |

`—` = lần chạy đó chưa đo cột này, **không phải** đo ra 0.
`*` = tổng gộp sản phẩm + judge, từ lần chạy trước khi tách `seconds_by_stage`; phần judge trong đó thấp hơn thực tế.

### Chất độc đi tới đâu (giải thích cột fabric./leak ở trên)

| Cấu hình | trong context retrieval | vào prompt sinh | còn trong hồ sơ |
|---|---|---|---|
| **V4+V5+A2** | — | — | fab 0 · leak 0 |
| — chỉ capability | — | — | fab 0 · leak 0 |
| — chỉ precedent (kênh sinh) | — | — | fab 0 · leak 0 |
| — không có capability sheet làm trọng tài | — | — | fab 0 · leak 0 |
| — tắt cả quarantine lúc ingest | — | — | fab 0 · leak 0 |

> **Đọc bảng ablation (guard & ingest):**
> Cấu hình `tắt cả quarantine lúc ingest` chưa có số đo đường đi của câu bịa (`fabrication_in_context` / `fabrication_in_prompt`). Thiếu hai số đó thì `fabrication = 0` **không diễn giải được**: không phân biệt nổi guard chặn, retriever không xếp lên, hay câu bịa vắng mặt trong index. Chạy lại eval bằng code hiện tại rồi sinh lại báo cáo.