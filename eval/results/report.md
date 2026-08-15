Judge: gpt-5.4-mini, effort=low, n=3 (mean±std)
Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~94 requirement atom
Tổng token Sản phẩm: 92,631 | Tổng token Đo lường (Judge): 6,447,440

| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A0 | Naive baseline | | | | / | | | | | | | | | |
| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |
| V4+A1 | +MMR | | | | / | | | | | | | | | |
| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |
| V6+A2 | +HyDE | | | | / | | | | | | | | | |
| **V4+V5+A2** | **đề xuất** | 0 | 0 | 0 | 0.429 / 0.575 | 1.000 | 0.377±0.003 | 0.333±0.000 | 0.338±0.007 | 0.997±0.004 | 0.601±0.003 | — | 380.0s | N/A |
| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |
| — chỉ capability | (ablation nguồn) | 0 | 0 | 0 | 0.296 / 0.725 | 1.000 | 0.563±0.006 | 0.500±0.000 | 0.007±0.010 | 1.000±0.000 | 0.631±0.018 | — | 294.6s | N/A |
| — chỉ precedent (kênh sinh) | (ablation nguồn) | 0 | 0 | 0 | 0.143 / 0.625 | 1.000 | 0.085±0.006 | 0.000±0.000 | 0.936±0.009 | 1.000±0.000 | 0.541±0.009 | — | 159.6s | N/A |
| — không có capability sheet làm trọng tài | (ablation nguồn & guard) | 0 | 0 | 0 | 0.143 / 0.625 | 1.000 | 0.090±0.010 | 0.000±0.000 | 0.949±0.024 | 1.000±0.000 | 0.550±0.001 | — | 140.3s | N/A |