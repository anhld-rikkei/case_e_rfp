# case_e_rfp — sinh hồ sơ thầu tiếng Nhật từ RFP

Đầu vào là một RFP dạng text, đầu ra là hồ sơ thầu tiếng Nhật. Hai ràng buộc quyết định toàn bộ
thiết kế: **mỗi câu truy được về nguồn** (`capability_sheet.json` hoặc một câu precedent có `source_id`),
và **không câu nào vượt quá năng lực thật của công ty** — thiếu bằng chứng thì ghi
`INSUFFICIENT_EVIDENCE`, không bịa. Số liệu (%, năm, số người) chỉ đến từ capability sheet hoặc
nguyên văn câu nguồn; LLM không được tự sinh.

## Trạng thái

**v1.3.0** · **107 test pass** (`pytest eval/`) · nhánh phát hành: `main`

| Tag | Nội dung chính | Test |
|---|---|---|
| [v1.0.0](../../releases/tag/v1.0.0) | Mốc chốt trước đợt nâng cấp | 22 |
| [v1.1.0](../../releases/tag/v1.1.0) | Guard bắt **biến thể** chứng chỉ cấm · bảng ablation có cặp k=5 chứng minh guard · thang retrieval V0/V1/V4 có số thật · retry backoff + checkpoint resume | 34 |
| [v1.2.0](../../releases/tag/v1.2.0) | Cache mức node với vân tay corpus + capability, tắt cứng ở eval · review loop chỉ soi chất lượng văn bản | 83 |
| [v1.3.0](../../releases/tag/v1.3.0) | Upload RFP · export markdown 5 mục với nhãn nháp **nằm trong file** · review 2 persona song song | 107 |

Hoãn có chủ đích: chat-refine từng mục, section lock, version history — chúng chạm chuỗi phụ thuộc
`used_fact_keys` trong `generate_per_section`, phải là release riêng có eval riêng (lý do đầy đủ ở
docstring `src/rfp/cache.py`).

### Ba lệnh hay dùng

```powershell
streamlit run app.py                                  # chạy app (http://localhost:8642)
.venv\Scripts\python.exe -m pytest eval/ -q           # full test suite — 107 passed
.venv\Scripts\python.exe -m eval.generate_report      # sinh lại eval/results/report.md
```

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env      # rồi điền OPENAI_API_KEY vào .env
Copy-Item scripts\check_secrets.sh .git\hooks\pre-commit -Force
```

> **Dòng cuối không bỏ được.** Git **không** đồng bộ `.git/hooks/`, nên mỗi người clone về phải tự
> cài hook. Hook này quét dòng `+` trong diff đã stage và chặn commit nếu thấy API key. Quên cài thì
> phòng tuyến chặn key chỉ còn Secret Detection ở CI — tức là key đã lên server rồi mới bị phát hiện,
> và lúc đó việc đầu tiên phải làm là **thu hồi key**, không phải xoá khỏi git.

Ba cờ `LLM_TOKEN_ARG` / `LLM_USE_TEMP` / `LLM_USE_EFFORT` trong `.env` phải điền theo kết quả
`python tools/probe_model.py`, không đoán — sai một cờ là mọi lệnh gọi trả 400.

### Kiểm tra nhanh sau khi cài

Lệnh này **không cần API key** — nó chỉ dựng index và kiểm data sạch:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m rfp.stores.sentence_index --assert-clean
```

Kết quả đúng:

```
indexed=273 · blocklist=0 · client_leak=0 · quarantine=4 · leak_dropped=14
```

Lần chạy đầu sẽ tải model embedding (~458MB) từ HuggingFace, cần mạng.
`PYTHONPATH=src` là bắt buộc với các lệnh dạng `python -m rfp.*` vì gói nằm trong `src/`;
`app.py` và mọi thứ trong `eval/` thì tự nạp đường dẫn nên không cần.

Bộ test tầng 1 cũng chạy được **không cần key** — client LLM khởi tạo lười, chỉ dựng khi thật sự
có lệnh gọi:

```powershell
.venv\Scripts\python.exe -m pytest eval/test_gates.py -q
```

Không key: `1 failed, 23 passed` — ca fail duy nhất là `test_redteam_27017` vì nó sinh hồ sơ thật.
Có key: `24 passed`.

Toàn bộ `eval/` là **107 test**: `test_gates` 24 · `test_cache` 29 · `test_review` 28 ·
`test_export` 16 · `test_resilience` 9 · `test_deepeval` 1. Trừ `test_redteam_27017` và
`test_deepeval`, tất cả đều cách ly — không mạng, không dựng FAISS.

> ⚠️ **Sinh hồ sơ và chạy eval thì bắt buộc có key.** Thiếu key, lỗi nổ đúng lúc gọi LLM với
> thông báo rõ ràng (`Missing credentials`), không phải một traceback lúc khởi động.

## Chạy

```powershell
streamlit run app.py
```

Mở http://localhost:8642 (cổng đặt trong `.streamlit/config.toml`). Dán RFP, tải file `.txt`
(UTF-8), hoặc chọn một trong 3 RFP mẫu. Hồ sơ sinh ra tải về được dưới dạng markdown — **nhãn "bản
nháp" và checklist người rà soát nằm trong chính file**, không chỉ trên màn hình, vì file rời khỏi
app là mất banner UI.

Chạy hồ sơ từ dòng lệnh, có checkpoint để chạy tiếp khi bị ngắt giữa chừng:

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m rfp.graph --rfp synthetic\rfps\RFP-2025-001.txt --job-id job1
.venv\Scripts\python.exe -m rfp.graph --rfp synthetic\rfps\RFP-2025-001.txt --job-id job1 --resume
```

### Cấu hình chạy (trong `config/settings.py`, không phải `.env`)

| Hằng số | Mặc định | Tác dụng |
|---|---|---|
| `CACHE_ENABLED` · `CACHE_DIR` · `CACHE_TTL_SECONDS` | `True` · `cache/` · 7 ngày | Cache kết quả 2 node retrieval/sinh. Mọi đường chạy `eval/` **luôn tắt cache**, không đọc cờ này |
| `PROMPT_VERSION` · `TEMPLATE_VERSION` | `"1"` | Nằm trong cache key — **bump tay** khi sửa prompt, quên bump là dùng lại kết quả của prompt cũ |
| `REVIEW_ENABLED` · `MAX_REVIEW_ROUNDS` | `True` · `3` | Vòng review chất lượng văn bản. Tắt để tiết kiệm ~115% token (xem bảng dưới) |

Bỏ qua cache cho một lần chạy: thêm `--force-regen` (vẫn ghi lại kết quả mới).

## Chạy eval

Ba tầng đo, tách theo chi phí:

```powershell
# Tầng 1 — cổng regex + coverage + groundedness. Vài giây, 0 lệnh gọi LLM.
.venv\Scripts\python.exe -m pytest eval/test_gates.py -q

# Tầng 2 — sinh hồ sơ thật rồi đo deterministic. Tốn token sản phẩm, 0 token judge.
.venv\Scripts\python.exe -m eval.run_ablation --deterministic-only

# Tầng 3 — thêm RAGAS (n=3). Đây là chỗ tốn tiền.
.venv\Scripts\python.exe -m eval.run_ragas --rfp <đường-dẫn-rfp.txt> --out eval/results/<tên>.json

# Sinh lại bảng §11.3
.venv\Scripts\python.exe -m eval.generate_report
```

**Chi phí đo được** trên một RFP (nguồn: `eval/results/review_cost.json` và header
`eval/results/report.md`):

| | token / hồ sơ | lệnh gọi LLM |
|---|---|---|
| sinh hồ sơ, review tắt | ~5,0k | 19 |
| sinh hồ sơ, review 2 persona | ~10,8k | 29 |
| judge RAGAS (đo lường) | ~299k | — |

Judge tốn **~63 lần** token sản phẩm. Đó là lý do hai nhóm token được đo tách nhau trong
`eval/results/*.json`; gộp chung là thổi phồng chi phí sản phẩm lên hàng chục lần. Đơn giá để
`None` trong `config/settings.py` nên báo cáo ghi "chưa cấu hình đơn giá" thay vì bịa ra số tiền.

Review loop tốn **+115% token** và **+10 lệnh gọi** (5 mục × 2 persona) nhưng chỉ +12,6 giây nhờ chạy
song song. Thực đo dừng ở **1 vòng** dù `MAX_REVIEW_ROUNDS=3` — đây là chi phí cố định của một lượt
soi, không phải chi phí sửa lỗi.

## Tài liệu

| File | Nội dung |
|---|---|
| [BUILD_GUIDE.md](BUILD_GUIDE.md) | 12+ bước thi công, mỗi bước có tiêu chí nghiệm thu — tài liệu chính |
| [ARCHITECTURE.md](ARCHITECTURE.md) | *vì sao* thiết kế như vậy |
| [EVAL.md](EVAL.md) | thiết kế đánh giá RAGAS/DeepEval |
| [eval/results/report.md](eval/results/report.md) | bảng ablation §11.3 · thang retrieval · chi phí review · "chất độc đi tới đâu" |

Ba con số đáng đọc nhất trong report:

1. **Cặp k=5 — guard làm được việc thật.** Hai dòng khác nhau đúng một biến: tắt guard thì hồ sơ chứa
   **16 lần chuỗi cấm + 6 lần tên khách hàng cũ**; bật guard thì **0**, và `GuardViolation` chặn xuất
   bản 8/9 RFP. Ô `fabrication = 0` đó **không phải hồ sơ sạch mà là không có hồ sơ** — guard là cầu
   dao, không phải bộ lọc.
2. **BM25: giả thuyết ban đầu sai.** Chấm trên **cùng 24 atom**, `context_recall` **không đổi** như
   dự đoán trong EVAL.md. Giá trị thật nằm chỗ khác và lớn hơn: coverage **0.362 → 0.429**, abstain
   **0.725 → 0.575**.
3. **Hiệu ứng thành phần mẫu.** RAGAS chỉ chấm atom *có answer*, nên so hai cấu hình khác
   `abstain_rate` là so trên hai tập mẫu khác nhau. Bảng gộp làm naive dense-only trông như thắng;
   chấm cùng tập mẫu thì đảo chiều.
