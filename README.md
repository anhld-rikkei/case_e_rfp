# case_e_rfp — sinh hồ sơ thầu tiếng Nhật từ RFP

Đầu vào là một RFP dạng text, đầu ra là hồ sơ thầu tiếng Nhật. Hai ràng buộc quyết định toàn bộ
thiết kế: **mỗi câu truy được về nguồn** (`capability_sheet.json` hoặc một câu precedent có `source_id`),
và **không câu nào vượt quá năng lực thật của công ty** — thiếu bằng chứng thì ghi
`INSUFFICIENT_EVIDENCE`, không bịa. Số liệu (%, năm, số người) chỉ đến từ capability sheet hoặc
nguyên văn câu nguồn; LLM không được tự sinh.

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

> ⚠️ **Mọi thứ còn lại đều cần API key**, kể cả `pytest`. Client LLM được tạo lúc *import*
> `src/rfp/llm.py`, nên thiếu key thì `pytest eval/test_gates.py` **dừng ngay ở bước collection**
> chứ không chạy được test nào — dù bản thân các test tầng 1 không hề gọi LLM.

## Chạy

```powershell
streamlit run app.py
```

Mở http://localhost:8642 (cổng đặt trong `.streamlit/config.toml`).

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

**Chi phí đo được** trên một RFP (`eval/results/rfp001.json`):

| | token |
|---|---|
| sinh hồ sơ (sản phẩm) | ~5.2k / hồ sơ |
| judge RAGAS (đo lường) | ~372k / RFP |

Judge tốn **~72 lần** token sản phẩm. Đó là lý do hai nhóm token được đo tách nhau trong
`eval/results/*.json`; gộp chung là thổi phồng chi phí sản phẩm lên hàng chục lần. Đơn giá để
`None` trong `config/settings.py` nên báo cáo ghi "chưa cấu hình đơn giá" thay vì bịa ra số tiền.

## Tài liệu

| File | Nội dung |
|---|---|
| [BUILD_GUIDE.md](BUILD_GUIDE.md) | 12+ bước thi công, mỗi bước có tiêu chí nghiệm thu — tài liệu chính |
| [ARCHITECTURE.md](ARCHITECTURE.md) | *vì sao* thiết kế như vậy |
| [EVAL.md](EVAL.md) | thiết kế đánh giá RAGAS/DeepEval |
| [eval/results/report.md](eval/results/report.md) | bảng ablation §11.3 |
