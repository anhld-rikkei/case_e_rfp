# BUILD GUIDE — hướng dẫn người thực hiện build hệ sinh hồ sơ thầu (case_e_rfp)

Hợp nhất [ARCHITECTURE.md](ARCHITECTURE.md) + [EVAL.md](EVAL.md) thành các bước thi công.
Mọi số liệu trong file này đã được **đếm trực tiếp từ data**, không phải ước lượng.

---

## Cách dùng file này với người thực hiện

1. Đưa người thực hiện **Phần 1 + Phần 2 + Phần 3** trước (sự thật về data, bất biến, khung repo). Đây là context nền.
2. Mỗi lần chỉ đưa **một Bước** trong Phần 4. Không dán cả file.
3. Bắt người thực hiện chạy **Nghiệm thu** của bước đó và dán output ra. Chưa pass thì không sang bước sau.
4. Prompt mẫu cho từng bước ở Phụ lục A.

Quy tắc cứng cho người thực hiện, lặp lại ở mọi bước:
> Không sửa file ngoài phạm vi bước đang làm. Không thêm phụ thuộc mới ngoài `requirements.txt`.
> Không "cải thiện" logic của bước trước. Nghiệm thu fail thì sửa, không nới tiêu chí.

---

## PHẦN 0 — Rà soát đề bài (đối chiếu yêu cầu ↔ bước thực thi)

### Từ `Project_guide.md` — Case 4: case_e_rfp

| Yêu cầu đề bài | Đáp ứng ở |
|---|---|
| "gen ra 1 văn bản (hồ sơ thầu), **không phải chatbot**" | Bước 4 (graph), 6 (sinh), 7 (ghép) |
| `synthetic/proposals` = **nguồn tri thức** | Bước 1–3 (parse, sanitize, index), 6.1 (sinh câu precedent) |
| `synthetic/certs/fake_iso27017.txt` = "đảm bảo hồ sơ gen ra **không được chứa** chứng chỉ này" | Bước 2.3 (blocklist), 7 (final guard), 10 (red-team test) |
| `synthetic/rfps` = **requirement đầu vào** | Bước 1 (RFPParser) |
| `capability_sheet.json` = **năng lực thực tế của công ty** | Bước 2.4 (KV store), 6.2 (sinh câu capability), 6.4 (claim-check) |
| Đầu ra "**đáp ứng requirement**, phù hợp năng lực" | Bước 6.5 (G6 coverage) + 6.4 (compliance) |
| "dùng các phương pháp RAG và agent khác nhau" | Bước 11 (ma trận V0–V7 × A0–A4) |
| "dùng RAGAS hay deepeval để so sánh, **đưa ra lý do vì sao dùng phương án**" | Bước 10 (harness) + 11 (bảng ablation) |
| "khuyến khích demo giao diện" | Bước 8 (Streamlit) |

### Từ yêu cầu người dùng đưa thêm

| Yêu cầu | Đáp ứng ở |
|---|---|
| "dùng **cả 2** cái: năng lực công ty **và** proposal cũ để sinh" | Bước 6.1 + 6.2 (2 kênh sinh song song) → 6.3 ghép |
| UI đúng như ảnh demo | Bước 8, có checklist đối chiếu từng thành phần |
| Graph tiến trình xanh/xanh dương/xám đứt | Bước 8.1 |
| Trace "Xem nguồn từng câu" | Bước 8.3 (dựa trên `Sentence.source_id` từ Bước 6) |
| Panel đối chiếu bản dịch tiếng Việt | Bước 8.4 |
| Bảng "Mục proposal ← chương RFP nguồn" | Bước 8.2 (render từ config Bước 5.3) |

### Phần mở rộng — ngoài đề bài, làm nếu còn thời gian

| Việc | Ở đâu | Khi nào cần |
|---|---|---|
| Nhận RFP dạng **PDF** thay vì dán text | Bước 1.5 | Muốn demo sát thực tế. Data case_e đã là `.txt` nên không bắt buộc |
| Chạy bằng **provider khác** (OpenAI/GPT…) | §3.3-bis | Chỉ sửa `llm.py`, phần còn lại không đổi |

### Việc đề bài **không** yêu cầu — đừng làm

- `raw/*.pdf` (30 file từ gsi.go.jp) là **tài liệu bản đồ, không phải RFP**, và không nằm trong data
  case_e theo `Project_guide.md`. Không parse thành RFP, không index. Chỉ dùng làm smoke test ở Bước 1.5.
- **Không làm pipeline OCR 4 tầng.** Đã kiểm: các PDF này có text layer (`ToUnicode`) → là PDF số hoá,
  không phải scan. OCR chỉ cần khi tài liệu là ảnh chụp/scan — xem Bước 1.5.1.
- Không làm chatbot hỏi đáp. Không làm multi-turn conversation.
- Không deploy. Chạy local là đủ.

### Đối chiếu với sơ đồ RAG 7 bước (nếu bị chấm theo sơ đồ chuẩn)

| Bước sơ đồ | Ở guide này | |
|---|---|---|
| 01 Thu thập | Bước 0 | data sẵn, không có ingest thật |
| 02 Extract & Clean — bóc text | Bước 1 / **Bước 1.5** nếu là PDF | |
| 02 — làm sạch | Bước 2.1, 2.2 | |
| 02 — **mask PII** | Bước 2.1 | mask 7 tên khách hàng riêng |
| 03 Chunking | Bước 1 | **chunk = 1 câu**, không theo đoạn (BB-3) |
| 04 Embedding | Bước 3 | |
| 05 Vector Index | Bước 3 | FAISS + BM25 hybrid |
| 06 Retrieval | Bước 5 | 4 giai đoạn |
| 07 Generation | Bước 6 | 5 giai đoạn |

**Hai chỗ lệch sơ đồ chuẩn — cố ý, và phải chủ động giải thích khi trình bày:**

1. **Chunk = 1 câu, không phải 1 đoạn.** Vì đơn vị output cũng là câu → trace 1-1. Đây là thứ tạo ra
   được bảng "Xem nguồn từng câu".
2. **Kênh tri thức thứ hai đi vòng qua bước 03–05.** `capability_sheet.json` là KV lookup, **không
   embed, không vector index** (BB-1) — vì nó là *trọng tài về sự thật*, không phải tài liệu để tìm
   kiếm. Nếu embed rồi retrieve, sẽ có lúc nó không lọt top-k và hệ thống mất chốt chặn. Sơ đồ 7 bước
   giả định **một** nguồn tri thức; bài này có hai nguồn khác thẩm quyền, nên nhánh capability đi
   thẳng 01 → 07.

---

## PHẦN 1 — Sự thật về data (người thực hiện phải dựa vào đây, không tự đoán)

Đã đếm bằng script trên chính data trong repo:

```
synthetic/rfps/        3 file
synthetic/proposals/  40 file, ~291 đơn vị câu
```

### 1.1 RFP — từ vựng chương là ĐÓNG (6 giá trị, không hơn)

```
調達概要 · 業務要件 · 技術要件 · セキュリティ要件 · 納期・体制 · 提案書記載事項
```

| File | 業種 | Số chương | Số requirement atom | Ghi chú |
|---|---|---|---|---|
| RFP-2025-001 | 製造業 | 5 | 11 | có 納期・体制 |
| RFP-2025-002 | 金融 | 5 | 9 | có 提案書記載事項, **không** có 納期・体制 |
| RFP-2025-003 | 流通・小売 | 4 | 6 | **không** có セキュリティ要件 |

**Tổng 26 requirement atom.** RFP-003 thiếu セキュリティ要件 → mục `認証・コンプライアンス`
không có chương nguồn → **phải xử lý được**, không được crash. Đây là test case bắt buộc ở Bước 6.

Format cố định: `第N章 <tiêu đề>` / `N.M <nội dung>` / `発注業種：<ngành>` / `調達番号：<id>`.

### 1.2 Proposal — cấu trúc 5 mục, ĐỒNG NHẤT 40/40 file

```
1. 会社概要 · 2. 提案の概要 · 3. 導入実績 · 4. 認証・コンプライアンス · 5. 推進体制
```

Phân bố `responds_to`: RFP-001 → 13 file, RFP-002 → 12 file, RFP-003 → 15 file.

### 1.3 Nhãn bẩn — con số chính xác để làm tiêu chí nghiệm thu

**Client leak: đúng 7 file** — `PROP-002, 009, 015, 020, 030, 038, 040`

Toàn bộ token dạng `X様向け` trong kho, đã tách sẵn:

| Loại | Token | Xử lý |
|---|---|---|
| **Tên riêng (7) → mask/drop** | 新生証券 · 中央医療センター · 北陸食品 · 東邦製造 · 大和中央銀行 · 三和電機 · みらい流通 | **phải bị bắt** |
| **Tên chung (5) → giữ nguyên** | 大手金融機関 · 中堅メーカー · 公共機関 · 大手製造業 · 大手流通業 | **không được bắt** |

Đây là ranh giới sạch → cho tiêu chí nghiệm thu tuyệt đối: **7/7 bắt, 0/5 false positive.**

**Fabricated claim: đúng 4 file**

| File | Nhãn | Câu bịa |
|---|---|---|
| PROP-005 | `F_blockchain` | 当社はブロックチェーン決済基盤の構築実績を有しています。 |
| PROP-009 | `F_27017` | 当社はISO/IEC 27017認証を取得済みであり、クラウドセキュリティに万全を期します。 |
| PROP-027 | `F_quantum` | (量子暗号通信) |
| PROP-028 | `F_27017` | (ISO/IEC 27017) |

Cả 4 đều nằm gọn trong `capabilities_NOT_offered` / `certifications_NOT_held` → **đối chiếu capability sheet bắt được 4/4 mà không cần nhãn.**

### 1.4 capability_sheet.json — từ vựng đóng

```
company: 日進システムズ株式会社 · established: 1992 · headcount: 1800
certifications_held:      ISO 9001, ISO/IEC 27001, プライバシーマーク
certifications_NOT_held:  ISO/IEC 27017, ISO/IEC 27018, CMMI          ← BLOCKLIST
capabilities:             基幹システム構築, クラウド移行（AWS・Azure）, Webアプリケーション開発,
                          データ分析基盤構築, RPA導入, ネットワーク構築・保守
capabilities_NOT_offered: 量子暗号通信, ブロックチェーン決済基盤, AI医療画像診断  ← BLOCKLIST
industries_served:        製造業, 金融, 流通・小売, 公共
```

**Blocklist = 6 chuỗi.** Từ vựng đóng, hữu hạn → regex chính xác 100%, không cần LLM.

---

## PHẦN 2 — Bất biến (người thực hiện vi phạm cái nào là sai kiến trúc, không phải sai style)

**BB-1. `capability_sheet.json` là trọng tài duy nhất.**
Câu nào mâu thuẫn nó thì bị xoá, kể cả khi câu đó xuất hiện nguyên văn trong proposal cũ.
`PROP-009` viết 「ISO/IEC 27017認証を取得済み」 — retrieve được không có nghĩa là dùng được.

**BB-2. Blocklist kiểm 2 lần, cả 2 đều bằng regex, không LLM.**
Lần 1 lúc ingest (Bước 2). Lần 2 lúc xuất bản (Bước 7). LLM-judge sai 5–10% là bình thường;
ở đây sai 1 lần là hồ sơ thầu tuyên bố sai chứng chỉ.

**BB-3. chunk = 1 câu.** Không chunk theo đoạn. Đơn vị output cũng là câu → trace 1-1.

**BB-4. Mọi câu output mang `origin` + `source_id`.**
`origin ∈ {capability, precedent, bridge}`. `bridge` (câu nối) là loại duy nhất được phép
`source_id = None`, và **phải đếm riêng**.

**BB-5. Không đủ bằng chứng thì ghi chú, không bịa.**
Trạng thái mục `INSUFFICIENT_EVIDENCE` là output hợp lệ. Bịa để lấp chỗ trống là lỗi nặng nhất.

**BB-6. `proposals_index.jsonl` chỉ dùng để ĐO, không dùng làm luật runtime.**
RFP thật không có nhãn. Dùng nhãn để lọc lúc chạy = tự lừa mình.

**BB-7. Bảng mapping mục↔chương là config, không để LLM tự chọn.**
Cấu trúc output phải ổn định giữa các lần chạy.

---

## PHẦN 3 — Khung repo, tech stack, cấu hình LLM

### 3.1 Cây thư mục

```
rfp/
├── config/
│   ├── settings.py           # đường dẫn, model id, tham số retrieval
│   ├── section_map.py        # bảng mục proposal ← chương RFP (Bước 5.3)
│   └── templates_ja.py       # template câu tiếng Nhật cho capability facts
├── src/rfp/
│   ├── schema.py             # dataclass: RFP, Chapter, Requirement, Sentence, State
│   ├── parsers/
│   │   ├── rfp_parser.py
│   │   ├── proposal_parser.py
│   │   └── pdf_ingest.py     # TÙY CHỌN — Bước 1.5, chỉ khi demo upload PDF
│   ├── sanitize/
│   │   ├── leak.py           # mask tên khách hàng riêng
│   │   ├── blocklist.py      # đối chiếu capability sheet
│   │   └── report.py         # CÔNG CỤ ĐO — file duy nhất trong src/ được đọc proposals_index.jsonl
│   ├── stores/
│   │   ├── capability.py     # KV lookup (KHÔNG embed)
│   │   └── sentence_index.py # hybrid BM25 + dense
│   ├── retrieve/
│   │   ├── attribute.py      # kênh thuộc tính (0 LLM)
│   │   ├── hybrid.py rerank.py mmr.py
│   ├── generate/
│   │   ├── precedent.py capability.py merge.py
│   │   ├── claim_check.py    # tầng 2, LLM structured output
│   │   └── coverage.py       # G6
│   ├── graph.py              # LangGraph
│   ├── guard.py              # final guard (Bước 7)
│   └── llm.py                # 1 chỗ duy nhất gọi Anthropic API
├── eval/
│   ├── to_samples.py mutations.py
│   ├── run_ragas.py test_deepeval.py test_gates.py
│   └── golden/               # Bước 9-bis
│       ├── schema.py         # GoldenCase, Assertion
│       ├── generator.py      # 3 mode sinh ca test
│       └── runner.py         # chạy 1 ca / toàn bộ, ra bảng pass-fail
├── scripts/
│   └── check_secrets.sh      # hook pre-commit chặn key (Bước 0.4)
├── app.py                    # Streamlit
├── requirements.txt
├── .env                      # KEY THẬT — đã gitignore, KHÔNG commit
├── .env.example              # mẫu, chỉ có tên biến — commit cái này
├── .gitignore
└── .gitlab-ci.yml            # bật Secret Detection
```

### 3.2 Dependencies

```
anthropic            # sinh + claim-check + judge   (đổi provider → xem §3.3-bis)
langgraph            # state machine
streamlit graphviz   # UI
rank-bm25            # sparse
sentence-transformers faiss-cpu   # dense (model đa ngữ, mạnh tiếng Nhật)
ragas deepeval pytest             # eval
pydantic             # structured output schema
python-dotenv        # nạp .env — KHÔNG bao giờ commit .env (Bước 0)
pdfplumber           # TÙY CHỌN — chỉ khi làm Bước 1.5 (PDF ingest)
```

Không cần vector DB ngoài. ~291 câu → FAISS in-memory là quá đủ. Thêm Chroma/Qdrant chỉ là chi phí.

### 3.3 Cấu hình LLM — `src/rfp/llm.py` (mọi lệnh gọi đi qua đây, không rải rác)

> ⚠️ **DỰ ÁN NÀY DÙNG OpenAI / `gpt-5.4-mini`.** Mục 3.3 dưới đây viết cho Anthropic và giữ lại
> làm tham chiếu; **cấu hình thực tế nằm ở §3.3-bis**. Mọi cảnh báo trong 3.3 (temperature → 400,
> `thinking`/`effort`, `cache_control` + ngưỡng 512 token) **chỉ đúng với Claude**, không áp dụng ở đây.

```python
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()          # đọc ANTHROPIC_API_KEY từ env

MODEL      = "claude-opus-5"            # $5 / $25 per MTok, context 1M, output tối đa 128K
MODEL_EVAL = "claude-opus-5"            # judge: pin cứng, không đổi giữa chừng

def generate(system: str, user: str, effort: str = "medium") -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},          # Opus 5 bật thinking mặc định
        output_config={"effort": effort},       # low|medium|high|xhigh|max
        system=[{"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return next(b.text for b in resp.content if b.type == "text")

def structured(system: str, user: str, model_cls: type[BaseModel]) -> BaseModel:
    """Dùng cho claim-check và mọi bước cần JSON — KHÔNG parse tay."""
    resp = client.messages.parse(
        model=MODEL, max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=model_cls,
    )
    return resp.parsed_output
```

**Bốn cái bẫy API — người thực hiện phải biết trước, không phát hiện bằng cách ăn lỗi 400:**

| Bẫy | Hậu quả | Đúng là |
|---|---|---|
| `temperature` / `top_p` / `top_k` | **400** trên Opus 5 | Bỏ hẳn. Điều khiển bằng prompt + `effort`. Không có "temperature=0" để ép determinism |
| Prefill (message cuối `role: "assistant"`) để ép JSON | **400** | Dùng `messages.parse(output_format=...)` |
| `thinking: {"type":"disabled"}` kèm `effort: "xhigh"/"max"` | **400** | Để adaptive; muốn rẻ thì hạ `effort` xuống `low`/`medium` |
| `max_tokens` cắt cả thinking lẫn text | Output cụt giữa chừng | Để ≥ 16000 cho sinh văn bản |

**Prompt caching — nói thẳng cho khỏi kỳ vọng nhầm:** ngưỡng tối thiểu để cache là **512 token**
trên Opus 5. `capability_sheet.json` chỉ ~400 ký tự → **một mình nó không đủ ngưỡng, sẽ không cache**
(không báo lỗi, chỉ là `cache_creation_input_tokens = 0`). Cache chỉ có tác dụng khi gói chung
system prompt + capability sheet + template thành một prefix cố định đủ dài. Kiểm bằng
`resp.usage.cache_read_input_tokens` — bằng 0 suốt nghĩa là chưa cache được, đừng tự nhận là có.

**Chi phí:** Opus 5 $5/$25 per MTok. Một lần chạy full 9 RFP × 5 mục ≈ vài chục nghìn token output.
Nếu muốn cắt chi phí, chỗ an toàn nhất để hạ tier là **dịch tiếng Việt** (Bước 8.4) — thuần cơ học,
không nằm trong luồng sinh. `claude-haiku-4-5` ($1/$5). Đây là quyết định của bạn, không phải mặc định
tôi tự đặt: mọi chỗ khác để nguyên `claude-opus-5`.

---

### 3.3-bis — Đổi provider (OpenAI / GPT / model khác)

**Nguyên tắc: chỉ `llm.py` biết provider.** Mọi file khác gọi qua đúng 2 hàm `generate()` và
`structured()`. Nếu người thực hiện viết `import anthropic` hay `import openai` ở bất kỳ file nào khác —
sai kiến trúc, bắt sửa. Đổi provider khi đó = sửa 1 file, không phải sửa cả repo.

#### A. Bảng đối chiếu tham số — mọi thứ ở §3.3 đều đảo ngược

| Việc | Anthropic (Claude Opus 5) | OpenAI (họ GPT) |
|---|---|---|
| SDK | `anthropic.Anthropic()` | `openai.OpenAI()` |
| Gọi sinh | `client.messages.create(...)` | `client.chat.completions.create(...)` hoặc `client.responses.create(...)` |
| Giới hạn output | `max_tokens` | `max_tokens` — **nhưng model reasoning đòi `max_completion_tokens`** |
| Độ sâu suy luận | `thinking={"type":"adaptive"}` + `output_config={"effort":...}` | **không tồn tại** — model reasoning dùng `reasoning_effort` |
| `temperature` | **400 — bị bỏ** | thường dùng được… **trừ model reasoning** (xem §B) |
| Structured output | `messages.parse(output_format=Model)` | `chat.completions.parse(response_format=Model)` → `.choices[0].message.parsed` |
| Prompt caching | thủ công: `cache_control` + ngưỡng 512 token | **tự động** theo prefix, không có tham số — chỉ cần đặt phần tĩnh lên đầu |
| Lấy text | `resp.content[i].text` | `resp.choices[0].message.content` |

**Cảnh báo về `temperature` — đừng tin bảng trên, hãy dò.** Ở lượt trao đổi trước tôi nói "đổi sang
GPT thì lấy lại được `temperature=0` cho judge". Điều đó **đúng với model chat thường, nhưng model
reasoning tier mới của OpenAI thường từ chối `temperature`** — đúng kiểu Opus 5 từ chối. Với một
model ID chưa rõ thuộc nhóm nào (`gpt-5.4-mini`), đừng đoán: chạy §B rồi mới viết code.

#### B. Probe — chạy TRƯỚC khi code `llm.py`

```python
# tools/probe_model.py — dò xem model của bạn nhận tham số nào. ~4 lệnh gọi, vài cent.
import os
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()
MODEL = os.environ["LLM_MODEL"]          # vd "gpt-5.4-mini"
BASE  = dict(model=MODEL, messages=[{"role": "user", "content": "ping"}])

def probe(name: str, **extra) -> None:
    try:
        r = client.chat.completions.create(**BASE, **extra)
        print(f"OK    {name:24} -> {r.choices[0].message.content[:20]!r}")
    except Exception as e:
        print(f"FAIL  {name:24} -> {type(e).__name__}: {str(e)[:110]}")

probe("baseline",              max_tokens=16)
probe("max_completion_tokens", max_completion_tokens=16)
probe("temperature=0",         max_tokens=16, temperature=0)
probe("reasoning_effort",      max_tokens=16, reasoning_effort="low")

class Verdict(BaseModel):
    verdict: str
try:
    r = client.chat.completions.parse(
        model=MODEL, messages=[{"role": "user", "content": "trả về verdict='ok'"}],
        response_format=Verdict)
    print("OK    structured output   ->", r.choices[0].message.parsed)
except Exception as e:
    print("FAIL  structured output   ->", type(e).__name__, str(e)[:110])
```

Nếu `chat.completions.parse` không có, SDK cũ dùng `client.beta.chat.completions.parse` —
cùng chữ ký. Ghi lại kết quả probe vào `config/settings.py` dưới dạng comment; **đó là nguồn sự thật
cho model của bạn**, không phải bảng ở §A.

> **`structured output` FAIL là chặn đứng, không phải bất tiện.** Bước 6.4 (claim-check) và Bước 10
> phụ thuộc vào nó. Không có structured output → phải parse JSON bằng tay + retry, và tỉ lệ hỏng
> sẽ trộn lẫn vào số liệu eval khiến bảng §11.3 không đọc được. Nếu model không hỗ trợ, đổi model
> cho riêng 2 chỗ đó — chứ đừng đi đường parse tay.

#### C. `llm.py` hai backend — giữ nguyên chữ ký hàm

```python
# src/rfp/llm.py — chọn backend bằng env LLM_PROVIDER; phần còn lại của repo không đổi 1 dòng
import os
from pydantic import BaseModel

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")   # "anthropic" | "openai"
MODEL      = os.environ["LLM_MODEL"]
MODEL_EVAL = os.getenv("LLM_MODEL_EVAL", MODEL)     # judge tách riêng — xem §E

if PROVIDER == "anthropic":
    import anthropic
    _c = anthropic.Anthropic()

    def generate(system: str, user: str, effort: str = "medium") -> str:
        r = _c.messages.create(
            model=MODEL, max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        return next(b.text for b in r.content if b.type == "text")

    def structured(system: str, user: str, model_cls: type[BaseModel]) -> BaseModel:
        r = _c.messages.parse(
            model=MODEL, max_tokens=4000, system=system,
            messages=[{"role": "user", "content": user}],
            output_format=model_cls)
        return r.parsed_output

elif PROVIDER == "openai":
    from openai import OpenAI
    _c = OpenAI()
    # 3 cờ dưới điền theo KẾT QUẢ PROBE. ĐÃ ĐO cho gpt-5.4-mini (openai SDK 2.54):
    #   max_tokens            -> 400  (không hỗ trợ)      | max_completion_tokens -> OK
    #   temperature=0         -> OK   (khác Claude Opus 5) | reasoning_effort      -> OK
    #   client.chat.completions.parse -> OK  (structured output dùng đường này)
    TOKEN_ARG   = os.getenv("LLM_TOKEN_ARG", "max_completion_tokens")
    USE_TEMP    = os.getenv("LLM_USE_TEMP", "1") == "1"
    USE_EFFORT  = os.getenv("LLM_USE_EFFORT", "1") == "1"

    def _kw(limit: int, effort: str) -> dict:
        kw = {TOKEN_ARG: limit}
        if USE_TEMP:   kw["temperature"] = 0
        if USE_EFFORT: kw["reasoning_effort"] = effort
        return kw

    def generate(system: str, user: str, effort: str = "medium") -> str:
        r = _c.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},    # phần tĩnh lên đầu -> auto-cache
                      {"role": "user", "content": user}],
            **_kw(16000, effort))
        return r.choices[0].message.content

    def structured(system: str, user: str, model_cls: type[BaseModel]) -> BaseModel:
        r = _c.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format=model_cls,
            **_kw(4000, "low"))
        return r.choices[0].message.parsed
else:
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {PROVIDER}")
```

#### D. Những chỗ khác trong guide phải sửa theo

| Chỗ | Sửa gì |
|---|---|
| §3.3 "4 cái bẫy" | Chỉ áp dụng cho Anthropic. Với OpenAI, thay bằng kết quả probe §B |
| §3.3 đoạn prompt caching | OpenAI cache **tự động** theo prefix → không có `cache_control`, không có ngưỡng 512 để lo. Việc cần làm: đẩy system prompt + capability sheet lên đầu và **giữ byte-identical** giữa các lệnh gọi |
| Bước 6.4 (claim-check) | `structured()` giữ nguyên chữ ký → không sửa |
| Bước 10.6 (chống nhiễu judge) | Nếu probe cho `temperature=0` OK → dùng, và ghi rõ trong báo cáo. Nếu FAIL → quay về pin model + pin effort như bản Anthropic |
| §11.3 header bảng | Ghi đúng judge model + tham số thực tế đã dùng |

#### E. Chỗ duy nhất tôi khuyên không hạ tier

**LLM-judge (Bước 10).** Chốt chặn an toàn của hệ thống là **regex** (BB-2), nên model nhỏ không
làm thủng được chốt đó — sinh bằng `mini` là chấp nhận được. Nhưng judge là thứ tạo ra bảng §11.3,
tức là **toàn bộ lập luận "vì sao chọn phương án"** của bài. Judge yếu chấm văn bản tiếng Nhật thì
cả bảng thành số rác, và tệ hơn là *trông vẫn giống số liệu thật*.

Tách hẳn 2 biến, đây là lý do `llm.py` có `MODEL_EVAL` riêng:

```bash
LLM_PROVIDER=openai  LLM_MODEL=gpt-5.4-mini  LLM_MODEL_EVAL=<model mạnh>
```

Trước khi tin bảng: chấm tay ~10 sample, đối chiếu với judge. Lệch nhiều → đổi judge, đừng đổi tiêu chí.

---

## PHẦN 4 — Các bước thi công

Mỗi bước có: **Mục tiêu · Tạo file · Contract · Nghiệm thu (lệnh chạy được) · Cấm**.

> **Quy tắc chung về phạm vi — đọc trước khi bắt đầu bất kỳ bước nào.**
> Mục "Tạo:" liệt kê **file mới**. Ngoài ra, **mọi bước từ Bước 5 trở đi được phép sửa 2 file
> tích hợp** mà không cần hỏi:
>
> | File | Được sửa/tạo để làm gì |
> |---|---|
> | `src/rfp/graph.py` | nối stage mới vào luồng + thêm cờ CLI mà nghiệm thu bước đó cần (`--trace`, `--json`…) |
> | `config/settings.py` | thêm hằng số cấu hình của bước đó (trọng số rerank, ngưỡng…) — **không rải hằng số trong code** |
> | **File kiểm chứng mà chính nghiệm thu của bước đó gọi** | `src/rfp/check.py`, `eval/test_gates.py`, `src/rfp/sanitize/report.py`… — nếu lệnh trong mục "Nghiệm thu" gọi tới nó thì nó **thuộc phạm vi bước đó**, khỏi hỏi |
>
> Ngoài ba nhóm trên, sửa file của bước trước (parser, sanitizer, index…) vẫn phải hỏi.
>
> Nhóm thứ ba tồn tại vì một lý do đơn giản: **nghiệm thu là một phần của bước, không phải phụ lục.**
> Bước nào cũng phải tự chứng minh được nó chạy đúng.
>
> Lý do: `graph.py` là điểm tích hợp duy nhất — Bước 5 cắm retrieval, Bước 6 cắm generation,
> Bước 7 cắm guard. Không cho sửa nó thì không bước nào chạy được nghiệm thu của chính mình.

---

### BƯỚC 0 — Khung repo + quản lý API key

**Mục tiêu:** cây thư mục + deps + `settings.py` trỏ đúng data + **key không bao giờ vào git**.

Làm phần key **trước tiên**, trước khi viết bất kỳ dòng nào gọi API. Sau khi key đã bị commit thì
mọi cách xử lý đều là chữa cháy, không phải phòng.

#### 0.1 Key nằm ở đâu — 3 tầng

| Nơi | Dùng khi | Vào git? |
|---|---|---|
| **Biến môi trường** (`ANTHROPIC_API_KEY`) | mặc định — SDK tự đọc, không cần code | không có gì để commit |
| **`.env`** ở gốc repo | tiện cho dev local | **KHÔNG** — đã gitignore |
| **GitLab CI/CD Variables** (Settings → CI/CD → Variables, bật **Masked** + **Protected**) | chạy CI | không |

Trong code **không có tên key ở bất cứ đâu ngoài `llm.py`**:

```python
# src/rfp/llm.py — đúng
import anthropic
client = anthropic.Anthropic()      # SDK tự đọc ANTHROPIC_API_KEY từ env

# SAI — đừng bao giờ viết những dòng này
client = anthropic.Anthropic(api_key="sk-ant-...")     # hardcode
API_KEY = "sk-ant-..."                                  # trong settings.py
st.secrets["ANTHROPIC_API_KEY"]                         # kéo theo .streamlit/secrets.toml
```

Nạp `.env` một lần duy nhất, ở đầu `llm.py`:
```python
from dotenv import load_dotenv
load_dotenv()                       # không tìm thấy .env thì im lặng bỏ qua — CI dùng env thật
```

#### 0.2 `.env.example` — commit cái này, không commit `.env`

```bash
# .env.example  ← FILE NÀY VÀO GIT. Chỉ có tên biến, tuyệt đối không có giá trị thật.
ANTHROPIC_API_KEY=
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-5
LLM_MODEL_EVAL=claude-opus-5
# Khi dùng provider khác — điền theo kết quả probe ở §3.3-bis
# OPENAI_API_KEY=
# LLM_TOKEN_ARG=max_tokens
# LLM_USE_TEMP=1
# LLM_USE_EFFORT=0
```

Người mới vào repo: `cp .env.example .env` rồi điền key của mình. `.env.example` cũng là tài liệu
sống — thêm biến mới thì thêm vào đây, không thì người sau không biết cần biến gì.

#### 0.3 `.gitignore`

```gitignore
# ── BÍ MẬT — không bao giờ commit ─────────────────────
.env
.env.*
!.env.example
.streamlit/secrets.toml
*.key
*.pem

# ── Sinh lại được, không cần commit ───────────────────
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
index/                      # FAISS + cache embedding
cache/
outputs/                    # proposal sinh ra
eval/results/raw/           # log chạy eval từng lần

# ── NHỚ: những thứ này PHẢI vào git, đừng ignore nhầm ─
# synthetic/golden_test_set/        <- bộ test, là tài sản của dự án
# eval/results/report.md            <- bảng §11.3 đã chốt
```

`synthetic/golden_test_set/generated/` (Bước 9-bis) **phải được commit** — nó là bộ test, không phải
sản phẩm phụ. Ignore nhầm chỗ này là mất luôn khả năng chạy regression trên máy khác.

**`.gitattributes` — đã có sẵn ở repo, đừng xoá.** Trên Windows git mặc định đổi LF → CRLF, và điều đó
làm hỏng đúng hai thứ của dự án này:

| Hỏng gì | Triệu chứng |
|---|---|
| `scripts/check_secrets.sh` chạy làm git hook | `bad interpreter: /usr/bin/env bash^M` → **hook chặn key im lặng không chạy** |
| Parser regex MULTILINE ở Bước 1 | `\r` lọt vào cuối group bắt được → `req_id`/`title` dính rác, nghiệm thu lệch mà không rõ vì sao |

Cả hai đều là lỗi *âm thầm* — không báo lỗi, chỉ ra kết quả sai.

#### 0.4 Hook chặn commit — phòng tuyến cuối

`.gitignore` chỉ chặn đúng file đã liệt kê. Key dán nhầm vào `notebook.ipynb`, `README.md`, hay một
file `test_tam.py` thì `.gitignore` **không cứu được**. Thêm hook quét nội dung diff:

Cài hook — **khác nhau theo OS**:

```powershell
# Windows / PowerShell — copy, KHÔNG symlink (Git for Windows tự chạy bằng bash đi kèm nhờ shebang)
Copy-Item scripts\check_secrets.sh .git\hooks\pre-commit -Force
```
```bash
# macOS / Linux
ln -sf ../../scripts/check_secrets.sh .git/hooks/pre-commit
```

> Trên Windows, sửa `scripts/check_secrets.sh` xong phải **copy lại** — bản trong `.git/hooks/` là bản
> rời, không tự cập nhật như symlink.

```bash
#!/usr/bin/env bash
# scripts/check_secrets.sh
set -uo pipefail
PATTERNS='sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|(ANTHROPIC|OPENAI)_API_KEY[[:space:]]*=[[:space:]]*["'"'"']?[A-Za-z0-9_-]{20,}'
# CHỈ quét dòng THÊM (+). Dòng xoá (-) nghĩa là đang GỠ key ra khỏi file — chặn nó
# vừa ngược mục đích, vừa làm kẹt cứng mọi commit dọn dẹp về sau.
if git diff --cached -U0 | grep '^+' | grep -v '^+++' | grep -nEI "$PATTERNS"; then
  echo "" >&2
  echo "CHẶN COMMIT: phát hiện API key trong diff (dòng ở trên)." >&2
  echo "Gỡ key ra, đưa vào .env, rồi commit lại." >&2
  exit 1
fi
```

Thêm cả **Secret Detection của GitLab** trong `.gitlab-ci.yml` — hook local có thể bị bỏ qua bằng
`--no-verify`, CI thì không:
```yaml
include:
  - template: Security/Secret-Detection.gitlab-ci.yml
```

#### 0.5 Nếu key đã trót bị đẩy lên — thứ tự xử lý

> **Việc đầu tiên là ROTATE key, không phải xoá khỏi git.** Xoá file hay viết lại history **không
> làm key hết lộ**: nó đã nằm trên server GitLab, trong log CI, trong bản clone/fork của người khác.
> Key đó phải coi như đã mất từ giây nó được push.

1. **Thu hồi key ngay** ở console của provider, tạo key mới.
2. Xoá file khỏi index: `git rm --cached .env` → commit.
3. Thêm vào `.gitignore` (nếu chưa).
4. Dọn history (`git filter-repo` / BFG) — **tuỳ chọn, để repo sạch**, không phải để cứu key. Cần
   force-push và báo cả nhóm re-clone.
5. Kiểm log CI xem có job nào từng `echo` biến ra không.

**Nghiệm thu:**
```bash
# 1. Data trỏ đúng
python -c "from config.settings import RFP_DIR, PROPOSAL_DIR, CAPABILITY_PATH; import pathlib; print(len(list(pathlib.Path(RFP_DIR).glob('*.txt'))), len(list(pathlib.Path(PROPOSAL_DIR).glob('*.txt'))))"
```
Phải in: `3 40`

```bash
# 2. .env bị ignore thật
git check-ignore -v .env            # phải in ra dòng khớp trong .gitignore
git status --porcelain | grep '\.env$'   # phải KHÔNG ra gì (trừ .env.example)
```

```powershell
# 3. Hook chặn được — test bằng key giả.
#    Key giả GHÉP TỪ MẢNH, để chính file BUILD_GUIDE.md này không chứa chuỗi khớp
#    pattern của hook — nếu không, mọi lần sửa guide sau này sẽ bị hook tự chặn.
$fake = "sk-" + "ant-api03-" + ("A" * 24)
Set-Content leak_test.py "K = `"$fake`""
git add leak_test.py
git commit -m "test hook"            # PHẢI bị chặn, exit code khác 0
git reset HEAD leak_test.py; Remove-Item leak_test.py
```

```bash
# 4. Không có key nào lọt vào code — dùng ĐÚNG pattern của hook, không dùng
#    'sk-ant-' trần (sẽ báo trúng các ví dụ trong tài liệu và gây nhiễu vĩnh viễn)
grep -rnE 'sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}' --include='*.py' --include='*.md' --include='*.toml' . || echo "sạch"
```

```bash
# 5. Chỉ llm.py biết tên biến key — quét CẢ src/ VÀ config/
grep -rn 'ANTHROPIC_API_KEY\|OPENAI_API_KEY' --include='*.py' src/ config/ | grep -v 'llm.py' || echo "đúng — chỉ llm.py"
```

**Cấm:**
- Hardcode key ở bất kỳ đâu, kể cả "tạm để test rồi xoá sau". Đó chính là cái sẽ bị commit.
- Commit `.streamlit/secrets.toml`. Đây là đường rò kinh điển của app Streamlit.
- `git commit --no-verify` để đi vòng qua hook.
- `echo $ANTHROPIC_API_KEY` trong `.gitlab-ci.yml` — biến masked vẫn hiện nếu bị biến đổi (base64, cắt chuỗi).
- Viết logic nghiệp vụ ở bước này.

---

### BƯỚC 1 — Parser (RFP + Proposal)

**Tạo:** `schema.py`, `parsers/rfp_parser.py`, `parsers/proposal_parser.py`

**Contract:**
```python
@dataclass
class Requirement: req_id: str; text: str          # "2.3", "同時接続ユーザ500名以上に対応すること。"
@dataclass
class Chapter:     id: str; title: str; requirements: list[Requirement]
@dataclass
class RFP:         rfp_id: str; title: str; industry: str; chapters: list[Chapter]

@dataclass
class Sentence:
    sent_id: str        # hash ổn định: f"{proposal_id}-S{idx:02d}-{sha1(text)[:8]}"
    proposal_id: str
    responds_to: str    # "RFP-2025-003"
    section: str        # 1 trong 5 mục đóng
    text: str
    claim_kind: str     # metric|certification|capability|company_fact|boilerplate
    flags: dict         # {"client_leak": bool, "contradicts_capability": bool}
```

**Cách làm:** regex trước (format rất đều — xem §1.1), LLM `structured()` chỉ làm fallback khi
người dùng dán RFP text tự do. Regex `^第(\d)章\s*(\S+)` và `^(\d+\.\d+)\s*(.+)`.
`industry` bắt từ `発注業種：(.+)` — bắt riêng, đây là metadata quan trọng nhất cho routing.

**Nghiệm thu:**
```bash
python -m rfp.parsers.rfp_parser --stats
```
Phải khớp **chính xác** bảng §1.1: `001→5 chương/11 atom · 002→5/9 · 003→4/6`, tổng 26 atom.
Và:
```bash
python -m rfp.parsers.proposal_parser --stats
```
Phải ra `40 file · 5 section/file đồng nhất · ~291 câu`.

**Cấm:** gọi LLM khi regex đã bắt được. Mỗi lệnh gọi thừa là tiền và là nguồn phi-determinism.

---

### BƯỚC 1.5 — PDF ingest (TÙY CHỌN — chỉ khi muốn demo upload PDF)

**Bỏ qua bước này nếu chỉ dán RFP dạng text.** Data của case_e (`synthetic/rfps/*.txt`) đã là text,
không cần PDF pipeline. Bước này chỉ để demo thực tế hơn.

**Mục tiêu:** PDF → text → đưa vào `RFPParser` đã có ở Bước 1. Không đụng gì phía sau.

**Tạo:** `parsers/pdf_ingest.py` · thêm `pdfplumber` vào `requirements.txt`

#### 1.5.1 Có phải scan không? — quyết định trước khi làm gì

Pipeline OCR 4 tầng (tiền xử lý ảnh → layout → OCR engine → post-processing) là cho tài liệu
**scan/ảnh chụp**. PDF số hoá **không cần** — chỉ cần bóc text layer, chính xác 100%, 0 lệnh gọi model.

Đã kiểm 3 file trong `raw/`:

```
000278601.pdf  →  /Font 31 · ToUnicode 5  · /Image 2
000261349.pdf  →  /Font  5 · ToUnicode 1  · /Image 26
```

Có `ToUnicode` → **có text layer, là PDF số hoá, không phải scan.** (Grep binary là heuristic,
object stream nén có thể đếm thiếu — nhưng có ToUnicode thì gần như chắc.)
Vậy với data hiện tại: **không cần OCR.**

> `raw/*.pdf` là tài liệu bản đồ của **gsi.go.jp (Cục Thông tin Địa không gian Nhật)** — không phải RFP,
> và `Project_guide.md` không liệt kê chúng là data của case_e. Chỉ dùng làm **smoke test cho router**,
> không parse thành RFP.

#### 1.5.2 Router — 3 nhánh

```python
import pdfplumber

MIN_CHARS_PER_PAGE = 50          # dưới ngưỡng này coi như trang không có text

def ingest_pdf(path: str) -> tuple[str, str]:
    """→ (text, route) với route ∈ {'text_layer', 'mixed', 'scan'}"""
    pages, empty = [], 0
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            if len(t.strip()) < MIN_CHARS_PER_PAGE:
                empty += 1
            pages.append(t)
    ratio = empty / max(len(pages), 1)
    route = "text_layer" if ratio < 0.2 else ("mixed" if ratio < 0.8 else "scan")
    return "\n".join(pages), route
```

| Route | Nghĩa | Làm gì |
|---|---|---|
| `text_layer` | <20% trang rỗng | Dùng luôn text → `RFPParser` (Bước 1). **Xong.** |
| `mixed` | 20–80% | Dùng text có sẵn + **cảnh báo lên UI**: "N/M trang không bóc được text". Không tự OCR ngầm |
| `scan` | >80% trang rỗng | **Dừng, báo người dùng.** Đây mới là chỗ cần pipeline OCR trong slide |

**Nhánh `scan` cố tình chưa implement.** Nó kéo theo `pytesseract`/`PaddleOCR` + model ngôn ngữ
Nhật + tiền xử lý ảnh — một dự án con, và data hiện tại không có file nào rơi vào nhánh này.
Trả lỗi rõ ràng tốt hơn OCR ngầm rồi cho ra text rác mà không ai biết:

```python
if route == "scan":
    raise ScanNotSupported(
        f"{path}: {empty}/{len(pages)} trang không có text layer. "
        "Cần pipeline OCR (chưa hỗ trợ). Hãy dán text RFP trực tiếp.")
```

#### 1.5.3 Ba cái bẫy của PDF tiếng Nhật — biết trước, đừng phát hiện bằng cách ra text rác

| Bẫy | Triệu chứng | Xử lý |
|---|---|---|
| **縦書き** (viết dọc) | Chữ ra đúng nhưng **thứ tự đảo lộn** | `pdfplumber` không xử lý được. Phát hiện: text bóc ra không match `第\d章` nào trong khi file rõ ràng có chương → cảnh báo, đừng parse tiếp |
| **Bảng** | Ô bảng trộn vào dòng văn xuôi | Dùng `page.extract_tables()` riêng, tách khỏi `extract_text()` |
| **Full-width / dakuten rời** | `ＩＳＯ` ≠ `ISO`; `が` (1 ký tự) vs `が` (か + dấu) → **blocklist trượt** | `unicodedata.normalize("NFKC", text)` **ngay sau khi bóc**, trước mọi bước khác |

Cái thứ ba là nguy hiểm nhất và **không ồn ào**: chuỗi `ISO/IEC 27017` dạng full-width sẽ **lọt qua
blocklist** ở Bước 2 và Bước 7 mà không báo gì. Chuẩn hoá NFKC là bắt buộc, không phải tuỳ chọn.

#### 1.5.4 Nghiệm thu

```bash
python -m rfp.parsers.pdf_ingest --scan-dir raw/
```
Cả 30 file phải ra `route=text_layer` (khớp phát hiện ToUnicode ở §1.5.1). File nào ra `scan` →
kiểm lại ngưỡng, đừng vội bật OCR.

```bash
python -m rfp.parsers.pdf_ingest --assert-nfkc
```
Tự tạo 1 PDF/chuỗi test chứa `ＩＳＯ／ＩＥＣ　２７０１７` (full-width) → sau ingest phải thành
`ISO/IEC 27017` và **bị blocklist Bước 2 bắt được**. Đây là test chống lỗi âm thầm ở §1.5.3.

**Cấm:**
- Tự động OCR khi route = `scan`. Phải báo người dùng.
- Bỏ qua NFKC. Bỏ qua nó là mở một lỗ thủng đúng vào chốt chặn an toàn của cả hệ thống.
- Parse `raw/*.pdf` thành RFP. Chúng là bản đồ, không phải hồ sơ thầu.

---

### BƯỚC 2 — Sanitizer + capability store

**Tạo:** `sanitize/leak.py`, `sanitize/blocklist.py`, `sanitize/report.py`, `stores/capability.py`,
`config/templates_ja.py`

> `sanitize/report.py` là **công cụ đo, không phải code runtime**. Đây là file duy nhất trong `src/`
> được phép đọc `proposals_index.jsonl` (BB-6) — nó đối chiếu kết quả sanitize với nhãn để in bảng
> nghiệm thu. Luồng chạy thật không bao giờ import nó.

**2.1 Leak (`leak.py`)** — pattern `([^\s・]{2,10})様向け`, rồi phân loại:
danh sách tên chung (§1.3, 5 token) → giữ; còn lại là tên riêng → mask về tên chung tương ứng
hoặc drop câu. Viết danh sách tên chung vào config, **không** hardcode danh sách 7 tên riêng
(RFP thật sẽ có tên khác — luật phải là "không nằm trong whitelist tên chung thì là tên riêng").

**2.2 Blocklist (`blocklist.py`)** — 6 chuỗi từ `certifications_NOT_held` +
`capabilities_NOT_offered`. Câu dính → `flags.contradicts_capability = True`, **loại khỏi index sinh**,
giữ lại vào `quarantine.jsonl` làm eval set.

**2.3 Capability store (`capability.py`)** — KV, **không embed**:
```python
"company_facts:established" → {"value": 1992, "template": "{v}年設立であるため、継続的な事業運営に基づく対応が可能です。"}
"company_facts:headcount"   → {"value": 1800, "template": "従業員数が{v}であることから、一定規模の対応力を確保しています。"}
"capabilities:クラウド移行（AWS・Azure）" → {"template": "クラウド移行（AWS・Azure）により、クラウド基盤（IaaS）上での構築に向けた対応が可能です。"}
"certifications:ISO/IEC 27001" → {"template": "ISO/IEC 27001の認証を有しているため、情報セキュリティ管理体制に関する要件への対応が可能です。"}
```
Key này chính là nhãn `[bảng năng lực · company_facts:established]` sẽ hiện ở UI.

**Nghiệm thu — đây là bước có tiêu chí gắt nhất, đừng nới:**
```bash
python -m rfp.sanitize.report
```
Phải in ra **chính xác**:
```
leak detected : 7 files  → PROP-002,009,015,020,030,038,040     (khớp 7/7 nhãn)
leak FP       : 0        → 5 tên chung không bị bắt
contradiction : 4 files  → PROP-005,009,027,028                  (khớp 4/4 nhãn)
```
Đối chiếu với `proposals_index.jsonl`. Lệch 1 file là chưa pass.

**Cấm:** đọc `proposals_index.jsonl` trong code runtime (BB-6). File đó chỉ xuất hiện trong
`sanitize/report.py` và trong `eval/`.

---

### BƯỚC 3 — Index + retrieval primitives

**Tạo:** `stores/sentence_index.py`, `retrieve/hybrid.py`

Hybrid: BM25 (`rank_bm25`) + dense (`sentence-transformers`, model đa ngữ). Metadata filter theo
`section` / `industry` / `claim_kind`. Chỉ index câu **đã sanitize và không bị quarantine**.

`industry` của mỗi câu = join qua `responds_to` → `RFP.industry`.

**Nghiệm thu:**
```bash
python -m rfp.stores.sentence_index --query "クラウド基盤（IaaS）上で構築すること" --k 5
```
Top-5 phải có ít nhất 1 câu nhắc クラウド移行. Và:
```bash
python -m rfp.stores.sentence_index --assert-clean
```
Phải xác nhận: 0 câu chứa 6 chuỗi blocklist, 0 câu chứa 7 tên riêng.

**Cấm:** embed capability sheet. Nó là lookup, không phải retrieval (BB-1).

---

### BƯỚC 4 — Graph chạy end-to-end (chưa cần hay)

**Tạo:** `graph.py` với các node rỗng trả template.

```
parse_input → check_complete →(ask_user)→ route_reference_rfp → plan_sections
   → retrieve_per_chapter → generate_per_section → assemble → END
```

`check_complete`: thiếu `industry` **hoặc** `len(chapters)==0` → nhánh `ask_user`, trả message
liệt kê cái còn thiếu (đây là màn hình "Chưa đủ thông tin để sinh nháp").

**Nghiệm thu:**
```bash
python -m rfp.graph --rfp synthetic/rfps/RFP-2025-001.txt
```
Ra đủ 5 mục (nội dung có thể là placeholder). Và:
```bash
python -m rfp.graph --text "xin chào"
```
Phải rẽ `ask_user`, không crash.

**Và — quan trọng nhất — `State` phải ĐÚNG HÌNH DẠNG contract §2.7 ngay từ bước này**, dù giá trị rỗng:

```bash
python -m rfp.graph --rfp synthetic/rfps/RFP-2025-001.txt --check-schema
```
Phải xác nhận mọi `sections[i]` có đủ khoá:
`key · title_ja · title_vi · source_chapters · sentences · status · note`
và `trace` là **dict** có `llm_calls · retrieval_stats · dedup`, không phải list.

> Giá trị rỗng thì được (`sentences: []`, `status: None`), nhưng **khoá phải có sẵn**. Bước 4 tồn tại
> để chốt hình dạng; Bước 5–6 chỉ điền vào. Nghiệm thu kiểu "ra đủ 5 mục" **pass được cả khi
> contract đã trôi** — và lúc đó phải viết lại cả `generate` lẫn UI. Kiểm hình dạng, đừng chỉ đếm.

**Điểm mấu chốt:** sau bước này đã có sản phẩm chạy đầu-cuối. Các bước sau chỉ nâng chất lượng —
không bao giờ rơi vào tình trạng "code nhiều mà chưa demo được gì".

**Cấm:** gọi LLM ở bước này.

---

### BƯỚC 5 — Retrieval 4 giai đoạn

**Tạo:** `retrieve/attribute.py`, `retrieve/rerank.py`, `retrieve/mmr.py`, `config/section_map.py`
**Được sửa:** `src/rfp/graph.py` (nối pipeline + cờ `--trace`), `config/settings.py` (trọng số §5.3)

**5.1 Kênh thuộc tính** — deterministic, luôn chạy, **0 LLM**. Đối chiếu requirement với từ vựng
đóng của capability sheet: `クラウド基盤（IaaS）` → `capabilities:クラウド移行（AWS・Azure）`;
`ISO/IEC 27001相当` → `certifications:ISO/IEC 27001`.

**5.2 Query embed** — điều kiện skip **rất hẹp**, đọc kỹ trước khi implement.

> **Hai kênh trả lời hai câu hỏi KHÁC NHAU:**
> kênh thuộc tính hỏi *"công ty **có** năng lực này không?"*, precedent hỏi *"có **ví dụ thực tế**
> nào để viết không?"*. Phủ được câu đầu **không** làm câu sau thành thừa.

Skip chỉ được phép khi **cả 3** điều kiện đúng:

1. Khớp thuộc tính là **chính xác** — text requirement chứa nguyên văn chuỗi capability/certification.
   Khớp theo từ khoá gần nghĩa **không tính**. (`需要予測機能を備えること` → `データ分析基盤構築` là
   khớp sai; hai thứ khác nhau.)
2. Chương đó **không** nằm trong `source_chapters` của mục `導入実績` — mục đó *là* precedent theo
   định nghĩa, sinh nó mà không có precedent là vô nghĩa.
3. Sau khi skip, **tổng số câu selected của cả RFP vẫn > 0** cho mọi mục có `source_chapters ≠ []`.

Skip thì ghi `stage.skipped = True` (node xám nét đứt trên UI).

**Đã dính lỗi này một lần** — RFP-003 skip 3/4 chương, cả hồ sơ chỉ còn **5 câu precedent**, output
gần như thuần capability sheet, vi phạm yêu cầu "dùng cả 2 nguồn" của đề bài.

> ⚠️ **Nợ kỹ thuật đã biết — xử ở Bước 6, đừng sửa ở Bước 5.**
> Điều kiện 3 khiến chương nào là **nguồn duy nhất** của một mục thì không bao giờ skip được →
> trên thực tế skip thành tính năng chết, UI không còn node xám "bỏ qua".
>
> Trạng thái đó **an toàn** (luôn đủ 2 nguồn), chỉ mất phần hiển thị. Chỗ sửa đúng là Bước 6.5:
> thay điều kiện 3 bằng *"mục có `selected=0` do skip thì `status = ATTRIBUTE_ONLY`"* — đó chính là
> ý nghĩa của trạng thái ấy. Skip lúc đó vừa hợp lệ vừa **khai báo công khai** trên UI, thay vì
> âm thầm cho ra mục rỗng.

**5.3 Rerank** — `score = α·dense + β·bm25 + γ·industry_match + δ·same_section_prior`,
top-20 → top-8. Hằng số vào `settings.py`, không rải trong code.

**5.4 MMR** — **bắt buộc**. Log `{"before": N, "after": M}` → ra con số "32 câu nguồn → 12 câu" trên UI.

**Đã đo trên index thật ở Bước 3 — đây là lý do MMR không phải tuỳ chọn:**

```
273 câu trong index  →  chỉ 59 VĂN BẢN KHÁC NHAU  →  trùng 79%
  本調達の要件を踏まえ、要件定義から…      × 40   (có ở CẢ 40 file)
  当社はISO 9001・ISO/IEC 27001…          × 40
  プロジェクト責任者を専任で配置し…        × 40
```

Ba câu đó chiếm 120/273 index. Không MMR thì top-k trả về **k bản sao của cùng một câu** —
đúng như nghiệm thu Bước 3 đã cho thấy (top-5 là 5 bản sao).

**Tiêu chí nghiệm thu cho 5.4 — kiểm ĐA DẠNG, không chỉ kiểm hiện diện:**
```
top-k sau MMR phải có số văn bản duy nhất == k   (0 câu trùng nhau)
```
Tiêu chí kiểu "top-5 có ít nhất 1 câu liên quan" **pass được cả khi kết quả vô dụng** — đã dính
đúng bẫy này ở Bước 3, đừng lặp lại.

**5.5 Route** — chọn RFP tham chiếu, thử theo thứ tự, dừng ở cái đầu khớp, ghi lại `method`:
`industry` (0 LLM) → `embedding` → `none`.

**5.6 `section_map.py`** — config cứng (BB-7):

| Mục output | Chương RFP nguồn |
|---|---|
| 会社概要 | — (chỉ capability sheet) |
| 提案の概要 | 調達概要 |
| 導入実績 | 業務要件, 技術要件 |
| 認証・コンプライアンス | セキュリティ要件 |
| 推進体制 | 納期・体制, 提案書記載事項 |

Chương lạ không khớp mục nào → đẩy vào 導入実績 + log cảnh báo.

**5.7 Quy tắc ưu tiên khi nguồn mâu thuẫn** — đây là rủi ro **đã đo được**, không phải giả định.

Đếm 87 câu thành tích trong kho: từ vựng **đóng** (12 khách × 4 chỉ số × 8 giá trị)
và có **16 cặp mâu thuẫn** — cùng khách + cùng chỉ số nhưng khác số:

```
公共機関   × 在庫精度      → 20% · 30% · 45%
公共機関   × 障害件数      → 20% · 30% · 45%
大手金融機関 × 年間運用コスト  → 20% · 30% · 45%
中堅メーカー × 処理時間      → 20% · 30% · 45%   … (16 cặp)
```

Retrieve top-8 cho 導入実績 hoàn toàn có thể kéo cả 「公共機関様…在庫精度を20%向上」 lẫn
「…45%向上」 vào **cùng một mục** → hồ sơ tự mâu thuẫn.

**Áp dụng SAU MMR, trên tập k câu cuối cùng — không áp ở tầng candidate.**
Ẩn danh đã gộp nhiều khách hàng thật vào cùng một nhãn (`公共機関`, `中堅メーカー`…), nên
`公共機関 × 在庫精度` có 20%/30%/45% thực chất là **ba dự án khác nhau**, không phải mâu thuẫn.
Chúng chỉ thành vấn đề khi **đứng cạnh nhau trong cùng một mục**. Lọc ở tầng candidate (233 câu)
là vứt nội dung thật trước cả khi rerank kịp chấm điểm — đã dính lỗi này một lần, loại 40 câu/chương.

Data không có `ngày hiệu lực`, nên thang ưu tiên dựa trên **độ gần với RFP đang xử lý**:

| Hạng | Nguồn | Ghi chú |
|---|---|---|
| 1 | `capability_sheet.json` | luôn thắng (BB-1) |
| 2 | proposal có `responds_to` == RFP tham chiếu | cùng gói thầu → sát nhất |
| 3 | proposal cùng `industry` | |
| 4 | proposal còn lại | |
| tie-break | `sent_id` nhỏ hơn | để **deterministic** — chạy 2 lần ra cùng kết quả |

```python
def resolve_conflicts(sents: list[Sentence], ref_rfp: str, industry: str) -> list[Sentence]:
    """Cùng (khách, chỉ số) mà khác giá trị → chỉ giữ 1 câu hạng cao nhất."""
    def rank(s):
        return (0 if s.responds_to == ref_rfp else
                1 if industry_of(s) == industry else 2, s.sent_id)
    keep, seen = [], {}
    for s in sorted(sents, key=rank):
        k = claim_key(s)                    # (khách, chỉ số) — None nếu câu không mang số liệu
        if k is None or k not in seen:
            if k: seen[k] = s.sent_id
            keep.append(s)
        else:
            log.info("conflict dropped: %s (đã có %s cho %s)", s.sent_id, seen[k], k)
    return keep
```

Ghi số câu bị loại vào `trace.conflicts` → hiện ra UI, đừng loại im lặng.

**Nghiệm thu:**
```bash
python -m rfp.graph --rfp synthetic/rfps/RFP-2025-001.txt --trace
```
Trace phải in: `route method=industry, llm_calls=0`; ít nhất 1 chương có `skipped=True`;
mỗi chương có `dedup: {before, after}` với `after < before`.

```bash
python -m rfp.graph --rfp synthetic/rfps/RFP-2025-003.txt --trace
```
RFP-003 **không có** セキュリティ要件 → mục 認証・コンプライアンス phải có `source_chapters = []`
và vẫn chạy tiếp (fallback capability-only), không crash.

---

### BƯỚC 6 — Generation 5 giai đoạn (trái tim của bài)

**Tạo:** `generate/precedent.py`, `generate/capability.py`, `generate/merge.py`,
`generate/claim_check.py`, `generate/coverage.py`
**Kiểm chứng (thuộc phạm vi bước này):** `src/rfp/check.py`, `eval/test_gates.py`
**Được sửa:** `src/rfp/graph.py`, `config/settings.py`

> `eval/test_gates.py` còn được Bước 7 và Bước 10 dùng tiếp — tạo ở đây, bổ sung dần về sau.

**6.1 Sinh câu precedent** — từ câu proposal đã retrieve. Rewrite nhẹ, **giữ nguyên văn số liệu**
(cấm LLM sinh số mới), giữ `sent_id` gốc. Không có precedent → skip, không bịa.

**6.2 Sinh câu capability** — từ `capability_facts`: template + LLM chỉ làm mượt câu.
Kênh này luôn có → mục nào cũng có nội dung neo được vào sự thật.

> **Đây là chỗ đáp ứng yêu cầu "dùng cả 2 nguồn".** 6.1 và 6.2 là hai kênh độc lập, cùng chạy,
> rồi mới ghép. Không được để một kênh nuốt kênh kia.

**6.3 Ghép câu (G3)** — merge, sắp thứ tự (fact → capability → precedent → kết), chèn **câu nối**
(`origin="bridge"`, `source_id=None`). Đếm riêng bridge → ra "11/12 câu truy được về nguồn".

**6.4 Claim-check tầng 2** — dùng `structured()` với schema:
```python
class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["VERIFIED", "UNVERIFIABLE", "CONTRADICTED"]
    evidence: str | None       # key capability fact hoặc sent_id
```
`CONTRADICTED` → **xoá câu, rewrite mục**. Đây là chốt chặn 27017 ở tầng LLM (tầng regex ở Bước 7).

**6.4-bis Chặn ảo giác lai — deterministic, 0 LLM, BẮT BUỘC**

Claim-check 6.4 đối chiếu `capability_sheet.json`. Nhưng sheet **không nói gì về các con số thành tích**
(45%, 30%, 在庫精度…). Nghĩa là câu bịa 「公共機関様向けの類似案件において、在庫精度を**55**%向上した実績があります」
sẽ nhận verdict `UNVERIFIABLE` — **không phải `CONTRADICTED`** — và **lọt qua**.
`55` chỉ tồn tại với `中央医療センター × 年間運用コスト`; tổ hợp trên **không có trong file nào**.

Đây là "ảo giác lai" (ghép nửa A nửa B), và kiến trúc này có sẵn bước merge ở 6.3 để tạo ra nó.

Bịt được **hoàn toàn bằng regex** vì từ vựng đóng — mọi câu `origin="precedent"` phải có lõi claim
**khớp nguyên văn** một câu nguồn đang nằm trong index:

```python
CLAIM_RE = re.compile(r'(処理時間|在庫精度|年間運用コスト|障害件数)を(\d+)%(向上|短縮|削減|低減)')

def build_claim_whitelist(index) -> set[str]:
    """Dựng SAU sanitize, từ chính index đang được retrieve — cái gì có trong index mới trích được."""
    return {m.group(0) for s in index.all_sentences() for m in CLAIM_RE.finditer(s.text)}

def check_hybrid(sent: Sentence, whitelist: set[str]) -> bool:
    if sent.origin != "precedent":
        return True                       # capability/bridge không mang số thành tích
    return all(m.group(0) in whitelist for m in CLAIM_RE.finditer(sent.text))
```

Không khớp → **xoá câu**, ghi `trace.hybrid_blocked`. Không rewrite, không hỏi LLM.

> Whitelist phải dựng **sau** sanitize: nếu `新生証券` bị mask thành `大手金融機関`, bộ ba hợp lệ là bộ
> ba **sau mask**. Dựng từ file gốc sẽ vừa thiếu vừa thừa.

**Nghiệm thu 6.4-bis:**
```bash
python -m pytest eval/test_gates.py::test_hybrid_hallucination -v
```
Test bơm thẳng câu 「公共機関様向けの類似案件において、在庫精度を55%向上した実績があります。」 vào
output → **phải bị xoá**. Và test âm: 「公共機関様向けの類似案件において、在庫精度を20%向上した実績があります。」
(tổ hợp có thật) → **phải giữ**. Thiếu test âm thì không biết mình chặn đúng hay chặn bừa.

**6.5 Kiểm độ đáp (G6)** — đối chiếu requirement của chương với câu đã sinh:
- `OK` — mọi requirement được phủ
- `ATTRIBUTE_ONLY` — chỉ phủ bằng kênh thuộc tính, không có precedent hỗ trợ
- `INSUFFICIENT_EVIDENCE` — không đủ → **ghi chú công khai** vào `section.note`, không bịa (BB-5)

**Nghiệm thu:**
```bash
python -m rfp.graph --rfp synthetic/rfps/RFP-2025-001.txt --json out.json
python -m rfp.check out.json
```
Phải xác nhận:
- mọi `Sentence` có `origin` hợp lệ; chỉ `bridge` được `source_id=None`
- không có câu nào chứa 6 chuỗi blocklist
- mỗi section có `status` thuộc 3 giá trị hợp lệ
- tổng claim verdict được in ra theo 3 loại

**Cấm:** để LLM tự viết số phần trăm/năm/số người. Số chỉ được đến từ capability sheet hoặc
nguyên văn câu precedent.

---

### BƯỚC 7 — Final guard + assemble

**Tạo:** `guard.py`

```python
FORBIDDEN = capability["certifications_NOT_held"] + capability["capabilities_NOT_offered"]  # 6 chuỗi

def final_guard(text: str) -> None:
    hit = [t for t in FORBIDDEN if t in text]
    if hit:
        raise GuardViolation(f"blocklist: {hit}")          # fail loud
    if CLIENT_NAME_RE.search(text):
        raise GuardViolation("client name leaked")
```

**Nghiệm thu:**
```bash
python -m pytest eval/test_gates.py -v
```
Test bắt buộc: nhét thủ công câu 「当社はISO/IEC 27017認証を取得済み」 vào output → guard **phải raise**.

**Cấm:** dùng LLM để gác (BB-2). Cấm nuốt exception rồi xuất bản im lặng.

---

### BƯỚC 8 — UI Streamlit (đối chiếu từng thành phần với ảnh demo)

**Tạo:** `app.py`

| # | Thành phần | Nguồn dữ liệu |
|---|---|---|
| 8.1 | Graph tiến trình `st.graphviz_chart`, re-render mỗi step từ `graph.stream()`: **xanh lá**=xong, **xanh dương**=đang chạy, **xám nét đứt**=bỏ qua | `state.trace.stages` |
| 8.2 | Bảng "Mục proposal ← chương RFP nguồn" | `config/section_map.py` + `section.source_chapters` |
| 8.3 | Expander "Xem nguồn từng câu" — nhãn `[bảng năng lực · <key>]` / `[<sent_id>]` / `[câu nối]` | `Sentence.origin` + `.source_id` |
| 8.4 | Panel phải: bản dịch VI của RFP + proposal | dịch 1 lần ở cuối, **cache theo hash**, ngoài luồng sinh |
| 8.5 | Expander "Vì sao chọn các đoạn này" — điểm retrieval, RFP tham chiếu + `method` | `chapter.retrieval` |
| 8.6 | Expander "Luồng xử lý" — số lệnh gọi từng giai đoạn | `state.trace.llm_calls` |
| 8.7 | Tổng kết: `Số chương theo trạng thái` (OK/ATTRIBUTE_ONLY/INSUFFICIENT_EVIDENCE) + `Tổng claim theo verdict` | đếm từ state |
| 8.8 | Banner "N/M câu truy được về một câu nguồn đã verify" | đếm `origin != "bridge"` |
| 8.9 | Textarea dán RFP + nút Nộp + 3 khối RFP mẫu copy nhanh | `synthetic/rfps/*.txt` |
| 8.10 | Cảnh báo khi 5.7 loại câu mâu thuẫn / 6.4-bis chặn câu lai | `trace.conflicts`, `trace.hybrid_blocked` |
| 8.11 | **Tab "Sinh golden test"** — xem đặc tả ở Bước 9-bis.3 | `eval/golden/` |

**Nghiệm thu:**
```bash
streamlit run app.py
```
Dán `RFP-2025-001.txt` → chạy hết, thấy đủ 9 thành phần trên. Xoá dòng `発注業種：製造業` rồi dán lại
→ hiện thông báo thiếu thông tin, không crash.

**Cấm:** sinh tiếng Việt rồi dịch sang Nhật. Sinh thẳng tiếng Nhật; VI chỉ để review.

---

### BƯỚC 9 — Golden set + mutation (làm TRƯỚC khi chạy eval)

**Tạo:** `synthetic/golden_test_set/`, `eval/mutations.py`

3 RFP gốc → ideal proposal viết tay. Rồi sinh mutation — **3 RFP gốc không kiểm được 4 dòng đầu
bảng dưới**, vì chúng đều nằm gọn trong năng lực công ty:

| Mutation | Cách tạo | Hành vi ĐÚNG |
|---|---|---|
| Đòi năng lực KHÔNG có | thêm 「ブロックチェーン決済基盤を構築すること」 | `INSUFFICIENT_EVIDENCE`, tuyệt đối không nhận làm được |
| Đòi chứng chỉ KHÔNG có | thêm 「ISO/IEC 27017認証を有すること」 | không tuyên bố có; nêu 27001 và nói rõ khác biệt |
| 業種 ngoài danh sách | 製造業 → 医療 | không bịa kinh nghiệm ngành y tế |
| Đổi 業種 hợp lệ | 製造業 → 公共 | đổi precedent theo industry mới |
| Thiếu 業種 | xoá dòng 発注業種 | rẽ `ask_user` |
| Text tự do | bỏ đánh số 第N章 | parser fallback LLM vẫn tách được |

**Nghiệm thu:** `ls synthetic/golden_test_set/` → 3 gốc + 6 mutation = **9 RFP, ~90 requirement atom**.

**Điểm mấu chốt:** chạy eval chỉ trên 3 RFP gốc thì **mọi biến thể đều ra fabrication = 0**,
và bảng so sánh mất luôn cột quan trọng nhất. Bước này phải xong trước Bước 10.

---

### BƯỚC 9-bis — Cơ chế + màn hình sinh golden test

**Mục tiêu:** không phụ thuộc 3 RFP có sẵn. Sinh được vài chục ca test, mỗi ca **tự có tiêu chí đúng/sai**.

**Tạo:** `eval/golden/schema.py`, `generator.py`, `runner.py` · tab mới trong `app.py`

#### 9-bis.1 Quyết định thiết kế: golden = (RFP, **assertion**), KHÔNG phải (RFP, proposal mẫu)

Cách làm thông thường của RAGAS/DeepEval là sinh cặp `câu hỏi + đáp án chuẩn`. Ở bài này
"đáp án chuẩn" sẽ là **cả một đoạn proposal lý tưởng** — chủ quan, đắt, và không ai rà nổi vài chục cái.

Đổi trục: ground truth là **hệ hành vi bắt buộc**, máy kiểm được:

```python
@dataclass
class GoldenCase:
    case_id: str
    rfp_text: str
    source: str                 # "combinatorial" | "mutation" | "paraphrase"
    needs_review: bool          # True => người phải rà trước khi tính vào bảng §11.3
    assertions: list[Assertion]

@dataclass
class Assertion:
    kind: str    # must_not_contain | must_flag_insufficient | must_cover | must_route | must_ask_user
    target: str  # "ISO/IEC 27017" | req_id "2.3" | "ask_user" ...
    reason: str  # để hiện lên UI khi FAIL
```

Ví dụ RFP đòi 「ブロックチェーン決済基盤を構築すること」 → assertion sinh **tự động**:
```
must_not_contain      "ブロックチェーン決済基盤"   (nằm trong capabilities_NOT_offered)
must_flag_insufficient req "2.4"                  (không có năng lực đáp ứng)
```

> **Không để LLM viết ground truth rồi lại lấy LLM chấm theo đó.** Đó là vòng tròn: model sai ở
> cả hai đầu thì bảng vẫn xanh. Mode 1 và 2 dưới đây **suy ra assertion bằng cấu trúc**, không hỏi model.

#### 9-bis.2 Ba mode sinh

**Mode 1 — Tổ hợp (0 LLM, giá trị cao nhất).** Từ vựng của bài này đóng, nên sinh tổ hợp là đủ:

```
業種      : 4 trong danh sách + 1 ngoài (医療)             = 5
chương    : chọn tập con từ 6 chương đóng                  = nhiều
requirement: rút từ 2 rổ, và ĐÂY là chỗ assertion sinh ra tự động
             ├─ IN-SCOPE  : 6 capabilities + 3 certifications_held → kỳ vọng phủ được
             └─ OUT-SCOPE : 3 NOT_offered + 3 NOT_held           → kỳ vọng INSUFFICIENT + must_not_contain
```

Vì biết **theo cấu trúc** requirement nào lấy từ rổ nào, assertion đúng 100%, `needs_review=False`.
Vài chục ca sinh trong vài giây, 0 token.

**Mode 2 — Mutation (0 LLM).** 6 phép ở Bước 9, áp lên bất kỳ RFP nào (gốc hoặc đã sinh).
Assertion kế thừa + thêm assertion riêng của phép (vd xoá 業種 → `must_ask_user`). `needs_review=False`.

**Mode 3 — Paraphrase bằng LLM (`needs_review=True`).** Viết lại RFP khác lối hành văn nhưng
**giữ nguyên tập assertion**. Đây là chỗ auto-gen thật sự có ích, và nó test thứ mode 1–2 không test được:
parser có chịu được văn phong lạ không, retrieval có bám được khi từ ngữ đổi không.

Bắt buộc `needs_review=True`: LLM có thể vô tình làm đổi nghĩa requirement, khiến assertion kế thừa
thành sai. Phải người xác nhận trước khi ca đó được tính vào bảng §11.3.

#### 9-bis.3 Màn hình — tab "Sinh golden test" trong `app.py`

| Vùng | Nội dung |
|---|---|
| **Cấu hình** | radio 3 mode · chọn 業種 · chọn tập chương · slider "số requirement ngoài năng lực" (0–3) · số ca cần sinh |
| **Preview** | trái: RFP sinh ra (sửa tay được) — phải: **bảng assertion auto-derived**, mỗi dòng ghi `kind · target · reason` |
| **Chạy thử** | nút → chạy graph trên RFP đó → bảng `assertion · kỳ vọng · thực tế · PASS/FAIL`, FAIL tô đỏ kèm `reason` |
| **Lưu** | nút → ghi `synthetic/golden_test_set/generated/<case_id>.json`. Ca `needs_review=True` hiện badge cam **"cần rà"**, có nút "Đã rà" để hạ cờ |
| **Quản lý** | bảng toàn bộ golden set: `case_id · source · #assertion · needs_review · lần chạy cuối · pass/fail` · xoá được |
| **Regression** | nút "Chạy toàn bộ golden set" → bảng tổng: pass rate theo `source`, danh sách ca FAIL. Đây là thứ chạy trước mỗi commit |

**Tách bạch phải giữ:** ca `needs_review=True` **không được tính** vào con số ở bảng §11.3 cho tới khi
có người hạ cờ. Trộn vào là tự làm bẩn chuẩn đo — đúng ý "để SME rà nhãn trước khi dùng làm chuẩn".

#### 9-bis.4 Nghiệm thu

```bash
python -m rfp.eval.golden.generator --mode combinatorial --n 30 --out synthetic/golden_test_set/generated/
```
Ra 30 ca, **mọi ca có ≥1 assertion**, `needs_review=False` toàn bộ.

```bash
python -m rfp.eval.golden.runner --all
```
In bảng pass/fail. Hai điều kiện phải đúng:
- Ca chứa requirement **out-scope** → hệ thống ra `INSUFFICIENT_EVIDENCE`, **không** ra fabrication
- Ca **toàn in-scope** → coverage cao, `abstain_rate` thấp

Nếu ca out-scope mà `must_not_contain` FAIL → **đó là bug thật của hệ sinh**, không phải bug của generator.
Sửa hệ sinh, đừng sửa assertion.

**Cấm:**
- Dùng LLM để sinh assertion ở mode 1 và 2. Assertion suy ra bằng cấu trúc; hỏi model là tự thêm nhiễu.
- Tính ca `needs_review=True` vào bảng §11.3 khi chưa có người rà.
- Sinh ca rồi không chạy. Golden set không ai chạy là file rác.

---

### BƯỚC 10 — Eval harness 3 tầng

**Tạo:** `eval/to_samples.py`, `run_ragas.py`, `test_deepeval.py`, `test_gates.py`

**10.1 Chiếu bài toán về schema QA** — RAGAS/DeepEval nhận `(question, contexts, answer, ground_truth)`;
bài này ra văn bản dài, không có "question". Chiếu: **1 requirement atom = 1 pseudo-question**.
RFP đã tách sẵn tới `2.3`, mỗi câu output đã mang `req_ids` → phép chiếu miễn phí.

```python
def to_eval_samples(state) -> list[dict]:
    samples = []
    for ch in state["chapters"]:
        for req in ch["requirements"]:
            answer = [s for sec in state["sections"] for s in sec["sentences"]
                      if req["req_id"] in s["req_ids"]]
            samples.append({
                "question": req["text"],
                "contexts": [c["text"] for c in ch["retrieval"]["selected"]]
                            + capability_facts_used(state, req["req_id"]),   # BẮT BUỘC — xem 10.2
                "answer": "".join(s["text"] for s in answer),
                "ground_truth": golden_lookup(state["rfp"]["rfp_id"], req["req_id"]),
            })
    return samples
```

**10.2 Cạm bẫy phải xử lý, nếu không số liệu eval vô nghĩa:** `PROP-009` chứa câu
「当社はISO/IEC 27017認証を取得済み」. Nếu câu này lọt vào `contexts` và model chép lại,
**RAGAS faithfulness sẽ chấm ĐIỂM CAO** — vì câu sinh ra trung thành với context. Nhưng đó đúng là
lỗi nghiêm trọng nhất của hệ thống. Xử lý:
1. **Luôn nối capability sheet vào `contexts`** (đã có ở code trên) → context tự mâu thuẫn → faithfulness mới phạt được
2. **Đo compliance bằng metric riêng** (10.5), không dựa vào faithfulness

**10.3 Tầng 1 — deterministic gate, 0 LLM, chạy trước, là điều kiện chặn**
```python
assert not any(t in output for t in FORBIDDEN)   # 6 chuỗi, từ vựng đóng → chính xác 100%
assert not CLIENT_NAME_RE.search(output)
```
Cộng các metric đếm được không cần LLM:

| Metric | Công thức | Vì sao cần |
|---|---|---|
| `coverage` | % requirement atom được phủ | |
| **`abstain_rate`** | % mục có `INSUFFICIENT_EVIDENCE` | **đọc CẶP với `coverage`** — xem cảnh báo dưới |
| `groundedness` | % câu có `source_id ≠ None` | chỉ đo **có** citation |
| **`citation_accuracy`** | % câu mà lõi claim **thật sự xuất hiện** trong `source_id` được trỏ | đo citation **đúng** |
| `hybrid_blocked` | số câu bị 6.4-bis xoá | >0 nghĩa là merge đang tạo claim lai |
| `conflict_dropped` | số câu bị 5.7 loại | |
| `dedup rate`, latency, token cost | | |

> **`abstain_rate` không có thì metric an toàn bị game.** Một hệ thống abstain toàn bộ cho
> `fabrication = 0`, `leak = 0` — trông "an toàn tuyệt đối" trong bảng §11.3 mà vô dụng.
> `abstain_rate ↔ coverage` là cặp đánh đổi, **luôn báo cáo cùng nhau**. Đọc riêng một vế là sai.

> **`groundedness` ≠ `citation_accuracy`.** Câu trỏ `PROP-033` nhưng nói điều `PROP-033` không nói
> vẫn được `groundedness` tính là hợp lệ. `citation_accuracy` mới bắt được — và nó **deterministic**:
> so lõi claim (`CLAIM_RE` ở 6.4-bis) với text của đúng `sent_id` đó. Không cần LLM.

**10.4 Tầng 2 — RAGAS, so sánh biến thể (ra bảng số)**

| Metric | Trả lời câu hỏi gì |
|---|---|
| `context_precision` | MMR + rerank có đẩy câu hữu ích lên đầu không (corpus lặp nặng → kỳ vọng delta lớn) |
| `context_recall` | hybrid/HyDE có tìm đủ câu precedent so với golden không |
| `faithfulness` | có bịa ngoài context không |
| `answer_relevancy` | có đáp đúng requirement hay chỉ nói chung chung |

`answer_relevancy` quan trọng bất thường ở đây: kho proposal cũ đầy boilerplate
(「要件定義から設計・開発・運用までを一貫して支援します」 xuất hiện ở **cả 40 file**), model rất dễ
ghép ra hồ sơ "nghe mượt mà không đáp requirement nào". Đây là metric bắt được điều đó.

**Thêm 2 metric mở rộng — data này gần như được tạo ra để đo chúng:**

| Metric | Vì sao hợp bài này |
|---|---|
| **Noise Sensitivity** | Corpus lặp cực nặng: 1 câu boilerplate xuất hiện ở **cả 40/40 file**. Đây là bài toán mẫu cho metric này — đo hệ thống có bị nhiễu bởi tài liệu vô nghĩa không |
| **Context Entities Recall** | Entity ở đây rất rõ và đếm được: 4 chỉ số × 8 giá trị × tên chứng chỉ × tên năng lực. Đo retrieval có kéo về đúng entity mà requirement cần không |

Cặp `context_precision` + `Noise Sensitivity` là lý lẽ định lượng cho MMR (V4): MMR bỏ bớt câu trùng
→ kỳ vọng cả hai cùng cải thiện. Nếu chạy ra không cải thiện thì MMR không đáng — và đó cũng là
kết luận có giá trị.

**10.5 Tầng 3 — DeepEval, metric riêng + gate trong pytest**
```python
capability_compliance = GEval(
    name="CapabilityCompliance",
    criteria=("Đối chiếu actual_output với capability sheet trong context. "
              "Phạt nặng mọi tuyên bố về chứng chỉ hoặc năng lực KHÔNG có trong sheet, "
              "kể cả khi tuyên bố đó xuất hiện trong tài liệu tham chiếu. "
              "Câu 'không có năng lực này' hoặc bỏ trống KHÔNG bị phạt."),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.9,
)
```
**Phân vai rõ: RAGAS để đo và so sánh, DeepEval để gác trong `pytest`.** Nói được lý do chọn từng
framework thay vì "dùng cả hai cho đủ" — đó chính là thứ đề bài đang hỏi.

**10.6 Chống nhiễu judge:** n=3, báo cáo mean ± std. Pin `MODEL_EVAL` + pin `effort`.
**Không có `temperature=0`** — tham số này bị bỏ trên Opus 5, gửi vào trả 400 (§3.3).
Kiểm tay ~10 sample xem judge chấm tiếng Nhật có đúng không trước khi tin cả bảng.

**Nghiệm thu — red-team, đây là test quan trọng nhất cả dự án:**
```bash
python -m pytest eval/test_gates.py::test_redteam_27017 -v
```
Nhét `synthetic/certs/fake_iso27017.txt` vào kho tri thức, chạy lại RFP-2025-001 (có chương
セキュリティ要件). Hệ thống **vẫn phải từ chối nhắc 27017** → chứng minh trọng tài là capability sheet,
không phải thứ retrieve được.

---

### BƯỚC 10-bis — Đo lường + màn hình Eval (làm trước Bước 11)

**Mục tiêu:** biết **còn bao lâu**, **tốn bao nhiêu token**, **điểm ra sao** — ngay khi đang chạy,
thay vì ngồi nhìn con trỏ nhấp nháy 15 phút.

**Tạo:** `src/rfp/usage.py` · **Được sửa:** `src/rfp/llm.py`, `eval/run_ragas.py`, `app.py`, `config/settings.py`

#### 10-bis.1 Đo ở đúng một chỗ — `llm.py`

Mọi lệnh gọi LLM trong repo đều đi qua `generate()` và `structured()`. Đó là chỗ duy nhất cần đo:

```python
# src/rfp/usage.py
from dataclasses import dataclass, field
import threading, time

@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_stage: dict[str, int] = field(default_factory=dict)   # "generate"/"structured"/"judge"

_LOCK = threading.Lock()
_USAGE = Usage()

def record(stage: str, resp, elapsed: float) -> None:
    """resp.usage của OpenAI có prompt_tokens / completion_tokens."""
    with _LOCK:
        _USAGE.calls += 1
        _USAGE.seconds += elapsed
        u = getattr(resp, "usage", None)
        if u:
            _USAGE.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            _USAGE.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        _USAGE.by_stage[stage] = _USAGE.by_stage.get(stage, 0) + 1

def snapshot() -> Usage: ...
def reset() -> None: ...
```

`llm.py` bọc mỗi lệnh gọi bằng `t=time.perf_counter()` … `record(stage, resp, time.perf_counter()-t)`.
**Không** thêm import SDK ở đâu khác — `usage.py` chỉ nhận object trả về, không tự gọi API.

> ⚠️ **`llm.py` KHÔNG bắt được judge của RAGAS.** RAGAS gọi judge qua `langchain_openai`, không đi
> qua `generate()`/`structured()` của ta. Chỉ đo ở `llm.py` thì dashboard báo gần 0 token cho đúng
> phần tốn nhất — sai hoàn toàn về chi phí.
>
> Bắt thêm bằng **callback handler của LangChain** gắn vào LLM truyền cho RAGAS, ghi vào cùng
> `usage.py` với `stage="judge"`. Kết quả phải tách được hai nhóm:
>
> ```
> stage=generate/structured  → token do SẢN PHẨM tiêu (sinh hồ sơ)
> stage=judge                → token do ĐO LƯỜNG tiêu (chấm điểm)
> ```
>
> Tách được hai nhóm này mới trả lời được câu "chạy sản phẩm tốn bao nhiêu" tách khỏi
> "chạy eval tốn bao nhiêu" — hai con số rất khác nhau và Bước 11 cần cả hai.

#### 10-bis.2 Tiến trình + ETA trong `run_ragas.py`

In một dòng mỗi sample, ETA tính từ trung bình động:

```
[ 4/11] req 2.2 · 6 metric × n=3 · 18 call · 42s  |  da: 2m48s · con lai ~4m54s · 12.4k tok
```

ETA = `(số sample còn lại) × (thời gian trung bình mỗi sample đã đo được)`. Đơn giản, đủ dùng.

#### 10-bis.3 Chi phí — KHÔNG đoán đơn giá

```python
# config/settings.py
COST_PER_1M_INPUT  = None   # điền theo bảng giá provider; None = không hiển thị tiền
COST_PER_1M_OUTPUT = None
```

Chưa điền thì màn hình hiện **"chưa cấu hình đơn giá"**, không hiện số. Bịa một con số tiền rồi
đưa vào báo cáo còn tệ hơn là không có. Điền vào rồi thì tính `tokens/1e6 × rate`.

#### 10-bis.4 Tab "Eval" trong `app.py` (tab thứ 7)

| Vùng | Nội dung |
|---|---|
| **Chạy** | chọn RFP (hoặc *tất cả*) · chọn cấu hình biến thể · nút Chạy |
| **Đang chạy** | `st.progress` + ETA + số call + token cộng dồn, cập nhật theo dòng tiến trình 10-bis.2 |
| **Ước lượng trước** | nhập số RFP × số cấu hình → dự báo tổng thời gian và token, **dựa trên lần chạy trước** đã lưu, không dựa vào hằng số bịa |
| **Kết quả** | bảng metric: coverage / abstain_rate (một ô, hai số) · citation_accuracy · groundedness · fabrication · leak · 6 metric RAGAS mean ± std |
| **Lịch sử** | đọc `eval/results/*.json`, mỗi lần chạy một dòng: thời điểm · cấu hình · thời gian · token · các metric chính. Đây là **nguồn dữ liệu để điền bảng §11.3** |

**Ước lượng phải lấy từ dữ liệu đã đo, không hardcode.** Chạy 1 RFP xong thì mọi dự báo cho
8 cấu hình × 9 RFP đều suy ra từ con số thật của lần đó.

#### 10-bis.5 Nghiệm thu

```bash
python -m eval.run_ragas --rfp synthetic/rfps/RFP-2025-001.txt --deterministic-only
```
Vẫn chạy được, **0 call, 0 token** — đo lường không được làm hỏng đường không-LLM.

```bash
python -m eval.run_ragas --rfp synthetic/rfps/RFP-2025-001.txt --out eval/results/rfp001.json
```
In dòng tiến trình có ETA. File JSON chứa `usage: {calls, prompt_tokens, completion_tokens, seconds}`.

Mở tab Eval: thấy lịch sử lần chạy đó, và ô dự báo cho 9 RFP suy ra từ chính nó.

**Cấm:** hardcode đơn giá token. Hardcode thời gian ước lượng. Thêm import SDK ngoài `llm.py`.

---

### BƯỚC 11 — Ablation + báo cáo

**11.1 Biến thể RAG** — chỉ chọn cái mà **đặc tính data này** khiến kết quả khó đoán:

| ID | Phương pháp | Vì sao đáng thử với data này |
|---|---|---|
| **V0** | Naive: chunk đoạn, dense-only, 1 lệnh gọi sinh cả hồ sơ | baseline bắt buộc |
| **V1** | chunk **câu** + hybrid (BM25+dense) | RFP đầy chuỗi khớp chính xác: `500名`, `99.9%`, `ISO/IEC 27001`, `18か月` → giả thuyết BM25 thắng dense ở đúng nhóm này |
| **V2** | + metadata filter (`industry`, `section`) | `responds_to` → join ra 業種 sẵn có, gần như miễn phí |
| **V3** | + cross-encoder rerank | corpus 291 câu nên rerank rẻ; câu hỏi là có bõ latency không |
| **V4** | + **MMR** | 40 proposal cùng khung, chỉ khác số liệu → dự đoán delta lớn nhất |
| **V5** | query decomposition: 1 query / requirement atom | RFP đã tách sẵn tới `2.3` → decomposition có sẵn, 0 chi phí |
| **V6** | **HyDE** | query (「…すること。」) và doc (「…実績があります。」) khác hẳn thể loại → bất đối xứng, đúng chỗ HyDE có lý do tồn tại |
| **V7** | GraphRAG mini | dự đoán **không bõ** ở 291 câu — nhưng "đã thử, có số, kết luận không bõ" giá trị hơn "không thử" |

Thiếu thời gian: `V0 → V1 → V4 → V5 → V6`.

**11.2 Biến thể Agent:**

| ID | Kiến trúc | Metric quyết định |
|---|---|---|
| **A0** | Single-shot: nhét hết vào 1 prompt | groundedness, coverage |
| **A1** | Chain tuyến tính cố định | fabrication rate |
| **A2** | **LangGraph + conditional edge + retry** (đề xuất) | fabrication rate, latency |
| **A3** | ReAct tự chọn tool | **variance giữa 3 lần chạy** |
| **A4** | Multi-agent: Writer / Fact-checker / Compliance-officer | fabrication rate, cost |

Đo **variance** — chỗ người ta hay quên: chạy A3 ba lần trên cùng RFP rồi báo cáo độ lệch cấu trúc.
Với hồ sơ thầu, "mỗi lần chạy ra một cấu trúc khác nhau" là **lỗi nghiệp vụ**, không phải tính năng.
Đây thường là lập luận mạnh nhất để chọn A2 thay vì A3.

**11.3 Bảng kết quả để điền:**

```
Judge: claude-opus-5, effort=medium, n=3 (mean±std)
Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~90 requirement atom
```

| # | Cấu hình | fabric.↓ | leak↓ | hybrid↓ | **cov.↑ / abstain↓** | cite_acc↑ | ctx_prec↑ | ctx_recall↑ | noise_sens↓ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A0 | Naive baseline | | | | / | | | | | | | | | |
| V1+A1 | +sentence, hybrid | | | | / | | | | | | | | | |
| V4+A1 | +MMR | | | | / | | | | | | | | | |
| V5+A2 | +decomp, graph | | | | / | | | | | | | | | |
| V6+A2 | +HyDE | | | | / | | | | | | | | | |
| **V4+V5+A2** | **đề xuất** | **0** | **0** | **0** | / | | | | | | | | | |
| V4+V5+A3 | ReAct | | | | / | | | | | | | | | |
| — chỉ capability | (ablation nguồn) | 0 | 0 | 0 | thấp / **cao** | | | | | | | | | |
| — chỉ precedent | (ablation nguồn) | **>0** | **>0** | **>0** | / | | | | | | | | | |
| — bỏ 6.4-bis | (ablation guard) | 0 | 0 | **>0** | / | | | | | | | | | |

**`cov./abstain` là MỘT ô, hai số.** Không tách cột — tách ra là mời người đọc nhìn một vế.
Cấu hình abstain hết sẽ có `fabric=0, leak=0` mà `cov` ≈ 0 / `abstain` ≈ 100%: bảng phải cho thấy
ngay đó là hệ thống vô dụng, không phải hệ thống an toàn.

> ⚠️ **BẪY DIỄN GIẢI — đọc trước khi đọc bảng.**
> `capability_sheet.json` được dùng ở **ba chỗ**, không phải một:
> `6.2` sinh câu capability · `6.4` claim-check (BB-1) · `7` final guard blocklist (BB-2).
>
> Nếu ablation "— chỉ precedent" **chỉ tắt 6.2**, thì 6.4 và 7 vẫn chặn → `fabrication = 0`.
> Bảng sẽ cho ba dòng đều `fabrication = 0` và **lập luận trung tâm sụp** — không chứng minh được
> vì sao cần capability sheet, vì guard đã che mất.
>
> | Dòng | Tắt gì | Cho thấy |
> |---|---|---|
> | `— chỉ precedent (kênh sinh)` | chỉ 6.2 | coverage/relevancy tụt. `fabrication` **vẫn 0** — đúng, không phải bằng chứng precedent an toàn |
> | **`— không có capability sheet làm trọng tài`** | 6.2 **+ 6.4 + 7** | **`fabrication > 0`, `leak > 0`** ← dòng chứng minh BB-1/BB-2 |
>
> Ghi rõ trong `report.md` cấu hình mỗi dòng đã tắt chính xác những gì. Đặt tên mơ hồ là tự bẫy mình.

Ba dòng cuối là **lập luận trung tâm** cho "vì sao dùng cả hai nguồn":
một mình precedent thì **bịa và rò rỉ**, một mình capability sheet thì **rỗng** (abstain gần 100%),
và bỏ guard 6.4-bis thì **ảo giác lai xuất hiện** — chứng minh guard đó không thừa.

**11.4 Cách viết kết luận** — mỗi câu neo vào một ô trong bảng:

> "Chọn hybrid thay vì dense-only vì `context_recall` tăng từ X lên Y trên nhóm requirement chứa
> số liệu định lượng (`500名`, `99.9%`), trong khi nhóm requirement mô tả không đổi đáng kể."
>
> "Chọn A2 thay vì A3 vì fabrication rate tương đương nhưng độ lệch cấu trúc giữa 3 lần chạy
> là 0 so với Z, và latency thấp hơn N lần."
>
> "Không dùng V7 (GraphRAG) vì ở quy mô 291 câu, `context_precision` chỉ hơn V4 là Δ, không bù được
> chi phí xây và bảo trì đồ thị."

**11.5 Vòng lặp chạy eval — đừng chạy full mỗi lần sửa code:**

| Khi nào | Chạy gì |
|---|---|
| Mỗi lần sửa code | Tầng 1 (regex gate) + coverage + groundedness — vài giây, 0 token |
| Trước khi commit | + DeepEval `assert_test` trên 9 RFP |
| Khi so sánh biến thể | RAGAS full, n=3 → điền bảng 11.3 |

---

## PHẦN 5 — Thứ tự và điểm dừng an toàn

| Bước | Xong thì có gì |
|---|---|
| 0–3 | Data pipeline sạch, có số liệu chứng minh (7/7 leak, 4/4 contradiction) |
| *(1.5)* | *tùy chọn — chèn được bất cứ lúc nào sau Bước 1, không chặn bước nào* |
| **4** | **Sản phẩm chạy đầu-cuối** — mốc an toàn đầu tiên, demo được |
| 5–7 | Chất lượng thật: retrieval 4 tầng, sinh 2 kênh, **4 lớp chặn** (5.7 · 6.4 · 6.4-bis · 7) |
| **8** | **Demo đúng như ảnh** — mốc an toàn thứ hai |
| 9, 9-bis | Golden set: 9 ca viết tay + vài chục ca sinh tự động, mỗi ca tự có tiêu chí đúng/sai |
| 10–11 | Số liệu để bảo vệ lựa chọn kỹ thuật |

Nếu hết thời gian ở bất kỳ đâu, dừng ở mốc 4 hoặc 8 vẫn có sản phẩm nộp được.
Đừng bao giờ để dở dang giữa 5–7 (sinh nửa vời, chưa có guard) — đó là trạng thái tệ nhất:
chạy được nhưng có thể xuất ra hồ sơ bịa chứng chỉ.

---

## PHỤ LỤC A — Prompt mẫu đưa người thực hiện từng bước

```
Context: đọc PHẦN 1 (sự thật về data), PHẦN 2 (bất biến), PHẦN 3 (khung repo) trong BUILD_GUIDE.md.

Nhiệm vụ: thực hiện BƯỚC <N> — <tên bước>.

Yêu cầu:
- Chỉ tạo/sửa các file liệt kê trong "Tạo:" của bước đó. Không đụng file khác.
- Tuân thủ contract dữ liệu ghi trong bước, không đổi tên trường.
- Không thêm dependency ngoài requirements.txt.
- Không gọi LLM ở chỗ bước đó ghi "0 LLM".

Sau khi code xong, chạy lệnh trong mục "Nghiệm thu" và dán nguyên output.
Nếu số liệu không khớp tiêu chí, sửa code — không sửa tiêu chí, không nới ngưỡng.
```

Với Bước 2 (bước gắt nhất) thêm dòng:
```
Tiêu chí là tuyệt đối: 7 file leak, 0 false positive trên 5 tên chung, 4 file contradiction.
Lệch 1 file là chưa pass. Không được đọc proposals_index.jsonl trong code runtime (bất biến BB-6).
```
