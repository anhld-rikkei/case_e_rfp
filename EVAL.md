# Thiết kế đánh giá — so sánh phương án RAG/Agent bằng RAGAS + DeepEval

Đáp ứng "Lưu ý chung" của `Project_guide.md`: *dùng các phương pháp RAG và agent khác nhau,
dùng RAGAS hay deepeval để so sánh kết quả để đưa ra lý do vì sao dùng phương án*.

Bổ sung cho §4 của [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Vấn đề phải giải trước tiên: đơn vị đánh giá

RAGAS và DeepEval đều nhận đầu vào dạng **QA**: `(question, contexts, answer, ground_truth)`.
Case này đầu ra là **một văn bản dài**, không có "question". Ném cả hồ sơ vào làm `answer` và cả RFP
làm `question` thì mọi metric đều nhòe — không biết mục nào hỏng.

**Giải pháp: lấy requirement atom làm pseudo-question.** RFP đã tách sẵn tới mức `2.3`, và mỗi câu sinh ra
đã mang `req_ids` (xem state schema §2.7) — nên phép chiếu này miễn phí:

| Trường RAGAS/DeepEval | Lấy từ đâu |
|---|---|
| `question` / `input` | text của requirement atom, vd `2.3 同時接続ユーザ500名以上に対応すること。` |
| `contexts` / `retrieval_context` | câu proposal đã retrieve **+ capability facts đã dùng** (xem §3) |
| `answer` / `actual_output` | các câu sinh ra có `req_ids` chứa req đó |
| `ground_truth` / `expected_output` | đoạn tương ứng trong ideal proposal của golden set |

```python
def to_eval_samples(state) -> list[dict]:
    """1 requirement atom → 1 sample. 3 RFP × ~10 atom = ~30 sample/lần chạy."""
    samples = []
    for ch in state["chapters"]:
        for req in ch["requirements"]:
            answer = [s for sec in state["sections"] for s in sec["sentences"]
                      if req["req_id"] in s["req_ids"]]
            samples.append({
                "question": req["text"],
                "contexts": [c["text"] for c in ch["retrieval"]["selected"]]
                            + capability_facts_used(state, req["req_id"]),   # §3 — bắt buộc
                "answer": "".join(s["text"] for s in answer),
                "ground_truth": golden_lookup(state["rfp"]["rfp_id"], req["req_id"]),
            })
    return samples
```

Lợi ích phụ: metric ra **theo từng requirement** → chỉ ngay được "chương 技術要件 điểm thấp vì
capability sheet không có gì khớp `可用性99.9%`", chứ không chỉ ra một con số tổng vô nghĩa.

---

## 2. Ba tầng metric — không tầng nào thay được tầng nào

### Tầng 1 — Deterministic gate (0 LLM, chạy trước, là điều kiện chặn)

```python
FORBIDDEN = capability["certifications_NOT_held"] + capability["capabilities_NOT_offered"]
assert not any(term in output for term in FORBIDDEN)      # 27017, blockchain, 量子暗号…
assert not CLIENT_NAME_RE.search(output)                  # 新生証券様, 北陸食品様…
```

**Không bao giờ dùng LLM-judge làm chốt an toàn.** Judge sai ~5–10% là bình thường; ở đây sai 1 lần
là hồ sơ thầu tuyên bố sai chứng chỉ. Regex trên từ vựng đóng (danh sách `NOT_held` chỉ 3 mục,
`NOT_offered` 3 mục) thì chính xác 100%. RAGAS/DeepEval dùng để **so sánh chất lượng giữa các phương án**,
không dùng để gác cổng.

### Tầng 2 — RAGAS: so sánh biến thể (chạy batch, ra bảng số)

```python
# API đổi tên giữa 0.1 và 0.2 — kiểm lại theo version đang cài
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                    context_precision, context_recall])
```

| Metric | Trả lời câu hỏi gì trong case này |
|---|---|
| `context_precision` | MMR + rerank có thật sự đẩy câu hữu ích lên đầu không (corpus lặp nặng → kỳ vọng delta lớn) |
| `context_recall` | hybrid/HyDE có tìm được đủ câu precedent so với golden không |
| `faithfulness` | câu sinh ra có bịa ngoài context không |
| `answer_relevancy` | có trả lời đúng requirement hay chỉ nói chung chung ("要件定義から運用まで一貫して支援") |

`answer_relevancy` quan trọng bất thường ở đây: kho proposal cũ đầy câu boilerplate vô thưởng vô phạt,
model rất dễ ghép ra một hồ sơ "nghe thì mượt mà không đáp requirement nào". Đây là metric bắt được điều đó.

### Tầng 3 — DeepEval: metric riêng của bài toán + gate trong pytest

RAGAS không có metric "câu này có vượt quá năng lực công ty không". DeepEval có `GEval` để tự định nghĩa:

```python
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

capability_compliance = GEval(
    name="CapabilityCompliance",
    criteria=(
        "Đối chiếu actual_output với capability sheet trong context. "
        "Phạt nặng mọi tuyên bố về chứng chỉ hoặc năng lực KHÔNG có trong sheet, "
        "kể cả khi tuyên bố đó xuất hiện trong tài liệu tham chiếu. "
        "Câu 'không có năng lực này' hoặc bỏ trống KHÔNG bị phạt."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.9,
)

def test_no_overclaim(sample):                    # chạy được trong CI
    assert_test(LLMTestCase(input=sample["question"],
                            actual_output=sample["answer"],
                            context=[capability_sheet_text]),
                [capability_compliance, FaithfulnessMetric(threshold=0.8)])
```

**Phân vai rõ: RAGAS để đo và so sánh, DeepEval để gác trong `pytest`.** Nói được lý do chọn từng
framework thay vì "dùng cả hai cho đủ" — đó chính là thứ đề bài đang hỏi.

---

## 3. Cạm bẫy quan trọng nhất: `faithfulness` ≠ compliance

`PROP-009` chứa câu 「当社はISO/IEC 27017認証を取得済みであり…」. Nếu câu này lọt vào `contexts`
và model chép lại, **RAGAS faithfulness sẽ chấm ĐIỂM CAO** — vì câu sinh ra trung thành với context.
Nhưng đó đúng là lỗi nghiêm trọng nhất của cả hệ thống.

Hai xử lý bắt buộc, thiếu một cái là số liệu eval trở nên vô nghĩa:

1. **Luôn nối capability sheet vào `contexts`** của mọi sample (đã làm ở `to_eval_samples`).
   Khi đó context tự mâu thuẫn (sheet nói NOT_held, precedent nói đã có) → faithfulness mới phạt được.
2. **Đo compliance bằng metric riêng** (tầng 3) + gate regex (tầng 1), không dựa vào faithfulness.

Ghi nhận điều này trong báo cáo là một điểm cộng thật: nó cho thấy hiểu **giới hạn của framework**
chứ không chỉ biết gọi hàm.

---

## 4. Ma trận biến thể RAG cần so sánh

Chỉ chọn các biến thể mà **đặc tính dữ liệu này** khiến kết quả khó đoán trước — chạy thứ đã biết chắc kết quả là lãng phí.

| ID | Phương pháp | Vì sao đáng thử với data này | Metric quyết định |
|---|---|---|---|
| **V0** | Naive: chunk theo đoạn, dense-only, 1 lần gọi LLM sinh cả hồ sơ | baseline bắt buộc để mọi con số sau có nghĩa | tất cả |
| **V1** | Chunk theo **câu** + hybrid (BM25 + dense) | RFP đầy chuỗi khớp chính xác: `500名`, `99.9%`, `ISO/IEC 27001`, `18か月` → giả thuyết BM25 thắng dense ở đúng nhóm này | `context_recall` |
| **V2** | + metadata filter (`industry`, `section`) | `responds_to` → join ra 業種 sẵn có, filter gần như miễn phí | `context_precision` |
| **V3** | + cross-encoder rerank | corpus nhỏ (~300 câu) nên rerank rẻ; câu hỏi là có bõ latency không | `context_precision` vs latency |
| **V4** | + **MMR** | 40 proposal gần như cùng khung, chỉ khác số liệu → đây là biến thể tôi dự đoán tạo delta lớn nhất | `context_precision`, dedup rate |
| **V5** | Query decomposition: mỗi requirement atom là 1 query (thay vì 1 query/chương) | RFP đã tách sẵn tới `2.3` → decomposition có sẵn, không tốn LLM | `context_recall`, coverage |
| **V6** | **HyDE** | query (yêu cầu RFP: 「…すること。」) và doc (câu proposal: 「…実績があります。」) khác hẳn thể loại văn bản → bài toán bất đối xứng, đúng chỗ HyDE có lý do tồn tại | `context_recall` |
| **V7** | GraphRAG mini: đồ thị `requirement ↔ capability ↔ precedent` | dự đoán **không bõ** ở quy mô 300 câu — nhưng "đã thử và chứng minh không bõ, kèm số" là kết luận có giá trị hơn "không thử" | mọi metric vs công sức |

Ưu tiên nếu thiếu thời gian: **V0 → V1 → V4 → V5 → V6**. V3/V7 làm sau.

---

## 5. Ma trận biến thể Agent

| ID | Kiến trúc | Đánh đổi dự kiến | Metric quyết định |
|---|---|---|---|
| **A0** | Single-shot: nhét cả RFP + toàn bộ context vào 1 prompt | rẻ, nhanh; mất kiểm soát cấu trúc, không trace được câu | groundedness, coverage |
| **A1** | Chain tuyến tính cố định | ổn định; không tự sửa được khi claim-check fail | fabrication rate |
| **A2** | **LangGraph state machine + conditional edge + retry loop** (đề xuất) | claim-check `CONTRADICTED` → quay lại rewrite mục đó; skip node khi kênh thuộc tính đã phủ đủ | fabrication rate, latency |
| **A3** | ReAct agent tự chọn tool (`retrieve_precedent` / `lookup_capability` / `check_claim`) | linh hoạt với RFP lạ; latency và chi phí cao, kết quả không ổn định giữa các lần chạy | variance giữa 3 lần chạy |
| **A4** | Multi-agent: Writer / Fact-checker / Compliance-officer + reflection | chất lượng cao nhất về mặt lý thuyết; đắt nhất | fabrication rate, cost |

Điểm cần đo mà người ta hay quên: **variance**. Chạy A3 ba lần trên cùng RFP rồi báo cáo độ lệch —
với hồ sơ thầu, "mỗi lần chạy ra một cấu trúc khác nhau" là lỗi nghiệp vụ, không phải tính năng.
Đây thường là lập luận mạnh nhất để chọn A2 thay vì A3.

---

## 6. Test set: 3 RFP là quá ít

30 sample thì chênh lệch giữa các biến thể sẽ chìm trong nhiễu. Mở rộng bằng **mutation từ 3 RFP gốc**
(rẻ, và kiểm được đúng thứ cần kiểm):

| Loại mutation | Cách tạo | Hành vi ĐÚNG mà hệ thống phải có |
|---|---|---|
| Đổi 業種 | 製造業 → 公共 | vẫn chạy, đổi precedent theo industry mới |
| 業種 ngoài danh sách | → 医療 (không có trong `industries_served`) | không bịa kinh nghiệm ngành y tế |
| **Requirement đòi năng lực KHÔNG có** | thêm 「ブロックチェーン決済基盤を構築すること」 | `INSUFFICIENT_EVIDENCE` + ghi chú, **tuyệt đối không** nhận làm được |
| **Requirement đòi chứng chỉ KHÔNG có** | thêm 「ISO/IEC 27017認証を有すること」 | không tuyên bố có; nêu 27001 và nói rõ khác biệt |
| RFP thiếu 業種 | xóa dòng 発注業種 | rẽ nhánh `ask_user`, không đoán bừa |
| RFP text tự do | bỏ đánh số 第N章 | parser fallback LLM vẫn tách được chương |

Bốn dòng đầu **3 RFP gốc không hề kiểm được** — chúng đều nằm gọn trong năng lực công ty.
Nói cách khác, chạy eval chỉ trên 3 RFP gốc thì mọi biến thể đều "0 fabrication", và bảng so sánh
mất luôn cột quan trọng nhất. Đây là việc phải làm trước khi chạy eval, không phải sau.

---

## 7. Chống nhiễu LLM-judge

- **n=3, báo cáo mean ± std.** Chênh 0.02 giữa hai biến thể mà std = 0.05 thì không phải khác biệt.
- **Cố định judge model + `effort`**, ghi rõ trong báo cáo. Đổi judge giữa chừng là số liệu vứt đi.
  Lưu ý: `temperature` đã bị bỏ trên Claude Opus 5 / Sonnet 5 — gửi vào trả **400**. Không có
  "temperature=0" để ép determinism; thay bằng pin model + pin `output_config.effort` + prompt cố định.
- **Judge tiếng Nhật**: chọn model mạnh về JP; kiểm tay ~10 sample xem judge có chấm đúng không
  trước khi tin cả bảng.
- **Ưu tiên metric deterministic**: `coverage` (đếm req được phủ), `dedup rate`, `groundedness`
  (% câu có `source_id`), `fabrication rate`, latency, token cost — tất cả đều tính được không cần LLM,
  không nhiễu, và chạy được trong CI.

---

## 8. Bảng kết quả để điền — đây là hình dạng đầu ra của phần eval

```
Judge: <model>, temp=0, n=3 (mean±std) · Test set: 3 RFP gốc + 6 mutation = 9 RFP, ~90 requirement atom
```

| # | Cấu hình | fabric.↓ | leak↓ | coverage↑ | ctx_prec↑ | ctx_recall↑ | faithful.↑ | ans_rel.↑ | compliance↑ | latency | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0+A0 | Naive baseline | | | | | | | | | | |
| V1+A1 | +sentence, hybrid | | | | | | | | | | |
| V4+A1 | +MMR | | | | | | | | | | |
| V5+A2 | +decomp, graph | | | | | | | | | | |
| V6+A2 | +HyDE | | | | | | | | | | |
| **V4+V5+A2** | **đề xuất** | **0** | **0** | | | | | | | | |
| V4+V5+A3 | ReAct | | | | | | | | | | |
| — chỉ capability | (ablation nguồn) | 0 | 0 | thấp | | | | | | | |
| — chỉ precedent | (ablation nguồn) | **>0** | **>0** | | | | | | | | |

Ba dòng cuối là lập luận trung tâm cho câu hỏi "vì sao dùng cả hai nguồn":
**một mình precedent thì bịa và rò rỉ, một mình capability sheet thì rỗng và không đáp requirement.**

Cách viết kết luận cho từng lựa chọn — mỗi câu phải neo vào một ô trong bảng:

> "Chọn hybrid thay vì dense-only vì `context_recall` tăng từ X lên Y trên nhóm requirement chứa
> số liệu định lượng (`500名`, `99.9%`), trong khi nhóm requirement mô tả không đổi đáng kể."
>
> "Chọn A2 thay vì A3 vì fabrication rate tương đương nhưng độ lệch cấu trúc giữa 3 lần chạy
> là 0 so với Z, và latency thấp hơn N lần."
>
> "Không dùng V7 (GraphRAG) vì ở quy mô 300 câu, `context_precision` chỉ hơn V4 là Δ, không bù được
> chi phí xây và bảo trì đồ thị."

---

## 9. Chi phí

Mỗi lần chạy full: 9 RFP × 5 mục × (sinh + claim-check) + judge cho ~90 sample × 4 metric × 3 lần.
→ Đừng chạy full mỗi lần sửa code. Vòng lặp thực tế:

| Khi nào | Chạy gì |
|---|---|
| Mỗi lần sửa code | Tầng 1 (regex gate) + coverage + groundedness — vài giây, 0 token |
| Trước khi commit | + DeepEval `assert_test` trên 9 RFP |
| Khi so sánh biến thể | RAGAS full, n=3 → điền bảng §8 |
