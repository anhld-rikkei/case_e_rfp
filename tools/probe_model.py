"""Dò xem model OpenAI của bạn nhận tham số nào — chạy TRƯỚC khi viết nhánh openai trong llm.py.

    .venv\\Scripts\\python.exe tools\\probe_model.py

Kết quả in ra chính là giá trị điền vào .env (LLM_TOKEN_ARG / LLM_USE_TEMP / LLM_USE_EFFORT).
Tốn ~5 lệnh gọi rất ngắn.
"""

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "").strip()
if not MODEL:
    sys.exit("Thiếu LLM_MODEL trong .env")
if not os.getenv("OPENAI_API_KEY", "").strip():
    sys.exit("Thiếu OPENAI_API_KEY trong .env")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Chưa cài SDK. Chạy: pip install openai")

client = OpenAI()
PING = [{"role": "user", "content": "ping"}]
result = {}


def probe(name: str, **extra) -> bool:
    try:
        r = client.chat.completions.create(model=MODEL, messages=PING, **extra)
        txt = (r.choices[0].message.content or "")[:24]
        print(f"  OK    {name:24} -> {txt!r}")
        return True
    except Exception as e:
        print(f"  FAIL  {name:24} -> {type(e).__name__}: {str(e)[:100]}")
        return False


print(f"\nModel: {MODEL}\n")
print("[1] Tham số giới hạn token")
if probe("max_tokens", max_tokens=16):
    result["LLM_TOKEN_ARG"] = "max_tokens"
elif probe("max_completion_tokens", max_completion_tokens=16):
    result["LLM_TOKEN_ARG"] = "max_completion_tokens"
else:
    result["LLM_TOKEN_ARG"] = "??? cả hai đều lỗi — đọc kỹ thông báo trên"

tok = {result["LLM_TOKEN_ARG"]: 16} if result["LLM_TOKEN_ARG"].startswith("max") else {}

print("\n[2] temperature (dùng cho LLM-judge ở Bước 10)")
result["LLM_USE_TEMP"] = "1" if probe("temperature=0", temperature=0, **tok) else "0"

print("\n[3] reasoning_effort")
result["LLM_USE_EFFORT"] = "1" if probe("reasoning_effort", reasoning_effort="low", **tok) else "0"

# Đo TỔ HỢP, không chỉ đo từng cái. gpt-5.4-mini nhận temperature và reasoning_effort
# riêng lẻ nhưng TỪ CHỐI khi gửi chung -> nếu chỉ đo riêng sẽ điền cả hai cờ = 1 rồi 400.
if result["LLM_USE_TEMP"] == "1" and result["LLM_USE_EFFORT"] == "1":
    print("\n[3b] temperature + reasoning_effort CÙNG LÚC")
    if probe("temp + effort", temperature=0, reasoning_effort="low", **tok):
        result["_COMBO"] = "cả hai gửi chung được"
    else:
        result["_COMBO"] = (
            "KHÔNG gửi chung được -> llm.py phải chọn 1 tuỳ ngữ cảnh:\n"
            "      generate()   -> reasoning_effort (chất lượng suy luận)\n"
            "      structured() -> temperature=0    (claim-check & judge cần determinism)"
        )

print("\n[4] structured output — BẮT BUỘC PHẢI OK, xem ghi chú cuối")


class Verdict(BaseModel):
    verdict: str


ok_struct = False
for path, fn in (
    ("chat.completions.parse", lambda: client.chat.completions.parse),
    ("beta.chat.completions.parse", lambda: client.beta.chat.completions.parse),
):
    try:
        r = fn()(
            model=MODEL,
            messages=[{"role": "user", "content": "trả về verdict='ok'"}],
            response_format=Verdict,
            **tok,
        )
        print(f"  OK    {path:24} -> {r.choices[0].message.parsed}")
        result["_STRUCTURED_PATH"] = path
        ok_struct = True
        break
    except AttributeError:
        print(f"  N/A   {path:24} -> SDK không có đường này")
    except Exception as e:
        print(f"  FAIL  {path:24} -> {type(e).__name__}: {str(e)[:100]}")

print("\n" + "=" * 62)
print("ĐIỀN VÀO .env:")
for k, v in result.items():
    if not k.startswith("_"):
        print(f"  {k}={v}")
if "_STRUCTURED_PATH" in result:
    print(f"\n  (llm.py dùng: client.{result['_STRUCTURED_PATH']})")
if "_COMBO" in result:
    print(f"\n  temperature + reasoning_effort: {result['_COMBO']}")
if not ok_struct:
    print(
        "\n  ⚠ STRUCTURED OUTPUT KHÔNG CHẠY — đây là CHẶN ĐỨNG, không phải bất tiện.\n"
        "    Bước 6.4 (claim-check) và Bước 10 (eval) phụ thuộc nó. Parse JSON tay + retry\n"
        "    sẽ trộn tỉ lệ hỏng vào số liệu eval khiến bảng §11.3 không đọc được.\n"
        "    Cách xử: đổi model riêng cho 2 chỗ đó, ĐỪNG đi đường parse tay."
    )
print("=" * 62)
