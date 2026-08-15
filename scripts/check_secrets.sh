#!/usr/bin/env bash
set -uo pipefail

PATTERNS='sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|(ANTHROPIC|OPENAI)_API_KEY[[:space:]]*=[[:space:]]*["'"'"']?[A-Za-z0-9_-]{20,}'
# CHỈ quét dòng THÊM (+). Dòng xoá (-) nghĩa là đang GỠ key ra khỏi file —
# chặn nó là ngược mục đích, và làm kẹt luôn mọi commit dọn dẹp.
if git diff --cached -U0 | grep '^+' | grep -v '^+++' | grep -nEI "$PATTERNS"; then
  echo "" >&2
  echo "CHẶN COMMIT: phát hiện API key trong diff (dòng ở trên)." >&2
  echo "Gỡ key ra, đưa vào .env, rồi commit lại." >&2
  exit 1
fi
