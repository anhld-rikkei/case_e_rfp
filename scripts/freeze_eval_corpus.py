"""Đóng băng corpus hiện tại để đo lường không trôi theo dữ liệu vận hành.

Chạy script này **khi bắt đầu thêm dữ liệu vận hành** vào `synthetic/proposals`
hoặc `synthetic/rfps`. Nó chép nguyên trạng corpus hiện tại sang
`synthetic/eval_corpus/`, sau đó sửa `config/settings.py`:

    EVAL_PROPOSAL_DIR = SYNTHETIC_DIR / "eval_corpus" / "proposals"
    EVAL_RFP_DIR      = SYNTHETIC_DIR / "eval_corpus" / "rfps"

Vì sao cần: bảng ablation §11.3, golden set 9 ca, và `test_hybrid_hallucination`
(cần câu 「在庫精度を20%向上」 có thật trong corpus) đều neo theo dữ liệu hiện
tại. Thêm hồ sơ mới vào corpus vận hành là số đo trôi và test đỏ, mà nguyên nhân
thì rất khó truy ngược.

Không chạy sẵn lúc build v1.5: nhân đôi 40 file khi chưa có nhu cầu là rác, và
hai hằng số đang trỏ cùng chỗ nên hành vi hôm nay không đổi.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config.settings import PROPOSAL_DIR, RFP_DIR, SYNTHETIC_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=SYNTHETIC_DIR / "eval_corpus",
        help="thư mục đích (mặc định synthetic/eval_corpus)",
    )
    parser.add_argument("--force", action="store_true", help="ghi đè nếu đã có")
    args = parser.parse_args()

    if args.dest.exists() and not args.force:
        raise SystemExit(
            f"{args.dest} đã tồn tại. Dùng --force nếu thật sự muốn đóng băng lại "
            "— đè lên nghĩa là mọi số đo cũ không so được với số mới nữa."
        )

    for name, source in (("proposals", PROPOSAL_DIR), ("rfps", RFP_DIR)):
        target = args.dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        print(f"{name}: chép {len(list(target.glob('*.txt')))} file -> {target}")

    print()
    print("Xong. Bước tiếp theo — sửa config/settings.py:")
    print('    EVAL_PROPOSAL_DIR = SYNTHETIC_DIR / "eval_corpus" / "proposals"')
    print('    EVAL_RFP_DIR      = SYNTHETIC_DIR / "eval_corpus" / "rfps"')
    print("rồi chạy lại `pytest eval/ -q` để xác nhận số đo không đổi.")


if __name__ == "__main__":
    main()
