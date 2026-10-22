"""학습/검증 분할 (데이터 B). 분할 파일은 커밋해서 팀 전원이 같은 검증셋을 쓴다.

실행:
    uv run python scripts/split_data.py --config configs/base.yaml --val-ratio 0.2

결과: data/splits/{version}/train.txt, val.txt (이미지 파일 이름 한 줄씩)

분할 기준: 이미지에서 가장 희귀한 클래스를 기준으로 층화(stratify)한다.
→ 드문 알약이 검증셋에만 몰리거나 학습셋에서 빠지는 것을 줄인다.
10/27 확정 이후 검증셋(val.txt)은 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
from collections import Counter

from sklearn.model_selection import train_test_split

from src.dataset import load_records
from src.interfaces import SEED
from src.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--version", default=None, help="기본값: config의 data.version")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    version = args.version or d["version"]

    records, _ = load_records(d["train_annotations"])
    freq = Counter(c for r in records for c in r["category_ids"])
    # 이미지마다 가장 드문 클래스를 대표값으로. 대표값이 1장뿐이면 층화가 불가능하므로 'rare'로 묶음
    keys = [
        min(r["category_ids"], key=lambda c: freq[c]) if r["category_ids"] else -1 for r in records
    ]
    key_count = Counter(keys)
    keys = [k if key_count[k] >= 2 else "rare" for k in keys]
    stratify = keys if Counter(keys).most_common()[-1][1] >= 2 else None

    names = [r["file_name"] for r in records]
    train, val = train_test_split(
        names, test_size=args.val_ratio, random_state=SEED, stratify=stratify
    )

    out = resolve_path(d["split_dir"]) / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(sorted(train)) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(sorted(val)) + "\n", encoding="utf-8")

    val_set = set(val)
    train_classes = {c for r in records if r["file_name"] not in val_set for c in r["category_ids"]}
    val_only = {
        c for r in records if r["file_name"] in val_set for c in r["category_ids"]
    } - train_classes
    print(f"train {len(train)}장 / val {len(val)}장 → {out}")
    print(f"층화 분할: {'예' if stratify else '아니오 (클래스당 이미지가 너무 적음)'}")
    if val_only:
        print(f"[주의] 검증셋에만 있는 클래스 {len(val_only)}개: {sorted(val_only)[:10]}")


if __name__ == "__main__":
    main()
