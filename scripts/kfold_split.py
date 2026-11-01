"""알약 조합 단위 K-fold 분할 (과적합·재현성 테스트용).

실행:
    uv run python scripts/kfold_split.py --config configs/base.yaml --k 5

결과: data/splits/{version}_f{i}/train.txt, val.txt  (i = 0 .. k-1)

왜 조합 단위로 나누나:
    같은 알약 조합(파일명 첫 '_' 앞부분, 예: K-001900-016548-019607-029451)을
    카메라 각도(70/75/90)만 바꿔 여러 장 찍었다. 이미지 단위로 나누면 거의 같은 사진이
    학습과 검증에 나뉘어 들어가 검증 점수가 부풀려진다 (데이터 누수).
    그래서 같은 조합의 사진은 항상 같은 fold에 들어가게 묶는다.

공식 분할(data/splits/{version})은 건드리지 않는다. K-fold는 검증용 별도 버전이다.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.dataset import load_records
from src.interfaces import SEED
from src.utils import load_config, resolve_path


def combo_of(file_name: str) -> str:
    """파일명에서 알약 조합 부분만 꺼낸다 (카메라 각도 등 촬영 조건 제외)."""
    return file_name.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--version", default=None, help="기본값: config의 data.version (예: v0)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    version = args.version or d["version"]

    records, _ = load_records(d["train_annotations"])
    names = [r["file_name"] for r in records]
    groups = [combo_of(n) for n in names]

    # 층화 기준: 이미지에서 가장 드문 클래스 (split_data.py와 같은 방식)
    freq = Counter(c for r in records for c in r["category_ids"])
    rarest = [
        min(r["category_ids"], key=lambda c: freq[c]) if r["category_ids"] else -1 for r in records
    ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 클래스당 이미지가 k보다 적다는 경고
            splitter = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
            folds = list(splitter.split(names, rarest, groups))
        method = "StratifiedGroupKFold (조합 단위 + 드문 클래스 층화)"
    except ValueError:
        folds = list(GroupKFold(n_splits=args.k).split(names, groups=groups))
        method = "GroupKFold (조합 단위)"

    print(f"이미지 {len(names)}장, 알약 조합 {len(set(groups))}개 → {args.k}-fold, {method}")
    print(f"{'fold':>4} {'train':>6} {'val':>5} {'겹치는 조합':>8} {'val에만 있는 클래스':>14}")
    for i, (tr_idx, va_idx) in enumerate(folds):
        train = sorted(names[j] for j in tr_idx)
        val = sorted(names[j] for j in va_idx)
        out = resolve_path(d["split_dir"]) / f"{version}_f{i}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

        overlap = {combo_of(n) for n in train} & {combo_of(n) for n in val}
        train_cls = {c for j in tr_idx for c in records[j]["category_ids"]}
        val_only = {c for j in va_idx for c in records[j]["category_ids"]} - train_cls
        print(f"{i:>4} {len(train):>6} {len(val):>5} {len(overlap):>8} {len(val_only):>14}")
        if overlap:
            print(f"     [주의] fold {i}: 학습과 검증에 같은 조합 {len(overlap)}개")

    print(f"저장: {resolve_path(d['split_dir'])}/{version}_f0 ~ {version}_f{args.k - 1}")


if __name__ == "__main__":
    main()
