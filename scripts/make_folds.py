"""K-fold 교차검증 분할과 전체 학습용 분할 만들기 (실험 D7, 데이터 B).

검증셋이 수십 장뿐이라 점수가 우연히 흔들린다. 전체 이미지를 K조각으로 나눠
K번 학습·검증한 평균으로 판단하면 더 믿을 수 있고, K개 모델은 그대로 앙상블(P2)에 쓴다.

실행:
    uv run python scripts/make_folds.py --config configs/base.yaml --k 5

결과 (모두 커밋, 파일 이름만 들어 있음):
    data/splits/{version}_f1 ~ _f5/   train.txt, val.txt  (전체 이미지를 5조각, 조각마다 검증 1개)
    data/splits/{version}_full/       train.txt = 전체 이미지, val.txt = 원래 검증셋
                                      (최종 제출용 학습. 검증 점수는 학습에 쓴 사진이라 의미 없음)

분할 기준은 split_data.py와 같다: 이미지에서 가장 드문 클래스로 층화한다.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

from src.dataset import load_records
from src.interfaces import SEED
from src.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--k", type=int, default=5, help="조각 수")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    version = d["version"]
    split_dir = resolve_path(d["split_dir"])

    records, _ = load_records(d["train_annotations"])
    names = [r["file_name"] for r in records]
    freq = Counter(c for r in records for c in r["category_ids"])
    keys = [
        min(r["category_ids"], key=lambda c: freq[c]) if r["category_ids"] else -1 for r in records
    ]
    key_count = Counter(keys)
    keys = [k if key_count[k] >= args.k else "rare" for k in keys]  # 조각 수보다 적으면 묶음
    if min(Counter(keys).values()) >= args.k:
        folds = StratifiedKFold(args.k, shuffle=True, random_state=SEED).split(names, keys)
        how = "층화"
    else:
        folds = KFold(args.k, shuffle=True, random_state=SEED).split(names)
        how = "무작위 (클래스당 이미지가 너무 적음)"

    for i, (tr, va) in enumerate(folds, start=1):
        out = split_dir / f"{version}_f{i}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text(
            "\n".join(sorted(names[j] for j in tr)) + "\n", encoding="utf-8"
        )
        (out / "val.txt").write_text(
            "\n".join(sorted(names[j] for j in va)) + "\n", encoding="utf-8"
        )
        missing = {c for j in va for c in records[j]["category_ids"]} - {
            c for j in tr for c in records[j]["category_ids"]
        }
        warn = f"  [주의] 검증 조각에만 있는 클래스 {len(missing)}개" if missing else ""
        print(f"{out.name}: train {len(tr)}장 / val {len(va)}장{warn}")

    full = split_dir / f"{version}_full"
    full.mkdir(parents=True, exist_ok=True)
    (full / "train.txt").write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    shutil.copy(split_dir / version / "val.txt", full / "val.txt")
    print(f"{full.name}: train {len(names)}장 (전체) / val = 원래 검증셋 (점수 참고용 아님)")
    print(f"분할 방식: {how}")


if __name__ == "__main__":
    main()
    