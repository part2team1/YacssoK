"""외부 데이터(AI Hub)를 더한 버전의 K-fold 분할을 만든다.

조합 단위 K-fold(kfold_split.py)는 대회 이미지만으로 나눠야 한다. AI Hub 이미지가 검증에 섞이면
대회 점수와 다른 것을 재게 되기 때문이다. 그래서 대회 데이터 fold({base}_f{i})의
train에만 외부 이미지를 더하고, val은 그대로 복사한다.

실행 예:
    uv run python scripts/kfold_split.py --config configs/yolo_v1_clean.yaml --k 5     # v1_f0 ~ v1_f4
    uv run python scripts/make_ext_folds.py --base v1 --ext v3allc --k 5              # v3allc_f0 ~ v3allc_f4
결과: data/splits/{ext}_f{i}/train.txt, val.txt
"""

from __future__ import annotations

import argparse
import csv

from src.utils import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="split_dir 위치를 읽을 설정")
    ap.add_argument("--base", default="v1", help="대회 데이터 fold 버전 (예: v1 → v1_f0 ...)")
    ap.add_argument(
        "--ext", default="v3allc", help="외부 데이터 버전 (annotations/{ext}_external.csv)"
    )
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    d = load_config(args.config)["data"]
    split_dir = resolve_path(d["split_dir"])
    ext_csv = resolve_path(d["class_map"]).parent / "annotations" / f"{args.ext}_external.csv"
    with open(ext_csv, encoding="utf-8") as f:
        ext_names = [row["file_name"] for row in csv.DictReader(f)]

    for i in range(args.k):
        src = split_dir / f"{args.base}_f{i}"
        train = (src / "train.txt").read_text(encoding="utf-8").split()
        val = (src / "val.txt").read_text(encoding="utf-8").split()
        assert not set(val) & set(ext_names), "외부 이미지가 검증셋에 들어가면 안 된다"
        out = split_dir / f"{args.ext}_f{i}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text("\n".join(train + ext_names) + "\n", encoding="utf-8")
        (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
        print(
            f"{out.name}: train {len(train)} + 외부 {len(ext_names)} = {len(train) + len(ext_names)}, val {len(val)}"
        )


if __name__ == "__main__":
    main()
