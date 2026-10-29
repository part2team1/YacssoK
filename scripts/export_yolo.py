"""팀 표준 데이터를 YOLO(ultralytics) 형식으로 내보낸다 (데이터 B).

실행:
    uv run python scripts/export_yolo.py --config configs/yolo_baseline.yaml

결과: data/yolo/{version}/
    images/train, images/val   (원본 이미지 링크 또는 복사본)
    labels/train, labels/val   (이미지별 txt: "label_idx cx cy w h", 0~1 정규화)
    data.yaml                  (ultralytics 학습 설정)
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch

from src.dataset import (  #v3.D1.3 ← 뒤에 두 이름 추가
    load_records,
    load_split,
    oversample,
    repeat_factors,
)
from src.utils import (
    category_to_label,
    label_names,
    load_class_map,
    load_config,
    resolve_path,
    save_yaml,
    xyxy_to_yolo,
    yolo_data_yaml,  # v3.D1.3 ← from src.utils import ( … ) 목록 맨 아래에 추가
)


def _link_or_copy(src, dst) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)  # 용량 절약. Windows에서 권한이 없으면 복사
    except OSError:
        shutil.copy(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo_baseline.yaml")
    ap.add_argument("--set", nargs="*", default=None, help="설정 값 바꾸기 (예: data.version=v0_f1)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    class_map = load_class_map(d["class_map"])
    cat2label = category_to_label(class_map)
    records, _ = load_records(d["train_annotations"])
    root = resolve_path(d["yolo_dir"]) / d["version"]
    image_dir = resolve_path(d["train_images"])

    train_records = [] # v3.D1.3  (나) for split 줄 위에 한 줄
    for split in ("train", "val"):
        split_records = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
        if split == "train":
            train_records = split_records
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in split_records:
            src = image_dir / r["file_name"]
            _link_or_copy(src, root / "images" / split / r["file_name"])
            boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
            yolo = xyxy_to_yolo(boxes, r["width"], r["height"]).clamp(0, 1)
            lines = [
                f"{cat2label[c]} " + " ".join(f"{v:.6f}" for v in b.tolist())
                for b, c in zip(yolo, r["category_ids"], strict=True)
            ]
            label_file = root / "labels" / split / (src.stem + ".txt")
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{split}: {len(split_records)}장")

    save_yaml(
        {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(label_names(class_map))),
        },
        root / "data.yaml",
    )
    print(f"저장: {root / 'data.yaml'}")

    # 가이드v2-v3 실험 D1: 드문 클래스가 든 이미지를 목록에 여러 번 적어 더 자주 학습 (baseline data.yaml은 그대로)
    thr = float(d.get("repeat_threshold", 0) or 0)
    if thr > 0:
        repeated = oversample(train_records, thr, seed=cfg.get("seed", 42))
        list_file = root / f"train_rfs{thr:g}.txt"
        # abspath: 링크를 따라가지 않아야 ultralytics가 labels/ 폴더를 찾는다
        list_file.write_text(
            "\n".join(os.path.abspath(root / "images" / "train" / r["file_name"]) for r in repeated)
            + "\n",
            encoding="utf-8",
        )
        rf = repeat_factors(train_records, thr)
        out_yaml = yolo_data_yaml(cfg)
        save_yaml(
            {
                "path": str(root.resolve()),
                "train": list_file.name,
                "val": "images/val",
                "names": dict(enumerate(label_names(class_map))),
            },
            out_yaml,
        )
        print(
            f"오버샘플링(threshold={thr:g}): 학습 이미지 {len(train_records)}장 → {len(repeated)}장 "
            f"(반복된 이미지 {sum(f > 1 for f in rf)}장, 최대 {max(rf, default=1):.1f}배)"
        )
        print(f"저장: {out_yaml}")


if __name__ == "__main__":
    main()
