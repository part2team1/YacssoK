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

from src.dataset import load_records, load_split
from src.utils import (
    category_to_label,
    label_names,
    load_class_map,
    load_config,
    resolve_path,
    save_yaml,
    xyxy_to_yolo,
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
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    class_map = load_class_map(d["class_map"])
    cat2label = category_to_label(class_map)
    records, _ = load_records(d["train_annotations"])
    root = resolve_path(d["yolo_dir"]) / d["version"]
    image_dir = resolve_path(d["train_images"])

    for split in ("train", "val"):
        split_records = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
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


if __name__ == "__main__":
    main()
