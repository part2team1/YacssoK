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
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from src.dataset import (  # v3.D1.3 ← 뒤에 두 이름 추가
    load_records,
    load_split,
    oversample,
    repeat_factors,
)
from src.utils import (
    apply_overrides,
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
    ap.add_argument(
        "--set", nargs="*", default=None, help="설정 값 바꾸기 (예: data.version=v0_f1)"
    )
    args = ap.parse_args()
    # cfg = load_config(args.config)
    cfg = apply_overrides(load_config(args.config), args.set)
    export_dataset(cfg)


def _write_export(cfg: dict, root: Path, destination: Path, splits: dict) -> None:
    """검증한 분할을 임시 폴더에 쓴다. YAML 경로는 최종 목적지를 가리킨다."""
    d = cfg["data"]
    class_map = load_class_map(d["class_map"])
    cat2label = category_to_label(class_map)
    image_dir = resolve_path(d["train_images"])

    train_records = splits["train"]
    for split in ("train", "val"):
        split_records = splits[split]
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
            "path": str(destination.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(label_names(class_map))),
        },
        root / "data.yaml",
    )
    print(f"저장: {root / 'data.yaml'}")

    _write_repeat(cfg, root, destination, train_records, class_map)


def _write_repeat(cfg, root, destination, train_records, class_map):
    """같은 버전에서 사용 중인 모든 RFS 설정도 새 분할에 맞춰 다시 쓴다."""
    d = cfg["data"]
    thr = float(d.get("repeat_threshold", 0) or 0)
    if thr > 0:
        repeated = oversample(train_records, thr, seed=cfg.get("seed", 42))
        list_file = root / f"train_rfs{thr:g}.txt"
        # abspath: 링크를 따라가지 않아야 ultralytics가 labels/ 폴더를 찾는다
        list_file.write_text(
            "\n".join(
                os.path.abspath(destination / "images" / "train" / r["file_name"]) for r in repeated
            )
            + "\n",
            encoding="utf-8",
        )
        rf = repeat_factors(train_records, thr)
        out_yaml = root / yolo_data_yaml(cfg).name
        save_yaml(
            {
                "path": str(destination.resolve()),
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


def export_dataset(cfg: dict) -> Path:
    """한 버전을 완성한 뒤 교체해 이전 사진·라벨·Ultralytics 캐시를 남기지 않는다."""
    d = cfg["data"]
    version = str(d["version"])
    if version in ("", ".", "..") or Path(version).name != version:
        raise ValueError("data.version은 폴더 이름 하나여야 합니다.")
    destination = resolve_path(d["yolo_dir"]) / version
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"생성 폴더가 아닌 경로는 교체하지 않습니다: {destination}")
    image_dir = resolve_path(d["train_images"]).resolve()
    inputs = [
        image_dir,
        resolve_path(d["train_annotations"]).resolve(),
        resolve_path(d["class_map"]).resolve(),
        (resolve_path(d["split_dir"]) / version).resolve(),
    ]
    target = destination.resolve()
    if any(
        source == target or target in source.parents or source in target.parents
        for source in inputs
    ):
        raise ValueError("내보내기 폴더와 원본 이미지·정답·번호표·분할 경로는 겹칠 수 없습니다.")
    records, _ = load_records(d["train_annotations"])
    splits = {}
    for split in ("train", "val"):
        split_path = resolve_path(f"{d['split_dir']}/{version}/{split}.txt")
        requested = {name.strip() for name in split_path.read_text().splitlines() if name.strip()}
        selected = load_split(records, split_path)
        names = [record["file_name"] for record in selected]
        if requested != set(names) or len(names) != len(set(names)):
            raise ValueError(f"{split}: 분할 목록에 정답이 없거나 파일명이 중복됩니다.")
        stems = set()
        for name in names:
            if Path(name).name != name or not (image_dir / name).is_file():
                raise ValueError(f"{split}: 원본 이미지 경로를 확인하세요: {name}")
            if Path(name).stem.casefold() in stems:
                raise ValueError(f"{split}: 같은 라벨 파일명을 만드는 이미지가 있습니다: {name}")
            stems.add(Path(name).stem.casefold())
        splits[split] = selected
    overlap = {r["file_name"] for r in splits["train"]} & {r["file_name"] for r in splits["val"]}
    if overlap:
        raise ValueError(f"학습·검증에 같은 사진이 있습니다: {sorted(overlap)[:5]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_thresholds = set()
    if destination.exists():
        for path in destination.glob("data_rfs*.yaml"):
            match = re.fullmatch(r"data_rfs([0-9.eE+-]+)\.yaml", path.name)
            if match and float(match[1]) > 0:
                previous_thresholds.add(float(match[1]))
    # 같은 파일시스템에 생성한다. 준비 중 오류가 나면 기존 출력은 그대로 유지된다.
    with tempfile.TemporaryDirectory(prefix=f".{version}-export-", dir=destination.parent) as temp:
        staging, backup = Path(temp) / "ready", Path(temp) / "previous"
        _write_export(cfg, staging, destination, splits)
        class_map = load_class_map(d["class_map"])
        for threshold in previous_thresholds:
            variant = deepcopy(cfg)
            variant["data"]["repeat_threshold"] = threshold
            _write_repeat(variant, staging, destination, splits["train"], class_map)
        had_previous = destination.exists()
        if had_previous:
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if had_previous:
                backup.rename(destination)
            raise
    print(f"저장: {destination / 'data.yaml'}")
    return destination


if __name__ == "__main__":
    main()
