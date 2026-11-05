"""드문 클래스 합성 데이터 만들기: 알약을 잘라 다른 학습 이미지에 붙인다 (실험 D2, 데이터 B).

실행:
    uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2
    uv run python scripts/make_copy_paste.py --config configs/yolo_v1_clean.yaml --version v2   # 정제 v1 위에

결과:
    data/synth/v2/images/          원본 이미지(하드링크 또는 복사) + 합성 이미지 cp_*.png  — 커밋하지 않음
    data/annotations/v2.json       원본 + 합성 어노테이션 (COCO)                          — 커밋하지 않음
    data/annotations/v2_copy_paste.csv  어떤 알약을 어디에 붙였는지 기록 (파일 이름만)      — 커밋
    data/splits/v2/                train = 원본 train + 합성, val = 원본 val 그대로        — 커밋

규칙
  - 학습셋(train)의 알약만 잘라 학습셋 이미지에만 붙인다. 검증셋 이미지는 절대 쓰지 않는다(점수 부풀림 방지).
  - 박스 수가 --min-boxes 보다 적은 클래스만 채운다. 이미지당 알약은 과제 조건(4개)을 넘지 않는다.
  - 붙일 자리는 기존 박스와 겹치지 않는 곳만 고른다. 잘라낸 사각형 = 새 박스라서 박스가 정확하다.
  - 잘라낸 알약은 90도 단위로 돌리거나 뒤집어 다양하게 붙인다.
  - 붙일 자리의 배경 원본에 배경색과 다른 픽셀이 많으면(라벨이 빠진 알약 등) 그 자리는 쓰지 않는다.
  - --paste mask(기본)는 조각에서 알약 부분만 골라 테두리를 부드럽게 붙인다(네모난 경계 줄이기).
    --paste rect는 예전처럼 사각형 그대로 붙인다.
붙인 결과는 scripts/review_labels.py --images synth 로 그림과 점검표를 만들어 확인한다.
설정 파일에서 data 블록의 train_images·train_annotations·version 세 줄을 바꾸면 이 데이터로 학습한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter

import cv2
import numpy as np

from src.dataset import load_records, load_split, read_image_rgb, records_to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_config, resolve_path, save_json


def _link_or_copy(src, dst) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 하드링크: 용량을 쓰지 않고, Windows에서도 관리자 권한 없이 된다
    except OSError:
        shutil.copy(src, dst)


def _overlaps(box, boxes, margin: int = 4) -> bool:
    x1, y1, x2, y2 = box
    for bx1, by1, bx2, by2 in boxes:
        if x1 < bx2 + margin and bx1 < x2 + margin and y1 < by2 + margin and by1 < y2 + margin:
            return True
    return False


def _background_color(img: np.ndarray) -> np.ndarray:
    """사진 가장자리 띠의 중앙값 색 = 촬영 배경색."""
    m = max(4, min(img.shape[:2]) // 40)
    border = np.concatenate(
        [img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3), img[:, :m].reshape(-1, 3)]
    )
    return np.median(border, axis=0)


def _is_empty(region: np.ndarray, bg: np.ndarray, tol: float = 40.0, ratio: float = 0.02) -> bool:
    """영역에 배경색과 다른 픽셀이 ratio 미만이면 비어 있다고 본다 (라벨 없는 알약 덮기 방지)."""
    diff = np.abs(region.astype(np.float32) - bg).sum(axis=2)
    return float((diff > tol).mean()) < ratio


def _pill_alpha(crop: np.ndarray, tol: float = 40.0) -> np.ndarray:
    """조각에서 알약 부분 = 조각 테두리 색과 다른 픽셀. 가장자리를 흐리게 한 0~1 가중치를 돌려준다."""
    edge = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.float32)
    diff = np.abs(crop.astype(np.float32) - np.median(edge, axis=0)).sum(axis=2)
    mask = (diff > tol).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if mask.mean() < 0.2:  # 알약을 못 골라냈으면 사각형 전체를 쓴다
        return np.ones(crop.shape[:2], np.float32)
    return cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)


def _augment_crop(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    crop = np.rot90(crop, rng.choice([0, 1, 2, 3]))
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    return np.ascontiguousarray(crop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--version", default="v2", help="새 데이터 버전 이름")
    ap.add_argument(
        "--min-boxes", type=int, default=10, help="학습셋 박스가 이보다 적은 클래스를 채움"
    )
    ap.add_argument(
        "--per-image", type=int, default=2, help="합성 이미지 한 장에 붙일 최대 알약 수"
    )
    ap.add_argument(
        "--paste", choices=["mask", "rect"], default="mask", help="알약만 / 사각형 그대로 붙이기"
    )
    ap.add_argument(
        "--empty-ratio",
        type=float,
        default=0.02,
        help="붙일 자리에 배경과 다른 픽셀이 이 비율 이상이면 건너뜀 (1.0이면 검사 안 함)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cfg = load_config(args.config)
    d = cfg["data"]
    records, categories = load_records(d["train_annotations"])
    split_dir = resolve_path(d["split_dir"])
    train = load_split(records, split_dir / d["version"] / "train.txt")
    src_dir = resolve_path(d["train_images"])

    counts = Counter(c for r in train for c in r["category_ids"])
    need = {c: args.min_boxes - n for c, n in counts.items() if n < args.min_boxes}
    crops = {c: [] for c in need}
    for r in train:
        for box, c in zip(r["boxes"], r["category_ids"], strict=True):
            if c in crops:
                crops[c].append((r, box))
    backgrounds = [r for r in train if len(r["boxes"]) < MAX_PILLS_PER_IMAGE]
    print(f"학습셋 클래스 {len(counts)}개 중 박스 {args.min_boxes}개 미만: {len(need)}개")
    if not need or not backgrounds:
        print("채울 클래스가 없거나 붙일 배경 이미지가 없습니다.")
        return

    # 실제 데이터는 data/, 합성 연습 데이터는 data/sample/ 아래에 만든다
    data_root = resolve_path(d["class_map"]).parent
    out_dir = data_root / "synth" / args.version / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:  # 원본 전체(검증 포함)를 같은 폴더에 연결해 train_images 하나로 읽게 한다
        _link_or_copy(src_dir / r["file_name"], out_dir / r["file_name"])

    cache: dict[str, np.ndarray] = {}

    def image(rec: Record) -> np.ndarray:
        if rec["file_name"] not in cache:
            cache[rec["file_name"]] = read_image_rgb(src_dir / rec["file_name"])
        return cache[rec["file_name"]]

    synth: list[Record] = []
    next_id = max((r["image_id"] for r in records), default=0) + 1
    log = []
    tries = 0
    while any(v > 0 for v in need.values()) and tries < 5000:
        tries += 1
        bg = rng.choice(backgrounds)
        canvas = image(bg).copy()
        h, w = canvas.shape[:2]
        bg_color = _background_color(canvas)
        boxes = [list(b) for b in bg["boxes"]]
        cats = list(bg["category_ids"])
        room = min(args.per_image, MAX_PILLS_PER_IMAGE - len(boxes))
        pasted = 0
        for c in sorted(need, key=lambda k: -need[k]):
            if pasted >= room or need[c] <= 0:
                continue
            src_rec, (x1, y1, x2, y2) = rng.choice(crops[c])
            crop = image(src_rec)[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                continue
            crop = _augment_crop(crop, rng)
            ch, cw = crop.shape[:2]
            if ch >= h or cw >= w:
                continue
            for _ in range(50):
                px, py = rng.randint(0, w - cw), rng.randint(0, h - ch)
                box = [px, py, px + cw, py + ch]
                region = canvas[py : py + ch, px : px + cw]
                if not _overlaps(box, boxes) and _is_empty(
                    region, bg_color, ratio=args.empty_ratio
                ):
                    if args.paste == "mask":
                        a = _pill_alpha(crop)[..., None]
                        canvas[py : py + ch, px : px + cw] = (a * crop + (1 - a) * region).astype(
                            np.uint8
                        )
                    else:
                        canvas[py : py + ch, px : px + cw] = crop
                    boxes.append([float(v) for v in box])
                    cats.append(c)
                    need[c] -= 1
                    pasted += 1
                    log.append({"source_image": src_rec["file_name"], "category_id": c})
                    break
        if pasted == 0:
            continue
        name = f"cp_{len(synth) + 1:04d}_{os.path.splitext(bg['file_name'])[0]}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        buf.tofile(str(out_dir / name))  # 한글 경로에서도 저장되게
        synth.append(
            Record(
                image_id=next_id + len(synth),  # 원본 id와 겹치지 않는 새 번호
                file_name=name,
                width=w,
                height=h,
                boxes=boxes,
                category_ids=cats,
            )
        )
        for row in log[-pasted:]:
            row.update({"synth_image": name, "background_image": bg["file_name"]})

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    save_json(records_to_coco(records + synth, categories), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_copy_paste.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["synth_image", "background_image", "source_image", "category_id"]
        )
        writer.writeheader()
        writer.writerows(log)

    new_split = split_dir / args.version
    new_split.mkdir(parents=True, exist_ok=True)
    (new_split / "train.txt").write_text(
        "\n".join([r["file_name"] for r in train] + [r["file_name"] for r in synth]) + "\n",
        encoding="utf-8",
    )
    shutil.copy(split_dir / d["version"] / "val.txt", new_split / "val.txt")

    after = Counter(c for r in train + synth for c in r["category_ids"])
    still = [c for c in counts if after[c] < args.min_boxes]
    print(f"합성 이미지 {len(synth)}장, 붙인 알약 {len(log)}개")
    print(f"드문 클래스 최소 박스 수: {min(counts.values())} → {min(after.values())}")
    if still:
        print(
            f"목표를 다 못 채운 클래스 {len(still)}개 (붙일 자리 부족). --per-image를 늘려 보세요."
        )
    print(f"저장: {out_dir}, {ann_dir / (args.version + '.json')}, {new_split}")
    rel = data_root.relative_to(resolve_path(".")).as_posix()
    print("설정 파일 data 블록:")
    print(f"  train_images: {rel}/synth/{args.version}/images")
    print(f"  train_annotations: {rel}/annotations/{args.version}.json")
    print(f"  version: {args.version}")


if __name__ == "__main__":
    main()
