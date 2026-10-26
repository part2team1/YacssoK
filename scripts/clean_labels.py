"""라벨 정제: 원본 어노테이션을 규칙대로 고쳐 새 데이터 버전(v1, v2 ...)을 만든다 (데이터 B).

실행:
    uv run python scripts/clean_labels.py --config configs/base.yaml --version v1

결과:
    data/annotations/v1.json        정제된 어노테이션 (COCO 형식 JSON 하나) — 커밋하지 않음 (대회 데이터 재배포 금지)
    data/annotations/v1_report.csv  무엇을 고쳤는지 기록 (보고서용) — 커밋
    data/splits/v1/                 v0 분할을 그대로 복사 (검증셋 고정) — 커밋
팀원은 이 스크립트와 manual_drop.csv를 받아 같은 명령을 실행하면 똑같은 v1.json이 만들어진다.

사용: 설정 파일에서 아래 두 줄만 바꾸면 모델이 정제 데이터로 학습한다.
    data:
      train_annotations: data/annotations/v1.json
      version: v1

규칙을 바꾸거나 추가하려면 clean() 함수만 고친다. 사람이 직접 확인한 오류는
data/annotations/manual_drop.csv (file_name,category_id) 에 적으면 해당 박스를 뺀다.
"""

from __future__ import annotations

import argparse
import csv
import shutil

from src.dataset import load_records
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_config, resolve_path, save_json

MIN_SIZE = 3  # 너비·높이가 이 픽셀보다 작으면 잘못된 박스로 본다


def load_manual_drop(path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {(row["file_name"], int(row["category_id"])) for row in csv.DictReader(f)}


# 특정 파일 bbox swap 코드 추가
def clean(records: list[Record], manual_drop: set) -> tuple[list[Record], list[dict]]:
    """정제 규칙을 적용하고 변경 내역을 기록한다. 원본 records는 수정하지 않는다."""
    swap_file = (
        "K-003483-025367-027733-035206_0_2_0_2_75_000_200.png"
    )
    swap_cids = (27733, 35206)

    cleaned, log = [], []

    for r in records:
        file_name = r["file_name"]

        # 이미지 전체 수동 제외
        if (file_name, "all") in manual_drop:
            log.append({
                "file_name": file_name,
                "category_id": None,
                "action": "drop_image_manual",
            })
            continue

        w, h = r["width"], r["height"]
        source_boxes = [list(box) for box in r["boxes"]]
        source_cats = list(r["category_ids"])

        # 수동 교정: 지정한 두 클래스의 bbox 교환
        if file_name == swap_file:
            cid_a, cid_b = swap_cids

            if source_cats.count(cid_a) != 1 or source_cats.count(cid_b) != 1:
                raise ValueError(
                    f"bbox 교환 대상이 각각 하나씩 있어야 합니다: "
                    f"{file_name}, category_ids={source_cats}"
                )

            i = source_cats.index(cid_a)
            j = source_cats.index(cid_b)

            source_boxes[i], source_boxes[j] = (
                source_boxes[j],
                source_boxes[i],
            )

            for cid in swap_cids:
                log.append({
                    "file_name": file_name,
                    "category_id": cid,
                    "action": "swap_bbox",
                })

        boxes, cats = [], []

        for (x1, y1, x2, y2), cid in zip(
            source_boxes, source_cats, strict=True
        ):
            if (file_name, cid) in manual_drop:
                log.append({
                    "file_name": file_name,
                    "category_id": cid,
                    "action": "drop_manual",
                })
                continue

            # 규칙 1: 이미지 밖으로 나간 부분을 이미지 경계로 자른다
            nx1, ny1 = max(0.0, x1), max(0.0, y1)
            nx2 = min(float(w), x2) if w else x2
            ny2 = min(float(h), y2) if h else y2

            if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                log.append({
                    "file_name": file_name,
                    "category_id": cid,
                    "action": "clip",
                })

            # 규칙 2: 너무 작거나 넓이가 없는 박스를 뺀다
            if nx2 - nx1 < MIN_SIZE or ny2 - ny1 < MIN_SIZE:
                log.append({
                    "file_name": file_name,
                    "category_id": cid,
                    "action": "drop_tiny",
                })
                continue

            boxes.append([nx1, ny1, nx2, ny2])
            cats.append(cid)

        # 규칙 3: 객체가 너무 많으면 검토 대상으로 기록
        if len(boxes) > MAX_PILLS_PER_IMAGE:
            log.append({
                "file_name": file_name,
                "category_id": None,
                "action": "check_too_many",
            })

        # 규칙 4: 박스가 하나도 없는 이미지는 제외
        if not boxes:
            log.append({
                "file_name": file_name,
                "category_id": None,
                "action": "drop_image_no_box",
            })
            continue

        cleaned.append({
            **r,
            "boxes": boxes,
            "category_ids": cats,
        })

    return cleaned, log


def to_coco(records: list[Record], categories: dict[int, str]) -> dict:
    images, annotations = [], []
    for r in records:
        images.append(
            {
                "id": r["image_id"],
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
            }
        )
        for (x1, y1, x2, y2), cid in zip(r["boxes"], r["category_ids"], strict=True):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": r["image_id"],
                    "category_id": cid,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "area": round((x2 - x1) * (y2 - y1), 1),
                    "iscrowd": 0,
                }
            )
    cats = [
        {"id": cid, "name": name, "supercategory": "pill"}
        for cid, name in sorted(categories.items())
    ]
    return {"images": images, "annotations": annotations, "categories": cats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--source", default=None, help="원본 어노테이션 (기본: config의 train_annotations)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    records, categories = load_records(args.source or d["train_annotations"])
    ann_dir = resolve_path(d["class_map"]).parent / "annotations"  # 기본: data/annotations
    cleaned, log = clean(records, load_manual_drop(ann_dir / "manual_drop.csv"))

    save_json(to_coco(cleaned, categories), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_report.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "category_id", "action"])
        writer.writeheader()
        writer.writerows(log)

    # 검증셋 고정: v0 분할을 새 버전으로 복사 (정제로 빠진 이미지는 학습 때 자동 제외됨)
    src_split = resolve_path(d["split_dir"]) / "v0"
    dst_split = resolve_path(d["split_dir"]) / args.version
    if src_split.exists() and not dst_split.exists():
        shutil.copytree(src_split, dst_split)

    actions: dict[str, int] = {}
    for row in log:
        actions[row["action"]] = actions.get(row["action"], 0) + 1
    print(f"원본 {len(records)}장 → 정제 {len(cleaned)}장, 수정 내역: {actions or '없음'}")
    print(
        f"저장: {ann_dir / (args.version + '.json')}, {args.version}_report.csv, splits/{args.version}/"
    )


if __name__ == "__main__":
    main()
