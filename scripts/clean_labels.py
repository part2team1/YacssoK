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
빼지 않고 고치려면 data/annotations/manual_fix.csv (실험 D10) 에 적는다:
    file_name,category_id,action,new_value
    a.png,1899,class,3482                  ← 1899로 잘못 적힌 박스를 3482로
    a.png,3482,box,100.0 120.0 260.0 300.0  ← 박스 좌표를 x1 y1 x2 y2(픽셀)로 바꿈
    b.png,1899,add,40.0 50.0 200.0 210.0    ← 빠진 알약 박스 추가
고치기가 먼저 적용되고, manual_drop은 고친 뒤의 category_id 기준으로 뺀다.

수정:
manual_fix에 class·box·add가 있는 이미지는 manual_drop의 이미지 전체
제외 지시를 무시한다. 수정 기록은 해당 이미지의 삭제 사유를 모두
해결한 경우에만 작성한다. 클래스별 박스 삭제 지시는 그대로 적용한다.
"""

from __future__ import annotations

import argparse
import csv
import shutil

from src.dataset import load_records
from src.dataset import records_to_coco as to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_config, resolve_path, save_json

MIN_SIZE = 3  # 너비·높이가 이 픽셀보다 작으면 잘못된 박스로 본다


def load_manual_drop(path) -> set[tuple[str, int | None]]:
    """(file_name, category_id) 목록. category_id를 all 로 적으면 그 이미지 전체를 뺀다
    (라벨이 빠진 알약이 있는 이미지처럼 박스 하나로는 고칠 수 없는 경우)."""
    if not path.exists():
        return set()
    out: set[tuple[str, int | None]] = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("category_id") or "").strip()
            out.add((row["file_name"].strip(), None if cid.lower() in ("", "all") else int(cid)))
    return out


def load_manual_fix(path) -> list[dict]:
    """manual_fix.csv 의 줄 목록 (실험 D10). 파일이 없으면 빈 목록."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [
            {k: (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(f)
            if (row.get("file_name") or "").strip()
        ]


def apply_fixes(r: Record, fixes: list[dict], log: list[dict]) -> Record:
    """한 이미지에 manual_fix 줄을 적용한 사본을 돌려준다 (실험 D10)."""
    boxes = [list(b) for b in r["boxes"]]
    cats = list(r["category_ids"])
    for fx in fixes:
        cid, action, value = int(fx["category_id"]), fx["action"], fx.get("new_value", "")
        idx = [i for i, c in enumerate(cats) if c == cid]
        if action == "add":
            boxes.append([float(v) for v in value.split()])
            cats.append(cid)
        elif (
            action in {"class", "box"} and not idx
        ):  # 수정 대상이 없는데도 삭제만 취소되는 상황 방지
            raise ValueError(
                f"manual_fix.csv 수정 대상 없음: "
                f"{r['file_name']}, category_id={cid}, action={action}"
            )
        elif action == "class":
            for i in idx:
                cats[i] = int(value)
        elif action == "box":
            boxes[idx[0]] = [float(v) for v in value.split()]
        else:
            continue  # check 등 고치지 않는 줄
        log.append({"file_name": r["file_name"], "category_id": cid, "action": f"fix_{action}"})
    return {**r, "boxes": boxes, "category_ids": cats}


def clean(
    records: list[Record], manual_drop: set, manual_fix: list[dict] | None = None
) -> tuple[list[Record], list[dict]]:
    """규칙대로 박스를 고치거나 빼고, 바꾼 내용을 log로 남긴다."""
    cleaned, log = [], []
    fixes_by_file: dict[str, list[dict]] = {}
    for fx in manual_fix or []:
        fixes_by_file.setdefault(fx["file_name"], []).append(fx)

    for r in records:
        file_name = r["file_name"]
        fixes = fixes_by_file.get(file_name, [])

        # 실제 수정 지시가 있으면 이미지 전체 삭제를 취소한다.
        # check만 있는 이미지는 삭제를 유지한다.
        has_fix = any(fx["action"] in {"class", "box", "add"} for fx in fixes)

        if (file_name, None) in manual_drop and not has_fix:
            log.append(
                {
                    "file_name": file_name,
                    "category_id": None,
                    "action": "drop_image_manual",
                }
            )
            continue

        if fixes:
            r = apply_fixes(r, fixes, log)

        w, h = r["width"], r["height"]
        boxes, cats = [], []
        for (x1, y1, x2, y2), cid in zip(r["boxes"], r["category_ids"], strict=True):
            if (r["file_name"], cid) in manual_drop:
                log.append(
                    {"file_name": r["file_name"], "category_id": cid, "action": "drop_manual"}
                )
                continue
            # 규칙 1: 이미지 밖으로 나간 부분은 이미지 경계로 자른다
            nx1, ny1 = max(0.0, x1), max(0.0, y1)
            nx2, ny2 = (min(float(w), x2) if w else x2), (min(float(h), y2) if h else y2)
            if (nx1, ny1, nx2, ny2) != (x1, y1, x2, y2):
                log.append({"file_name": r["file_name"], "category_id": cid, "action": "clip"})
            # 규칙 2: 너무 작거나 넓이가 없는 박스는 뺀다
            if nx2 - nx1 < MIN_SIZE or ny2 - ny1 < MIN_SIZE:
                log.append({"file_name": r["file_name"], "category_id": cid, "action": "drop_tiny"})
                continue
            boxes.append([nx1, ny1, nx2, ny2])
            cats.append(cid)
        # 규칙 3: 과제 조건(최대 4개)을 넘으면 기록만 남긴다 (자동 삭제하지 않고 사람이 확인)
        if len(boxes) > MAX_PILLS_PER_IMAGE:
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "check_too_many"}
            )
        # 규칙 4: 박스가 하나도 없는 이미지는 학습에서 뺀다
        if not boxes:
            log.append(
                {"file_name": r["file_name"], "category_id": None, "action": "drop_image_no_box"}
            )
            continue
        cleaned.append({**r, "boxes": boxes, "category_ids": cats})
    return cleaned, log


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
    manual_fix = load_manual_fix(ann_dir / "manual_fix.csv")
    unknown = {
        int(v)
        for fx in manual_fix
        for v in ([fx["category_id"]] + ([fx["new_value"]] if fx["action"] == "class" else []))
        if int(v) not in categories
    }
    if unknown:
        raise SystemExit(
            f"manual_fix.csv에 학습 데이터에 없는 category_id가 있습니다: {sorted(unknown)}"
        )
    cleaned, log = clean(records, load_manual_drop(ann_dir / "manual_drop.csv"), manual_fix)

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
