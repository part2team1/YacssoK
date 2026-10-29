"""외부 데이터(AI Hub 경구약제 이미지)를 학습셋에 더한 새 데이터 버전 만들기 (실험 D6, 데이터 B).

준비 (데이터 가이드 v2의 D6 ①~②):
    data/aihub/labels/TL_1_조합/...   AI Hub 라벨 JSON (압축 푼 폴더, 여러 개 가능)
    data/aihub/images/TS_1_조합/...   AI Hub 이미지 PNG
    ※ TL_2_조합·TS_2_조합은 대회 원본이라 내려받지도, 쓰지도 않는다.

실행:
    # 기존 클래스만 보강 (class_map 그대로, 먼저 이것부터)
    uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known
    # 학습에 없던 알약까지 추가 (data/class_map_v3all.json 이 새로 생김 → 모든 모델 재학습)
    uv run python scripts/make_external.py --config configs/base.yaml --version v3all --classes all

결과:
    data/external/v3/images/          원본 + AI Hub 이미지 (하드링크 또는 복사)  — 커밋하지 않음
    data/annotations/v3.json          원본 + AI Hub 어노테이션 (COCO)            — 커밋하지 않음
    data/annotations/v3_external.csv  추가한 AI Hub 이미지 목록 (파일 이름만)     — 커밋
    data/annotations/external_drop.csv (있으면 읽음) 라벨이 틀려 뺄 AI Hub 사진 file_name 목록 — 커밋
    data/splits/v3/                   train = 원본 train + AI Hub, val = 원본 val — 커밋
    data/class_map_{version}.json     (--classes all 일 때만) 늘어난 클래스 표    — 커밋

규칙
  - AI Hub 이미지는 학습셋에만 넣는다. 검증셋은 원본 그대로라 이전 실험과 점수를 비교할 수 있다.
  - 대회 학습·테스트 이미지와 내용이 같은 사진은 뺀다 (누수 방지).
  - 박스가 이미지 밖에 있거나, bbox가 4개 숫자가 아니거나, 알약이 4개를 넘는 사진은 뺀다.
  - AI Hub JSON의 categories는 모두 1 "Drug"라서, 알약 번호는 images의 drug_N(K-코드) 또는 dl_idx에서 만든다.
    어느 칸을 쓸지는 대회 데이터와 약 이름이 같은 알약으로 자동 확인한다 (--id-field로 직접 지정 가능).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.dataset import build_class_map, load_records, load_split, records_to_coco
from src.interfaces import MAX_PILLS_PER_IMAGE, Record
from src.utils import load_class_map, load_config, load_json, resolve_path, save_json

BANNED = ("TL_2_조합", "TS_2_조합")  # 대회 원본: 사용 금지


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # 하드링크: 같은 드라이브면 용량을 쓰지 않는다
    except OSError:
        shutil.copy(src, dst)


def _num(value) -> int | None:
    digits = re.sub(r"\D", "", str(value or ""))
    return int(digits) if digits else None


def _angle(file_name: str) -> str | None:
    """AI Hub 파일 이름 예: K-001900-016548_0_2_0_2_75_000_200.png → '75' (6번째 칸)."""
    parts = Path(file_name).stem.split("_")
    return parts[5] if len(parts) >= 8 and parts[5].isdigit() else None


def _thumb(path: Path) -> np.ndarray | None:
    """64x64 흑백 축소본 (내용 비교용)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)


def _dhash(thumb: np.ndarray) -> int:
    """밝기 차이로 만든 64비트 지문. 거의 같은 사진이면 지문도 거의 같다 (1차 후보 찾기용)."""
    small = cv2.resize(thumb.astype(np.uint8), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _same_photo(a: np.ndarray, b: np.ndarray) -> bool:
    """2차 확인: 축소본 픽셀 차이 평균이 아주 작으면 같은 사진으로 본다."""
    return float(np.abs(a - b).mean()) < 4.0


def read_aihub(label_dirs: list[Path]) -> tuple[list[dict], Counter]:
    """AI Hub 라벨 JSON(이미지 x 알약 하나당 한 개)을 읽어 알약 단위 목록으로."""
    pills, skipped = [], Counter()
    for root in label_dirs:
        if any(b in str(root) for b in BANNED):
            raise SystemExit(f"사용 금지 폴더입니다 (대회 원본): {root}")
        for jf in sorted(root.rglob("*.json")):
            try:
                data = load_json(jf)
                img = data["images"][0]
                ann = data["annotations"][0]
            except Exception:  # 깨진 JSON
                skipped["JSON 읽기 실패"] += 1
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped["bbox가 4개 숫자가 아님"] += 1
                continue
            pills.append(
                {
                    "file_name": img["file_name"],
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                    "bbox": [float(v) for v in bbox],
                    "dl_name": str(img.get("dl_name") or ""),
                    "drug_N": _num(img.get("drug_N")),
                    "dl_idx": _num(img.get("dl_idx")),
                }
            )
    return pills, skipped


def choose_id_rule(pills: list[dict], comp_names: dict[str, int], field: str, offset: int | None):
    """약 이름이 대회 데이터와 같은 알약으로 '어느 칸 + 몇을 더하면 대회 번호가 되는지' 고른다."""
    if field != "auto" and offset is not None:
        return field, offset, None
    fields = ["drug_N", "dl_idx"] if field == "auto" else [field]
    named = {p["dl_name"]: p for p in pills if p["dl_name"] in comp_names}
    best = None
    for f in fields:
        for k in (0, -1, 1) if offset is None else (offset,):
            ok = sum(
                1 for name, p in named.items() if p[f] is not None and p[f] + k == comp_names[name]
            )
            if best is None or ok > best[2]:
                best = (f, k, ok)
    return best[0], best[1], (best[2], len(named))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="출발 데이터 버전의 설정")
    ap.add_argument("--version", default="v3", help="새 데이터 버전 이름")
    ap.add_argument("--labels", nargs="+", default=["data/aihub/labels"], help="AI Hub 라벨 폴더")
    ap.add_argument("--images", nargs="+", default=["data/aihub/images"], help="AI Hub 이미지 폴더")
    ap.add_argument(
        "--classes",
        choices=["known", "all"],
        default="known",
        help="known: 대회 학습 클래스로만 된 사진만 / all: 새 알약도 추가 (class_map이 바뀜)",
    )
    ap.add_argument(
        "--angles", nargs="*", default=["70", "75", "90"], help="쓸 촬영 각도 (빈 값이면 전부)"
    )
    ap.add_argument(
        "--max-per-class", type=int, default=0, help="클래스당 추가 이미지 상한 (0=없음)"
    )
    ap.add_argument("--max-images", type=int, default=0, help="추가 이미지 전체 상한 (0=없음)")
    ap.add_argument("--id-field", choices=["auto", "drug_N", "dl_idx"], default="auto")
    ap.add_argument("--id-offset", type=int, default=None, help="--id-field와 함께 직접 지정")
    ap.add_argument("--no-dedup", action="store_true", help="테스트 이미지와 내용 비교를 건너뜀")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    records, categories = load_records(d["train_annotations"])
    split_dir = resolve_path(d["split_dir"])
    train = load_split(records, split_dir / d["version"] / "train.txt")
    comp_class_map = load_class_map(d["class_map"])
    known = {c["category_id"] for c in comp_class_map}
    comp_names = {c["name"]: c["category_id"] for c in comp_class_map}

    # 1) AI Hub 라벨 읽기 → 알약 번호 규칙 정하기
    pills, skipped = read_aihub([resolve_path(p) for p in args.labels])
    if not pills:
        raise SystemExit(f"AI Hub 라벨 JSON을 찾지 못했습니다: {args.labels}")
    field, offset, check = choose_id_rule(pills, comp_names, args.id_field, args.id_offset)
    print(f"AI Hub 라벨: 알약 {len(pills)}개 읽음")
    print(f"알약 번호 = {field} + ({offset})", end="")
    if check:
        print(f"  [이름이 같은 알약 {check[1]}종 중 {check[0]}종 번호 일치]")
        if check[1] and check[0] < check[1] * 0.9:
            print(
                "  [주의] 번호가 맞지 않는 알약이 많습니다. --id-field / --id-offset을 확인하세요."
            )
    else:
        print("  (직접 지정)")

    # 2) 이미지 단위로 묶기 + 검사
    by_image: dict[str, list[dict]] = defaultdict(list)
    for p in pills:
        by_image[p["file_name"]].append(p)
    image_paths = {}
    for root in args.images:
        root = resolve_path(root)
        if any(b in str(root) for b in BANNED):
            raise SystemExit(f"사용 금지 폴더입니다 (대회 원본): {root}")
        for path in root.rglob("*.png"):
            image_paths.setdefault(path.name, path)
    comp_files = {r["file_name"] for r in records}
    test_dir = resolve_path(d["test_images"])
    test_files = {p.name for p in test_dir.glob("*.png")} if test_dir.exists() else set()

    names = dict(categories)
    candidates = []
    angles = set(args.angles or [])
    # 사람이 확인해 뺄 AI Hub 사진 (review_labels·find_label_issues로 찾은 라벨 오류). 파일 이름만 — 커밋
    drop_path = resolve_path(d["class_map"]).parent / "annotations" / "external_drop.csv"
    drop: set[str] = set()
    if drop_path.exists():
        with open(drop_path, encoding="utf-8") as f:
            drop = {row["file_name"].strip() for row in csv.DictReader(f) if row.get("file_name")}
    for file_name, items in sorted(by_image.items()):
        if file_name in drop:
            skipped["사람이 뺌 (external_drop.csv)"] += 1
            continue
        if file_name in comp_files or file_name in test_files:
            skipped["대회 이미지와 이름이 같음"] += 1
            continue
        if file_name not in image_paths:
            skipped["이미지 파일 없음"] += 1
            continue
        if angles and _angle(file_name) not in angles:
            skipped["촬영 각도 제외"] += 1
            continue
        w, h = items[0]["width"], items[0]["height"]
        boxes, cats, bad = [], [], False
        for p in items:
            x, y, bw, bh = p["bbox"]
            cid = None if p[field] is None else p[field] + offset
            if cid is None or bw <= 1 or bh <= 1 or x < 0 or y < 0 or x + bw > w or y + bh > h:
                bad = True
                break
            box = [round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1)]
            if box not in boxes:
                boxes.append(box)
                cats.append(cid)
                names.setdefault(cid, p["dl_name"] or str(cid))
        if bad:
            skipped["박스 오류(이미지 밖·크기 0·번호 없음)"] += 1
            continue
        if len(boxes) > MAX_PILLS_PER_IMAGE:
            skipped[f"알약 {MAX_PILLS_PER_IMAGE}개 초과"] += 1
            continue
        if args.classes == "known" and not set(cats) <= known:
            skipped["학습에 없는 알약 포함 (--classes known)"] += 1
            continue
        candidates.append((file_name, w, h, boxes, cats))

    # 3) 테스트 이미지와 내용이 같은 사진 빼기 (지문으로 후보 → 축소본 픽셀로 확인)
    if not args.no_dedup and test_files and candidates:
        print(f"테스트 이미지와 내용 비교 중... (후보 {len(candidates)}장, 몇 분 걸릴 수 있음)")
        tests = [t for p in sorted(test_dir.glob("*.png")) if (t := _thumb(p)) is not None]
        test_hashes = [_dhash(t) for t in tests]
        kept = []
        for c in candidates:
            thumb = _thumb(image_paths[c[0]])
            if thumb is not None:
                h = _dhash(thumb)
                if any(
                    (h ^ th).bit_count() <= 6 and _same_photo(thumb, t)
                    for th, t in zip(test_hashes, tests, strict=True)
                ):
                    skipped["테스트 이미지와 같은 사진"] += 1
                    continue
            kept.append(c)
        candidates = kept

    # 4) 상한 적용 (드문 클래스가 든 사진부터)
    base_count = Counter(c for r in train for c in r["category_ids"])
    candidates.sort(key=lambda c: min(base_count.get(cid, 0) for cid in c[4]))
    added_count: Counter = Counter()
    next_id = max((r["image_id"] for r in records), default=0) + 1
    external: list[Record] = []
    for file_name, w, h, boxes, cats in candidates:
        if args.max_images and len(external) >= args.max_images:
            break
        if args.max_per_class and all(added_count[c] >= args.max_per_class for c in cats):
            continue
        added_count.update(set(cats))
        external.append(
            Record(
                image_id=next_id + len(external),  # 원본 id와 겹치지 않는 새 번호
                file_name=file_name,
                width=w,
                height=h,
                boxes=boxes,
                category_ids=cats,
            )
        )
    if not external:
        raise SystemExit(
            "추가할 이미지가 없습니다. 아래 제외 사유를 확인하세요.\n" + str(dict(skipped))
        )

    # 5) 저장: 이미지 폴더, 어노테이션, 분할, (all이면) 새 class_map
    data_root = resolve_path(d["class_map"]).parent
    out_dir = data_root / "external" / args.version / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = resolve_path(d["train_images"])
    for r in records:  # 원본 전체(검증 포함)도 같은 폴더에 연결해 train_images 하나로 읽게 한다
        _link_or_copy(src_dir / r["file_name"], out_dir / r["file_name"])
    for r in external:
        _link_or_copy(image_paths[r["file_name"]], out_dir / r["file_name"])

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    all_records = records + external
    save_json(records_to_coco(all_records, names), ann_dir / f"{args.version}.json")
    with open(ann_dir / f"{args.version}_external.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "n_pills", "category_ids"])
        for r in external:
            writer.writerow(
                [r["file_name"], len(r["boxes"]), " ".join(map(str, r["category_ids"]))]
            )

    new_split = split_dir / args.version
    new_split.mkdir(parents=True, exist_ok=True)
    (new_split / "train.txt").write_text(
        "\n".join([r["file_name"] for r in train] + [r["file_name"] for r in external]) + "\n",
        encoding="utf-8",
    )
    shutil.copy(split_dir / d["version"] / "val.txt", new_split / "val.txt")

    class_map_path = d["class_map"]
    new_classes = sorted({c for r in external for c in r["category_ids"]} - known)
    if args.classes == "all":
        # 기존 클래스 + 새 클래스. build_class_map은 번호순으로 label_idx를 다시 매긴다
        class_map = build_class_map(all_records, names)
        cm_file = data_root / f"class_map_{args.version}.json"
        save_json(class_map, cm_file)
        class_map_path = cm_file.relative_to(resolve_path(".")).as_posix()

    print("\n제외한 AI Hub 이미지:", dict(skipped) or "없음")
    print(
        f"추가한 AI Hub 이미지 {len(external)}장 (알약 {sum(len(r['boxes']) for r in external)}개)"
    )
    print(f"학습셋: {len(train)}장 → {len(train) + len(external)}장, 검증셋은 그대로")
    print(
        f"새 클래스 {len(new_classes)}개"
        + (" (class_map 그대로)" if args.classes == "known" else "")
    )
    rel = data_root.relative_to(resolve_path(".")).as_posix()
    print("설정 파일 data 블록:")
    print(f"  train_images: {rel}/external/{args.version}/images")
    print(f"  train_annotations: {rel}/annotations/{args.version}.json")
    print(f"  class_map: {class_map_path}")
    print(f"  version: {args.version}")


if __name__ == "__main__":
    main()
