"""정답 박스·라벨을 이미지에 그려 눈으로 확인한다. 합성 이미지(D2)는 붙이기 전과 나란히 그리고 자동 점검한다.

실행:
    uv run python scripts/review_labels.py --config configs/base.yaml                  # 원본 v0 전체
    uv run python scripts/review_labels.py --config configs/base.yaml --split val      # 검증셋만
    uv run python scripts/review_labels.py --config configs/yolo_v1_clean.yaml         # 정제 v1 확인 (D5·D10)
    uv run python scripts/review_labels.py --config configs/yolo_v2_copypaste.yaml --images synth   # 합성만 (D2)
    uv run python scripts/review_labels.py --config configs/yolo_v2_copypaste.yaml --images synth --only-flagged

옵션
    --split   all / train / val            분할 파일 기준으로 고르기
    --images  all / original / synth        원본만, 합성(cp_*)만, 둘 다
    --only-flagged                          합성 이미지 중 자동 점검에 걸린 것만 그림 저장

결과 (대회 이미지이므로 커밋하지 않음): outputs/label_review/<버전>_<split>_<images>/
    <이미지>.jpg   원본: 박스 + label_idx 번호, 아래쪽에 파일 이름·알약 수·번호 목록
                   합성: 왼쪽 = 붙이기 전 배경 원본, 오른쪽 = 합성 결과
                         초록 = 배경에 원래 있던 정답, 빨강 = 새로 붙인 알약,
                         노랑 ? = 붙인 자리 아래에 무언가 있었던 것으로 의심되는 곳
    index.csv      이미지별 알약 수, 번호, 약 이름
    synth_check.csv  (합성이 있을 때) 붙인 알약마다 covered·edge_diff·flag

합성 자동 점검
    covered    붙인 자리의 배경 원본에서 배경색과 다른 픽셀 비율. --covered(기본 0.15) 이상이면 flag
               → 라벨이 빠진 알약을 덮었을 가능성 (그 배경 사진은 D5·D10으로 정제)
    edge_diff  붙인 조각 테두리 색과 배경색 차이. 클수록 네모난 경계가 눈에 띔
확인할 것 (원본): 박스 없는 알약(라벨 빠짐), 알약을 벗어난 박스, 같은 모양인데 번호가 다른 알약.
찾은 것은 manual_drop.csv(빼기, D5) 또는 manual_fix.csv(고치기, D10)에 적는다.
"""

from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

from src.dataset import load_records, load_split, read_image_rgb
from src.interfaces import OUTPUT_DIR
from src.utils import category_to_label, load_class_map, load_config, resolve_path

COLORS = [(75, 25, 230), (75, 180, 60), (200, 130, 0), (48, 130, 245), (180, 30, 145)]  # BGR
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 230), (0, 210, 255)  # BGR


def background_color(img: np.ndarray) -> np.ndarray:
    """사진 가장자리 띠의 중앙값 색 = 촬영 배경색으로 본다."""
    m = max(4, min(img.shape[:2]) // 40)
    border = np.concatenate(
        [img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3), img[:, :m].reshape(-1, 3)]
    )
    return np.median(border, axis=0)


def foreground_ratio(region: np.ndarray, bg: np.ndarray, tol: float = 40.0) -> float:
    """영역 안에서 배경색과 tol 이상 다른 픽셀의 비율 (0~1)."""
    if region.size == 0:
        return 0.0
    diff = np.abs(region.astype(np.float32) - bg).sum(axis=2)
    return float((diff > tol).mean())


def edge_diff(crop: np.ndarray, bg: np.ndarray) -> float:
    """붙인 조각의 테두리 1픽셀 평균색과 배경색의 차이 (색 3개 합)."""
    if crop.size == 0:
        return 0.0
    edge = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.float32)
    return float(np.abs(edge.mean(axis=0) - bg).sum())


def draw(img: np.ndarray, boxes, labels, color=None, thick: int = 3) -> np.ndarray:
    """BGR 이미지에 박스와 번호를 그린 사본. color가 없으면 번호마다 다른 색."""
    out = img.copy()
    for (x1, y1, x2, y2), lbl in zip(boxes, labels, strict=True):
        c = color if color is not None else COLORS[int(lbl) % len(COLORS)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, thick)
        cv2.putText(out, str(lbl), (int(x1) + 4, int(y1) + 30), 0, 1.0, c, 3)
    return out


def footer(img: np.ndarray, text: str) -> np.ndarray:
    y = img.shape[0] - 15
    cv2.putText(img, text, (10, y), 0, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), 0, 0.7, (255, 255, 255), 2)
    return img


def save(img: np.ndarray, path) -> None:
    ok, buf = cv2.imencode(".jpg", img)  # 한글 경로에서도 저장되게
    buf.tofile(str(path))


def load_backgrounds(d: dict) -> dict[str, str]:
    """합성 이미지 → 배경 원본 이름 (make_copy_paste.py가 남긴 <버전>_copy_paste.csv)."""
    log = resolve_path(d["class_map"]).parent / "annotations" / f"{d['version']}_copy_paste.csv"
    if not log.exists():
        return {}
    with open(log, encoding="utf-8") as f:
        return {row["synth_image"]: row["background_image"] for row in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", choices=["all", "train", "val"], default="all")
    ap.add_argument("--images", choices=["all", "original", "synth"], default="all")
    ap.add_argument(
        "--only-flagged", action="store_true", help="합성 중 점검에 걸린 것만 그림 저장"
    )
    ap.add_argument("--covered", type=float, default=0.15, help="합성 covered 의심 기준 비율")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    all_records, _ = load_records(d["train_annotations"])
    by_name = {r["file_name"]: r for r in all_records}
    records = all_records
    if args.split != "all":
        records = load_split(records, f"{d['split_dir']}/{d['version']}/{args.split}.txt")
    backgrounds = load_backgrounds(d)
    if args.images == "synth":
        records = [r for r in records if r["file_name"] in backgrounds]
        if not records:
            raise SystemExit(
                "합성 이미지가 없습니다. D2 설정 파일(yolo_v2_copypaste.yaml)인지 확인하세요."
            )
    elif args.images == "original":
        records = [r for r in records if r["file_name"] not in backgrounds]

    class_map = load_class_map(d["class_map"])
    c2l = category_to_label(class_map)
    names = {c["label_idx"]: c["name"] for c in class_map}
    image_dir = resolve_path(d["train_images"])
    out_dir = OUTPUT_DIR / "label_review" / f"{d['version']}_{args.split}_{args.images}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):  # 이전 실행 그림이 섞이지 않게
        old.unlink()

    rows, synth_rows = [], []
    n_flagged = 0
    for r in records:
        img = cv2.cvtColor(read_image_rgb(image_dir / r["file_name"]), cv2.COLOR_RGB2BGR)
        labels = [c2l[c] for c in r["category_ids"]]
        rows.append(
            {
                "file_name": r["file_name"],
                "n_pills": len(labels),
                "label_idx": " ".join(map(str, labels)),
                "names": " | ".join(names[lbl] for lbl in labels),
                "synth": "Y" if r["file_name"] in backgrounds else "",
            }
        )
        info = f"{r['file_name'][:60]}  pills={len(labels)}  labels={labels}"
        if r["file_name"] not in backgrounds:  # 원본 이미지
            save(
                footer(draw(img, r["boxes"], labels), info),
                out_dir / f"{r['file_name'].rsplit('.', 1)[0]}.jpg",
            )
            continue

        # 합성 이미지: 붙이기 전 배경과 나란히 + 자동 점검
        b_rec = by_name[backgrounds[r["file_name"]]]
        b_img = cv2.cvtColor(read_image_rgb(image_dir / b_rec["file_name"]), cv2.COLOR_RGB2BGR)
        bg = background_color(b_img)
        orig = [tuple(b) for b in b_rec["boxes"]]
        b_labels = [c2l[c] for c in b_rec["category_ids"]]
        pasted = [
            (b, c)
            for b, c in zip(r["boxes"], r["category_ids"], strict=True)
            if tuple(b) not in orig
        ]
        flagged = []
        for (x1, y1, x2, y2), cid in pasted:
            ys, xs = slice(int(y1), int(y2)), slice(int(x1), int(x2))
            cov = foreground_ratio(b_img[ys, xs], bg)
            if cov >= args.covered:
                flagged.append((x1, y1, x2, y2))
            synth_rows.append(
                {
                    "synth_image": r["file_name"],
                    "background_image": b_rec["file_name"],
                    "category_id": cid,
                    "box": " ".join(f"{v:.1f}" for v in (x1, y1, x2, y2)),
                    "covered": f"{cov:.2f}",
                    "edge_diff": f"{edge_diff(img[ys, xs], bg):.0f}",
                    "flag": "covered" if cov >= args.covered else "",
                }
            )
        n_flagged += len(flagged)
        if args.only_flagged and not flagged:
            continue
        left = draw(b_img, b_rec["boxes"], b_labels, GREEN)
        left = draw(left, flagged, ["?"] * len(flagged), YELLOW, thick=4)
        right = draw(img, b_rec["boxes"], b_labels, GREEN)
        right = draw(right, [b for b, _ in pasted], [c2l[c] for _, c in pasted], RED)
        gap = np.full((img.shape[0], 12, 3), 255, np.uint8)
        pair = np.concatenate([footer(left, "before"), gap, footer(right, info)], axis=1)
        save(pair, out_dir / f"{r['file_name'].rsplit('.', 1)[0]}.jpg")

    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file_name"])
        writer.writeheader()
        writer.writerows(rows)
    n_synth = sum(1 for row in rows if row["synth"])
    print(f"이미지 {len(rows)}장 (원본 {len(rows) - n_synth}, 합성 {n_synth}) 저장: {out_dir}")
    if synth_rows:
        with open(out_dir / "synth_check.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(synth_rows[0]))
            writer.writeheader()
            writer.writerows(synth_rows)
        print(
            f"합성 점검: 붙인 알약 {len(synth_rows)}개, covered 의심 {n_flagged}개 → synth_check.csv"
        )


if __name__ == "__main__":
    main()