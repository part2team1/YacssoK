"""스켈레톤 동작 확인용 합성 알약 데이터 생성기.

실제 대회 데이터 없이도 EDA → 학습 → 제출 → 데모가 끝까지 도는지 확인하기 위해 쓴다.
대회 데이터와 같은 COCO 형식이며, 알약 하나당 JSON 한 개로 나눠 저장해 "여러 JSON 합치기"도 함께 검증한다.

실행:
    uv run python scripts/make_sample_data.py --n-train 80 --n-test 12
결과: data/sample/ (train_images, train_annotations, test_images)
"""

from __future__ import annotations

import argparse
import random

import cv2
import numpy as np

from src.utils import resolve_path, save_json

# (category_id, 이름, 모양, 색 RGB)
PILLS = [
    (1001, "샘플정 원형흰색", "circle", (245, 245, 240)),
    (1002, "샘플정 원형노랑", "circle", (240, 215, 90)),
    (1003, "샘플정 타원분홍", "ellipse", (235, 160, 180)),
    (1004, "샘플정 타원주황", "ellipse", (240, 150, 70)),
    (1005, "샘플캡슐 청백", "capsule", (70, 110, 210)),
    (1006, "샘플캡슐 적황", "capsule", (200, 60, 60)),
    (1007, "샘플정 사각녹색", "square", (110, 180, 110)),
    (1008, "샘플정 원형갈색", "circle", (150, 100, 60)),
]


def draw_pill(
    img: np.ndarray, shape: str, color, cx: int, cy: int, size: int, angle: float
) -> tuple:
    c = tuple(int(v) for v in color)
    if shape == "circle":
        cv2.circle(img, (cx, cy), size, c, -1)
        cv2.circle(img, (cx, cy), size, (90, 90, 90), 1)
        return cx - size, cy - size, cx + size, cy + size
    if shape in ("ellipse", "capsule"):
        a, b = size, int(size * 0.55)
        cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, c, -1)
        if shape == "capsule":  # 반쪽 색을 다르게
            cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 180, (235, 235, 225), -1)
        cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, (90, 90, 90), 1)
        pts = cv2.ellipse2Poly((cx, cy), (a, b), int(angle), 0, 360, 10)
    else:
        rect = ((cx, cy), (size * 1.6, size * 1.6), angle)
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(img, [pts], c)
    x, y, w, h = cv2.boundingRect(pts)
    return x, y, x + w, y + h


def make_image(w: int, h: int, n_pills: int, rng: random.Random):
    bg = np.full((h, w, 3), rng.randint(40, 90), dtype=np.uint8)
    bg = cv2.GaussianBlur((bg + np.random.randint(0, 20, bg.shape)).astype(np.uint8), (5, 5), 0)
    items, placed = [], []
    for _ in range(n_pills * 10):
        if len(items) == n_pills:
            break
        cid, _, shape, color = rng.choice(PILLS)
        size = rng.randint(int(min(w, h) * 0.07), int(min(w, h) * 0.12))
        cx, cy = rng.randint(size * 2, w - size * 2), rng.randint(size * 2, h - size * 2)
        if any(abs(cx - px) < size * 2.6 and abs(cy - py) < size * 2.6 for px, py in placed):
            continue
        x1, y1, x2, y2 = draw_pill(bg, shape, color, cx, cy, size, rng.uniform(0, 180))
        placed.append((cx, cy))
        items.append((cid, [x1, y1, x2 - x1, y2 - y1]))
    return bg, items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--out", default="data/sample")
    args = ap.parse_args()
    rng = random.Random(42)
    np.random.seed(42)
    out = resolve_path(args.out)
    (out / "train_images").mkdir(parents=True, exist_ok=True)
    (out / "test_images").mkdir(parents=True, exist_ok=True)
    categories = [{"id": cid, "name": name, "supercategory": "pill"} for cid, name, _, _ in PILLS]
    w, h = 640, 480

    ann_id = 1
    for i in range(1, args.n_train + 1):
        img, items = make_image(w, h, rng.randint(1, 4), rng)
        file_name = f"sample_train_{i:04d}.png"
        cv2.imwrite(str(out / "train_images" / file_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for cid, bbox in items:  # 알약 하나당 JSON 하나 (대회 데이터처럼 흩어진 구조)
            save_json(
                {
                    "images": [{"id": i, "file_name": file_name, "width": w, "height": h}],
                    "annotations": [
                        {
                            "id": ann_id,
                            "image_id": i,
                            "category_id": cid,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [c for c in categories if c["id"] == cid],
                },
                out
                / "train_annotations"
                / f"{file_name[:-4]}_json"
                / f"{cid}"
                / f"{file_name[:-4]}_{cid}.json",
            )
            ann_id += 1
    for i in range(1, args.n_test + 1):
        img, _ = make_image(w, h, rng.randint(1, 4), rng)
        cv2.imwrite(str(out / "test_images" / f"{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(
        f"합성 데이터 생성: {out} (train {args.n_train}장, test {args.n_test}장, 클래스 {len(PILLS)}개)"
    )


if __name__ == "__main__":
    main()
