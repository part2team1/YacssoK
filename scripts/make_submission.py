"""Kaggle 제출 파일 생성 (Experimentation Lead).

실행:
    uv run python scripts/make_submission.py --exp 1028_modelA_yolov8s_v0_baseline
    uv run python scripts/make_submission.py --exp expA expB          # 두 모델 앙상블

결과: submissions/{exp}.csv

제출 형식은 대회 sample_submission.csv와 반드시 비교해서 맞춘다.
- 박스: 팀 표준 xyxy → 대회 형식 (기본: COCO xywh)
- 클래스: label_idx → 대회 원본 category_id
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.ensemble import ensemble_nms
from src.infer import predict_test
from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
from src.utils import (
    label_to_category,
    load_class_map,
    load_config,
    load_json,
    predictions_from_json,
    xyxy_to_xywh,
)

COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="실험 이름 (여러 개면 앙상블)")
    ap.add_argument("--reuse", action="store_true", help="이미 있는 preds_test.json 재사용")
    args = ap.parse_args()

    pred_lists = []
    for exp in args.exp:
        path = OUTPUT_DIR / exp / "preds_test.json"
        if args.reuse and path.exists():
            pred_lists.append(predictions_from_json(load_json(path)))
        else:
            pred_lists.append(predict_test(exp))
    preds = pred_lists[0] if len(pred_lists) == 1 else ensemble_nms(pred_lists)

    cfg = load_config(OUTPUT_DIR / args.exp[0] / "config.yaml")
    l2c = label_to_category(load_class_map(cfg["data"]["class_map"]))
    conf_thr = cfg.get("submission", {}).get("conf_thr", 0.0)

    rows = []
    for p in sorted(preds, key=lambda x: x["image_id"]):
        keep = p["scores"] >= conf_thr
        for box, score, label in zip(
            xyxy_to_xywh(p["boxes"][keep]), p["scores"][keep], p["labels"][keep], strict=True
        ):
            x, y, w, h = (round(float(v), 2) for v in box)
            rows.append(
                {
                    "annotation_id": len(rows) + 1,
                    "image_id": p["image_id"],
                    "category_id": l2c[int(label)],
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "score": round(float(score), 5),
                }
            )
    df = pd.DataFrame(rows, columns=COLUMNS)

    out = PROJECT_ROOT / "submissions" / ("+".join(args.exp) + ".csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"제출 파일: {out} ({len(df)}행, 이미지 {df['image_id'].nunique()}장)")
    print("Kaggle 제출 전 sample_submission.csv와 컬럼 이름·순서가 같은지 확인하세요.")


if __name__ == "__main__":
    main()
