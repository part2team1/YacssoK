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


def submission_configs(experiments: list[str]) -> list[dict]:
    """모델 내부 라벨 번호가 같은 약을 뜻할 때만 앙상블을 허용한다."""
    configs = [load_config(OUTPUT_DIR / exp / "config.yaml") for exp in experiments]
    mappings = [label_to_category(load_class_map(cfg["data"]["class_map"])) for cfg in configs]
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError(
            "서로 다른 클래스 번호표의 모델은 합칠 수 없습니다. "
            "같은 label_idx→category_id를 쓰는 실험만 선택하세요."
        )
    return configs


def validate_predictions(pred_lists, valid_labels) -> None:
    """캐시가 다른 사진이나 잘못된 라벨을 포함하면 CSV를 만들기 전에 중단한다."""
    reference = None
    for preds in pred_lists:
        images = [(p["file_name"], p["image_id"]) for p in preds]
        if len({name for name, _ in images}) != len(images):
            raise ValueError("예측 목록에 같은 사진이 여러 번 있습니다.")
        if reference is not None and set(images) != reference:
            raise ValueError("앙상블의 테스트 사진 목록 또는 image_id가 다릅니다.")
        reference = set(images)
        for pred in preds:
            if any(int(label) not in valid_labels for label in pred["labels"]):
                raise ValueError("예측 라벨이 실험의 클래스 번호표에 없습니다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True, help="실험 이름 (여러 개면 앙상블)")
    ap.add_argument("--reuse", action="store_true", help="이미 있는 preds_test.json 재사용")
    args = ap.parse_args()

    configs = submission_configs(args.exp)  # 추론·출력보다 먼저 검사한다.
    cfg = configs[0]
    l2c = label_to_category(load_class_map(cfg["data"]["class_map"]))
    pred_lists = []
    for exp in args.exp:
        path = OUTPUT_DIR / exp / "preds_test.json"
        if args.reuse and path.exists():
            pred_lists.append(predictions_from_json(load_json(path)))
        else:
            pred_lists.append(predict_test(exp))
    validate_predictions(pred_lists, l2c)
    preds = pred_lists[0] if len(pred_lists) == 1 else ensemble_nms(pred_lists)

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
