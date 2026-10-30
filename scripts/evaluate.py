"""학습 없이 평가만 다시 하기 (모델 A·B, 실험 P1: 임계값·NMS 조정).

실행:
    uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline
    uv run python scripts/evaluate.py --exp 1028_modelA_yolov8s_v0_baseline --nms-iou 0.6 --max-det 4
    uv run python scripts/evaluate.py --exp <실험> --class-metrics      # 클래스별 AP까지

결과: 화면 출력 + outputs/{exp}/eval_{설정}.json
학습된 가중치(best.pt)는 그대로 두고, 예측 설정만 바꿔 검증 점수를 다시 계산한다.
좋은 값을 찾으면 설정 파일의 eval 블록에 반영해 다음 학습·제출에 쓴다.
"""

from __future__ import annotations

import argparse

from src.infer import load_experiment
from src.interfaces import OUTPUT_DIR
from src.metrics import compute_map
from src.train import val_targets
from src.utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--conf", type=float, default=None, help="eval.conf_thr 대신 쓸 값")
    ap.add_argument("--nms-iou", type=float, default=None, help="eval.nms_iou 대신 쓸 값 (YOLO)")
    ap.add_argument("--max-det", type=int, default=None, help="eval.max_det 대신 쓸 값")
    ap.add_argument("--class-metrics", action="store_true", help="클래스별 AP 계산 (느림)")
    ap.add_argument("--val-config", default=None, help="이 설정의 data 블록(검증셋·정답)으로 채점")
    
    args = ap.parse_args()

    cfg, model = load_experiment(args.exp)
    ev = cfg["eval"]
    for key, value in (
        ("conf_thr", args.conf),
        ("nms_iou", args.nms_iou),
        ("max_det", args.max_det),
    ):
        if value is not None:
            ev[key] = value
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):  # Faster R-CNN
        model.model.roi_heads.detections_per_img = ev["max_det"]

    #paths, targets = val_targets(cfg)
    data_cfg = cfg
    if args.val_config:  # 모델은 그대로, 채점할 검증 데이터만 바꾼다
        data_cfg = {**cfg, "data": {**cfg["data"], **load_config(args.val_config)["data"]}}
        paths, targets = val_targets(data_cfg)    
    
    preds = model.predict(paths)
    metrics = compute_map(
        preds, targets, ev.get("iou_thresholds"), class_metrics=args.class_metrics
    )

    tag = f"conf{ev['conf_thr']}_nms{ev.get('nms_iou')}_det{ev['max_det']}"
    if args.val_config:
        tag += "_val" + data_cfg["data"]["version"]   
    save_json(
        {"exp_name": args.exp, "eval": ev, **metrics}, OUTPUT_DIR / args.exp / f"eval_{tag}.json"
    )
    print(f"[{args.exp}] {tag}")
    print(f"  mAP@[0.75:0.95] = {metrics.get('map')}   mAP75 = {metrics.get('map_75')}")
    print(f"  저장: outputs/{args.exp}/eval_{tag}.json")


if __name__ == "__main__":
    main()
