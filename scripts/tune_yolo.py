"""YOLO 학습 설정 자동 탐색 (실험 T3, 모델 A). 시간이 많이 든다 (짧은 학습 x 반복 횟수).

Ultralytics의 model.tune()이 학습률·손실 가중치·증강 값을 조금씩 바꿔 가며 짧게 학습하고,
가장 좋았던 값을 best_hyperparameters.yaml로 남긴다. 그 값을 설정 파일의 yolo_args에 옮겨 본 학습한다.
(Ultralytics의 기준 점수는 0.1 x mAP50 + 0.9 x mAP50-95로 대회 지표와 다르지만 방향은 비슷하다)

실행:
    uv run python scripts/export_yolo.py --config configs/yolo_baseline.yaml   # data.yaml 준비
    uv run python scripts/tune_yolo.py --config configs/yolo_baseline.yaml --epochs 15 --iterations 30

결과: outputs/tune_<실험 이름>/tune/best_hyperparameters.yaml, tune_results.ndjson  (커밋하지 않음)
"""

from __future__ import annotations

import argparse

from src.interfaces import OUTPUT_DIR
from src.utils import apply_overrides, get_device, load_config, yolo_data_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=15, help="반복 한 번의 학습 epoch")
    ap.add_argument("--iterations", type=int, default=30, help="몇 가지 설정을 시도할지")
    ap.add_argument("--set", nargs="*", default=None, help="설정 값 바꾸기")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = apply_overrides(load_config(args.config), args.set)
    data_yaml = yolo_data_yaml(cfg)
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} 이 없습니다. 먼저 export_yolo.py를 실행하세요.")
    dev = get_device(cfg["train"].get("device"))
    out = OUTPUT_DIR / f"tune_{cfg['exp_name']}"
    model = YOLO(cfg["model"]["weights"])
    tune_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "iterations": args.iterations,
        "imgsz": cfg["data"]["img_size"],
        "batch": cfg["train"]["batch_size"],
        "workers": cfg["train"].get("num_workers", 4),
        "device": dev.index if dev.type == "cuda" else dev.type,
        "optimizer": cfg["train"].get("optimizer", "auto"),
        "project": str(out),
        "name": "tune",
        "plots": False,
        "save": False,
        "val": True,
    }
    tune_args.update(cfg.get("yolo_args") or {})  # 설정 파일의 yolo_args가 이긴다 (M0과 같은 방식)
    tune_args.update({"plots": False, "save": False})  # 반복마다 그림·가중치를 남기지 않음
    model.tune(**tune_args)
    print(
        f"\n결과: {out}/tune/best_hyperparameters.yaml 의 값을 새 설정 파일 yolo_args에 옮겨 본 학습하세요."
    )


if __name__ == "__main__":
    main()
