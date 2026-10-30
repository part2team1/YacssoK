"""학습 실행 진입점.

실행:
    uv run python scripts/run_train.py --config configs/yolo_baseline.yaml
    uv run python scripts/run_train.py --config configs/yolo_baseline.yaml --epochs 3   # 빠른 확인

결과: outputs/{exp_name}/ (config.yaml, best.pt, metrics.json, preds_val.json)
"""

from __future__ import annotations

import argparse

from src.train import run
#from src.utils import load_config
from src.utils import apply_overrides, load_config

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--epochs", type=int, default=None, help="설정값 대신 쓸 epoch 수 (빠른 확인용)"
    )
    ap.add_argument(
        "--exp-name", default=None, help="설정값 대신 쓸 실험 이름"
        "--set", nargs="*", default=None, help="설정 값 바꾸기 (예: data.version=v0_f1 seed=7)"
    )
    args = ap.parse_args()

    #cfg = load_config(args.config)
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.exp_name:
        cfg["exp_name"] = args.exp_name
    run(cfg)


if __name__ == "__main__":
    main()
