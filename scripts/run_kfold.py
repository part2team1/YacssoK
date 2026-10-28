"""K-fold · seed 반복 학습 (과적합·재현성 테스트).

실행 예:
    # 1) 조합 단위 5-fold  (먼저: uv run python scripts/kfold_split.py --k 5)
    uv run python scripts/run_kfold.py --config configs/yolo_m_v8m.yaml --folds 0 1 2 3 4 --tag kfold
    # 2) 같은 분할(공식 v0)에서 seed만 바꿔 3번
    uv run python scripts/run_kfold.py --config configs/yolo_m_v8m.yaml --seeds 42 0 1 --tag seed
    # 빠른 동작 확인
    uv run python scripts/run_kfold.py --config configs/yolo_m_v8m.yaml --folds 0 --epochs 3 --tag test

결과:
    outputs/{exp_name}/                 실행마다 일반 학습과 같은 형식 (results.csv 포함)
    outputs/experiment_log.jsonl        실행마다 한 줄씩 기록
    outputs/kfold/{tag}_summary.csv     실행별 점수 + 평균 ± 표준편차

읽는 법:
    - 표준편차가 크면 점수가 분할·seed에 따라 많이 흔들린다 → 모델 간 작은 차이는 믿기 어렵다
    - fold별 학습 곡선(notebooks/scratch/pegstar_curves.ipynb)으로 과적합 여부를 본다
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from copy import deepcopy

from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
from src.train import run
from src.utils import load_config, resolve_path, save_yaml, yolo_data_yaml


def ensure_yolo_data(cfg: dict) -> None:
    """이 데이터 버전의 YOLO 데이터가 없으면 export_yolo.py로 만든다."""
    if cfg["model"]["family"] != "yolo" or yolo_data_yaml(cfg).exists():
        return
    tmp = OUTPUT_DIR / "kfold" / "configs" / f"{cfg['exp_name']}.yaml"
    save_yaml(cfg, tmp)  # base 없이 완성된 설정 (export_yolo.py가 그대로 읽음)
    subprocess.run(
        [sys.executable, "scripts/export_yolo.py", "--config", str(tmp)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="예: 0 1 2 3 4")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="예: 42 0 1")
    ap.add_argument("--epochs", type=int, default=None, help="설정값 대신 쓸 epoch (빠른 확인용)")
    ap.add_argument("--tag", default="kfold", help="실험 이름과 요약 파일에 붙는 이름")
    args = ap.parse_args()

    base = load_config(args.config)
    base_version = base["data"]["version"]
    versions = [f"{base_version}_f{i}" for i in args.folds] if args.folds else [base_version]
    seeds = args.seeds or [base.get("seed", 42)]
    prefix = f"{base['exp_name']}_{args.tag}"

    rows = []
    for version in versions:
        split = resolve_path(base["data"]["split_dir"]) / version / "val.txt"
        if not split.exists():
            hint = (
                "먼저 실행: uv run python scripts/kfold_split.py --k 5"
                if args.folds
                else "공식 분할이 없습니다. git pull로 데이터 B의 분할을 받으세요"
            )
            raise SystemExit(f"{split} 이 없습니다. {hint}")
        for seed in seeds:
            cfg = deepcopy(base)
            cfg["data"]["version"] = version
            cfg["seed"] = seed
            if args.epochs:
                cfg["train"]["epochs"] = args.epochs
            cfg["exp_name"] = f"{prefix}_{version}_s{seed}"
            cfg["note"] = f"{base.get('note', '')} | {args.tag}: version={version}, seed={seed}"
            print(f"\n===== {cfg['exp_name']} =====")
            ensure_yolo_data(cfg)
            s = run(cfg)
            rows.append(
                {
                    "exp_name": cfg["exp_name"],
                    "version": version,
                    "seed": seed,
                    "map": s.get("map"),
                    "map_75": s.get("map_75"),
                    "train_min": s.get("train_min"),
                }
            )

    out = OUTPUT_DIR / "kfold" / f"{prefix}_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n===== 요약 ({len(rows)}회) =====")
    for r in rows:
        print(
            f"{r['version']:>8}  seed {r['seed']:>3}  mAP@[0.75:0.95]={r['map']:.4f}  mAP75={r['map_75']:.4f}"
        )
    for key in ("map", "map_75"):
        vals = [r[key] for r in rows if r[key] is not None]
        if len(vals) > 1:
            print(
                f"{key}: 평균 {statistics.mean(vals):.4f} ± 표준편차 {statistics.stdev(vals):.4f}  (최소 {min(vals):.4f}, 최대 {max(vals):.4f})"
            )
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
