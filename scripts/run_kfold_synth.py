"""외부 데이터 + copy-paste 합성 버전(v3allcp 등)의 조합 단위 K-fold.

copy-paste는 학습셋 알약을 잘라 붙이므로, 공식 분할로 만든 합성 데이터를 K-fold에 그대로 쓰면
검증 fold 이미지의 알약이 학습에 섞인다(점수 부풀림). 그래서 fold마다 그 fold의 train으로
합성을 다시 만든 뒤 학습한다.

먼저:
    uv run python scripts/kfold_split.py --config configs/yolo_v1_clean.yaml --k 5   # v1_f0~4 (대회 이미지만)
    uv run python scripts/make_ext_folds.py --base v1 --ext v3allc --k 5            # v3allc_f0~4 (train에 AI Hub 추가)
실행:
    uv run python scripts/run_kfold_synth.py --config configs/yolo_m_26m_v3allcp.yaml --folds 0 1 2 3 4
    uv run python scripts/run_kfold_synth.py --config configs/yolo_m_26m_v3allcp.yaml --folds 0 --epochs 1   # 동작 확인
결과:
    data/synth/{synth}_f{i}/, data/annotations/{synth}_f{i}.json, data/splits/{synth}_f{i}/  (fold별 합성)
    outputs/{exp_name}_kfold_{synth}_f{i}_s{seed}/                                           (fold별 학습)
    outputs/kfold/{exp_name}_kfold_summary.csv                                               (평균 ± 표준편차)
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


def _call(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True, help="합성 버전으로 학습하는 설정 (예: yolo_m_26m_v3allcp)"
    )
    ap.add_argument(
        "--ext", default="v3allc", help="합성 전 외부 데이터 버전 (fold 분할 {ext}_f{i} 필요)"
    )
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None, help="설정값 대신 쓸 epoch (빠른 확인용)")
    ap.add_argument("--tag", default="kfold")
    args = ap.parse_args()

    base = load_config(args.config)
    synth = base["data"]["version"]  # 예: v3allcp
    split_dir = resolve_path(base["data"]["split_dir"])
    tmp_dir = OUTPUT_DIR / "kfold" / "configs"
    prefix = f"{base['exp_name']}_{args.tag}"
    if args.epochs:
        prefix += f"_e{args.epochs}"

    rows = []
    for i in args.folds:
        ext_v, syn_v = f"{args.ext}_f{i}", f"{synth}_f{i}"
        if not (split_dir / ext_v / "train.txt").exists():
            raise SystemExit(
                f"{split_dir / ext_v} 이 없습니다. kfold_split.py → make_ext_folds.py 먼저"
            )

        # 1) 이 fold의 train 알약으로만 합성 (검증 fold 이미지는 쓰지 않음)
        ann = resolve_path(f"data/annotations/{syn_v}.json")
        if not ann.exists():
            src = deepcopy(base)
            src["data"].update(
                train_images=f"data/external/{args.ext}/images",
                train_annotations=f"data/annotations/{args.ext}.json",
                version=ext_v,
            )
            save_yaml(src, tmp_dir / f"data_{ext_v}.yaml")
            _call(
                [
                    "scripts/make_copy_paste.py",
                    "--config",
                    str(tmp_dir / f"data_{ext_v}.yaml"),
                    "--version",
                    syn_v,
                ]
            )

        # 2) 학습 설정: fold 합성 데이터
        cfg = deepcopy(base)
        cfg["data"].update(
            train_images=f"data/synth/{syn_v}/images",
            train_annotations=f"data/annotations/{syn_v}.json",
            version=syn_v,
        )
        if args.epochs:
            cfg["train"]["epochs"] = args.epochs
        seed = cfg.get("seed", 42)
        cfg["exp_name"] = f"{prefix}_{syn_v}_s{seed}"
        cfg["note"] = (
            f"{base.get('note', '')} | {args.tag}: fold {i}, 합성은 fold train으로 새로 만듦"
        )
        if not yolo_data_yaml(cfg).exists():
            save_yaml(cfg, tmp_dir / f"{cfg['exp_name']}.yaml")
            _call(["scripts/export_yolo.py", "--config", str(tmp_dir / f"{cfg['exp_name']}.yaml")])

        print(f"\n===== {cfg['exp_name']} =====")
        s = run(cfg)
        rows.append(
            {
                "exp_name": cfg["exp_name"],
                "fold": i,
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
    print(f"\n===== 요약 ({len(rows)} fold) =====")
    for r in rows:
        print(
            f"fold {r['fold']}  mAP@[0.75:0.95]={r['map']:.4f}  mAP75={r['map_75']:.4f}  {r['train_min']}분"
        )
    vals = [r["map"] for r in rows]
    if len(vals) > 1:
        print(f"평균 {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
