"""K-fold 결과를 팀 공통 결과 표(마크다운)로 만든다.

실행:
    uv run python scripts/kfold_table.py --prefix 1029_pegstar_yolo26m_v0_S1_kfold --model YOLO26m --stage S1

입력: outputs/kfold/{prefix}_summary.csv (run_kfold.py 결과)
      + 실행별 outputs/{exp}/yolo_run/results.csv (실제 완료 epoch), config.yaml, metrics.json
      + data/splits/{version}/train.txt, val.txt (장수)
결과: 화면 출력 + outputs/kfold/{prefix}_table.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

HEADER = (
    "| 담당자 | 모델 | 단계 | fold | seed | train 장수 | val 장수 | 실제 완료 에폭 "
    "| mAP@[0.75:0.95] | mAP75 | 학습 시간(분) | 성공/실패·비고 |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def count_lines(path: Path) -> int | str:
    if not path.exists():
        return "-"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def epochs_done(exp: str) -> int | str:
    path = OUT / exp / "yolo_run" / "results.csv"
    if not path.exists():
        return "-"
    with open(path, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def split_dir(exp: str) -> Path:
    cfg_path = OUT / exp / "config.yaml"
    for line in cfg_path.read_text(encoding="utf-8").splitlines() if cfg_path.exists() else []:
        if line.strip().startswith("split_dir:"):
            return ROOT / line.split(":", 1)[1].strip()
    return ROOT / "data" / "splits"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="outputs/kfold/{prefix}_summary.csv 의 prefix")
    ap.add_argument("--owner", default="pegstar")
    ap.add_argument("--model", required=True, help="정확한 모델 이름 (예: YOLO26m)")
    ap.add_argument("--stage", required=True, help="S1 / S2 / S3 / S4")
    ap.add_argument("--epochs", type=int, default=50, help="설정 epoch (완료 여부 판단용)")
    args = ap.parse_args()

    summary = OUT / "kfold" / f"{args.prefix}_summary.csv"
    with open(summary, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    lines = [HEADER, SEP]
    maps, maps75 = [], []
    for r in rows:
        exp, version = r["exp_name"], r["version"]
        fold = int(version.rsplit("_f", 1)[1]) + 1 if "_f" in version else "-"
        sd = split_dir(exp) / version
        n_ep = epochs_done(exp)
        ok = (OUT / exp / "metrics.json").exists() and r["map"] not in ("", None)
        note = "성공" if ok else "실패"
        if ok and isinstance(n_ep, int) and n_ep < args.epochs:
            note += f" (조기 종료 {n_ep}/{args.epochs})"
        m = float(r["map"]) if ok else None
        m75 = float(r["map_75"]) if ok else None
        if m is not None:
            maps.append(m)
            maps75.append(m75)
        lines.append(
            f"| {args.owner} | {args.model} | {args.stage} | {fold} | {r['seed']} "
            f"| {count_lines(sd / 'train.txt')} | {count_lines(sd / 'val.txt')} | {n_ep} "
            f"| {m:.4f} | {m75:.4f} | {r['train_min']} | {note} |"
            if ok
            else f"| {args.owner} | {args.model} | {args.stage} | {fold} | {r['seed']} "
            f"| {count_lines(sd / 'train.txt')} | {count_lines(sd / 'val.txt')} | {n_ep} "
            f"| - | - | {r['train_min']} | {note} |"
        )
    if len(maps) > 1:
        lines.append(
            f"| {args.owner} | {args.model} | {args.stage} | 평균 ± 표준편차 | | | | "
            f"| {statistics.mean(maps):.4f} ± {statistics.stdev(maps):.4f} "
            f"| {statistics.mean(maps75):.4f} ± {statistics.stdev(maps75):.4f} | | |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    out = OUT / "kfold" / f"{args.prefix}_table.md"
    out.write_text(text, encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
