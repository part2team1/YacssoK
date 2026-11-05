"""Kaggle 제출용 파일 3개를 한 폴더에 모은다 (제출은 PM이 한다).

실행 (먼저 make_submission.py로 CSV를 만든 뒤):
    uv run python scripts/pack_submission.py --exp 1029_pegstar_yolo26m_v0_S2 \
        --kfold 1029_pegstar_yolo26m_v0_S2_kfold

결과: submissions/{exp}/
    {exp}.csv      제출용 CSV
    config.yaml    학습에 쓴 설정 (outputs/{exp}/config.yaml)
    score.md       모델·실험 이름과 검증 점수 (공식 v0 + K-fold 평균)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="제출에 쓴 실험 이름")
    ap.add_argument("--kfold", default=None, help="같은 설정의 K-fold 요약 prefix (점수 기록용)")
    args = ap.parse_args()

    csv_path = ROOT / "submissions" / f"{args.exp}.csv"
    cfg_path = OUT / args.exp / "config.yaml"
    metrics_path = OUT / args.exp / "metrics.json"
    for p in (csv_path, cfg_path, metrics_path):
        if not p.exists():
            raise SystemExit(
                f"{p} 이 없습니다. 학습(run_train.py)과 make_submission.py를 먼저 실행하세요."
            )

    dest = ROOT / "submissions" / args.exp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_path, dest / csv_path.name)
    shutil.copy(cfg_path, dest / "config.yaml")

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines = [
        f"# {args.exp}",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 실험 이름 | {args.exp} |",
        f"| 모델 | {m.get('model')} |",
        f"| 담당자 | {m.get('owner')} |",
        f"| 데이터 버전 | {m.get('data_version')} |",
        f"| epoch / img | {m.get('epochs')} / {m.get('img_size')} |",
        f"| 검증 mAP@[0.75:0.95] (공식 v0, 47장) | {m.get('map')} |",
        f"| 검증 mAP75 (공식 v0) | {m.get('map_75')} |",
        f"| 학습 시간(분) | {m.get('train_min')} |",
        f"| 학습 완료 | {m.get('finished_at')} |",
        f"| 메모 | {m.get('note')} |",
    ]
    if args.kfold:
        with open(OUT / "kfold" / f"{args.kfold}_summary.csv", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r["map"]]
        vals = [float(r["map"]) for r in rows]
        vals75 = [float(r["map_75"]) for r in rows]
        lines += [
            f"| 조합 단위 {len(vals)}-fold mAP@[0.75:0.95] | "
            f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} |",
            f"| 조합 단위 {len(vals)}-fold mAP75 | "
            f"{statistics.mean(vals75):.4f} ± {statistics.stdev(vals75):.4f} |",
        ]
    lines += [
        "",
        "공식 v0 점수는 같은 조합이 학습·검증에 섞인 분할이라 실제보다 높게 나온다. "
        "모델 비교는 K-fold 점수로 한다.",
    ]
    (dest / "score.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"제출 묶음: {dest}")
    for p in sorted(dest.iterdir()):
        print(f"  {p.name}")
    print("CSV는 커밋하지 말고 PM에게 전달하세요.")


if __name__ == "__main__":
    main()
