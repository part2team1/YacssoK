"""여러 모델 × 여러 데이터 실험을 한 번에 학습하고 결과를 표로 모은다 (데이터 실험 적용 가이드).

계획 파일(configs/matrix_data.yaml)에 모델 설정과 데이터 실험 묶음(--set 값)을 적어 두면
모델마다 [기준 + 실험들]을 차례로 export_yolo → run_train 한다. bash·PowerShell 어디서나 같은 명령이다.

실행:
    uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --dry-run      # 명령만 보기
    uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --epochs 1     # 1 epoch 동작 확인
    uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml                # 전체
    uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --models yolo11s --runs base D2
    uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --summary      # 학습 없이 표만

동작
  - 실험 이름 = {prefix}_{모델}_{실험}. outputs/<이름>/metrics.json 이 있으면 건너뛴다(이어 하기).
    다시 학습하려면 --force.
  - 실험 하나가 실패해도 다음 실험으로 넘어가고, 끝에 실패 목록을 보여 준다.
  - 데이터 파일(train_annotations)이 없는 실험은 "데이터 없음"으로 건너뛴다 (데이터 버전을 먼저 만든다).
  - --epochs 를 주면 실험 이름 끝에 _e{N}이 붙어 본 실험 결과와 섞이지 않는다.
  - runs에 {ocr: <실험>} 으로 적은 줄은 D11: 학습 없이 그 실험 모델 위에 각인 OCR 재분류를 얹는다
    (uv sync --extra ocr, data/drug_info.json 필요. 탐지 실험이 먼저 끝나 있어야 함).
결과: 화면 표 + outputs/matrix_{prefix}.csv (모델 × 실험의 map, 기준 대비 변화) — 커밋하지 않음
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys

from src.interfaces import OUTPUT_DIR
from src.utils import apply_overrides, load_config, load_json, resolve_path

PY = [sys.executable]


def _sets(plan: dict, run: str) -> list[str]:
    spec = plan["runs"][run]
    if isinstance(spec, dict):  # D11 {ocr: <탐지 실험>}: 그 탐지 실험과 같은 데이터 묶음
        return _sets(plan, spec["ocr"])
    items: list[str] = []
    for bundle in spec or []:
        if bundle not in plan["bundles"]:
            raise SystemExit(f"실험 {run}: 묶음 {bundle} 이 bundles에 없습니다.")
        items += [str(v) for v in plan["bundles"][bundle]]
    return items


def _call(cmd: list[str], dry: bool) -> bool:
    shown = [c.replace(str(resolve_path(".")), ".") for c in cmd[1:]]
    print("  $ python " + " ".join(shown))
    if dry:
        return True
    return subprocess.run(cmd, cwd=resolve_path(".")).returncode == 0


def _done(exp: str) -> bool:
    """학습이 끝난 실험 (점수가 없어도 metrics.json이 있으면 끝난 것)."""
    return (OUTPUT_DIR / exp / "metrics.json").exists()


def _map(exp: str):
    path = OUTPUT_DIR / exp / "metrics.json"
    return load_json(path).get("map") if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/matrix_data.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="이 모델만 (계획 파일의 이름)")
    ap.add_argument("--runs", nargs="*", default=None, help="이 실험만 (계획 파일의 이름)")
    ap.add_argument("--epochs", type=int, default=None, help="빠른 확인용 epoch 수")
    ap.add_argument("--force", action="store_true", help="이미 끝난 실험도 다시 학습")
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력")
    ap.add_argument("--summary", action="store_true", help="학습 없이 결과 표만")
    args = ap.parse_args()

    plan = load_config(args.plan)
    prefix = plan["prefix"]
    models = {k: v for k, v in plan["models"].items() if not args.models or k in args.models}
    runs = [r for r in plan["runs"] if not args.runs or r in args.runs]
    suffix = f"_e{args.epochs}" if args.epochs else ""
    if "base" not in runs and not args.summary:
        print("[주의] base(기준)가 목록에 없으면 기준 대비 변화를 계산할 수 없습니다.")

    failed, skipped = [], []
    exported: set[tuple] = set()
    for model, model_cfg in models.items():
        family = load_config(model_cfg)["model"]["family"]
        for run in runs:
            exp = f"{prefix}_{model}_{run}{suffix}"
            items = _sets(plan, run)
            if args.summary:
                continue
            spec = plan["runs"][run]
            if isinstance(spec, dict):  # D11 OCR 재분류: 학습 없이 탐지 실험 위에 얹는다
                det = f"{prefix}_{model}_{spec['ocr']}{suffix}"
                if _done(exp) and not args.force:
                    print(f"[{exp}] 이미 있음 → 건너뜀")
                    continue
                if not _done(det) and not args.dry_run:
                    print(f"[{exp}] 탐지 실험 {det} 이 아직 없음 → 건너뜀")
                    skipped.append(exp)
                    continue
                print(f"\n[{exp}] OCR 재분류 (탐지: {det})")
                cmd = PY + [
                    str(resolve_path("scripts/run_train.py")),
                    "--config",
                    plan.get("ocr_config", "configs/ocr_rerank.yaml"),
                    "--exp-name",
                    exp,
                    "--set",
                    *items,
                    f"model.detector_exp={det}",
                ]
                if not _call(cmd, args.dry_run):
                    failed.append(exp)
                continue
            if _done(exp) and not args.force:
                print(f"[{exp}] 이미 있음 → 건너뜀")
                continue
            cfg = apply_overrides(load_config(model_cfg), items)
            ann = resolve_path(cfg["data"]["train_annotations"])
            if not ann.exists():
                print(f"[{exp}] 데이터 없음 ({ann}) → 건너뜀. 데이터 버전을 먼저 만드세요.")
                skipped.append(exp)
                continue
            print(f"\n[{exp}] {' '.join(items) or '(기준)'}")
            set_args = ["--set", *items] if items else []
            if family == "yolo":
                d = cfg["data"]
                key = (d["version"], d.get("repeat_threshold"), d.get("class_map"))
                if key not in exported:
                    ok = _call(
                        PY
                        + [
                            str(resolve_path("scripts/export_yolo.py")),
                            "--config",
                            model_cfg,
                            *set_args,
                        ],
                        args.dry_run,
                    )
                    if not ok:
                        failed.append(f"{exp} (export)")
                        continue
                    exported.add(key)
            cmd = PY + [
                str(resolve_path("scripts/run_train.py")),
                "--config",
                model_cfg,
                "--exp-name",
                exp,
                *set_args,
            ]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if not _call(cmd, args.dry_run):
                failed.append(exp)

    # 결과 표: 모델 × 실험
    rows = []
    for run in runs:
        spec = plan["runs"][run]
        label = f"OCR({spec['ocr']})" if isinstance(spec, dict) else "+".join(spec or []) or "-"
        row = {"run": run, "bundles": label}
        for model in models:
            m = _map(f"{prefix}_{model}_{run}{suffix}")
            b = _map(f"{prefix}_{model}_base{suffix}")
            row[model] = "" if m is None else f"{m:.4f}"
            row[f"{model}_diff"] = "" if m is None or b is None else f"{m - b:+.4f}"
        rows.append(row)
    print("\n결과 (map = mAP@[0.75:0.95], 괄호 = 같은 모델 base 대비)")
    print(f"{'실험':16s}" + "".join(f"{m:>20s}" for m in models))
    for row in rows:
        cells = [
            f"{row[m]} ({row[m + '_diff']})" if row[m_diff] else (row[m] or "-")
            for m, m_diff in ((m, m + "_diff") for m in models)
        ]
        print(f"{row['run']:16s}" + "".join(f"{c:>20s}" for c in cells))
    out = OUTPUT_DIR / f"matrix_{prefix}{suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out}")
    if skipped:
        print(f"데이터 없음으로 건너뜀 {len(skipped)}개: {', '.join(skipped)}")
    if failed:
        print(f"실패 {len(failed)}개: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
