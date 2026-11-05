"""데이터 실험 → 결합 → 하이퍼파라미터 탐색 → (OCR) → 그래프 → 제출 파일 → 보고서 → 웹 데모를 한 번에 (모델 4개).

계획 파일(configs/matrix_data.yaml)의 모델·데이터 실험을 그대로 쓰고, 아래 단계를 차례로 실행한다.
각 단계는 이미 끝난 실험을 건너뛰므로 중간에 멈춰도 같은 명령으로 이어서 할 수 있다.

  1 data_exp  모델마다 D0(base) + 데이터 실험 학습                     (run_matrix.py)
  2 combo     기준 대비 효과 있던 실험을 모아 결합 실험을 만들어 모든 모델에 학습
  3 tune      지금까지 1등 (모델, 실험)으로 하이퍼파라미터 탐색 → 찾은 값으로 다시 학습 (tune_yolo.py)
  4 ocr       최종 1등 모델 위에 각인 OCR 재분류 (D11, OCR 패키지가 있을 때만)
  5 plot      모델 × 실험 그래프 (plot_matrix.py)
  6 submit    최종 1등으로 Kaggle 제출 파일 (make_submission.py)
  7 report    한 페이지 보고서 outputs/pipeline_<prefix>/report.html
  8 demo      최종 1등 모델로 웹 데모 실행 (--demo 를 줄 때만, 끝내기 Ctrl+C)

실행 (bash·PowerShell 공통):
    uv run python scripts/run_pipeline.py --epochs 1 --tune-iterations 2 --tune-epochs 1   # 1 epoch 동작 확인
    uv run python scripts/run_pipeline.py                                                  # 본 실행 (오래 걸림)
    uv run python scripts/run_pipeline.py --demo                                           # 끝나면 데모까지
    uv run python scripts/run_pipeline.py --stages plot report                             # 일부 단계만
    uv run python scripts/run_pipeline.py --models yolo11s                                 # 모델 하나만

자동으로 하지 않는 것 (사람 판단이 필요): 데이터 버전 만들기(정제 v1의 manual_drop, AI Hub 내려받기 등),
v3all 채택, 의사 라벨(D8), K-fold(D7), 전체 데이터 최종 학습(P5). 데이터 버전이 없는 실험은 건너뛴다.
결과는 모두 outputs/ 아래라 커밋하지 않는다.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import subprocess
import sys
from datetime import datetime

import yaml

from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
from src.provenance import checkpoint_signature, evaluation_signature, experiment_signature
from src.utils import apply_overrides, load_config, load_json, resolve_path

STAGES = ["data_exp", "combo", "tune", "ocr", "plot", "submit", "report"]


def sh(cmd: list[str], log: list[str]) -> bool:
    """프로젝트 폴더에서 python 스크립트 실행. 명령과 성공 여부를 기록한다."""
    root = str(PROJECT_ROOT) + ("\\" if "\\" in str(PROJECT_ROOT) else "/")
    shown = "uv run python " + " ".join(c.replace(root, "") for c in cmd)
    print(f"\n▶ {shown}", flush=True)
    ok = subprocess.run([sys.executable, *cmd], cwd=PROJECT_ROOT).returncode == 0
    log.append(("✔ " if ok else "✖ ") + shown)
    return ok


def metric(exp: str, key: str = "map"):
    p = OUTPUT_DIR / exp / "metrics.json"
    return load_json(p).get(key) if p.exists() else None


def selection_result(exp: str) -> dict:
    folder = OUTPUT_DIR / exp
    path = folder / "selection_metrics.json"
    return load_json(path if path.exists() else folder / "metrics.json")


def run_items(plan: dict, run: str) -> list[str]:
    spec = plan["runs"][run]
    if isinstance(spec, dict):
        return run_items(plan, spec["ocr"])
    return [str(v) for b in spec or [] for v in plan["bundles"][b]]


def is_data_run(plan: dict, run: str) -> bool:
    return any(i.startswith("data.version=") for i in run_items(plan, run))


def changes_classes(plan: dict, run: str) -> bool:
    return any(i.startswith("data.class_map=") for i in run_items(plan, run))


def comparable_leaderboard(experiments: list[tuple[str, str, str]]) -> list[tuple[float, str, str]]:
    """정답·사진·번호표·IoU가 일치한다고 기록된 점수만 자동 순위에 사용한다.

    과거 실험은 evaluate.py --val-config <공통설정> --for-selection으로
    다시 평가할 수 있다. 학습이나 과거 metrics.json 덮어쓰기는 필요하지 않다.
    """
    rows, keys = [], set()
    for exp, model, run in experiments:
        folder = OUTPUT_DIR / exp
        path = folder / "selection_metrics.json"
        if not path.exists():
            path = folder / "metrics.json"
        if not path.exists():
            continue
        result = load_json(path)
        if result.get("map") is None:
            continue
        if not math.isfinite(float(result["map"])):
            raise ValueError(f"{exp}: 유효한 숫자가 아닌 검증 점수입니다.")
        provenance = result.get("provenance", {})
        key = provenance.get("evaluation")
        weights = folder / "best.pt"
        cfg = load_config(folder / "config.yaml")
        if (
            not key
            or not weights.is_file()
            or provenance.get("weights") != checkpoint_signature(cfg, weights)
        ):
            raise ValueError(
                f"{exp}: 검증 입력 또는 가중치 지문이 없습니다/달라졌습니다. "
                "evaluate.py --val-config <공통설정> --for-selection으로 다시 평가하세요."
            )
        keys.add(key)
        rows.append((float(result["map"]), model, run))
    if len(keys) > 1:
        raise ValueError(
            "서로 다른 검증 사진·정답·클래스 번호·IoU의 점수로 1등을 정할 수 없습니다. "
            "모든 후보를 같은 --val-config로 --for-selection 평가한 뒤 다시 실행하세요."
        )
    return sorted(rows, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/matrix_data.yaml", help="모델·데이터 실험 계획 파일")
    ap.add_argument("--models", nargs="*", default=None, help="이 모델만 (계획 파일의 이름)")
    ap.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES, help="실행할 단계")
    ap.add_argument("--epochs", type=int, default=None, help="빠른 확인용 epoch (이름 끝 _eN)")
    ap.add_argument("--min-gain", type=float, default=0.01, help="결합에 넣을 최소 향상폭")
    ap.add_argument(
        "--min-share", type=float, default=0.5, help="이 비율 이상 모델에서 향상돼야 함"
    )
    ap.add_argument("--tune-iterations", type=int, default=20, help="하이퍼파라미터 시도 횟수")
    ap.add_argument("--tune-epochs", type=int, default=15, help="시도 한 번의 epoch")
    ap.add_argument("--demo", action="store_true", help="마지막에 웹 데모 실행 (Ctrl+C로 종료)")
    ap.add_argument("--selection-config", help="모든 후보를 다시 평가할 공통 검증 data 설정")
    args = ap.parse_args()

    plan = load_config(args.plan)
    plan["runs"] = {k: v for k, v in plan["runs"].items() if not isinstance(v, dict)}  # D11은 4단계
    prefix, sfx = plan["prefix"], (f"_e{args.epochs}" if args.epochs else "")
    models = [m for m in plan["models"] if not args.models or m in args.models]
    out_dir = OUTPUT_DIR / f"pipeline_{prefix}{sfx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.yaml"  # 결합·튜닝 실험이 더해진 계획 (실행 기록)
    state_path = out_dir / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    log: list[str] = []
    name = lambda m, r: f"{prefix}_{m}_{r}{sfx}"  # noqa: E731

    def save_plan() -> None:
        plan_path.write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), "utf-8")

    def matrix(runs: list[str], only: list[str]) -> None:
        save_plan()
        cmd = [
            "scripts/run_matrix.py",
            "--plan",
            str(plan_path),
            "--models",
            *only,
            "--runs",
            *runs,
        ]
        if not sh(cmd + (["--epochs", str(args.epochs)] if args.epochs else []), log):
            raise SystemExit("실험 실행이 실패해 이전 결과를 이용한 후속 단계를 중단합니다.")

    def selection_scores() -> None:
        if not args.selection_config:
            return
        common_data = load_config(args.selection_config)["data"]
        for model in models:
            for run in plan["runs"]:
                exp = name(model, run)
                if metric(exp) is None:
                    continue
                folder = OUTPUT_DIR / exp
                cfg = load_config(folder / "config.yaml")
                data_cfg = {**cfg, "data": {**cfg["data"], **common_data}}
                expected = evaluation_signature(data_cfg)
                saved_path = folder / "selection_metrics.json"
                saved = load_json(saved_path).get("provenance", {}) if saved_path.exists() else {}
                if (
                    saved.get("evaluation") == expected
                    and saved.get("weights") == checkpoint_signature(cfg, folder / "best.pt")
                    and saved.get("inference_eval") == cfg["eval"]
                ):
                    continue
                if not sh(
                    [
                        "scripts/evaluate.py",
                        "--exp",
                        exp,
                        "--val-config",
                        args.selection_config,
                        "--for-selection",
                    ],
                    log,
                ):
                    raise SystemExit(f"공통 검증 실패: {exp}. 자동 모델 선택을 중단합니다.")

    def validate_plan_results() -> None:
        for model in models:
            for run in plan["runs"]:
                exp = name(model, run)
                if metric(exp) is None:
                    continue
                desired = apply_overrides(load_config(plan["models"][model]), run_items(plan, run))
                desired["exp_name"] = exp
                if args.epochs is not None:
                    desired["train"]["epochs"] = args.epochs
                saved = load_config(OUTPUT_DIR / exp / "config.yaml")
                if saved != desired:
                    raise ValueError(
                        f"{exp}: 저장된 설정과 현재 실험 계획이 다릅니다. 새 실험 이름을 쓰세요."
                    )
                fingerprint = (
                    load_json(OUTPUT_DIR / exp / "metrics.json")
                    .get("provenance", {})
                    .get("experiment")
                )
                if fingerprint and fingerprint != experiment_signature(desired):
                    raise ValueError(
                        f"{exp}: 실험 후 데이터·설정·코드가 변경됐습니다. 새 실험 이름을 쓰세요."
                    )

    def gains() -> dict[str, dict[str, float]]:
        """실험 → {모델: 기준 대비 변화}. v1처럼 검증 정답이 바뀐 실험은 다시 채점한 기준과 비교."""
        validate_plan_results()
        out = {}
        for run in plan["runs"]:
            if run == "base":
                continue
            row = {}
            for m in models:
                try:
                    scored = comparable_leaderboard(
                        [(name(m, run), m, run), (name(m, "base"), m, "base")]
                    )
                except ValueError as error:
                    print(f"[비교 보류] {error}")
                    continue
                values = {r: score for score, _, r in scored}
                if run in values and "base" in values:
                    row[m] = values[run] - values["base"]
            out[run] = row
        return out

    # 1) 데이터 실험
    if "data_exp" in args.stages:
        matrix(list(plan["runs"]), models)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    selection_scores()

    # 2) 결합: 절반 이상 모델에서 +min_gain 이상인 실험만. 데이터 버전은 하나만 (가장 좋은 것)
    if "combo" in args.stages:
        g = gains()
        good = []
        for run, row in g.items():
            if not row or run.startswith(("combo", "tuned")) or changes_classes(plan, run):
                continue
            share = sum(v >= args.min_gain for v in row.values()) / len(row)
            avg = sum(row.values()) / len(row)
            if share >= args.min_share and avg >= args.min_gain:
                good.append((avg, run))
        good.sort(reverse=True)
        data_runs = [r for _, r in good if is_data_run(plan, r)]
        chosen = data_runs[:1] + [r for _, r in good if not is_data_run(plan, r)]
        bundles: list[str] = []
        for r in chosen:
            bundles += [b for b in plan["runs"][r] if b not in bundles]
        state["combo"] = {"runs": chosen, "bundles": bundles, "gains": g}
        if len(chosen) >= 2:
            plan["runs"]["combo"] = bundles
            matrix(["combo"], models)
        else:
            print(f"\n[combo] 효과 있는 실험이 {len(chosen)}개라 결합 실험을 건너뜁니다: {chosen}")
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    elif "combo" in state and len(state["combo"]["runs"]) >= 2:
        plan["runs"]["combo"] = state["combo"]["bundles"]

    def leaderboard() -> list[tuple[float, str, str]]:
        try:
            validate_plan_results()
            selection_scores()
            return comparable_leaderboard(
                [(name(m, r), m, r) for m in models for r in plan["runs"]]
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    # 3) 하이퍼파라미터 탐색: 1등 (모델, 실험) 설정으로 tune → 찾은 값으로 다시 학습
    if "tune" in args.stages and leaderboard():
        untuned = [x for x in leaderboard() if not x[2].startswith("tuned")]
        if not untuned:
            raise SystemExit("튜닝할 기준 실험이 없습니다. 원본 실험 결과를 확인하세요.")
        _, bm, br = untuned[0]
        items = run_items(plan, br)
        cfg_path = plan["models"][bm]
        exported = sh(
            ["scripts/export_yolo.py", "--config", cfg_path, "--set", *items]
            if items
            else ["scripts/export_yolo.py", "--config", cfg_path],
            log,
        )
        if not exported:
            raise SystemExit("튜닝용 데이터 내보내기에 실패했습니다.")
        tune_cmd = [
            "scripts/tune_yolo.py",
            "--config",
            cfg_path,
            "--epochs",
            str(args.tune_epochs),
            "--iterations",
            str(args.tune_iterations),
        ]
        tune_cmd += ["--set", f"exp_name={name(bm, br)}", *items]
        if not sh(tune_cmd, log):
            raise SystemExit("튜닝이 실패해 이전 파라미터를 재사용하지 않습니다.")
        best_yaml = OUTPUT_DIR / f"tune_{name(bm, br)}" / "tune" / "best_hyperparameters.yaml"
        if best_yaml.exists():
            hp = yaml.safe_load(best_yaml.read_text("utf-8")) or {}
            plan["bundles"]["TUNED"] = [f"yolo_args.{k}={v}" for k, v in hp.items()]
            plan["runs"]["tuned"] = [*plan["runs"][br], "TUNED"]
            state["tuned"] = {"model": bm, "from": br, "params": hp}
            matrix(["tuned"], [bm])
        else:
            print(f"[tune] {best_yaml} 이 없어 다시 학습을 건너뜁니다.")
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    elif "tuned" in state:
        plan["bundles"]["TUNED"] = [
            f"yolo_args.{k}={v}" for k, v in state["tuned"]["params"].items()
        ]
        plan["runs"]["tuned"] = [*plan["runs"][state["tuned"]["from"]], "TUNED"]

    board = leaderboard()
    if not board:
        raise SystemExit("끝난 실험이 없습니다. 데이터 버전과 계획 파일을 확인하세요.")
    best_map, best_model, best_run = board[0]
    best = name(best_model, best_run)
    state["best"] = {"exp": best, "model": best_model, "run": best_run, "map": best_map}

    # 4) D11 OCR 재분류 (학습 없음). 점수가 오르면 최종 1등을 OCR 실험으로 바꾼다
    if "ocr" in args.stages:
        try:
            import rapidocr_onnxruntime  # noqa: F401

            has_ocr = (
                resolve_path(load_config(plan["models"][best_model])["data"]["class_map"]).parent
                / "drug_info.json"
            ).exists()
        except ImportError:
            has_ocr = False
        if has_ocr:
            plan["runs"][f"{best_run}_ocr"] = {"ocr": best_run}
            matrix([f"{best_run}_ocr"], [best_model])
            selection_scores()
            ocr_board = comparable_leaderboard(
                [
                    (best, best_model, best_run),
                    (name(best_model, f"{best_run}_ocr"), best_model, f"{best_run}_ocr"),
                ]
            )
            v = next((score for score, _, run in ocr_board if run.endswith("_ocr")), None)
            if v is not None and v > best_map:
                best = name(best_model, f"{best_run}_ocr")
                state["best"] = {
                    "exp": best,
                    "model": best_model,
                    "run": f"{best_run}_ocr",
                    "map": v,
                }
        else:
            print(
                "\n[ocr] OCR 패키지(uv sync --extra ocr) 또는 data/drug_info.json 이 없어 건너뜁니다."
            )
            log.append("- D11 OCR 건너뜀 (패키지 또는 drug_info.json 없음)")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    save_plan()

    # 5) 그래프
    if "plot" in args.stages:
        cmd = ["scripts/plot_matrix.py", "--plan", str(plan_path), "--models", *models]
        if not sh(cmd + (["--epochs", str(args.epochs)] if args.epochs else []), log):
            raise SystemExit("그래프 생성이 실패했습니다. 후속 단계를 중단합니다.")

    # 6) 제출 파일
    sub = PROJECT_ROOT / "submissions" / f"{best}.csv"
    if "submit" in args.stages:
        if not sh(["scripts/make_submission.py", "--exp", best], log):
            raise SystemExit("제출 CSV 생성이 실패했습니다.")
        eval_cmd = ["scripts/evaluate.py", "--exp", best, "--class-metrics"]
        if args.selection_config:
            eval_cmd += ["--val-config", args.selection_config, "--for-selection"]
        if not sh(eval_cmd, log):
            raise SystemExit("선택 모델의 상세 평가가 실패했습니다.")

    # 7) 보고서
    if "report" in args.stages:
        write_report(out_dir, plan, state, models, name, sub, log, args)

    print(f"\n최종 1등: {state['best']['exp']}  (map {state['best']['map']:.4f})")
    print(f"보고서: {out_dir / 'report.html'}")

    # 8) 웹 데모
    if args.demo:
        print("\n웹 데모: 브라우저에서 http://127.0.0.1:7860  (끝내기 Ctrl+C)")
        subprocess.run(
            [sys.executable, "app/demo.py", "--exp", state["best"]["exp"]], cwd=PROJECT_ROOT
        )


def write_report(out_dir, plan, state, models, name, sub, log, args) -> None:
    """최종 선택에 쓴 점수와 같은 평가 결과로 표·보조 지표를 만든다."""
    best = state["best"]
    tag = f"{plan['prefix']}{f'_e{args.epochs}' if args.epochs else ''}"
    fig_dir = OUTPUT_DIR / f"compare_{tag}"

    def img(path) -> str:
        if not path.exists():
            return ""
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"<img src='data:image/png;base64,{b64}'>"

    rows = []
    for r in plan["runs"]:
        cells = []
        for m in models:
            folder = OUTPUT_DIR / name(m, r)
            if not (folder / "metrics.json").exists():
                cells.append("<td>-</td>")
                continue
            result = selection_result(name(m, r))
            base_folder = OUTPUT_DIR / name(m, "base")
            base_result = (
                selection_result(name(m, "base")) if (base_folder / "metrics.json").exists() else {}
            )
            v, b = result.get("map"), base_result.get("map")
            if v is None:
                cells.append("<td>-</td>")
                continue
            exam = result.get("provenance", {}).get("evaluation")
            same_exam = bool(exam) and exam == base_result.get("provenance", {}).get("evaluation")
            d = (
                ""
                if r == "base" or b is None or not same_exam
                else f" <small>({v - b:+.4f})</small>"
            )
            mark = " class='best'" if name(m, r) == best["exp"] else ""
            cells.append(f"<td{mark}>{v:.4f}{d}</td>")
        spec = plan["runs"][r]
        label = (
            f"OCR({spec['ocr']})" if isinstance(spec, dict) else " + ".join(spec or []) or "기준"
        )
        rows.append(
            f"<tr><th>{html.escape(r)}</th><td>{html.escape(label)}</td>{''.join(cells)}</tr>"
        )
    head = "".join(f"<th>{html.escape(m)}</th>" for m in models)
    m_best = selection_result(best["exp"])
    training = load_json(OUTPUT_DIR / best["exp"] / "metrics.json")
    rescored = (OUTPUT_DIR / best["exp"] / "selection_metrics.json").exists()

    def score_text(key):
        value = m_best.get(key)
        return "미측정" if value is None else f"{value:.4f}"

    combo = state.get("combo", {})
    tuned = state.get("tuned")
    class_rows = ""
    if m_best:
        e = m_best
        if "classes" in e:
            cfg = load_config(OUTPUT_DIR / best["exp"] / "config.yaml")
            from src.utils import load_class_map

            names = {c["label_idx"]: c["name"] for c in load_class_map(cfg["data"]["class_map"])}
            pairs = sorted(zip(e["map_per_class"], e["classes"], strict=True))
            worst = "".join(
                f"<tr><td>{c}</td><td>{html.escape(names.get(c, str(c)))}</td><td>{v:.3f}</td></tr>"
                for v, c in pairs[:10]
            )
            class_rows = f"<h2>클래스별 AP 하위 10개 (다음 개선 대상)</h2><table><tr><th>label</th><th>약 이름</th><th>AP</th></tr>{worst}</table>"
    parts = [
        "<meta charset='utf-8'><title>실험 보고서</title><style>"
        "body{font-family:sans-serif;margin:24px;max-width:1400px;line-height:1.5}"
        "table{border-collapse:collapse;font-size:13px;margin:8px 0 18px}"
        "td,th{padding:4px 10px;border-bottom:1px solid #eee;text-align:left}"
        "td.best{background:#e5f4e4;font-weight:bold}img{max-width:100%;border:1px solid #ddd}"
        "code{background:#f4f4f4;padding:1px 4px}pre{white-space:pre-wrap;font-size:12px}</style>",
        f"<h1>데이터·모델 실험 보고서: {html.escape(tag)}</h1>",
        f"<p>{datetime.now():%Y-%m-%d %H:%M} 생성 · 계획 파일 <code>{html.escape(args.plan)}</code> · "
        f"모델 {len(models)}개 · 지표 mAP@[0.75:0.95] (검증셋)</p>",
        f"<h2>결론</h2><p><b>최종 1등: <code>{html.escape(best['exp'])}</code></b> "
        f"(모델 {html.escape(best['model'])}, 실험 {html.escape(best['run'])}) — map "
        f"<b>{best['map']:.4f}</b>, map_75 {score_text('map_75')}, "
        f"재현율 mar_100 {score_text('mar_100')}, 학습 {training.get('train_min', '-')}분.</p>",
        f"<p>결합 실험에 넣은 실험: <b>{html.escape(', '.join(combo.get('runs', [])) or '없음')}</b> "
        f"(기준: 모델의 {args.min_share:.0%} 이상에서 +{args.min_gain} 이상, 데이터 버전은 가장 좋은 하나). "
        + (
            f"하이퍼파라미터 탐색: {html.escape(tuned['model'])} · {html.escape(tuned['from'])} 기준 "
            f"{args.tune_iterations}회 × {args.tune_epochs} epoch."
            if tuned
            else "하이퍼파라미터 탐색: 없음."
        )
        + "</p>",
        f"<h2>모델 × 실험 점수</h2><table><tr><th>실험</th><th>묶음</th>{head}</tr>{''.join(rows)}</table>"
        "<p>괄호 = 같은 검증 사진·정답·번호표·IoU로 채점한 같은 모델 base 대비. "
        "초록 칸 = 최종 1등. 선택용 재평가 파일이 있으면 그 점수를 사용한다.</p>",
        "<h2>그래프</h2>",
        img(fig_dir / "scores.png") if not rescored else "",
        img(fig_dir / "metrics.png")
        if not rescored
        else "<p>이전 원점수 그래프는 공통 재평가 점수와 섞지 않기 위해 제외했다.</p>",
        *([img(fig_dir / f"curves_{m}.png") for m in models] if not rescored else []),
        class_rows,
    ]
    if tuned:
        hp = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
            for k, v in tuned["params"].items()
        )
        parts.append(f"<h2>찾은 하이퍼파라미터</h2><table>{hp}</table>")
    parts += [
        "<h2>산출물</h2><ul>",
        f"<li>가중치: <code>outputs/{html.escape(best['exp'])}/best.pt</code></li>",
        f"<li>Kaggle 제출 파일: <code>{html.escape(str(sub.relative_to(PROJECT_ROOT)))}</code>"
        f"{'' if sub.exists() else ' (아직 없음)'}</li>",
        f"<li>웹 데모: <code>uv run python app/demo.py --exp {html.escape(best['exp'])}</code></li>",
        f"<li>그래프 원본: <code>outputs/compare_{html.escape(tag)}/</code>, 실행 계획: "
        f"<code>{html.escape(str((out_dir / 'plan.yaml').relative_to(PROJECT_ROOT)))}</code></li></ul>",
        "<h2>이번 실행 기록</h2><pre>"
        + html.escape("\n".join(log) or "(이번 실행에서 새로 한 단계 없음)")
        + "</pre>",
        "<p>자동으로 하지 않은 것: 데이터 버전 만들기(v1 정제·AI Hub), v3all 채택, D7 K-fold, "
        "D8 의사 라벨, 전체 데이터 최종 학습(P5). 이 보고서는 대회 이미지를 담지 않아 공유할 수 있다.</p>",
    ]
    (out_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
