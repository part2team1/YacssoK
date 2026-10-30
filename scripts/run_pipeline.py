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
import subprocess
import sys
from datetime import datetime

import yaml

from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
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


def run_items(plan: dict, run: str) -> list[str]:
    spec = plan["runs"][run]
    if isinstance(spec, dict):
        return run_items(plan, spec["ocr"])
    return [str(v) for b in spec or [] for v in plan["bundles"][b]]


def is_data_run(plan: dict, run: str) -> bool:
    return any(i.startswith("data.version=") for i in run_items(plan, run))


def changes_classes(plan: dict, run: str) -> bool:
    return any(i.startswith("data.class_map=") for i in run_items(plan, run))


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
        sh(cmd + (["--epochs", str(args.epochs)] if args.epochs else []), log)

    def gains() -> dict[str, dict[str, float]]:
        """실험 → {모델: 기준 대비 변화}. v1처럼 검증 정답이 바뀐 실험은 다시 채점한 기준과 비교."""
        out = {}
        for run in plan["runs"]:
            if run == "base":
                continue
            row = {}
            for m in models:
                v, b = metric(name(m, run)), metric(name(m, "base"))
                rescored = state.get("rescored", {}).get(f"{m}:{run}")
                if v is not None and b is not None:
                    row[m] = v - (rescored if rescored is not None else b)
            out[run] = row
        return out

    # 1) 데이터 실험
    if "data_exp" in args.stages:
        matrix(list(plan["runs"]), models)
        for run in plan[
            "runs"
        ]:  # 학습은 끝났는데 점수가 없으면 그 데이터 버전의 검증 정답이 비어 있는 것
            for m in models:
                exp = name(m, run)
                if (OUTPUT_DIR / exp / "metrics.json").exists() and metric(exp) is None:
                    print(
                        f"[주의] {exp}: 점수가 없습니다. 이 데이터 버전의 검증 정답(val)이 비어 있습니다."
                    )
                    log.append(f"- {exp}: 점수 없음 (검증 정답 비어 있음, 데이터 버전 확인 필요)")
        # 검증 정답이 바뀌는 실험(정제 v1)은 기준을 그 정답으로 다시 채점해 공정하게 비교
        for run in plan["runs"]:
            items = run_items(plan, run)
            if not any(i == "data.version=v1" for i in items):
                continue
            val_cfg = out_dir / f"val_{run}.yaml"
            data = apply_overrides(load_config(plan["models"][models[0]]), items)["data"]
            val_cfg.write_text(yaml.safe_dump({"data": data}, allow_unicode=True), "utf-8")
            for m in models:
                base_exp = name(m, "base")
                # 기준과 정제 실험이 둘 다 점수를 낸 경우에만 다시 채점 (정제 데이터가 없으면 건너뜀)
                if metric(base_exp) is None or metric(name(m, run)) is None:
                    continue
                if sh(
                    ["scripts/evaluate.py", "--exp", base_exp, "--val-config", str(val_cfg)], log
                ):
                    evals = sorted((OUTPUT_DIR / base_exp).glob("eval_*_valv1.json"))
                    score = load_json(evals[-1]).get("map") if evals else None
                    if score is None:  # 검증 정답이 비어 있음: v1.json에 검증 이미지가 없음
                        print(
                            f"[주의] {run}: v1 검증 정답이 비어 있어 다시 채점하지 못했습니다. "
                            "data/annotations/v1.json 과 data/splits/v1/val.txt 를 확인하세요."
                        )
                        log.append(f"- {run}: v1 검증 정답 비어 있음 (기준 다시 채점 실패)")
                    else:
                        state.setdefault("rescored", {})[f"{m}:{run}"] = score
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")

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
        rows = [(metric(name(m, r)), m, r) for m in models for r in plan["runs"]]
        return sorted([x for x in rows if x[0] is not None], reverse=True)

    # 3) 하이퍼파라미터 탐색: 1등 (모델, 실험) 설정으로 tune → 찾은 값으로 다시 학습
    if "tune" in args.stages and leaderboard():
        _, bm, br = next(x for x in leaderboard() if not x[2].startswith("tuned"))
        items = run_items(plan, br)
        cfg_path = plan["models"][bm]
        sh(
            ["scripts/export_yolo.py", "--config", cfg_path, "--set", *items]
            if items
            else ["scripts/export_yolo.py", "--config", cfg_path],
            log,
        )
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
        sh(tune_cmd, log)
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
            v = metric(name(best_model, f"{best_run}_ocr"))
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
        sh(cmd + (["--epochs", str(args.epochs)] if args.epochs else []), log)

    # 6) 제출 파일
    sub = PROJECT_ROOT / "submissions" / f"{best}.csv"
    if "submit" in args.stages:
        sh(["scripts/make_submission.py", "--exp", best], log)
        sh(["scripts/evaluate.py", "--exp", best, "--class-metrics"], log)

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
    """한 페이지 HTML 보고서. 그림은 plot_matrix 결과를 넣고, 표는 metrics.json에서 만든다."""
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
            v = metric(name(m, r))
            b = metric(name(m, "base"))
            if v is None:
                cells.append("<td>-</td>")
                continue
            d = "" if r == "base" or b is None else f" <small>({v - b:+.4f})</small>"
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
    m_best = load_json(OUTPUT_DIR / best["exp"] / "metrics.json")
    combo = state.get("combo", {})
    tuned = state.get("tuned")
    ev = sorted((OUTPUT_DIR / best["exp"]).glob("eval_*.json"))
    class_rows = ""
    if ev:
        e = load_json(ev[-1])
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
        f"<b>{best['map']:.4f}</b>, map_75 {m_best.get('map_75', 0):.4f}, "
        f"재현율 mar_100 {m_best.get('mar_100', 0):.4f}, 학습 {m_best.get('train_min', '-')}분.</p>",
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
        "<p>괄호 = 같은 모델 base 대비. 초록 칸 = 최종 1등. 정제(v1) 실험은 검증 정답이 바뀌므로 "
        "결합 판단에는 v1 정답으로 다시 채점한 base와 비교했다.</p>",
        "<h2>그래프</h2>",
        img(fig_dir / "scores.png"),
        img(fig_dir / "metrics.png"),
        *[img(fig_dir / f"curves_{m}.png") for m in models],
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
