"""실험이 끝난 뒤 학습 곡선과 검증 점수를 그래프로 비교한다 (데이터 실험 적용 가이드).

실행:
    uv run python scripts/plot_matrix.py --plan configs/matrix_data.yaml           # run_matrix 결과 전체
    uv run python scripts/plot_matrix.py --plan configs/matrix_data.yaml --epochs 1  # --epochs 1 로 돌린 결과
    uv run python scripts/plot_matrix.py --exps <기준 실험> <실험2> <실험3>          # 실험 이름을 직접 (첫째가 기준)

결과: outputs/compare_<이름>/  (그래프에는 대회 이미지가 없어 PR·디스코드에 올려도 됨. 폴더는 커밋하지 않음)
    report.html        아래 그림과 점수표를 한 페이지에 (브라우저로 열기)
    scores.png         모델 × 실험 map 히트맵 (칸 색 = 같은 모델 기준 대비 변화, 초록 오름 / 빨강 내림)
    metrics.png        map·map_75·mar_100 막대 비교
    curves_<모델>.png  에폭별 train loss, val loss, 검증 mAP50-95 (실험마다 선 하나, 기준은 검정 굵은 선)
    scores.csv         그래프에 쓴 숫자
읽는 법은 데이터 실험 적용 가이드의 「과적합·과소적합 확인」「평가 지표 보는 법」 참고.
"""

from __future__ import annotations

import argparse
import base64
import html

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.interfaces import OUTPUT_DIR  # noqa: E402
from src.utils import load_config, load_json  # noqa: E402
from src.visualize import set_korean_font  # noqa: E402

METRICS = [
    ("map", "mAP@[0.75:0.95] (대회 지표)"),
    ("map_75", "mAP75"),
    ("mar_100", "재현율 mar_100"),
]


def load_curve(exp: str) -> pd.DataFrame | None:
    """에폭별 train_loss / val_loss / val_map. YOLO는 results.csv, R-CNN은 train_history.json."""
    yolo = OUTPUT_DIR / exp / "yolo_run" / "results.csv"
    if yolo.exists():
        df = pd.read_csv(yolo)
        df.columns = df.columns.str.strip()
        out = pd.DataFrame({"epoch": df["epoch"]})
        out["train_loss"] = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
        out["val_loss"] = df[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)
        out["val_map"] = df["metrics/mAP50-95(B)"]
        return out
    rcnn = OUTPUT_DIR / exp / "train_history.json"
    if rcnn.exists():
        return pd.DataFrame(load_json(rcnn))
    return None


def collect(args) -> tuple[str, list[str], list[str], dict]:
    """(이름, 모델 목록, 실험 목록, {(모델, 실험): 실험 이름})."""
    if args.exps:
        runs = args.exps
        return "custom", ["exps"], runs, {("exps", r): r for r in runs}
    plan = load_config(args.plan)
    suffix = f"_e{args.epochs}" if args.epochs else ""
    models = [m for m in plan["models"] if not args.models or m in args.models]
    runs = [r for r in plan["runs"] if not args.runs or r in args.runs]
    names = {(m, r): f"{plan['prefix']}_{m}_{r}{suffix}" for m in models for r in runs}
    return f"{plan['prefix']}{suffix}", models, runs, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/matrix_data.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="run_matrix --epochs 로 돌린 결과")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument(
        "--exps", nargs="*", default=None, help="계획 파일 대신 실험 이름 (첫째가 기준)"
    )
    args = ap.parse_args()
    set_korean_font()

    tag, models, runs, names = collect(args)
    base_run = runs[0] if args.exps else "base"
    rows = []
    for (model, run), exp in names.items():
        path = OUTPUT_DIR / exp / "metrics.json"
        m = load_json(path) if path.exists() else {}
        rows.append({"model": model, "run": run, "exp": exp, **{k: m.get(k) for k, _ in METRICS}})
        rows[-1]["train_min"] = m.get("train_min")
    df = pd.DataFrame(rows)
    done = df["map"].notna()
    if not done.any():
        raise SystemExit("끝난 실험(metrics.json)이 없습니다. 실험 이름과 --epochs를 확인하세요.")
    base = df[df["run"] == base_run].set_index("model")["map"]
    df["diff"] = df.apply(lambda r: r["map"] - base.get(r["model"], np.nan), axis=1)
    out_dir = OUTPUT_DIR / f"compare_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):  # 이전 실행 그림(빠진 모델 등)이 섞이지 않게
        old.unlink()
    df.to_csv(out_dir / "scores.csv", index=False, encoding="utf-8-sig")
    figs = []

    # 1) 히트맵: 칸 = map, 색 = 기준 대비 변화
    mat = df.pivot(index="run", columns="model", values="map").reindex(index=runs, columns=models)
    dif = df.pivot(index="run", columns="model", values="diff").reindex(index=runs, columns=models)
    lim = max(0.02, float(np.nanmax(np.abs(dif.values))) if dif.notna().any().any() else 0.02)
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(models), 1.2 + 0.55 * len(runs)))
    ax.imshow(dif.fillna(0).values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i, run in enumerate(runs):
        for j, model in enumerate(models):
            v, d = mat.loc[run, model], dif.loc[run, model]
            text = "-" if pd.isna(v) else f"{v:.3f}" + ("" if run == base_run else f"\n{d:+.3f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(runs)), runs)
    ax.set_title(f"검증 map (아래 숫자 = {base_run} 대비)")
    fig.tight_layout()
    fig.savefig(out_dir / "scores.png", dpi=120)
    figs.append(("scores.png", "모델 × 실험 점수"))
    plt.close(fig)

    # 2) 막대: map, map_75, mar_100
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 3.8), sharey=False)
    width = 0.8 / len(models)
    x = np.arange(len(runs))
    for ax, (key, title) in zip(axes, METRICS, strict=True):
        piv = df.pivot(index="run", columns="model", values=key).reindex(index=runs, columns=models)
        for j, model in enumerate(models):
            ax.bar(x + j * width - 0.4 + width / 2, piv[model].astype(float), width, label=model)
        ax.set_xticks(x, runs, rotation=40, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=120)
    figs.append(("metrics.png", "지표별 비교"))
    plt.close(fig)

    # 3) 학습 곡선: 모델마다 한 장, 실험마다 선 하나
    for model in models:
        curves = {run: load_curve(names[(model, run)]) for run in runs}
        curves = {r: c for r, c in curves.items() if c is not None}
        if not curves:
            continue
        panels = [
            ("train_loss", "train loss"),
            ("val_loss", "val loss"),
            ("val_map", "검증 mAP (추세용)"),
        ]
        panels = [p for p in panels if any(p[0] in c for c in curves.values())]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), squeeze=False)
        for ax, (key, title) in zip(axes[0], panels, strict=True):
            for run, c in curves.items():
                if key not in c:
                    continue
                is_base = run == base_run
                ax.plot(
                    c["epoch"],
                    c[key],
                    color="black" if is_base else None,
                    lw=2.5 if is_base else 1.3,
                    label=run,
                )
            ax.set_title(f"{model} · {title}")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes[0][-1].legend(fontsize=8)
        fig.tight_layout()
        fname = f"curves_{model}.png"
        fig.savefig(out_dir / fname, dpi=120)
        figs.append((fname, f"{model} 학습 곡선"))
        plt.close(fig)

    # 4) 한 페이지 리포트
    table = df[done][["model", "run", "map", "diff", "map_75", "mar_100", "train_min", "exp"]]
    table = table.round(4).to_html(index=False, na_rep="-", border=0)
    parts = [
        "<meta charset='utf-8'><title>실험 비교</title>",
        "<style>body{font-family:sans-serif;margin:24px;max-width:1400px}"
        "img{max-width:100%;border:1px solid #ddd;margin:6px 0 18px}"
        "table{border-collapse:collapse;font-size:13px}td,th{padding:4px 10px;border-bottom:1px solid #eee}"
        "</style>",
        f"<h1>실험 비교: {html.escape(tag)}</h1>",
        "<p>map = mAP@[0.75:0.95] (대회 지표), diff = 같은 모델 기준 대비. "
        "곡선의 검증 mAP는 ultralytics mAP50-95라 추세만 봅니다.</p>",
    ]
    for fname, title in figs:
        b64 = base64.b64encode((out_dir / fname).read_bytes()).decode()
        parts.append(f"<h2>{html.escape(title)}</h2><img src='data:image/png;base64,{b64}'>")
    parts.append("<h2>점수표</h2>" + table)
    (out_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")

    missing = df[~done]["exp"].tolist()
    print(f"끝난 실험 {int(done.sum())}개 / 전체 {len(df)}개")
    if missing:
        print(
            f"아직 없는 실험 {len(missing)}개: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}"
        )
    print(f"저장: {out_dir}  (report.html 을 브라우저로 여세요)")


if __name__ == "__main__":
    main()
