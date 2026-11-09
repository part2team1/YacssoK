"""시각화 함수 (데이터 A 담당). 노트북과 보고서 그림에서 공통으로 쓴다."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import read_image_rgb
from src.interfaces import ClassInfo, Prediction, Record
from src.utils import resolve_path

plt.rcParams["axes.unicode_minus"] = False


def set_korean_font() -> None:
    """그래프 한글 깨짐 방지. 설치된 한글 폰트를 찾아 쓴다."""
    from matplotlib import font_manager

    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            return


def _color(idx: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(idx + 7)
    return tuple(int(c) for c in rng.integers(40, 230, size=3))


def draw_boxes(
    img: np.ndarray,
    boxes,
    labels,
    names: list[str] | None = None,
    scores=None,
    thickness: int = 3,
) -> np.ndarray:
    """RGB 이미지에 xyxy 박스와 라벨을 그린 사본을 돌려준다.

    OpenCV 기본 글꼴은 한글을 못 그리므로 박스 위에는 번호(와 점수)만 쓰고,
    약 이름은 범례나 표로 따로 보여준다.
    """
    out = np.ascontiguousarray(img.copy())
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    for i, (b, lab) in enumerate(zip(boxes, labels, strict=True)):
        color = _color(int(lab))
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"#{int(lab)}"
        if scores is not None:
            text += f" {float(scores[i]):.2f}"
        scale = max(0.6, out.shape[1] / 1200)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 3, max(th, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2,
        )
    return out


def show_samples(
    records: list[Record],
    image_dir: str | Path,
    class_map: list[ClassInfo],
    n: int = 6,
    cols: int = 3,
    seed: int = 0,
) -> None:
    """정답 박스를 그린 샘플 이미지 n장."""
    cat2label = {c["category_id"]: c["label_idx"] for c in class_map}
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(records), size=min(n, len(records)), replace=False)
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), picks, strict=False):
        r = records[int(i)]
        img = read_image_rgb(resolve_path(image_dir) / r["file_name"])
        labels = [cat2label[c] for c in r["category_ids"]]
        ax.imshow(draw_boxes(img, r["boxes"], labels))
        ax.set_title(f"{r['file_name']}\nlabels: {labels}", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_prediction(
    img: np.ndarray,
    pred: Prediction,
    target_boxes=None,
    target_labels=None,
    title: str = "",
) -> None:
    """정답(왼쪽)과 예측(오른쪽)을 나란히."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if target_boxes is not None:
        axes[0].imshow(draw_boxes(img, target_boxes, target_labels))
        axes[0].set_title("정답")
    axes[1].imshow(draw_boxes(img, pred["boxes"], pred["labels"], scores=pred["scores"]))
    axes[1].set_title("예측")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_df: pd.DataFrame, top_n: int = 40) -> None:
    """eda.class_distribution 결과를 막대그래프로. 클래스가 많으면 상위 top_n개만."""
    df = class_df.head(top_n)
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.3), 4))
    ax.bar(df["category_id"].astype(str), df["n_boxes"], color="#3b6fd8")
    ax.set_xlabel("category_id")
    ax.set_ylabel("박스 수")
    ax.set_title(f"클래스별 박스 수 (상위 {len(df)}개 / 전체 {len(class_df)}개)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_hist(series: pd.Series, title: str, bins: int = 30, xlabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(series.dropna(), bins=bins, color="#3b6fd8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("개수")
    plt.tight_layout()
    plt.show()


def plot_box_centers(boxes_df: pd.DataFrame) -> None:
    """알약 중심 위치 분포 (이미지 크기로 정규화). 한쪽에 몰려 있으면 크롭·이동 증강 근거."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(boxes_df["cx_rel"], boxes_df["cy_rel"], s=4, alpha=0.4, color="#3b6fd8")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("박스 중심 위치")
    plt.tight_layout()
    plt.show()


def plot_experiments(exp_df: pd.DataFrame, metric: str = "map") -> None:
    """outputs 실험 비교표를 막대그래프로 (notebooks/03)."""
    df = exp_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.4)))
    ax.barh(df["exp_name"], df[metric], color="#3b6fd8")
    for y, v in enumerate(df[metric]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title("실험별 검증 점수")
    plt.tight_layout()
    plt.show()


def eda_grid(items, cols=5, rows=None, title=""):
    """이미지와 설명을 같은 격자에 표시한다."""
    import matplotlib.pyplot as plt

    rows = rows or max(1, (len(items) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image, caption) in zip(axes.flat, items, strict=False):
        ax.imshow(image)
        ax.set_title(caption, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_random_photos(folder, n=50, rows=None, seed=42):
    import random
    from pathlib import Path

    from PIL import Image

    paths = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    chosen = random.Random(seed).sample(paths, min(n, len(paths)))
    eda_grid(
        [(Image.open(p).convert("RGB"), p.name) for p in chosen],
        rows=rows,
        title=f"무작위 사진 {len(chosen)}장 · seed {seed}",
    )


def plot_eda_counts(table, x, y, title):
    import matplotlib.pyplot as plt

    table.plot.bar(x=x, y=y, figsize=(15, 4), legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_class_crops(ctx):
    from PIL import Image

    labels = {c["category_id"]: c["label_idx"] for c in ctx["class_map"]}
    crops = []
    for cid in sorted(ctx["categories"]):
        rec = next((r for r in ctx["records"] if cid in r["category_ids"]), None)
        if rec is None:
            continue
        i = rec["category_ids"].index(cid)
        image = Image.open(ctx["image_dir"] / rec["file_name"]).convert("RGB")
        crops.append(
            (
                image.crop(tuple(rec["boxes"][i])),
                f"label {labels[cid]} / code {cid}\n{ctx['categories'][cid]}",
            )
        )
    for start in range(0, len(crops), 20):
        eda_grid(crops[start : start + 20], title="클래스별 대표 알약 · 라벨 대조용")


def plot_flip_example(ctx):
    from PIL import Image, ImageDraw, ImageOps

    rec = ctx["records"][0]
    im = Image.open(ctx["image_dir"] / rec["file_name"]).convert("RGB")
    left = im.copy()
    right = ImageOps.mirror(im)
    for x1, y1, x2, y2 in rec["boxes"]:
        ImageDraw.Draw(left).rectangle((x1, y1, x2, y2), outline="lime", width=5)
        ImageDraw.Draw(right).rectangle(
            (im.width - x2, y1, im.width - x1, y2), outline="lime", width=5
        )
    eda_grid(
        [(left, "원본 + 정답 박스"), (right, "좌우 반전 + 변환한 박스")],
        cols=2,
        title="증강 시 사진과 박스를 함께 변환",
    )


def plot_quality_audit(audit, ctx, kind="labels"):
    """로컬 원본에서 교정 근거를 그린다. 생성 그림은 공개 저장소에 넣지 않는다."""
    from PIL import Image, ImageDraw

    if kind == "models":
        rows = audit["model_reference"]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.18, [r["before"] for r in rows], 0.36, label="교정 전")
        ax.bar(x + 0.18, [r["after"] for r in rows], 0.36, label="교정 후")
        ax.set_xticks(x, [r["condition"] for r in rows])
        ax.set_ylabel("mAP@[0.75:0.95]")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        plt.show()
        return
    entries = audit["label_corrections" if kind == "labels" else "added_validation_boxes"]
    for row in entries:
        with Image.open(ctx["image_dir"] / row["file_name"]) as source:
            image = source.convert("RGB")
        if kind == "labels":
            items = [(image.crop(tuple(row["box"])), "문제 사진 확대")]
            for ref, caption in zip(
                row["references"],
                ["원래 정답 이름의 참고 사진", "교정 이름의 참고 사진"],
                strict=True,
            ):
                with Image.open(ctx["image_dir"] / ref["file_name"]) as source:
                    crop = source.convert("RGB").crop(tuple(ref["box"]))
                items.append((crop, caption))
            title = row["old_name"] + " → " + row["corrected_name"]
        else:
            before, after = image.copy(), image.copy()
            for box in row["existing_boxes"]:
                ImageDraw.Draw(before).rectangle(tuple(box), outline="green", width=5)
                ImageDraw.Draw(after).rectangle(tuple(box), outline="green", width=5)
            ImageDraw.Draw(after).rectangle(tuple(row["new_box_xyxy"]), outline="orange", width=7)
            items = [(before, "기존 정답"), (after, "누락 박스 추가 · 주황")]
            title = row["name"]
        eda_grid(items, cols=len(items), rows=1, title=title)


def plot_missing_candidates(ctx, candidates):
    """비공개 육안 기록 파일 대신 원본과 파일명 후보를 나란히 보여준다.

    파일명 차집합은 확인 후보이며 이 그림만으로 라벨 누락을 확정하지 않는다.
    """
    from PIL import Image, ImageDraw

    records = {r["file_name"]: r for r in ctx["records"]}
    items = []
    for row in candidates.to_dict("records"):
        record = records[row["file_name"]]
        with Image.open(ctx["image_dir"] / record["file_name"]) as source:
            image = source.convert("RGB")
        for box in record["boxes"]:
            ImageDraw.Draw(image).rectangle(tuple(box), outline="green", width=5)
        items.append((image, row["file_name"] + "\n확인 후보: " + row["파일명 기준 누락 코드"]))
    eda_grid(items, cols=4, title="파일명 기준 후보와 원본 정답 박스 · 육안 확인 필요")
