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



def show_all_records(
    records: list[Record],
    image_dir: str | Path,
    class_map: list[ClassInfo],
    page: int = 1,
    per_page: int = 6,
    cols: int = 2,
) -> None:
    """전체 records를 파일명순으로 정렬해 페이지별로 정답 박스와 클래스 정보를 표시."""
    if not records:
        print("표시할 이미지가 없습니다.")
        return
    if per_page < 1 or cols < 1:
        raise ValueError("per_page와 cols는 1 이상이어야 합니다.")

    ordered = sorted(records, key=lambda r: r["file_name"])
    total_pages = (len(ordered) + per_page - 1) // per_page

    if not 1 <= page <= total_pages:
        raise ValueError(f"page는 1~{total_pages} 범위여야 합니다.")

    cat_info = {c["category_id"]: c for c in class_map}
    start = (page - 1) * per_page
    selected = ordered[start:start + per_page]

    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(8 * cols, 8 * rows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for index, (ax, record) in enumerate(
        zip(axes.ravel(), selected), start=start + 1
    ):
        img = read_image_rgb(
            resolve_path(image_dir) / record["file_name"]
        )

        labels = [
            cat_info[category_id]["label_idx"]
            for category_id in record["category_ids"]
        ]
        ax.imshow(draw_boxes(img, record["boxes"], labels))

        # 같은 클래스의 객체가 여러 개여도 목록에는 한 번만 표시
        descriptions = []
        for category_id in dict.fromkeys(record["category_ids"]):
            info = cat_info[category_id]
            descriptions.append(
                f"label={info['label_idx']} | "
                f"category_id={category_id} | {info['name']}"
            )

        ax.set_title(
            f"[{index}/{len(ordered)}] {record['file_name']}\n"
            + "\n".join(descriptions),
            fontsize=9,
        )

    fig.suptitle(
        f"정답 검토 — {page}/{total_pages} 페이지",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def show_by_filename(
    records: list[Record],
    image_dir: str | Path,
    class_map: list[ClassInfo],
    file_name: str,
) -> None:
    """파일명이 일치하는 이미지의 정답 박스를 표시한다."""
    matched = [r for r in records if r["file_name"] == file_name]

    if not matched:
        print(f"해당 파일을 records에서 찾을 수 없습니다: {file_name}")
        return

    show_samples(
        matched,
        image_dir,
        class_map,
        n=len(matched),
        cols=1,
    )


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
