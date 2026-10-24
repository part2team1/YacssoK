"""EDA 분석 함수 (데이터 A 담당).

노트북(notebooks/01_eda.ipynb)은 이 함수들을 불러 결과를 보고 해석만 한다.
모든 함수는 pandas DataFrame을 돌려주므로 그대로 표로 보거나 그래프로 그릴 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.interfaces import MAX_PILLS_PER_IMAGE, ClassInfo, Record
from src.utils import resolve_path


def records_to_frames(
    records: list[Record], class_map: list[ClassInfo] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Record 목록을 이미지 표(images_df)와 박스 표(boxes_df)로 펼친다."""
    names = {c["category_id"]: c["name"] for c in class_map} if class_map else {}
    labels = {c["category_id"]: c["label_idx"] for c in class_map} if class_map else {}
    images_df = pd.DataFrame(
        [
            {
                "image_id": r["image_id"],
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
                "n_objects": len(r["boxes"]),
            }
            for r in records
        ]
    )
    rows = []
    for r in records:
        for (x1, y1, x2, y2), cid in zip(r["boxes"], r["category_ids"], strict=True):
            w, h = x2 - x1, y2 - y1
            rows.append(
                {
                    "image_id": r["image_id"],
                    "file_name": r["file_name"],
                    "category_id": cid,
                    "label_idx": labels.get(cid),
                    "name": names.get(cid, str(cid)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_w": w,
                    "box_h": h,
                    "area": w * h,
                    "aspect": w / h if h else None,
                    "rel_area": (w * h) / (r["width"] * r["height"])
                    if r["width"] and r["height"]
                    else None,
                    "cx_rel": (x1 + w / 2) / r["width"] if r["width"] else None,
                    "cy_rel": (y1 + h / 2) / r["height"] if r["height"] else None,
                }
            )
    boxes_df = pd.DataFrame(rows)
    return images_df, boxes_df


def dataset_summary(images_df: pd.DataFrame, boxes_df: pd.DataFrame) -> pd.Series:
    """한눈에 보는 요약 숫자."""
    per_class = boxes_df["category_id"].value_counts()
    return pd.Series(
        {
            "이미지 수": len(images_df),
            "박스 수": len(boxes_df),
            "클래스 수": boxes_df["category_id"].nunique(),
            "이미지당 평균 알약 수": round(images_df["n_objects"].mean(), 2),
            "클래스당 박스 수 (최소)": int(per_class.min()) if len(per_class) else 0,
            "클래스당 박스 수 (중앙값)": float(per_class.median()) if len(per_class) else 0,
            "클래스당 박스 수 (최대)": int(per_class.max()) if len(per_class) else 0,
            "이미지 크기 종류": images_df[["width", "height"]].drop_duplicates().shape[0],
        }
    )


def class_distribution(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """클래스별 박스 수와 등장 이미지 수 (많은 순)."""
    out = (
        boxes_df.groupby(["category_id", "name"])
        .agg(n_boxes=("file_name", "size"), n_images=("file_name", "nunique"))
        .reset_index()
        .sort_values("n_boxes", ascending=False)
    )
    out["share_%"] = (out["n_boxes"] / out["n_boxes"].sum() * 100).round(2)
    return out.reset_index(drop=True)


def image_size_stats(images_df: pd.DataFrame) -> pd.DataFrame:
    """이미지 해상도 종류별 개수. 한 가지뿐이면 리사이즈 전략이 단순해진다."""
    return (
        images_df.groupby(["width", "height"])
        .size()
        .rename("n_images")
        .reset_index()
        .sort_values("n_images", ascending=False)
    )


def objects_per_image(images_df: pd.DataFrame) -> pd.DataFrame:
    """이미지당 알약 수 분포. 과제 조건(최대 4개)과 다른 이미지가 있는지 본다."""
    return images_df["n_objects"].value_counts().sort_index().rename("n_images").reset_index()


def bbox_stats(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """박스 크기·비율·상대 면적 통계. 작은 알약이 많으면 입력 해상도를 키우는 실험(D6) 근거가 된다."""
    cols = ["box_w", "box_h", "area", "aspect", "rel_area"]
    return boxes_df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(3)


def find_label_issues(
    images_df: pd.DataFrame, boxes_df: pd.DataFrame, image_dir: str | Path | None = None
) -> pd.DataFrame:
    """정제(데이터 B)가 확인할 의심 항목 목록.

    - out_of_bounds: 박스가 이미지 밖으로 나감
    - tiny_or_empty: 너비나 높이가 2픽셀 이하
    - too_many: 이미지당 알약이 과제 조건(4개)보다 많음
    - no_boxes: 박스가 하나도 없는 이미지
    - heavy_overlap: 같은 이미지 안에서 두 박스가 크게 겹침 (중복 라벨 의심)
    - missing_file: 이미지 파일이 없음
    """
    issues = []
    sizes = images_df.set_index("file_name")[["width", "height"]].to_dict("index")
    for row in boxes_df.itertuples():
        s = sizes.get(row.file_name, {})
        if s and (
            row.x1 < -1 or row.y1 < -1 or row.x2 > s["width"] + 1 or row.y2 > s["height"] + 1
        ):
            issues.append(
                {
                    "file_name": row.file_name,
                    "issue": "out_of_bounds",
                    "category_id": row.category_id,
                }
            )
        if row.box_w <= 2 or row.box_h <= 2:
            issues.append(
                {
                    "file_name": row.file_name,
                    "issue": "tiny_or_empty",
                    "category_id": row.category_id,
                }
            )
    for row in images_df.itertuples():
        if row.n_objects > MAX_PILLS_PER_IMAGE:
            issues.append({"file_name": row.file_name, "issue": "too_many", "category_id": None})
        if row.n_objects == 0:
            issues.append({"file_name": row.file_name, "issue": "no_boxes", "category_id": None})
    for file_name, g in boxes_df.groupby("file_name"):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if _iou(b[i], b[j]) > 0.7:
                    issues.append(
                        {"file_name": file_name, "issue": "heavy_overlap", "category_id": None}
                    )
    if image_dir is not None:
        image_dir = resolve_path(image_dir)
        for name in images_df["file_name"]:
            if not (image_dir / name).exists():
                issues.append({"file_name": name, "issue": "missing_file", "category_id": None})
    return pd.DataFrame(issues, columns=["file_name", "issue", "category_id"])


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_box_size(boxes_df: pd.DataFrame) -> pd.DataFrame:
    """클래스별 평균 박스 크기 (작은 순). 작은 알약은 해상도·증강 실험 후보."""
    out = (
        boxes_df.groupby(["category_id", "name"])
        .agg(
            n_boxes=("area", "size"),
            mean_w=("box_w", "mean"),
            mean_h=("box_h", "mean"),
            mean_rel_area=("rel_area", "mean"),
        )
        .reset_index()
        .sort_values("mean_rel_area")
    )
    return out.round(3).reset_index(drop=True)