"""EDA 분석 함수 (데이터 A 담당).

노트북(notebooks/01_eda.ipynb)은 이 함수들을 불러 결과를 보고 해석만 한다.
모든 함수는 pandas DataFrame을 돌려주므로 그대로 표로 보거나 그래프로 그릴 수 있다.
"""

from __future__ import annotations

import json
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
    """클래스별 평균 박스 크기 (작은 순). 작은 알약은 해상도-증강 실험 후보."""
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


def background_counts(ann_dir) -> pd.DataFrame:
    """어노테이션 images[]의 back_color(촬영 배경)별 이미지 수."""
    from src.utils import load_json, resolve_path

    rows = {}
    for path in resolve_path(ann_dir).rglob("*.json"):
        for img in load_json(path).get("images", []):
            rows[img["file_name"]] = img.get("back_color", "없음")
    return pd.Series(rows).value_counts().rename("n_images").reset_index()


def load_eda_context(config_path="configs/base.yaml"):
    """설정의 상대 경로로 원본만 읽는다. 개인 PC 경로를 코드에 넣지 않는다."""
    from src.dataset import build_class_map, load_records
    from src.utils import load_config

    cfg = load_config(config_path)["data"]
    image_dir = resolve_path(cfg["train_images"])
    ann_dir = resolve_path(cfg["train_annotations"])
    test_dir = resolve_path(cfg["test_images"])
    if not image_dir.is_dir() or not ann_dir.exists():
        raise FileNotFoundError("설정 파일의 원본 학습 이미지·JSON 경로를 확인하세요.")
    records, categories = load_records(ann_dir)
    class_map = build_class_map(records, categories)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    json_paths = sorted(ann_dir.rglob("*.json")) if ann_dir.is_dir() else [ann_dir]
    meta = []
    for path in json_paths:
        meta.extend(json.loads(path.read_text(encoding="utf-8-sig")).get("images", []))
    images, boxes = records_to_frames(records, class_map)
    return dict(
        records=records,
        categories=categories,
        class_map=class_map,
        paths=paths,
        json_paths=json_paths,
        meta=pd.DataFrame(meta),
        image_dir=image_dir,
        test_dir=test_dir,
        images=images,
        boxes=boxes,
    )


def filename_parts(name):
    """파일명 앞쪽 알약 코드와 나머지 토큰을 나눈다."""
    tokens = Path(name).stem.split("_")
    return [int(v) for v in tokens[0].removeprefix("K-").split("-") if v.isdigit()], tokens[1:]


def eda_inventory(ctx):
    return pd.DataFrame(
        {
            "항목": ["학습 사진", "JSON", "클래스", "정답 박스"],
            "개수": [
                len(ctx["paths"]),
                len(ctx["json_paths"]),
                len(ctx["class_map"]),
                len(ctx["boxes"]),
            ],
        }
    )


def filename_token_table(ctx):
    _, tokens = filename_parts(ctx["paths"][0].name)
    fields = [
        "drug_dir 대조 필요",
        "light_color 대조 필요",
        "back_color 대조 필요",
        "미확정",
        "camera_la 대조",
        "camera_lo 대조",
        "size 대조",
    ]
    return pd.DataFrame(
        {"토큰 위치": range(1, len(tokens) + 1), "값": tokens, "대조 항목": fields[: len(tokens)]}
    )


def filename_metadata(ctx):
    m = ctx["meta"]
    cols = [
        c
        for c in [
            "file_name",
            "dl_idx",
            "drug_dir",
            "light_color",
            "back_color",
            "camera_la",
            "camera_lo",
            "size",
        ]
        if c in m
    ]
    return m.loc[m.file_name.eq(ctx["paths"][0].name), cols].drop_duplicates()


def eda_class_counts(ctx):
    from collections import Counter

    counts = Counter(c for r in ctx["records"] for c in r["category_ids"])
    labels = {c["category_id"]: c["label_idx"] for c in ctx["class_map"]}
    return (
        pd.DataFrame(
            [
                {
                    "category_id": c,
                    "label_idx": labels[c],
                    "이름": ctx["categories"][c],
                    "박스 수": n,
                }
                for c, n in counts.items()
            ]
        )
        .sort_values("박스 수", ascending=False)
        .reset_index(drop=True)
    )


def filename_missing_candidates(ctx):
    rows = []
    for r in ctx["records"]:
        missing = sorted(set(filename_parts(r["file_name"])[0]) - set(r["category_ids"]))
        if missing:
            rows.append(
                {
                    "file_name": r["file_name"],
                    "파일명 기준 누락 코드": str(missing),
                    "현재 박스": len(r["boxes"]),
                }
            )
    return pd.DataFrame(rows, columns=["file_name", "파일명 기준 누락 코드", "현재 박스"])


def file_alignment(ctx):
    files = {p.name for p in ctx["paths"]}
    names = {r["file_name"] for r in ctx["records"]}
    return pd.DataFrame(
        {
            "검사": ["사진 있으나 JSON 이미지 기록 없음", "JSON 이미지 기록 있으나 사진 없음"],
            "개수": [len(files - names), len(names - files)],
        }
    )


def image_integrity(ctx):
    import hashlib

    from PIL import Image

    by_name = {r["file_name"]: r for r in ctx["records"]}
    corrupt = []
    mismatch = []
    hashes = {}
    for p in ctx["paths"]:
        try:
            with Image.open(p) as im:
                im.load()
                actual = im.size
            r = by_name.get(p.name)
            if r and actual != (r["width"], r["height"]):
                mismatch.append(p.name)
            hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p.name)
        except (OSError, ValueError):
            corrupt.append(p.name)
    return pd.DataFrame(
        {
            "검사": ["읽기 오류", "JSON 크기 불일치", "동일 바이트 중복 그룹"],
            "개수": [len(corrupt), len(mismatch), sum(len(v) > 1 for v in hashes.values())],
        }
    )


def metadata_counts(ctx, field):
    if field not in ctx["meta"]:
        return pd.DataFrame(columns=[field, "고유 사진 수"])
    return (
        ctx["meta"]
        .groupby(field, dropna=False)
        .file_name.nunique()
        .rename("고유 사진 수")
        .reset_index()
    )


def pill_combinations(ctx):
    from collections import Counter

    counts = Counter(tuple(sorted(filename_parts(r["file_name"])[0])) for r in ctx["records"])
    return pd.DataFrame([{"원본 코드 조합": str(c), "사진 수": n} for c, n in counts.most_common()])


def angle_alignment(ctx):
    rows = []
    m = ctx["meta"]
    for r in ctx["records"]:
        _, parts = filename_parts(r["file_name"])
        values = m.loc[m.file_name.eq(r["file_name"]), "camera_la"].dropna().unique()
        ok = len(parts) > 4 and len(values) > 0 and all(float(v) == float(parts[4]) for v in values)
        rows.append({"file_name": r["file_name"], "일치": ok})
    return pd.DataFrame(rows).groupby("일치").size().rename("사진 수").reset_index()


def test_image_inventory(ctx):
    from collections import Counter

    from PIL import Image

    folder = ctx["test_dir"]
    if not folder.is_dir():
        raise FileNotFoundError("data.test_images에 테스트 폴더를 연결하세요.")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    sizes = Counter()
    errors = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                im.load()
                sizes[im.size] += 1
        except (OSError, ValueError):
            errors += 1
    return pd.DataFrame(
        [
            {"가로": w, "세로": h, "사진 수": n, "전체 읽기 오류": errors}
            for (w, h), n in sizes.items()
        ]
    )


def load_quality_audit(ctx, config_path):
    """공개 파일에는 사진을 넣지 않고 로컬 원본·기존 교정표로 감사 자료를 만든다.

    원래 라벨을 기준으로 종류 교환을 찾으므로 교정 후 자료를 다시 교정하지 않는다.
    모델 점수는 별도 실행에서 검증된 집계 기록을 읽으며 여기서 재평가하지 않는다.
    """
    import csv

    from src.utils import load_config

    config = load_config(config_path)["eda_audit"]
    with resolve_path(config["manual_fix"]).open(encoding="utf-8-sig", newline="") as handle:
        fixes = list(csv.DictReader(handle))
    records = {r["file_name"]: r for r in ctx["records"]}
    classes = {c["category_id"]: c for c in ctx["class_map"]}
    excluded = {row["file_name"] for row in fixes if row["action"] == "class"}
    labels, missing = [], []
    for row in fixes:
        name, category = row["file_name"], int(row["category_id"])
        if name not in records:
            raise ValueError(f"교정표의 사진이 원본 자료에 없습니다: {name}")
        rec = records[name]
        if row["action"] == "class":
            indices = [i for i, value in enumerate(rec["category_ids"]) if value == category]
            if len(indices) != 1:
                raise ValueError(f"종류 교정 대상이 유일하지 않습니다: {name}, {category}")
            replacement = int(row["new_value"])
            references = []
            for cid in [category, replacement]:
                ref = next(
                    (
                        r
                        for r in sorted(ctx["records"], key=lambda r: r["file_name"])
                        if r["file_name"] not in excluded and cid in r["category_ids"]
                    ),
                    None,
                )
                if ref is None:
                    raise ValueError(f"참고 사진이 없습니다: {cid}")
                i = ref["category_ids"].index(cid)
                references.append({"file_name": ref["file_name"], "box": ref["boxes"][i]})
            labels.append(
                {
                    "file_name": name,
                    "old_name": classes[category]["name"],
                    "corrected_name": classes[replacement]["name"],
                    "old_category_id": category,
                    "new_category_id": replacement,
                    "box": rec["boxes"][indices[0]],
                    "references": references,
                }
            )
        elif row["action"] == "add":
            box = [float(v) for v in row["new_value"].split()]
            if len(box) != 4:
                raise ValueError(f"추가 박스는 xyxy 4개여야 합니다: {name}")
            missing.append(
                {
                    "file_name": name,
                    "category_id": category,
                    "name": classes[category]["name"],
                    "new_box_xyxy": box,
                    "existing_boxes": rec["boxes"],
                }
            )
    evaluation = json.loads(resolve_path(config["reevaluation"]).read_text(encoding="utf-8"))
    return {
        "label_corrections": labels,
        "added_validation_boxes": missing,
        "model_reference": evaluation["runs"],
        "evaluation_scope": evaluation["scope"],
    }


def quality_audit_table(audit, kind="labels"):
    """원시 박스·사진 참조와 요약 표를 분리한다."""
    keys = {
        "labels": "label_corrections",
        "missing": "added_validation_boxes",
        "models": "model_reference",
    }
    excluded = {"box", "references", "existing_boxes"}
    return pd.DataFrame(
        [{k: v for k, v in row.items() if k not in excluded} for row in audit[keys[kind]]]
    )
