"""팀 표준 형식 검사. 실행: uv run pytest

합성 샘플 데이터(data/sample)를 만들어 검사하므로 실제 대회 데이터 없이도 돈다.
dataset.py나 utils.py를 고친 PR은 이 테스트가 통과해야 병합한다.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from src.interfaces import PROJECT_ROOT
from src.utils import xywh_to_xyxy, xyxy_to_xywh, xyxy_to_yolo

SAMPLE = PROJECT_ROOT / "data" / "sample"


@pytest.fixture(scope="module")
def sample_records():
    if not (SAMPLE / "train_annotations").exists():
        subprocess.run(
            [sys.executable, "scripts/make_sample_data.py", "--n-train", "20", "--n-test", "3"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    from src.dataset import build_class_map, load_records

    records, cats = load_records(SAMPLE / "train_annotations")
    return records, build_class_map(records, cats)


def test_box_conversions_roundtrip():
    xyxy = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert torch.allclose(xywh_to_xyxy(xyxy_to_xywh(xyxy)), xyxy)
    yolo = xyxy_to_yolo(xyxy, img_w=100, img_h=100)
    assert torch.allclose(yolo, torch.tensor([[0.30, 0.50, 0.40, 0.60]]))


def test_records_are_xyxy(sample_records):
    records, _ = sample_records
    assert records
    for r in records:
        assert len(r["boxes"]) == len(r["category_ids"]) <= 4
        for x1, y1, x2, y2 in r["boxes"]:
            assert x2 > x1 and y2 > y1


def test_dataset_output_format(sample_records):
    from src.dataset import PillDataset

    records, class_map = sample_records
    ds = PillDataset(records, SAMPLE / "train_images", class_map)
    img, t = ds[0]
    assert img.dtype == torch.float32 and img.shape[0] == 3
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert t["boxes"].dtype == torch.float32 and t["boxes"].shape[1] == 4
    assert t["labels"].dtype == torch.int64
    assert int(t["labels"].min()) >= 0 and int(t["labels"].max()) < len(class_map)
    assert t["orig_size"] == tuple(img.shape[1:])


def test_perfect_prediction_scores_one(sample_records):
    """정답을 그대로 예측으로 넣으면 mAP@[0.75:0.95] = 1 이어야 한다 (평가 코드 검증)."""
    from src.dataset import PillDataset
    from src.interfaces import Prediction
    from src.metrics import compute_map

    records, class_map = sample_records
    ds = PillDataset(records[:10], SAMPLE / "train_images", class_map)
    targets = [ds[i][1] for i in range(len(ds))]
    preds = [
        Prediction(
            image_id=t["image_id"],
            file_name="",
            boxes=t["boxes"],
            scores=torch.ones(len(t["labels"])),
            labels=t["labels"],
        )
        for t in targets
    ]
    result = compute_map(preds, targets, iou_thresholds=[0.75, 0.8, 0.85, 0.9, 0.95])
    assert result["map"] == pytest.approx(1.0)
