"""추론 실행 로직 (PM·실험리드 Y 담당). 학습된 실험 폴더 하나로 테스트 이미지를 예측한다."""

from __future__ import annotations

from pathlib import Path

from src.dataset import list_images
from src.interfaces import OUTPUT_DIR, Prediction
from src.models import build_model
from src.models.base import BaseDetector
from src.utils import load_class_map, load_config, predictions_to_json, save_json


def load_experiment(exp_name: str) -> tuple[dict, BaseDetector]:
    """outputs/{exp_name}/config.yaml 과 best.pt 로 모델을 복원한다."""
    out_dir = OUTPUT_DIR / exp_name
    cfg = load_config(out_dir / "config.yaml")
    class_map = load_class_map(cfg["data"]["class_map"])
    model = build_model(cfg, num_classes=len(class_map))
    model.load(out_dir / "best.pt")
    return cfg, model


def predict_test(
    exp_name: str, image_dir: str | Path | None = None, conf_thr: float | None = None
) -> list[Prediction]:
    cfg, model = load_experiment(exp_name)
    image_dir = image_dir or cfg["data"]["test_images"]
    paths = list_images(image_dir)
    preds = model.predict(paths, conf_thr=conf_thr)
    save_json(predictions_to_json(preds), OUTPUT_DIR / exp_name / "preds_test.json")
    print(f"[{exp_name}] 테스트 이미지 {len(paths)}장 예측 → outputs/{exp_name}/preds_test.json")
    return preds
