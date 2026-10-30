"""모델 생성 창구. 새 모델을 추가하면 아래 MODELS에 한 줄만 추가한다.

모델 파일끼리는 서로 import 하지 않는다 (모델 A·B가 충돌 없이 동시에 작업하기 위해).
"""

from __future__ import annotations

from src.models.base import BaseDetector


def build_model(cfg: dict, num_classes: int) -> BaseDetector:
    family = cfg["model"]["family"]
    if family == "yolo":
        from src.models.yolo import YoloDetector

        return YoloDetector(cfg, num_classes)
    if family == "rcnn":
        from src.models.rcnn import RcnnDetector

        return RcnnDetector(cfg, num_classes)
    if family == "ocr_rerank":  # 실험 D11: 각인 OCR로 이름 다시 고르기
        from src.models.ocr_rerank import OcrRerankDetector

        return OcrRerankDetector(cfg, num_classes)
    raise ValueError(f"알 수 없는 model.family: {family} (yolo / rcnn / ocr_rerank)")
