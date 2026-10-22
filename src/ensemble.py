"""여러 모델의 예측 합치기 (모델 B 담당).

스켈레톤: 같은 이미지의 예측을 모두 모은 뒤 클래스별 NMS로 겹치는 박스를 정리한다.
고도화 아이디어 (실험 계획 P2, P3): WBF(Weighted Boxes Fusion), 모델별 가중치, TTA.
"""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms

from src.interfaces import MAX_PILLS_PER_IMAGE, Prediction


def ensemble_nms(
    pred_lists: list[list[Prediction]],
    iou_thr: float = 0.5,
    max_det: int = MAX_PILLS_PER_IMAGE,
    weights: list[float] | None = None,
) -> list[Prediction]:
    """pred_lists[k]는 모델 k의 예측 목록. 이미지는 file_name으로 맞춘다."""
    weights = weights or [1.0] * len(pred_lists)
    by_file: dict[str, list[tuple[Prediction, float]]] = {}
    for preds, w in zip(pred_lists, weights, strict=True):
        for p in preds:
            by_file.setdefault(p["file_name"], []).append((p, w))

    merged: list[Prediction] = []
    for file_name, items in by_file.items():
        boxes = torch.cat([p["boxes"] for p, _ in items])
        scores = torch.cat([p["scores"] * w for p, w in items]).clamp(max=1.0)
        labels = torch.cat([p["labels"] for p, _ in items])
        keep = (
            batched_nms(boxes, scores, labels, iou_thr)[:max_det]
            if len(boxes)
            else torch.empty(0, dtype=torch.long)
        )
        merged.append(
            Prediction(
                image_id=items[0][0]["image_id"],
                file_name=file_name,
                boxes=boxes[keep],
                scores=scores[keep],
                labels=labels[keep],
            )
        )
    return merged
