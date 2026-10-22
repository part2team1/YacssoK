"""평가 함수 (PM·실험리드 Y 담당).

검증 점수는 대회 리더보드 지표와 같은 방식으로 계산해야 내부 점수와 Kaggle 점수가 같은 방향으로 움직인다.
configs의 eval.iou_thresholds로 IoU 구간을 맞춘다 (null이면 COCO 기본 0.50:0.95).
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from src.interfaces import Prediction, Target


def compute_map(
    preds: list[Prediction],
    targets: list[Target],
    iou_thresholds: list[float] | None = None,
    class_metrics: bool = False,
) -> dict[str, float]:
    """preds와 targets는 같은 순서(같은 이미지)여야 한다."""
    if len(preds) != len(targets):
        raise ValueError(f"예측 {len(preds)}개와 정답 {len(targets)}개의 개수가 다릅니다.")
    if iou_thresholds is not None:
        iou_thresholds = [float(t) for t in iou_thresholds]
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=iou_thresholds,
        class_metrics=class_metrics,
    )
    metric.update(
        [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in preds],
        [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets],
    )
    result = metric.compute()
    out = {}
    for key, value in result.items():
        if key == "classes":
            continue
        if value.numel() == 1:
            if float(value) == -1:  # 계산 대상이 아닌 항목 (예: IoU 0.75~0.95에서는 map_50 없음)
                continue
            out[key] = round(float(value), 5)
        elif class_metrics and key.endswith("_per_class"):
            out[key] = [round(float(v), 5) for v in value]
    return out


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N,4] xyxy와 [M,4] xyxy의 IoU 행렬 [N, M]."""
    from torchvision.ops import box_iou as _box_iou

    return _box_iou(a, b)


def match_predictions(
    preds: list[Prediction],
    targets: list[Target],
    iou_thr: float = 0.5,
    conf_thr: float = 0.3,
) -> list[dict]:
    """오분류 분석용: 정답 박스마다 가장 많이 겹친 예측을 찾아 맞음/틀린 클래스/놓침으로 나눈다."""
    rows = []
    for p, t in zip(preds, targets, strict=True):
        keep = p["scores"] >= conf_thr
        pb, pl, ps = p["boxes"][keep], p["labels"][keep], p["scores"][keep]
        ious = (
            box_iou(t["boxes"], pb)
            if len(pb) and len(t["boxes"])
            else torch.zeros(len(t["boxes"]), 0)
        )
        for i, gt_label in enumerate(t["labels"].tolist()):
            if ious.shape[1] == 0 or float(ious[i].max()) < iou_thr:
                rows.append(
                    {
                        "image_id": t["image_id"],
                        "gt": gt_label,
                        "pred": None,
                        "score": None,
                        "result": "missed",
                    }
                )
                continue
            j = int(ious[i].argmax())
            pred_label = int(pl[j])
            rows.append(
                {
                    "image_id": t["image_id"],
                    "gt": gt_label,
                    "pred": pred_label,
                    "score": round(float(ps[j]), 4),
                    "result": "correct" if pred_label == gt_label else "wrong_class",
                }
            )
    return rows


def confusion_pairs(rows: list[dict], top_n: int = 20) -> list[tuple[int, int, int]]:
    """자주 헷갈리는 (정답, 예측, 횟수) 쌍."""
    counts: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["result"] == "wrong_class":
            key = (r["gt"], r["pred"])
            counts[key] = counts.get(key, 0) + 1
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    return [(g, p, n) for (g, p), n in pairs]


def summarize_scores(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {}
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}
