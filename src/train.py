"""학습 실행 로직 (PM·실험리드 Y 담당).

설정 파일 하나 → 모델 생성 → 학습 → 검증 예측 → 점수 계산 → outputs/{exp_name}/ 에 저장.
모든 실험이 같은 순서로 저장되므로 notebooks/03_experiment_compare.ipynb 에서 한 번에 비교할 수 있다.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from src.dataset import PillDataset, load_records, load_split
from src.interfaces import OUTPUT_DIR, Target
from src.metrics import compute_map
from src.models import build_model
from src.utils import (
    load_class_map,
    predictions_to_json,
    resolve_path,
    save_json,
    save_yaml,
    set_seed,
)


def val_targets(cfg: dict) -> tuple[list, list[Target]]:
    """검증셋 이미지 경로와 정답(Target)을 같은 순서로 돌려준다."""
    d = cfg["data"]
    records, _ = load_records(d["train_annotations"])
    val_records = load_split(records, f"{d['split_dir']}/{d['version']}/val.txt")
    ds = PillDataset(val_records, d["train_images"], load_class_map(d["class_map"]))
    paths = [resolve_path(d["train_images"]) / r["file_name"] for r in val_records]
    targets = []
    for i in range(len(ds)):
        _, t = ds[i]
        targets.append(t)
    return paths, targets


def evaluate(model, cfg: dict) -> tuple[dict, list]:
    paths, targets = val_targets(cfg)
    preds = model.predict(paths)
    metrics = compute_map(preds, targets, iou_thresholds=cfg["eval"].get("iou_thresholds"))
    return metrics, preds


def run(cfg: dict) -> dict:
    set_seed(cfg.get("seed", 42))
    exp_name = cfg["exp_name"]
    out_dir = OUTPUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config.yaml")  # 실행에 쓴 설정을 그대로 남김

    class_map = load_class_map(cfg["data"]["class_map"])
    model = build_model(cfg, num_classes=len(class_map))

    start = time.time()
    weights = model.fit(out_dir)
    train_min = round((time.time() - start) / 60, 2)

    model.load(weights)
    metrics, preds = evaluate(model, cfg)
    save_json(predictions_to_json(preds), out_dir / "preds_val.json")

    summary = {
        "exp_name": exp_name,
        "owner": cfg.get("owner", ""),
        "model": f"yolo:{cfg['model']['weights']}"
        if cfg["model"]["family"] == "yolo"
        else f"rcnn:{cfg['model']['arch']}",
        "data_version": cfg["data"]["version"],
        "augment": (cfg.get("augment") or {}).get("name", "none"),
        "epochs": cfg["train"]["epochs"],
        "img_size": cfg["data"].get("img_size"),
        "train_min": train_min,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": cfg.get("note", ""),
        **metrics,
    }
    save_json(summary, out_dir / "metrics.json")
    with open(OUTPUT_DIR / "experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(
        f"\n[{exp_name}] 검증 mAP@[0.75:0.95]={summary.get('map')}  mAP75={summary.get('map_75')}  ({train_min}분)"
    )
    print(f"결과 저장: {out_dir}")
    return summary
