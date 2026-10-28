"""팀 공통 함수. 다른 프로젝트 모듈은 interfaces.py만 import 한다 (의존 방향 규칙)."""

from __future__ import annotations

import json
import os
import random
import re
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.interfaces import PROJECT_ROOT, SEED, ClassInfo, Prediction

# ── 재현성 ───────────────────────────────────────────────────


def set_seed(seed: int = SEED) -> None:
    """random, numpy, torch 시드를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str | None = None) -> torch.device:
    """설정값이 없거나 'auto'면 사용 가능한 장치를 고른다 (cuda → mps → cpu)."""
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── 설정 파일 ────────────────────────────────────────────────


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """실험 YAML을 읽는다. `base:` 키가 있으면 그 파일을 먼저 읽고 덮어쓴다.

    예) configs/yolo_skeleton.yaml 에 `base: base.yaml`을 두면
        base.yaml의 값 위에 바뀐 값만 적으면 된다.
    """
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_update(deepcopy(base_cfg), cfg)
    return cfg


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def resolve_path(path: str | Path) -> Path:
    """설정 파일의 상대 경로는 레포 루트 기준으로 해석한다 (개인 PC 절대 경로 금지)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path

def yolo_data_yaml(cfg: dict) -> Path:
    """YOLO 학습에 쓸 data.yaml 경로. 오버샘플링(data.repeat_threshold)을 켜면 별도 파일을 쓴다.

    같은 데이터 버전을 쓰는 baseline 실험의 data.yaml을 덮어쓰지 않기 위해서다.
    """
    d = cfg["data"]
    root = resolve_path(d["yolo_dir"]) / d["version"]
    thr = float(d.get("repeat_threshold", 0) or 0)
    return root / ("data.yaml" if thr <= 0 else f"data_rfs{thr:g}.yaml")


# ── JSON ─────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 클래스 매핑 ──────────────────────────────────────────────


def load_class_map(path: str | Path) -> list[ClassInfo]:
    """data/class_map.json → label_idx 순서로 정렬된 리스트."""
    items = load_json(resolve_path(path))
    return sorted(items, key=lambda c: c["label_idx"])


def category_to_label(class_map: list[ClassInfo]) -> dict[int, int]:
    return {c["category_id"]: c["label_idx"] for c in class_map}


def label_to_category(class_map: list[ClassInfo]) -> dict[int, int]:
    return {c["label_idx"]: c["category_id"] for c in class_map}


def label_names(class_map: list[ClassInfo]) -> list[str]:
    return [c["name"] for c in class_map]


# ── 박스 변환 (모델 어댑터와 제출 스크립트에서만 사용) ─────────


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def xyxy_to_yolo(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """YOLO 라벨 파일용: 이미지 크기로 정규화한 중심 x, 중심 y, 너비, 높이."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h],
        dim=-1,
    )


# ── 이미지 ID ────────────────────────────────────────────────


def image_id_from_name(file_name: str) -> int:
    """테스트 이미지 파일 이름에서 image_id를 만든다.

    대회 테스트 이미지가 '1.png'처럼 숫자 이름이라고 가정한다.
    숫자가 없으면 이름의 해시를 쓴다. 대회 규칙과 다르면 이 함수만 고친다.
    """
    digits = re.findall(r"\d+", Path(file_name).stem)
    if digits:
        return int(digits[-1])
    return zlib.crc32(file_name.encode("utf-8"))  # 실행할 때마다 같은 값이 나오는 해시


# ── 예측 결과 저장·로드 (JSON은 텐서를 못 담으므로 리스트로 변환) ──


def predictions_to_json(preds: list[Prediction]) -> list[dict]:
    return [
        {
            "image_id": int(p["image_id"]),
            "file_name": p["file_name"],
            "boxes": p["boxes"].tolist(),
            "scores": p["scores"].tolist(),
            "labels": p["labels"].tolist(),
        }
        for p in preds
    ]


def predictions_from_json(items: list[dict]) -> list[Prediction]:
    return [
        Prediction(
            image_id=int(it["image_id"]),
            file_name=it["file_name"],
            boxes=torch.tensor(it["boxes"], dtype=torch.float32).reshape(-1, 4),
            scores=torch.tensor(it["scores"], dtype=torch.float32),
            labels=torch.tensor(it["labels"], dtype=torch.int64),
        )
        for it in items
    ]
