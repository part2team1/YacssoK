"""실험 재사용과 점수 비교에 필요한 입력 지문. 기존 결과는 임의로 덮어쓰지 않는다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.dataset import load_records, load_split
from src.interfaces import OUTPUT_DIR, PROJECT_ROOT
from src.utils import load_class_map, load_config, resolve_path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_digest(path: Path) -> str:
    if path.is_file():
        return file_digest(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return digest(
        [(str(p.relative_to(path)), file_digest(p)) for p in sorted(path.rglob("*.json"))]
    )


def experiment_signature(cfg: dict) -> str:
    """설정·정답·분할·번호표·학습 사진 메타데이터와 실행 코드를 함께 확인한다.

    사진은 크기·mtime_ns를 비교한다. 모든 원본 이미지 바이트의 동일성 보증은 아니다.
    """
    d = cfg["data"]
    files = {key: path_digest(resolve_path(d[key])) for key in ("train_annotations", "class_map")}
    for split in ("train", "val"):
        files[split] = path_digest(resolve_path(f"{d['split_dir']}/{d['version']}/{split}.txt"))
    photos = resolve_path(d["train_images"])
    image_stats = [
        (str(p.relative_to(photos)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(photos.rglob("*"))
        if p.is_file()
    ]
    code = [
        (str(p.relative_to(PROJECT_ROOT)), file_digest(p))
        for folder in ("src", "scripts")
        for p in sorted((PROJECT_ROOT / folder).rglob("*.py"))
    ]
    return digest(
        {
            "config": cfg,
            "files": files,
            "images": image_stats,
            "code": code,
            "model_dependencies": model_dependencies(cfg),
        }
    )


def model_dependencies(cfg: dict, seen=()) -> dict:
    """OCR 래퍼가 참조하는 탐지 모델·설정·번호표·각인표도 실제 내용으로 확인한다."""
    model = cfg["model"]
    if model["family"] != "ocr_rerank":
        return {}
    detector = str(model["detector_exp"])
    if detector in seen:
        raise ValueError("OCR 탐지 실험 의존성이 순환합니다.")
    folder = OUTPUT_DIR / detector
    detector_cfg = load_config(folder / "config.yaml")
    drug_info = (
        model.get("drug_info") or resolve_path(cfg["data"]["class_map"]).parent / "drug_info.json"
    )
    return {
        "detector_config": detector_cfg,
        "detector_weights": file_digest(folder / "best.pt"),
        "detector_classes": file_digest(resolve_path(detector_cfg["data"]["class_map"])),
        "nested": model_dependencies(detector_cfg, (*seen, detector)),
        "drug_info": file_digest(resolve_path(drug_info)),
    }


def checkpoint_signature(cfg: dict, weights: Path) -> str:
    dependencies = model_dependencies(cfg)
    checkpoint = file_digest(weights)
    return (
        digest({"checkpoint": checkpoint, "dependencies": dependencies})
        if dependencies
        else checkpoint
    )


def evaluation_signature(cfg: dict) -> str:
    """같은 실제 사진·정답·클래스 번호·IoU로 채점한 결과만 비교한다."""
    d = cfg["data"]
    records, _ = load_records(d["train_annotations"])
    records = load_split(records, f"{d['split_dir']}/{d['version']}/val.txt")
    if not records:
        raise ValueError("검증 사진이 없는 점수로는 모델을 선택할 수 없습니다.")
    records = sorted(records, key=lambda r: r["file_name"])
    photos = [
        (r["file_name"], file_digest(resolve_path(d["train_images"]) / r["file_name"]))
        for r in records
    ]
    return digest(
        {
            "records": records,
            "photos": photos,
            "classes": load_class_map(d["class_map"]),
            "iou_thresholds": cfg["eval"].get("iou_thresholds"),
        }
    )
