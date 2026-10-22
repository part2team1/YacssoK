"""데이터 추출: 어노테이션을 읽어 class_map.json과 records.json을 만든다 (데이터 B).

실행:
    uv run python scripts/prepare_data.py --config configs/base.yaml

결과:
    data/class_map.json        label_idx ↔ category_id ↔ 약 이름 (커밋 대상)
    data/processed/records.json  이미지별 박스 목록 (EDA·분할에서 사용)
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.dataset import build_class_map, load_records
from src.utils import load_config, resolve_path, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    records, categories = load_records(d["train_annotations"])
    if not records:
        raise SystemExit(f"어노테이션을 찾지 못했습니다: {d['train_annotations']}")
    class_map = build_class_map(records, categories)
    save_json(class_map, resolve_path(d["class_map"]))

    processed = resolve_path(d["class_map"]).parent / "processed" / "records.json"
    save_json(records, processed)

    n_boxes = sum(len(r["boxes"]) for r in records)
    per_image = Counter(len(r["boxes"]) for r in records)
    missing = [
        r["file_name"]
        for r in records
        if not (resolve_path(d["train_images"]) / r["file_name"]).exists()
    ]
    print(f"이미지 {len(records)}장, 박스 {n_boxes}개, 클래스 {len(class_map)}개")
    print(f"이미지당 알약 수 분포: {dict(sorted(per_image.items()))}")
    if missing:
        print(f"[주의] 이미지 파일이 없는 어노테이션 {len(missing)}건 (예: {missing[:3]})")
    print(f"저장: {d['class_map']}, {processed}")


if __name__ == "__main__":
    main()
