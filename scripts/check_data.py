"""실제 대회 데이터가 제자리에 있는지 확인한다 (팀원 전원, 데이터를 놓은 직후 한 번).

실행:
    uv run python scripts/check_data.py                          # 실제 데이터 (configs/base.yaml)
    uv run python scripts/check_data.py --config configs/sample_yolo.yaml   # 합성 연습 데이터

확인하는 것
  1. data/raw/ 아래 세 폴더와 파일 개수 (train 232장, 조합 폴더 114개, test 842장)
  2. main에서 받은 data/class_map.json, data/splits/{version}/ 이 있는지
  3. YOLO 형식(data/yolo/{version}/data.yaml)이 만들어졌는지
마지막 줄에 다음에 할 일을 알려 준다. 파일을 만들거나 고치지 않는다.
"""

from __future__ import annotations

import argparse

from src.utils import load_config, resolve_path, yolo_data_yaml  # v2-v3, D1-6

EXPECTED = {"train_images": 232, "train_annotations": 114, "test_images": 842}


def count(path, kind: str) -> int:
    if not path.exists():
        return -1
    if kind == "train_annotations":
        return sum(1 for p in path.iterdir() if p.is_dir()) if path.is_dir() else 1
    return sum(1 for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    real = args.config.endswith("base.yaml")
    todo: list[str] = []

    print("[1] 원본 데이터")
    missing_raw = False
    for key in ("train_images", "train_annotations", "test_images"):
        path = resolve_path(d[key])
        n = count(path, key)
        if n < 0:
            print(f"  [없음] {d[key]}")
            missing_raw = True
            continue
        exp = EXPECTED[key] if real else None
        ok = exp is None or n == exp
        mark = "OK" if ok else "확인"
        extra = "" if ok else f" (예상 {exp}개 — 폴더가 한 단계 더 들어가 있지 않은지 확인)"
        print(f"  [{mark}] {d[key]}: {n}개{extra}")

    if missing_raw:
        todo.append(
            "Kaggle 데이터를 내려받아 data/raw/ 에 세 폴더 놓기 (가이드 0장 ②)"
            if real
            else "uv run python scripts/make_sample_data.py 로 합성 데이터 만들기"
        )

    print("[2] 팀 공용 파일 (main에서 받음)")
    class_map = resolve_path(d["class_map"])
    split = resolve_path(d["split_dir"]) / d["version"]
    for path, label in (
        (class_map, "class_map.json"),
        (split / "val.txt", f"splits/{d['version']}"),
    ):
        print(f"  [{'OK' if path.exists() else '없음'}] {label}")
    if not class_map.exists() or not (split / "val.txt").exists():
        todo.append(
            "git pull origin main 으로 받기. main에도 없으면 데이터 B가 prepare_data.py·split_data.py 실행 후 PR"
            if real
            else f"prepare_data.py, split_data.py 를 --config {args.config} 로 실행"
        )

    print("[3] YOLO 형식 (각자 PC에서 생성, 커밋하지 않음)")
    # yolo = resolve_path(d["yolo_dir"]) / d["version"] / "data.yaml" #v2-v3 D1-6
    yolo = yolo_data_yaml(cfg)  # v2-v3 D1-6
    print(f"  [{'OK' if yolo.exists() else '없음'}] {yolo.relative_to(resolve_path('.'))}")
    if not yolo.exists():
        todo.append(f"YOLO 학습 전에: uv run python scripts/export_yolo.py --config {args.config}")

    print()
    if todo:
        print("다음에 할 일:")
        for i, t in enumerate(todo, 1):
            print(f"  {i}. {t}")
    else:
        print("모든 준비가 끝났습니다. 담당 가이드의 따라하기 실습으로 넘어가세요.")


if __name__ == "__main__":
    main()
