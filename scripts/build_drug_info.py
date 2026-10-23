"""데모 화면에 보여줄 약 정보표를 만든다 (PM·Experimentation Lead).

대회 어노테이션 JSON의 images[] 항목에는 약 정보(제품명, 제조사, 성분, 전문/일반 등)가 들어 있다.
알약 코드(dl_idx = category_id)별로 한 줄씩 모아 data/drug_info.json 으로 저장한다.

실행:
    uv run python scripts/build_drug_info.py --config configs/base.yaml
결과: data/drug_info.json → app/demo.py 가 있으면 자동으로 읽어 표에 표시
      (대회 데이터에서 뽑은 파일이므로 커밋하지 않고, 데모를 띄우는 PC에서 실행)
"""

from __future__ import annotations

import argparse

from src.utils import load_config, load_json, resolve_path, save_json

# 화면에 보여줄 항목: JSON 키 → 화면 이름 (필요하면 여기만 추가)
FIELDS = {
    "dl_name": "제품명",
    "dl_material": "성분",
    "dl_company": "제조사",
    "di_etc_otc_code": "전문/일반",
    "di_class_no": "분류",
    "drug_shape": "모양",
    "color_class1": "색상",
    "print_front": "식별문자(앞)",
    "print_back": "식별문자(뒤)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ann = resolve_path(cfg["data"]["train_annotations"])
    files = [ann] if ann.is_file() else sorted(ann.rglob("*.json"))

    info: dict[str, dict] = {}
    for path in files:
        data = load_json(path)
        for img in data.get("images", []):
            code = img.get("dl_idx")
            if code is None or str(code) in info:
                continue
            info[str(int(code))] = {
                label: img.get(key) for key, label in FIELDS.items() if img.get(key)
            }

    out = resolve_path(cfg["data"]["class_map"]).parent / "drug_info.json"
    save_json(info, out)
    print(f"약 {len(info)}종 정보 저장: {out}")


if __name__ == "__main__":
    main()
