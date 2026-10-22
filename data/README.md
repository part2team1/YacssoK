# data/

데이터는 Git에 올리지 않습니다. 각자 Kaggle에서 내려받아 아래 위치에 둡니다.
예외: `class_map.json`과 `splits/`는 팀 전원이 같은 값을 써야 하므로 **커밋합니다**.

```
data/
├── raw/                    # Kaggle 데이터 (원본, 수정 금지)
│   ├── train_images/       # 232장 PNG
│   ├── train_annotations/  # K-xxxxxx-..._json/ (조합) → K-xxxxxx/ (알약) → *.json (COCO 형식)
│   └── test_images/        # 842장 PNG, 파일 이름의 숫자 = 제출 image_id
├── class_map.json          # label_idx ↔ category_id(dl_idx) ↔ 약 이름  (prepare_data.py, 커밋)
├── splits/v0/              # train.txt, val.txt  (split_data.py, 커밋)
├── processed/records.json  # 이미지별 박스 목록 (prepare_data.py)
├── yolo/v0/                # YOLO 형식 내보내기 (export_yolo.py)
└── sample/                 # 합성 연습 데이터 (make_sample_data.py)
```

## 대회 데이터 참고사항
- 어노테이션은 알약 하나당 JSON 한 개로 나뉘어 있고, `load_records()`가 이미지 파일 이름 기준으로 합칩니다.
- `category_id`는 알약 식별자 `dl_idx`입니다. 제출할 때 이 값으로 되돌립니다.
- bbox는 COCO `[x, y, w, h]`이며, 길이가 4인 bbox만 유효합니다. 코드 안에서는 xyxy로 바꿔 씁니다.
- 테스트에는 학습에 없는 클래스가 있고, 제공 데이터에 오류가 섞여 있습니다 (`notebooks/01_eda.ipynb` 5장).
- AI Hub 데이터 추가 사용 가능. 단 **`TL_2_조합.zip`, `TS_2_조합.zip`은 사용 금지**.
