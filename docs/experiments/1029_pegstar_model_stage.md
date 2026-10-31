# 모델 비교·단계(S1/S2) 실험 결과 (10/29~10/30, pegstar)

모든 점수는 조합 단위 5-fold(`scripts/kfold_split.py --k 5`, seed 42) 평균 ± 표준편차, 대회 지표 mAP@[0.75:0.95].
fold별 val 장수: 48 / 46 / 47 / 46 / 45.

## 1. 모델·설정 비교 (batch 8, img 640, epoch 50, YOLO 기본 증강)

| 순위 | 설정 | config | mAP@[0.75:0.95] | mAP75 |
|---|---|---|---|---|
| 1 | YOLO26m + 재표본화 0.1 | `yolo_m_26m_rfs.yaml` | **0.860 ± 0.040** | 0.877 |
| 2 | YOLO11m + 재표본화 0.1 | `yolo_m_11m_rfs.yaml` | 0.852 ± 0.041 | 0.871 |
| 3 | YOLOv8m + 재표본화 0.1 | `yolo_m_v8m_rfs.yaml` | 0.848 ± 0.046 | 0.873 |
| 3 | YOLOv8s img 1024 | `yolo_s_v8s_img1024.yaml` | 0.848 ± 0.040 | 0.867 |
| 5 | YOLOv8s | `yolo_s_v8s.yaml` | 0.847 ± 0.042 | 0.872 |
| 6 | YOLOv8s + 재표본화 0.1 | `yolo_s_v8s_rfs.yaml` | 0.843 ± 0.043 | 0.872 |
| 7 | YOLOv8m | `yolo_m_v8m.yaml` | 0.829 ± 0.055 | 0.862 |
| 8 | YOLOv8s, 증강 없음 | `yolo_s_v8s_noaug.yaml` | 0.817 ± 0.062 | 0.837 |

- 26m은 공식 v0 분할(누수 있음)에서는 medium 4종 중 꼴찌(0.942)였지만 K-fold에서는 1위. v0 점수로 모델을 고르면 안 된다.
- YOLO 기본 증강을 끄면 5개 fold 모두 하락(v8s 기준 −0.030).
- `augment: name: none`은 YOLO 학습에 적용되지 않는다. YOLO 증강은 `yolo_args`로만 바뀐다.
- `optimizer: auto`라 config `lr` 0.01은 무시되고 실제 최대 학습률은 모든 실험에서 0.000158.
- 클래스 무시 NMS(agnostic)는 K-fold에서 효과 없음(10개 중 7개 동일, 3개 하락).

## 2. 팀 단계 비교: YOLO26m (batch 16, img 640, epoch 50, 재표본화 없음)

| 단계 | config | mAP@[0.75:0.95] | mAP75 | fold당 학습 시간 |
|---|---|---|---|---|
| S1 기준 (YOLO 내장 증강까지 끔) | `yolo_m_26m_S1.yaml` | **0.841 ± 0.051** | 0.858 ± 0.046 | 약 27분 |
| S2 안전 증강 (`yolo_aug_safe` 값) | `yolo_m_26m_S2.yaml` | 0.835 ± 0.052 | **0.871 ± 0.046** | 약 30분 |

- S2는 mAP75가 5개 fold 모두 상승(+0.013)했지만 대회 지표는 4개 fold에서 하락(−0.006).
  찾기·분류는 좋아지고 높은 IoU(0.90~0.95)의 박스 정밀도는 조금 떨어진 것으로 보인다 (scale·translate·mosaic 영향 추정).
- 8GB GPU에서 m 모델 batch 16은 GPU 메모리를 넘어 공유 메모리를 써서 batch 8보다 약 3배 느리다 (epoch당 30초 vs 9초).

## 3. 결과 표·제출 묶음 만들기

```
uv run python scripts/kfold_table.py --prefix 1029_pegstar_yolo26m_v0_S1_kfold --model YOLO26m --stage S1
uv run python scripts/pack_submission.py --exp <실험> --kfold <같은 설정의 kfold prefix>
```

- `kfold_table.py`: 팀 공통 결과 표(담당자, 모델, 단계, fold, seed, train/val 장수, 완료 에폭, 점수, 시간, 비고)를 `outputs/kfold/{prefix}_table.md`로 만든다.
- `pack_submission.py`: `submissions/{exp}/`에 제출 CSV, config.yaml, score.md(실험 이름·검증 점수)를 모은다. CSV는 커밋하지 않는다.
