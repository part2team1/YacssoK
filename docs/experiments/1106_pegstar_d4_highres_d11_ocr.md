# 1106 · D4 고해상도 학습 · D11 각인 OCR 재분류 결과

Nov 6, 2026 · pegstar · 지표는 모두 대회 지표 mAP@[0.75:0.95] (max_det 4)

## 결론

- **D4 고해상도(640 → 1024)는 효과가 없어 최종 조합에서 뺐다.** YOLO26m은 공식 검증에서 0.9416 → 0.9342로 오히려 낮아졌고, YOLOv8s는 같은 조합 단위 5-fold에서 0.847 → 0.848로 차이가 없었다(fold별 차이 +0.001 ± 0.015).
- **D11 각인 OCR 재분류도 효과가 없어 최종 조합에서 뺐다.** 세 가지 조건(26m v2c+RFS, 26m v3allcp, 26m v3allcp 5개 fold) 모두 점수가 소수 다섯째 자리까지 같았고, 이름이 바뀐 박스가 0개였다.
- 두 실험 모두 **"해 봤지만 점수가 오르지 않았다"**가 결과다. 미완료가 아니다.

## D4 · 고해상도 학습 (img 1024)

| 실험 | 모델 | 데이터 · 검증 | img | batch | mAP@[0.75:0.95] | 비교 기준 | 변화 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1102_pegstar_yolo26m_img1024` | YOLO26m | v0 · 공식 검증 47장 | 1024 | 4 | 0.9342 | `1102_pegstar_yolo26m_base` 0.9416 (img 640, batch 8) | −0.007 |
| `1029_pegstar_yolov8s_v0_img1024_kfold` | YOLOv8s | v0 · 조합 단위 5-fold `v0_f0~4` | 1024 | 8 | 0.848 ± 0.040 | `1028_pegstar_yolov8s_v0_scompare_kfold` 0.847 ± 0.041 (img 640) | +0.001 ± 0.015 |

YOLOv8s fold별 (같은 fold끼리 비교):

| fold | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| img 640 | 0.7992 | 0.8965 | 0.8104 | 0.8744 | 0.8545 |
| img 1024 | 0.7952 | 0.8994 | 0.8259 | 0.8526 | 0.8685 |
| 차이 | −0.004 | +0.003 | +0.015 | −0.022 | +0.014 |

- 1024가 이긴 fold 3개, 진 fold 2개로, 차이가 fold 간 흔들림(± 0.04)보다 훨씬 작다 → **차이 없음**.
- 26m은 8GB GPU에서 1024를 batch 8로 돌리면 메모리가 부족해 batch 4로 학습했다. batch가 달라 단독 원인을 1024로 단정할 수는 없지만, 점수가 오르지 않았다는 결론은 두 모델에서 같다.
- 26m 1024는 학습 시간이 약 2.7배(7분 → 19분)라 비용 대비 이득도 없다.

## D11 · 각인 OCR 재분류

탐지 모델은 다시 학습하지 않고, 확신도가 낮은 박스(0.6 미만)만 OCR로 각인을 읽어 알약 이름을 다시 고르는 후처리다(`configs/ocr_rerank.yaml`, `ocr_max_score 0.6`, `ocr_min_sim 0.8`, `ocr_weight 0.5`).

| 탐지 모델 | 검증 | OCR 전 | OCR 후 | 변화 |
| --- | --- | --- | --- | --- |
| `1102_pegstar_yolo26m_v2c_rfs01` (26m · 정제 + copy-paste + RFS) | v1 공식 검증 | 0.98827 | 0.98827 | 0 |
| `1103_yolo26m_v3allc_d123_yolo26m_d12_e50` (26m · v3allcp, 팀원 학습) | v1 공식 검증 | 0.99593 | 0.99593 | 0 |
| `1104_pegstar_yolo26m_v3allcp_kfold` fold 0~4 | 조합 단위 5-fold `v3allcp_f0~4` | 0.9446 / 0.9211 / 0.9698 / 0.9468 / 0.9164 | 0.9446 / 0.9211 / 0.9698 / 0.9468 / 0.9164 | 모두 0 |

- 두 번째 행에서 예측 파일을 비교한 결과, **이름이 바뀐 박스 0개**. OCR 대상(확신도 0.6 미만)은 167개 박스 중 11개였고, 대부분 알약 3개 사진의 남는 4번째 칸(확신도 0.01 미만)이라 원래 정답이 없는 자리였다.
- 즉 탐지 모델이 이미 거의 모든 알약을 높은 확신도로 맞혀서, OCR이 고칠 대상이 거의 없었다.
- `find_label_issues.py --ocr`(정답 라벨 오류 후보 찾기)는 이번 재분류 실험과 다른 진단 도구라 여기에 포함하지 않았다.

## 재현

```powershell
# D4: 26m 1024 (matrix의 img1024 run, batch 4)
uv run python scripts/run_matrix.py --plan configs/matrix_pegstar.yaml --runs base img1024
# D4: v8s 1024 조합 단위 5-fold (분할 v0_f0~4는 kfold_split.py로 생성)
uv run python scripts/kfold_split.py --config configs/base.yaml --k 5
uv run python scripts/run_kfold.py --config configs/yolo_s_v8s_img1024.yaml --folds 0 1 2 3 4 --tag kfold
# D11: 1위 조합 모델에 OCR 재분류 (학습 없음)
uv run python scripts/run_matrix.py --plan configs/matrix_pegstar_next.yaml --runs base v2c_rfs01_ocr
# D11: K-fold 각 fold에 OCR (가이드 docs/guides/04_grouped_kfold.md)
```

OCR 실행에는 `uv sync --extra tracking --extra ocr`과 `data/drug_info.json`이 필요하다.
