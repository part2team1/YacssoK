# 1105 · YOLO26s + v3allcp(D12) 조합 단위 5-fold

Nov 5, 2026 · pegstar · 절차는 [조합 단위 K-fold 가이드](../guides/04_grouped_kfold.md)

## 결론

- **YOLO26s + v3allcp(D12) + RFS 0.1의 조합 단위 5-fold mAP@[0.75:0.95]는 0.9428 ± 0.0126**이다. 학습 때 한 번도 못 본 알약 조합으로만 검증한 점수라, 실제 test 성능에 가까운 추정치로 쓴다.
- fold 간 범위는 0.927~0.959로, 검증 fold(44~45장)에 어떤 조합이 들어가느냐에 따라 ±0.013 흔들린다. 다른 실험과 비교할 때 이보다 작은 차이는 "차이 없음"으로 본다.
- 같은 설정의 공식 검증 점수는 0.991로, 학습셋과 같은 조합이 섞인 누수 때문에 K-fold보다 약 0.05 높다.
- 8GB GPU·batch 16에서 fold당 약 33분, 5-fold 약 2.7시간.

## 조건

| 항목 | 값 |
| --- | --- |
| 모델 | YOLO26s (`yolo26s.pt`), 설정 `configs/yolo_s_26s_v3allcp.yaml` |
| 데이터 | v3allcp = 정제 v1 + AI Hub(TL_1·TS_1, 새 클래스 포함) + copy-paste 합성, RFS 0.1 |
| 분할 | `v1_f0~4` (대회 이미지만 조합 단위, `kfold_split.py`, 겹치는 조합 0), AI Hub는 fold train에만(`make_ext_folds.py`) |
| 합성 | fold마다 그 fold train으로 새로 생성(`run_kfold_synth.py`), 검증 이미지를 쓴 합성 0장 |
| 학습 | 50 epoch, img 640, batch 16, seed 42 |
| 지표 | mAP@[0.75:0.95] (`metrics.json`의 `map`), max_det 4 |

## fold별 결과

| fold | 검증 장수 | mAP@[0.75:0.95] | mAP75 | 학습 시간 |
| --- | --- | --- | --- | --- |
| 0 | 45 | 0.9590 | 0.9688 | 39분 |
| 1 | 45 | 0.9406 | 0.9463 | 31분 |
| 2 | 44 | 0.9513 | 0.9762 | 34분 |
| 3 | 45 | 0.9355 | 0.9516 | 32분 |
| 4 | 44 | 0.9274 | 0.9350 | 29분 |
| **평균 ± 표준편차** | | **0.9428 ± 0.0126** | **0.9556 ± 0.0168** | 33분 |

## 재현

```powershell
uv run python scripts/kfold_split.py --config configs/yolo_v1_clean.yaml --k 5
uv run python scripts/make_ext_folds.py --base v1 --ext v3allc --k 5
uv run python scripts/run_kfold_synth.py --config configs/yolo_s_26s_v3allcp.yaml --folds 0 1 2 3 4
```

결과: `outputs/kfold/1105_pegstar_yolo26s_v3allcp_kfold_summary.csv`, fold별 모델 `outputs/1105_pegstar_yolo26s_v3allcp_kfold_v3allcp_f{i}_s42/best.pt` (커밋하지 않음, 드라이브로 공유)

## 참고

- 팀원 26s K-fold(0.9707 ± 0.0108)와는 분할 생성 방식이 달라 직접 비교하지 않는다. `data/splits/v1_f0/val.txt`와 AI Hub 장수(`v3allc_external.csv`)가 같은지 확인한 뒤 비교한다.
- 그림 `reports/figures/kfold_26s_v3allcp.png`, 요약 `kfold_26s_result_summary.png` (커밋하지 않음, Discord·PR 본문으로 공유, 대회 이미지 없음)
