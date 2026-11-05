# 조합 단위 K-fold 가이드 (모델팀용)

Nov 5, 2026 · pegstar

## 이 문서로 하는 일

**모델이나 데이터 실험이 정말 좋아졌는지를, 학습 때 한 번도 못 본 알약 조합으로만 검증해서 5번 평균으로 확인합니다.** 공식 검증셋(47장)은 학습셋과 같은 알약 조합이 많이 섞여 있어 점수가 높게 나오고(예: 26s v3allcp 0.991), 0.01 안쪽 차이는 우연과 구분되지 않습니다. 이 가이드의 K-fold는 실제 test(새 사진)에 더 가까운 조건입니다.

**규칙 네 가지**

1. **나누는 단위는 이미지가 아니라 알약 조합입니다.** 같은 조합(파일 이름 앞부분 `K-001900-016548-...`)의 사진은 모두 같은 fold에 들어갑니다.
2. **fold는 대회 이미지로만 나눕니다.** AI Hub 이미지·copy-paste 합성 이미지는 검증 fold에 절대 넣지 않고, 각 fold의 train에만 붙입니다.
3. **copy-paste 합성은 fold마다 새로 만듭니다.** 합성은 train 알약을 잘라 붙인 것이라, 공식 분할로 만든 합성을 그대로 쓰면 검증 fold 알약이 학습에 섞입니다.
4. **모델끼리 비교할 때는 같은 분할 파일(`data/splits/v1_f0~4`)·epoch·seed·데이터 조건을 씁니다.** 분할이 다르면 점수를 나란히 놓고 비교하지 않습니다.

> 팀 `make_folds.py`(`v0_f1~f5`)는 **이미지 단위**라 규칙 1을 어깁니다. 예: f1은 검증 47장 중 41장, f2는 37장이 학습셋과 같은 조합입니다. 이 가이드는 `kfold_split.py`(`v1_f0~f4`, 0부터 시작)를 씁니다.

---

## 어떤 스크립트를 쓰나

| 학습 데이터 | 분할 만들기 | 학습 | 예 |
| --- | --- | --- | --- |
| 대회 이미지만 (v0, v1) | `kfold_split.py` | `run_kfold.py` | 기준 모델, RFS(D1), 증강(D3) |
| + AI Hub (v3allc) | `kfold_split.py` → `make_ext_folds.py` | `run_kfold.py` | D6 |
| + copy-paste 합성 (v2c, v3allcp) | `kfold_split.py` → `make_ext_folds.py`(AI Hub가 있을 때) | **`run_kfold_synth.py`** | D2, D12 |

`make_ext_folds.py`·`run_kfold_synth.py`는 이 가이드와 함께 추가된 스크립트입니다.

---

## 빠른 시작: YOLO26s + v3allcp(D12) 5-fold (이 순서대로)

**▶ PowerShell** (pill-detection 폴더, 한 줄씩)

```powershell
# 0) 최신 코드 + 패키지
git switch <내이름>
git pull origin develop
uv sync --extra tracking --extra ocr

# 1) 정제 v1, AI Hub 추가 v3allc (처음 한 번. AI Hub TL_1·TS_1이 data/aihub/ 에 있어야 함)
uv run python scripts/clean_labels.py --config configs/base.yaml --version v1
uv run python scripts/make_external.py --config configs/yolo_v1_clean.yaml --version v3allc --classes all

# 2) 분할: 대회 이미지만 조합 단위 5-fold → 각 fold train에 AI Hub 추가
uv run python scripts/kfold_split.py --config configs/yolo_v1_clean.yaml --k 5
uv run python scripts/make_ext_folds.py --base v1 --ext v3allc --k 5

# 3) 동작 확인: fold 0을 1 epoch (몇 분). "요약"이 나오면 통과
uv run python scripts/run_kfold_synth.py --config configs/yolo_s_26s_v3allcp.yaml --folds 0 --epochs 1

# 4) 본 실행 (26s·8GB GPU·batch 16 기준 fold당 약 33분, 5-fold 약 3시간)
uv run python scripts/run_kfold_synth.py --config configs/yolo_s_26s_v3allcp.yaml --folds 0 1 2 3 4
```

**▶ 출력 예시** — 2)의 `kfold_split.py` (입력하는 것이 아님)

```text
이미지 223장, 알약 조합 ○○개 → 5-fold, StratifiedGroupKFold (조합 단위 + 드문 클래스 층화)
fold  train   val  겹치는 조합  val에만 있는 클래스
   0    178    45        0              ○      ← 실제 v1_f0: train 178, val 45, 겹치는 조합 0
```

**`겹치는 조합`이 모두 0이어야 합니다.** 0이 아니면 `[주의]`가 나오니 멈추고 확인합니다.

**▶ 출력 예시** — 2)의 `make_ext_folds.py`

```text
v3allc_f0: train 178 + 외부 1354 = 1532, val 45
```

검증(val) 수가 `kfold_split.py`와 같으면 됩니다. 외부 이미지가 val에 들어가면 스크립트가 바로 멈춥니다.

---

## 다른 모델·데이터로 돌리기

모델만 바꾸려면 설정 파일 하나를 새로 만들어  `model.weights`·`exp_name`·`batch_size`만 바꿉니다.

**▶ YAML** — `configs/yolo_s_11s_v3allcp.yaml` (예: YOLO11s)

```yaml
base: yolo_s_26s_v3allcp.yaml
exp_name: 1105_<이름>_yolo11s_v3allcp
note: 11s v3allcp rfs0.1
model:
  weights: yolo11s.pt
train:
  batch_size: 16   # m 모델은 8GB GPU에서 8
```

```powershell
uv run python scripts/run_kfold_synth.py --config configs/yolo_s_11s_v3allcp.yaml --folds 0 1 2 3 4
```

**합성이 없는 데이터**(v0·v1·v3allc)는 기존 `run_kfold.py`를 씁니다. 버전 이름만 fold 분할과 맞추면 됩니다.

```powershell
# v1 (정제) 기준 모델 5-fold
uv run python scripts/run_kfold.py --config configs/yolo_v1_clean.yaml --folds 0 1 2 3 4 --tag kfold
# v3allc (AI Hub, 합성 없음): 설정의 data.version 이 v3allc 여야 v3allc_f0~4 를 읽음
uv run python scripts/run_kfold.py --config configs/yolo_v3allc_aihub.yaml --folds 0 1 2 3 4 --tag kfold
```

---

## 결과 보기

| 위치 | 내용 |
| --- | --- |
| `outputs/kfold/<exp_name>_kfold_summary.csv` | fold별 mAP@[0.75:0.95]·mAP75·학습 시간 |
| 터미널 마지막 `평균 0.○○○○ ± 0.○○○○` | 5-fold 평균 ± 표준편차 |
| `outputs/<exp_name>_kfold_v3allcp_f{i}_s42/` | fold별 모델(`best.pt`), `metrics.json`, `config.yaml` |
| `outputs/kfold/configs/` | fold별로 실제 쓴 설정 (재현용) |

**판단 기준**

- 모델·실험은 **평균**으로 고르고, 가장 높은 fold 하나로 고르지 않습니다.
- 두 실험의 평균 차이가 **표준편차보다 작으면** "차이 없음"으로 봅니다. 같은 fold끼리 짝지어 몇 개 fold에서 이겼는지도 함께 적습니다.
- 공식 검증 점수(누수 있음)와 K-fold 점수를 섞어서 비교하지 않습니다.

---

## OCR 재분류를 fold마다 적용하기

OCR은 탐지 모델을 다시 학습하지 않는 후처리입니다. **fold i 모델은 fold i 검증셋으로만** 평가해 OCR 전후를 비교합니다. 설정 파일을 fold마다 고치지 않고 `--set`으로 넘깁니다(fold당 1분 이내).

**▶ PowerShell** — 5개 fold에 OCR 적용 (한 줄)

```powershell
$X = "1105_pegstar_yolo26s_v3allcp"   # ← 내 K-fold 설정의 exp_name
foreach ($i in 0..4) { uv run python scripts/run_train.py --config configs/ocr_rerank.yaml --exp-name "${X}_kfold_ocr_f$i" --set "model.detector_exp=${X}_kfold_v3allcp_f${i}_s42" "data.train_images=data/synth/v3allcp_f$i/images" "data.train_annotations=data/annotations/v3allcp_f$i.json" "data.version=v3allcp_f$i" "data.class_map=data/class_map_v3allc.json" }
```

**▶ PowerShell** — OCR 전후 비교표

```powershell
foreach ($i in 0..4) { $a = (Get-Content "outputs\${X}_kfold_v3allcp_f${i}_s42\metrics.json" | ConvertFrom-Json).map; $b = (Get-Content "outputs\${X}_kfold_ocr_f$i\metrics.json" | ConvertFrom-Json).map; "fold $i  전 $a  후 $b  변화 $([math]::Round($b - $a, 5))" }
```

| fold | 탐지 단독 | OCR 적용 | 변화량 |
| --- | --- | --- | --- |
| 0~4 | 기록 | 기록 | 후 − 전 |
| 평균 | | | 변화량 평균 ± 표준편차 |

> `find_label_issues.py --ocr`는 위의 OCR 재분류와 다른 **라벨 점검** 도구입니다. 정답 라벨이 틀렸을 수 있는 후보만 뽑고, 그림을 사람이 확인해 `manual_fix.csv`에 옮긴 뒤 재학습해야 점수에 반영됩니다.

---

## 다른 사람의 K-fold와 분할이 같은지 확인

점수를 비교하기 전에 검증 목록이 같은지 확인합니다. 아무것도 출력되지 않으면 같은 분할입니다.

```powershell
Compare-Object (Get-Content data\splits\v1_f0\val.txt) (Get-Content <받은 파일>\val.txt)
```

AI Hub 장수도 같아야 합니다. 같은 명령이어도 PC마다 받은 데이터·제외 기준이 다르면 장수가 달라집니다(예: develop 기록 2,657장, pegstar PC 1,354장). 아래 숫자를 서로 맞춰 봅니다.

```powershell
(Import-Csv data\annotations\v3allc_external.csv).Count
```

분할 파일은 파일 이름만 들어 있어 커밋할 수 있습니다(`.gitignore`가 `data/splits/`를 허용). **한 사람이 만든 `v1_f0~4`·`v3allc_f0~4`를 커밋해 팀이 같은 fold를 씁니다.** 각자 `kfold_split.py`를 돌려도 seed 42라 같게 나오지만, 정제(`v1`) 결과가 다르면 달라질 수 있습니다.

---

## 결과 (참고)

| 모델 | 데이터 | 분할 | 평균 ± 표준편차 | fold 0 / 1 / 2 / 3 / 4 | 조건 |
| --- | --- | --- | --- | --- | --- |
| YOLO26s | **v3allcp** + RFS 0.1 (D12) | 조합 단위 v1_f0~4, fold별 합성 | **0.9428 ± 0.0126** | 0.9590 / 0.9406 / 0.9513 / 0.9355 / 0.9274 | batch 16, 50 epoch, fold당 약 33분 |

같은 설정의 공식 검증 점수는 0.991로, 학습셋과 같은 조합이 섞인 누수 때문에 K-fold보다 약 0.05 높게 나옵니다. 모델·실험 선택은 K-fold 평균으로 합니다. 자세한 기록: `docs/experiments/1105_pegstar_kfold_26s_v3allcp.md`

---

## 문제 해결

| 증상 | 원인 · 해결 |
| --- | --- |
| `data/splits/v3allc_f0 이 없습니다` | 빠른 시작 2)의 두 명령을 먼저 실행 |
| `AI Hub 라벨 JSON을 찾지 못했습니다` | `data/aihub/labels/TL_1_조합`, `data/aihub/images/TS_1_조합` 위치 확인 |
| epoch이 3배쯤 느림 | 8GB GPU에서 m 모델 batch 16 → 메모리가 넘침. `train.batch_size: 8` |
| 중간에 꺼짐 | 끝난 fold는 빼고 남은 번호만: `--folds 3 4`. 합성·변환은 이미 만든 것을 다시 씀 |
| 합성을 다시 만들고 싶음 | `data/annotations/v3allcp_f{i}.json` 과 `data/yolo/v3allcp_f{i}` 를 지우고 다시 실행 |
| 실행 중 다른 명령을 쓰고 싶음 | **새 터미널**에서. 학습 터미널에는 입력하지 않음. 다른 학습은 GPU가 넘쳐 동시에 돌리지 않음 |

## 커밋하는 것 · 하지 않는 것

- **커밋**: `scripts/make_ext_folds.py`, `scripts/run_kfold_synth.py`, `configs/*.yaml`, `data/splits/v1_f*`·`v3allc_f*`, 이 문서, 결과 숫자(docs/experiments)
- **커밋하지 않음**: `outputs/`, `best.pt`, `data/aihub/`, `data/synth/`, `data/annotations/*.json`, `label_issues` 그림(대회 이미지). AI Hub 데이터는 재배포 금지라 팀 내부로만 주고받습니다.
