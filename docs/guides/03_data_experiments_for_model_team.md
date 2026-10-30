# 데이터 실험 적용 가이드 (모델팀용)

Oct 28, 2026

## 이 문서로 하는 일

**모델 담당자는 자기가 고른 모델 설정 하나를 기준으로 두고, 데이터 실험(D1\~D11)을 `--set` 묶음으로 하나씩 얹어 학습한 뒤, 효과 있던 묶음만 더해서 한 번 더 학습합니다.** 코드 수정은 없습니다. 데이터 쪽 코드와 데이터 버전은 [데이터 개선 실험 가이드 (v3)](https://claude.ai/artifact/6rT1TRmRuD2zPjctSiMuZr)대로 develop에 모두 반영된 상태를 전제로 합니다.

**시작 조건** (`git switch <내이름>` → `git pull origin develop` 후 확인)

- `uv run pytest -q`가 모두 통과
- 데이터 담당이 만든 데이터 버전이 내 PC에도 있음 (만드는 명령은 다음 장, 대회 데이터라 각자 PC에서 같은 명령으로 만듦)
- 내 기준 모델 설정 파일이 정해져 있음 (예: `configs/yolo11s.yaml`, `configs/rcnn_baseline.yaml`)

**규칙 세 가지**

1. 한 실험에 한 묶음만 바꿉니다. 에폭·시드·검증셋(`splits/v0/val.txt`)은 기준과 같게 둡니다.
2. 점수는 대회 지표 mAP@\[0.75:0.95\](`metrics.json`의 `map`)로 비교하고, 기준보다 \*\*0.01 이상 오르면 “효과 있음”\*\*으로 봅니다.
3. 실험 이름(`--exp-name`)은 매번 새로 붙입니다. 같은 이름이면 `outputs/`가 덮어써집니다.

## 준비: 데이터 버전, 기준 모델, 실험 묶음

### ① 데이터 버전 **만들기 **(처음 한 번, 각자 PC)

데이터는 대회 규칙상 레포에 없으므로, 아래 명령을 각자 PC에서 실행합니다. 분할·기록 파일(`data/splits/`, `manual_*.csv`, `*_copy_paste.csv`)은 develop에 있어서 누가 돌려도 같은 데이터가 나옵니다. 필요한 실험의 것만 만들면 됩니다.

| 데이터 버전 | 쓰는 실험 | 만드는 명령 |
| --- | --- | --- |
| v0 원본 | 기준, D1·D3·D4·D9 | `uv run python scripts/export_yolo.py --config configs/base.yaml` |
| v1 정제 | D5·D10 | `uv run python scripts/clean_labels.py --config configs/base.yaml --version v1` |
| v2 합성 | D2 | `uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2` |
| v3 외부 | D6 (기존 클래스 보강) | `uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known` |
| v3all 외부 | D6 (새 알약 추가) | `uv run python scripts/make_external.py --config configs/base.yaml --version v3all --classes all` |

**v3·v3all은 AI Hub 내려받기가 먼저입니다.** 데이터 가이드 v3의 D6 ①과 같은 내용을 모델 담당자가 바로 따라 할 수 있게 추렸습니다. 팀에서 외부 데이터 사용을 승인한 뒤에만 합니다.

1. [AI Hub 경구약제 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)에서 활용 신청 → 승인 후, Training » 경구약제 **조합** 데이터의 라벨 `TL_1_조합`(수 MB)과 이미지 `TS_1_조합`(약 3GB)만 받습니다. 효과가 보이면 3\~8번을 늘립니다.
2. **`TL_2_조합`·`TS_2_조합`은 대회 원본이라 받지도 쓰지도 않습니다.** 스크립트도 이 이름이 들어 있으면 멈춥니다. AI Hub 데이터도 재배포 금지이고, API 키는 `.env`에만 둡니다.
3. 아래 폴더에 압축을 힑니다. 안쪽 폴더 구조는 상관없고(스크립트가 끝까지 찾음), 레포와 같은 드라이브에 두어야 복사 대신 하드링크가 됩니다. 디스크는 받은 용량의 2\~3배를 비워 둡니다.

**▶ PowerShell 터미널** — AI Hub 압축 풀기 (Windows)

```powershell
# Windows: 탐색기 “압축 풀기”나 7-Zip도 됩니다 (한글 파일 이름이 깨지면 7-Zip 사용)
New-Item -ItemType Directory -Force data\aihub\labels, data\aihub\images | Out-Null
Expand-Archive "$HOME\Downloads\TL_1_조합.zip" -DestinationPath data\aihub\labels\TL_1_조합
Expand-Archive "$HOME\Downloads\TS_1_조합.zip" -DestinationPath data\aihub\images\TS_1_조합   # 3GB, 수 분
```

**▶ Bash 터미널** — AI Hub 압축 풀기 (Linux·WSL·Colab)

```bash
# Linux·WSL·Colab (한글 이름이 깨지면 unzip -O cp949)
mkdir -p data/aihub/labels data/aihub/images
unzip -q ~/Downloads/TL_1_조합.zip -d data/aihub/labels/TL_1_조합
unzip -q ~/Downloads/TS_1_조합.zip -d data/aihub/images/TS_1_조합
```

4. 위 표의 `make_external.py` 명령을 실행하고 출력을 확인합니다.

**▶ 출력 예시** — `make_external.py`를 실행하면 화면에 나오는 글 (입력하는 것이 아님)

```text
알약 번호 = drug_N + (0)  [이름이 같은 알약 8종 중 8종 번호 일치]      ← 90% 미만이면 멈추고 데이터 담당에게
제외한 AI Hub 이미지: {'학습에 없는 알약 포함 (--classes known)': 4, '촬영 각도 제외': 10, ...}
추가한 AI Hub 이미지 24장 (알약 47개)
```

(합성 연습 데이터 출력. `학습에 없는 알약 포함` 숫자가 크면 v3all로 얻을 것이 많다는 뜻입니다.)

**v3는 학습 전에 점검합니다.** 외부 데이터는 알약 번호를 규칙(`drug_N`)으로 만들었고 대회 데이터와 라벨러도 다르므로, 원본보다 라벨 오류 위험이 큽니다. 틀린 번호가 섬이면 데이터를 더해도 점수가 떨어집니다. 아래는 데이터 담당이 한 번 하고, 결과를 실험 이슈에 남깁니다.

| 점검 | 명령 | 볼 것 |
| --- | --- | --- |
| ① 형식 | `uv run python scripts/check_data.py --config configs/yolo_v3_aihub.yaml` | 모두 `[OK]` |
| ② 눈으로 (추가한 사진만) | `uv run python scripts/review_labels.py --config configs/yolo_v3_aihub.yaml --images external` | 박스가 알약에 맞는지, 같은 번호가 원본의 같은 번호 알약과 모양·색이 같은지, 박스 없는 알약이 없는지 |
| ③ 모델로 자동 점검 | `uv run python scripts/find_label_issues.py --exp <기준 실험> --data-config configs/yolo_v3_aihub.yaml --split train --external --draw` | `wrong_class`(번호가 틀렸을 수 있음), `loose_box`. v0로 학습한 기준 모델을 써야 틀린 라벨을 외운 모델이 되지 않음 |
| ④ 분포 (EDA) | `uv run python scripts/make_reports.py 01 --config configs/yolo_v3_aihub.yaml` | `reports/01_eda.html`에서 클래스 분포가 고르게 늘었는지, 이미지 크기가 대회와 같은지(976×1280), 박스 크기가 비슷한지. v0 리포트와 비교하려면 파일 이름을 바꿔 둡니다 |

틀린 사진을 찾으면 `data/annotations/external_drop.csv`에 파일 이름만 적고 `make_external.py`를 다시 실행합니다. 출력의 제외 이유에 `사람이 뺌 (external_drop.csv)`가 나오면 적용된 것입니다. AI Hub 사진은 대회 정답이 아니라서 `manual_fix.csv`로 고치지 않고 빼는 것이 안전합니다. 이 CSV는 파일 이름만 들어 있어 커밋합니다.

**▶ CSV 파일** — `data/annotations/external_drop.csv` (메모장·VS Code로 만듦)

```csv
file_name
K-001001_0_2_0_2_70_000_208.png
```

YOLO는 학습 전에 데이터 버전마다 `export_yolo.py`를 한 번 실행해야 합니다. 아래 ③의 묶음을 그대로 `--set`에 넘기면 됩니다. R-CNN은 JSON을 직접 읽어서 필요 없습니다.

### ② 기준 모델 정하기

PowerShell을 열 때마다 두 값을 정합니다. `$BASE`는 내가 고른 모델 설정, `$P`는 실험 이름 앞부분(`{MMDD}_{담당}_{모델}`)입니다.

**▶ PowerShell 터미널** — 변수 지정 + 기준 학습 (bash는 바로 아래 블록)

```powershell
$BASE = "configs/yolo11s.yaml"      # 내 기준 모델 설정 (R-CNN이면 configs/rcnn_baseline.yaml)
$P    = "1030_modelA_yolo11s"       # 실험 이름 앞부분

uv run python scripts/export_yolo.py --config $BASE
uv run python scripts/run_train.py   --config $BASE --exp-name "${P}_v0_base"
```

**▶ Bash 터미널** — 같은 내용 (Git Bash·Linux·WSL)

```bash
BASE="configs/yolo11s.yaml"      # = 앞뒤에 공백 없이, $ 없이 지정
P="1030_modelA_yolo11s"

uv run python scripts/export_yolo.py --config "$BASE"
uv run python scripts/run_train.py   --config "$BASE" --exp-name "${P}_v0_base"
```

**`$변수`는 두 터미널에서 같나요?** 쓰는 법(`$BASE`, `"${P}_v0_base"`)은 같고, **지정하는 법과 배열은 다릅니다.** PowerShell은 `$BASE = "…"`, bash는 `BASE="…"`(앞에 `$` 없음, `=` 양쪽 공백 없음)입니다. 그래서 이 문서의 블록 중 변수를 **쓰기만** 하는 것은 “공통”, **지정하거나 배열을 다루는** 것은 PowerShell·Bash를 따로 표시했습니다. 변수는 **그 터미널 창에서만** 유지되므로, 새 창을 열면 다시 지정합니다.

이 `${P}_v0_base`가 모든 데이터 실험의 비교 기준입니다. 모델 담당자마다 기준이 다르므로, 다른 사람의 기준과 비교하지 않습니다.

### ③ 실험 묶음 만들기 (PowerShell 변수)

각 데이터 실험이 바꾸는 설정 값을 **배열 하나**로 묶어 둡니다. `--set $D1`처럼 넘기면 배열의 값이 하나씩 나뉘어 들어가고, `--set ($D1 + $D3)`처럼 더하면 그대로 **결합 실험**이 됩니다. 설정 파일을 새로 만들 필요가 없고, 실제로 쓴 값은 `outputs/<실험>/config.yaml`에 남습니다. 아래를 통째로 복사해 두고 새 PowerShell을 열 때마다 붙여 넣습니다.

**묶음이란?** `run_train.py --set` 뒤에 붙이는 `설정키=값` 목록을 변수 하나에 담은 것입니다. 설정 파일(`$BASE`)을 그대로 읽고, 묶음에 적힌 키만 바꿔 학습합니다.

**▶ PowerShell 터미널** — 묶음 예제 (설명용, 실행하지 않아도 됨)

```powershell
# 묶음 = "설정키=값" 여러 개를 담은 배열. D4(해상도 1024)는 설정 두 개를 바꾼다
$D4 = @("data.img_size=1024", "train.batch_size=8")

# 아래 두 줄은 완전히 같은 명령이다 (묶음은 타이핑을 줄여 줄 뿐)
uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_img1024" --set $D4
uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_img1024" --set data.img_size=1024 train.batch_size=8

# 결과: configs/yolo11s.yaml 의 다른 값은 그대로, data.img_size 와 train.batch_size 만 바뀜
#       실제로 쓴 값은 outputs/1030_modelA_yolo11s_v0_img1024/config.yaml 에서 확인

# 두 묶음을 더하면 결합 실험: D1(오버샘플링) + D4(해상도)
uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_rfs01_img1024" --set ($D1 + $D4)
```

**이 방법은 모델 하나용입니다.** ③의 변수와 다음 장들의 명령은 `$BASE`에 지정한 **모델 하나**를 터미널에서 한 실험씩 돌리는 방법입니다. 다른 모델을 하려면 `$BASE`와 `$P`를 바꿔 다시 실행합니다. **여러 모델·여러 실험을 한 번에** 돌리려면 변수 없이 `run_matrix.py`(다음 장)를 씁니다. 같은 묶음을 계획 파일(`configs/matrix_data.yaml`)에 적어 두고 모델 4개를 차례로 학습합니다.

**▶ PowerShell 터미널** — 실험 묶음 변수 만들기 (통째로 붙여 넣기)

```powershell
# D1 드문 클래스 오버샘플링 (YOLO)
$D1  = @("data.repeat_threshold=0.1")
# D2 합성 데이터 v2
$D2  = @("data.version=v2", "data.train_images=data/synth/v2/images", "data.train_annotations=data/annotations/v2.json")
# D3 박스가 안전한 증강 (YOLO)
$D3  = @("yolo_args.fliplr=0.5", "yolo_args.flipud=0.5", "yolo_args.degrees=0.0", "yolo_args.hsv_h=0.0",
         "yolo_args.hsv_s=0.4", "yolo_args.hsv_v=0.3", "yolo_args.scale=0.3", "yolo_args.translate=0.1",
         "yolo_args.mosaic=1.0", "yolo_args.close_mosaic=10")
# D4 입력 해상도 1024 (메모리 때문에 batch 절반)
$D4  = @("data.img_size=1024", "train.batch_size=8")
# D5·D10 정제 v1 (manual_drop.csv + manual_fix.csv)
$D5  = @("data.version=v1", "data.train_annotations=data/annotations/v1.json")
# D6 외부 데이터 v3 (기존 클래스만)
$D6  = @("data.version=v3", "data.train_images=data/external/v3/images", "data.train_annotations=data/annotations/v3.json")
# D6 외부 데이터 v3all (새 알약 추가, 클래스 수가 바뀜)
$D6A = @("data.version=v3all", "data.train_images=data/external/v3all/images",
         "data.train_annotations=data/annotations/v3all.json", "data.class_map=data/class_map_v3all.json")
# D9 MixUp·넓은 크기 변화 (D3 위에)
$D9  = $D3 + @("yolo_args.mixup=0.1", "yolo_args.scale=0.5")
# R-CNN용: D1(샘플러)과 D3(transforms 증강)
$D1R = @("train.balance_power=0.5")
$D3R = @("augment.name=basic", "augment.hflip=0.5", "augment.vflip=0.5", "augment.rot90=0.5",
         "augment.brightness=0.2", "augment.sat_val=0.5", "augment.blur=0.2")
```

같은 키가 두 번 나오면 **뒤에 쓴 값이 이깁니다**(예: `$D9`의 `scale=0.5`가 `$D3`의 0.3을 덮음). 합성 데이터에서 `train.epochs=1`을 더해 `--set ($D1 + $D2 + $D3)`으로 1 epoch 학습을 돌려, 세 묶음이 모두 `config.yaml`에 들어간 것을 확인했습니다.

**bash(Git Bash·Linux·WSL)에서는** PowerShell 블록을 그대로 쓰면 묶음이 조용히 빈 값이 되어 기준 설정으로 학습됩니다. 아래 bash 판을 씁니다. 배열은 항상 따옴표를 붙여 `"${D1[@]}"`로 넘기고, 합칠 때는 `"${D1[@]}" "${D3[@]}"`처럼 나란히 씁니다. 파일(`my_sets.sh`, 커밋하지 않음)로 저장해 두고 `source my_sets.sh`로 불러오면 타이핑이 필요 없습니다.

**▶ Bash 터미널** — 실험 묶음 변수 + `run` 함수 (또는 `my_sets.sh`로 저장 후 `source my_sets.sh`)

```bash
BASE="configs/yolo11s.yaml"
P="1030_modelA_yolo11s"

D1=(data.repeat_threshold=0.1)
D2=(data.version=v2 data.train_images=data/synth/v2/images data.train_annotations=data/annotations/v2.json)
D3=(yolo_args.fliplr=0.5 yolo_args.flipud=0.5 yolo_args.degrees=0.0 yolo_args.hsv_h=0.0
    yolo_args.hsv_s=0.4 yolo_args.hsv_v=0.3 yolo_args.scale=0.3 yolo_args.translate=0.1
    yolo_args.mosaic=1.0 yolo_args.close_mosaic=10)
D4=(data.img_size=1024 train.batch_size=8)
D5=(data.version=v1 data.train_annotations=data/annotations/v1.json)
D6=(data.version=v3 data.train_images=data/external/v3/images data.train_annotations=data/annotations/v3.json)
D6A=(data.version=v3all data.train_images=data/external/v3all/images
     data.train_annotations=data/annotations/v3all.json data.class_map=data/class_map_v3all.json)
D9=("${D3[@]}" yolo_args.mixup=0.1 yolo_args.scale=0.5)
D1R=(train.balance_power=0.5)
D3R=(augment.name=basic augment.hflip=0.5 augment.vflip=0.5 augment.rot90=0.5
     augment.brightness=0.2 augment.sat_val=0.5 augment.blur=0.2)

run() {   # 사용: run <이름> <묶음...>   (R-CNN이면 export 줄을 지움)
  local name=$1; shift
  uv run python scripts/export_yolo.py --config "$BASE" --set "$@"
  uv run python scripts/run_train.py   --config "$BASE" --exp-name "${P}_${name}" --set "$@"
}
# 예: run v0_rfs01 "${D1[@]}"      결합: run v2_rfs01_augsafe "${D2[@]}" "${D1[@]}" "${D3[@]}"
```

| 하는 일 | PowerShell | bash |
| --- | --- | --- |
| 배열 | `$D1 = @("a=1", "b=2")` | `D1=(a=1 b=2)` |
| 넘기기 | `--set $D1` | `--set "${D1[@]}"` |
| 합치기 | `--set ($D1 + $D3)` | `--set "${D1[@]}" "${D3[@]}"` |
| 줄 잇기 | `` ` `` | `\` |

여러 모델·여러 실험을 한 번에 돌릴 때는 변수 없이 다음 장의 스크립트를 씁니다. 이 방법은 bash와 PowerShell에서 명령이 같습니다.

## 스크립트로 한 번에: 모델 4개 × 데이터 실험 (`run_matrix.py`)

**계획 파일 하나에 모델과 데이터 묶음을 적어 두면, `run_matrix.py`가 모델마다 \[기준 + 실험들\]을 차례로 학습하고 모델 × 실험 점수표를 만듭니다.** Python 스크립트라 bash와 PowerShell에서 명령이 같고, 변수를 칠 필요가 없습니다. 합성 데이터로 2 epoch 학습·이어 하기·데이터 없는 실험 건너뛰기를 확인했습니다.

**파일 두 개** (데이터 담당이 develop에 반영, 정답본 zip에도 있음)

- `scripts/run_matrix.py` — 실행기
- `configs/matrix_data.yaml` — 계획 파일 예시. 복사해서 `configs/matrix_<이름>.yaml`로 쓰고 `prefix`만 바꿉니다

**▶ YAML 설정 파일** — `configs/matrix_data.yaml` (계획 파일 전체, develop에 있는 파일과 같음)

```yaml
# 데이터 실험 × 모델 한 번에 돌리기 계획 (scripts/run_matrix.py, 데이터 실험 적용 가이드)
# 복사해서 내 이름으로 쓴다: configs/matrix_<이름>.yaml  (prefix만 바꾸면 됨)
#   uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --dry-run
prefix: 1030_modelA            # 실험 이름 = {prefix}_{모델}_{실험}

models:                        # 이름: 모델 설정 파일 (모두 같은 에폭·검증셋)
  yolov8s: configs/yolo_baseline.yaml
  yolo11s: configs/yolo11s.yaml
  yolo26s: configs/yolo26s.yaml
  yolov8m: configs/yolo_v8m.yaml

bundles:                       # 데이터 실험 묶음 = run_train.py --set 값
  D1: [data.repeat_threshold=0.1]
  D2: [data.version=v2, data.train_images=data/synth/v2/images, data.train_annotations=data/annotations/v2.json]
  D3: [yolo_args.fliplr=0.5, yolo_args.flipud=0.5, yolo_args.degrees=0.0, yolo_args.hsv_h=0.0,
       yolo_args.hsv_s=0.4, yolo_args.hsv_v=0.3, yolo_args.scale=0.3, yolo_args.translate=0.1,
       yolo_args.mosaic=1.0, yolo_args.close_mosaic=10]
  D4: [data.img_size=1024, train.batch_size=8]
  D5: [data.version=v1, data.train_annotations=data/annotations/v1.json]
  D6: [data.version=v3, data.train_images=data/external/v3/images, data.train_annotations=data/annotations/v3.json]
  D6A: [data.version=v3all, data.train_images=data/external/v3all/images,
        data.train_annotations=data/annotations/v3all.json, data.class_map=data/class_map_v3all.json]
  D9: [yolo_args.mixup=0.1, yolo_args.scale=0.5]          # D3과 같이 쓴다 (runs의 augmixup)
  V3CP: [data.version=v3cp, data.train_images=data/synth/v3cp/images, data.train_annotations=data/annotations/v3cp.json]

runs:                          # 실험 이름: 묶음 목록 (뒤에 쓴 묶음의 같은 키가 이김)
  base: []                     # D0 기준 (반드시 둔다)
  rfs01: [D1]
  copypaste: [D2]
  augsafe: [D3]
  img1024: [D4]
  clean: [D5]                  # 검증 정답이 바뀜 → base를 --val-config로 다시 채점해 비교
  aihub: [D6]
  augmixup: [D3, D9]
  # aihuball: [D6A]            # 팀 결정 후. 검증보다 Kaggle 점수로 판단
  # combo: [V3CP, D1, D3, D4]  # 단일 결과를 보고 효과 있던 것만 남겨 결합
  # D11 각인 OCR 재분류: 학습 없이 {ocr: <탐지 실험>} 모델 위에 얹는다 (몇 분).
  #   탐지 실험이 먼저 끝나 있어야 하므로 runs의 맨 끝에 둔다. 단일 결과를 본 뒤 가장 좋은 실험으로 바꾼다.
  #   필요: uv sync --extra ocr, data/drug_info.json (scripts/build_drug_info.py). 없으면 이 줄만 실패로 표시됨
  copypaste_ocr: {ocr: copypaste}
```

**묶음(bundles)과 실험(runs)의 관계**: `bundles`는 데이터 실험마다 바꿀 설정값을 정의하고, `runs`는 그 묶음을 골라 실험 하나로 만듭니다. `runs`에 쓴 묶음 이름(D1\~D9)은 모두 `bundles`에 있어야 하며, 없으면 실행 전에 “묶음 … 이 bundles에 없습니다”로 멈춥니다. D7(K-fold)과 D8(의사 라벨)은 팀 결정이 필요하거나 분할을 바꾸는 실험이라 기본 계획에 넣지 않았습니다.

**D11(OCR)도 넣을 수 있습니다.** D11은 학습이 아니라 이미 학습된 탐지 모델 위에 얹는 후처리여서, 다른 실험처럼 묶음으로 쓸 수 없고 `{ocr: <탐지 실험>}` 형태로 씁니다. `copypaste_ocr: {ocr: copypaste}`는 각 모델의 `copypaste` 결과 모델 위에 OCR을 얹어 채점합니다. 탐지 실험이 먼저 끝나야 하므로 `runs`의 맨 끝에 두고, OCR 패키지(`uv sync --extra ocr`)와 `data/drug_info.json`이 없으면 이 줄만 실패로 나오고 나머지 결과는 그대로입니다. 합성 데이터로 동작을 확인했습니다. 단일 결과를 본 뒤 가장 좋은 실험 이름으로 바꿔 다시 돌리면 그 줄만 실행됩니다.

**모델 하나만 해도 됩니다.** `models`에 하나만 남기거나 `--models yolo11s`를 주면, 그 모델의 D0(base)부터 D9·D11까지 차례로 학습하고 점수표(`--summary`)와 그래프(`plot_matrix.py --models yolo11s`)를 그 모델만으로 만듭니다.

**실행** (bash·PowerShell 공통)

**▶ 터미널 (PowerShell·Bash 공통)** — `run_matrix.py` 실행. 변수를 쓰지 않아 어느 터미널에서나 그대로 붙여 넣습니다

```bash
# ① 실행될 명령만 보기 (학습 없음). 계획 파일을 고친 뒤 오타 확인용
uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --dry-run

# ② 전체를 1 epoch으로 동작 확인 (실험 이름 끝에 _e1, 본 결과와 안 섞임). 본 학습 전 한 번
uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --epochs 1

# ③ 본 학습: 모델 4개 × runs 전부. 중간에 꺼지면 같은 명령을 다시 (끝난 실험은 건너뜀)
uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml

# ④ 일부만: 모델과 실험을 골라서 (아래 “예시 ①”)
uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --models yolo11s yolo26s --runs base copypaste

# ⑤ 학습 없이 지금까지의 점수표만 (아래 “예시 ②”)
uv run python scripts/run_matrix.py --plan configs/matrix_data.yaml --summary
```

**▶ 출력 예시 ①** — `--models yolo11s yolo26s --runs base copypaste`

**무엇을 위한 실행인가**: 계획 파일을 바꾸지 않고 **일부 모델·일부 실험만** 돌릴 때 씁니다. 예를 들어 “D2(합성)가 최신 모델에서도 효과가 있나”만 먼저 보거나, 모델을 담당자끼리 나눠 돌릴 때입니다. 이 명령은 yolo11s·yolo26s 각각의 `base`와 `copypaste` 4개만 학습합니다. **결과**는 그 4개의 점수표입니다(표 모양 예시. 숫자는 합성 연습 데이터의 값이라 실제 모델 성능과 무관).

```text
[1030_modelA_yolo11s_base] (기준)
  $ python ./scripts/export_yolo.py --config configs/yolo11s.yaml
  $ python ./scripts/run_train.py --config configs/yolo11s.yaml --exp-name 1030_modelA_yolo11s_base
  … (학습 로그)
[1030_modelA_yolo11s_copypaste] data.version=v2 data.train_images=data/synth/v2/images …
  …
결과 (map = mAP@[0.75:0.95], 괄호 = 같은 모델 base 대비)
실험                       yolo11s             yolo26s
base                0.2145 (+0.0000)    0.1567 (+0.0000)
copypaste           0.5369 (+0.3224)    0.6002 (+0.4435)
저장: …/outputs/matrix_1030_modelA.csv
```

**▶ 출력 예시 ②** — `--summary`

**용도**: 학습은 하지 않고 `outputs/`에 끝난 실험들의 `metrics.json`을 읽어 **모델 × 실험 점수표만 다시 만듭니다.** 여러 날 나눠 돌렸거나 다른 사람이 아직 학습 중일 때 지금까지의 결과를 볼 때 씁니다. 몇 초면 끝납니다. **결과**는 화면 표와 `outputs/matrix_<prefix>.csv`입니다. 아직 없는 칸은 `-`로 나옵니다.

```text
결과 (map = mAP@[0.75:0.95], 괄호 = 같은 모델 base 대비)
실험                       yolov8s             yolo11s             yolo26s             yolov8m
base                0.2145 (+0.0000)    0.1567 (+0.0000)                   -                   -
rfs01                              -                   -                   -                   -
copypaste           0.5369 (+0.3224)    0.6002 (+0.4435)                   -                   -
augsafe             0.2811 (+0.0666)    0.2000 (+0.0433)                   -                   -
…
copypaste_ocr       0.5369 (+0.3224)    0.6002 (+0.4435)                   -                   -
저장: …/outputs/matrix_1030_modelA.csv
```

(표의 모양을 보여 주는 예입니다. 숫자는 합성 연습 데이터로 다른 작은 모델을 돌린 값이라 실제 모델의 성능과 무관합니다. `-`는 아직 학습이 끝나지 않은 칸입니다. ④처럼 일부만 돌린 뒤에도 `--summary`를 다시 실행하면 전체 표가 나옵니다.)

| 동작 | 내용 |
| --- | --- |
| 이어 하기 | `outputs/<실험>/metrics.json`이 있으면 건너뜀. 중간에 꺼져도 같은 명령을 다시 실행. 다시 학습은 `--force` |
| 실패 | 하나가 실패해도 다음 실험 진행, 끝에 실패 목록 |
| 데이터 없음 | `train_annotations` 파일이 없는 실험은 “데이터 없음”으로 건너뜀 (예: v3를 아직 안 만듦) |
| 내보내기 | YOLO는 데이터 버전마다 `export_yolo.py`를 자동 실행 |
| `--epochs N` | 실험 이름 끝에 `_eN`이 붙어 본 결과와 섞이지 않음 |
| 결과 | 화면 표 + `outputs/matrix_<prefix>.csv` (모델 × 실험 map, 같은 모델 base 대비 변화) |

**▶ 출력 예시** — `run_matrix.py`가 끝에 보여 주는 표

```text
결과 (map = mAP@[0.75:0.95], 괄호 = 같은 모델 base 대비)
실험                     yolov8s             yolo11s   …
base                       0.xxxx              0.xxxx
copypaste        0.xxxx (+0.0xxx)    0.xxxx (-0.0xxx)
```

**시간**: 모델 4개 × 실험 8개 = 학습 32번입니다. 내 PC의 기준 학습 시간(`metrics.json`의 `train_min`) × 32가 대략의 총 시간이고, `img1024`와 `yolov8m`은 더 걸립니다. 하루 안에 끝나지 않으면 `--models`로 모델 담당자끼리 나눠 돌리고(각자 다른 prefix), 결과 CSV를 모읍니다.

### 끝난 뒤 그래프로 비교 (`plot_matrix.py`)

**학습이 끝나면 명령 하나로 학습 곡선과 검증 점수 그래프를 만들고, `report.html` 한 페이지에 모읍니다.** 일부 실험만 끝났어도 끝난 것만 그립니다.

**▶ 터미널 (PowerShell·Bash 공통)** — `plot_matrix.py` 실행. 셋째 줄의 `$P`는 「준비 ②」에서 지정한 값 (지정하지 않았다면 실험 이름을 직접 씁니다)

```bash
# ① run_matrix 결과 전체 (계획 파일의 모델 × 실험)
uv run python scripts/plot_matrix.py --plan configs/matrix_data.yaml

# ② 모델 하나만
uv run python scripts/plot_matrix.py --plan configs/matrix_data.yaml --models yolo11s

# ③ run_matrix --epochs 1 로 돌린 결과
uv run python scripts/plot_matrix.py --plan configs/matrix_data.yaml --epochs 1

# ④ 실험 이름을 직접 (첫째가 기준). 변수 없이 outputs/ 폴더 이름을 그대로 씁니다
#    run_matrix 결과 이름 = {prefix}_{모델}_{실험},  단일 실험 이름 = {$P}_{이름}
uv run python scripts/plot_matrix.py --exps 1030_modelA_yolo11s_base 1030_modelA_yolo11s_augsafe 1030_modelA_yolo11s_copypaste
```

결과는 `outputs/compare_<prefix>/`에 저장되고, `report.html`을 브라우저로 열면 됩니다.

| 그림 | 내용 | 보는 법 |
| --- | --- | --- |
| `scores.png` | 모델 × 실험 히트맵. 칸에 map과 기준 대비 변화 | 초록 = 기준보다 오름, 빨강 = 내림. 가로로 모두 초록인 실험이 모델과 상관없이 효과 있는 것 |
| `metrics.png` | map·map\_75·mar\_100 막대 | map\_75는 그대로인데 map만 오르면 박스가 더 정확해진 것 (평가 지표 ①) |
| `curves_<모델>.png` | 모델마다 에폭별 train loss, val loss, 검증 mAP. 실험마다 선 하나, 기준은 검정 굵은 선 | 과적합·과소적합 판단 (과적합·과소적합 확인 ①의 표) |
| `scores.csv` | 그래프에 쓴 숫자 | 실험 기록표에 옮길 때 |

그래프에는 대회 이미지가 없어서 `report.html`이나 PNG를 PR 본문과 디스코드에 올려도 됩니다. `outputs/` 폴더 자체는 커밋하지 않습니다. R-CNN 실험은 `train_history.json`에서 train loss와(있으면) 검증 map을 그립니다.

### 모델에 따라 데이터 실험 효과가 다를 수 있나요?

**네, 다를 수 있고 그래서 모델 × 실험 표를 만듭니다.** 흔한 이유는 세 가지입니다.

- **모델 크기**: 큰 모델(yolov8m)은 작은 데이터(185장)를 외우기 쉬워, 증강·데이터 추가(D2·D3·D6)가 더 크게 도움됩니다. 작은 모델은 같은 증강이 오히려 덜 배우게 만들 수 있습니다.
- **구조**: YOLO26은 NMS가 필요 없는 구조라 겹치는 박스를 다루는 방식이 v8·11과 다릅니다. 그래서 합성(D2)처럼 박스 배치가 바뀌는 실험의 효과가 다르게 나올 수 있습니다.
- **우연**: 검증셋이 47장이라 모델별 차이의 일부는 우연입니다(평가 지표 ④ 참고).

**표 읽는 법**: 모든 모델에서 +인 실험은 믿고 결합에 넣습니다. 한두 모델에서만 +인 실험은 **최종 모델을 정한 뒤 그 모델의 값으로 판단**합니다. 모델 선택은 각 모델의 `base`끼리가 아니라, 모델별 최고 조합끼리 비교합니다.

## 전부 한 번에: 실험 → 결합 → 튜닝 → 보고서 → 데모 (`run_pipeline.py`)

**명령 하나로 모델 4개의 데이터 실험부터 최고 성능 선정, 제출 파일, 보고서, 웹 데모까지 진행합니다.** 위 `run_matrix.py`·`plot_matrix.py`와 기존 `tune_yolo.py`·`make_submission.py`·`app/demo.py`를 차례로 부르는 스크립트이고, 계획 파일은 같은 `configs/matrix_data.yaml`을 씁니다. 합성 연습 데이터로 8단계를 끝까지 실행해 확인했습니다.

| 단계 | 하는 일 |
| --- | --- |
| 1 data\_exp | 모델마다 D0(base) + 계획 파일의 데이터 실험 학습. 정제(v1) 실험은 base를 v1 정답으로 다시 채점 |
| 2 combo | 모델의 절반 이상에서 +0.01 이상인 실험을 모아 결합 실험 학습 (데이터 버전은 가장 좋은 하나, v3all 제외) |
| 3 tune | 지금까지 1등 (모델, 실험)으로 하이퍼파라미터 탐색 → 찾은 값으로 다시 학습 (`tuned`) |
| 4 ocr | 최종 1등 위에 D11 OCR 재분류. 점수가 오르면 최종 1등을 바꿈 (OCR 패키지 없으면 건너뜀) |
| 5 plot | 모델 × 실험 그래프 (`outputs/compare_<prefix>/`) |
| 6 submit | 최종 1등으로 Kaggle 제출 파일(`submissions/<실험>.csv`)과 클래스별 AP |
| 7 report | 한 페이지 보고서 `outputs/pipeline_<prefix>/report.html` (결론, 점수표, 그래프, 하위 10개 클래스, 찾은 하이퍼파라미터, 산출물, 실행 기록) |
| 8 demo | `--demo`를 주면 최종 1등 모델로 웹 데모 실행 (http://127.0.0.1:7860, 끝내기 Ctrl+C) |

**▶ 터미널 (PowerShell·Bash 공통)** — `run_pipeline.py` 실행

```bash
# ① 동작 확인: 모든 단계를 1 epoch·튜닝 2회로 (이름 끝 _e1, 본 결과와 안 섞임)
uv run python scripts/run_pipeline.py --epochs 1 --tune-iterations 2 --tune-epochs 1

# ② 본 실행 (모델 4개, 하루 이상 걸릴 수 있음). 멈추면 같은 명령을 다시 → 끝난 실험은 건너뜀
uv run python scripts/run_pipeline.py

# ③ 끝나면 웹 데모까지
uv run python scripts/run_pipeline.py --demo

# ④ 모델 하나만 / 일부 단계만 (예: 학습은 끝났고 그래프·보고서만 다시)
uv run python scripts/run_pipeline.py --models yolo11s
uv run python scripts/run_pipeline.py --stages plot report
```

| 옵션 | 기본 | 뜻 |
| --- | --- | --- |
| `--plan` | `configs/matrix_data.yaml` | 모델·데이터 실험 계획 파일 |
| `--min-gain` / `--min-share` | 0.01 / 0.5 | 결합에 넣을 기준: 모델의 50% 이상에서 +0.01 이상 |
| `--tune-iterations` / `--tune-epochs` | 20 / 15 | 하이퍼파라미터 시도 횟수와 시도당 epoch. 늘릴수록 오래 걸림 |
| `--stages` | 전부 | `data_exp combo tune ocr plot submit report` 중 고르기 |

**자동으로 하지 않는 것** (사람 판단 필요): 데이터 버전 만들기(「준비 ①」, 특히 v1 정제와 AI Hub), v3all 채택, D7 K-fold, D8 의사 라벨, 전체 데이터 최종 학습(P5), Kaggle 제출 자체. 데이터 버전이 없는 실험은 건너뛰고, 보고서의 “실행 기록”에 남습니다. 보고서에는 대회 이미지가 없어 팀에 공유해도 됩니다.

## 단일 실험: 실험 기록표와 실행 방법

데이터 가이드 v3의 실험 기록표에 **실행 방법**을 더했습니다. 모든 줄의 실행은 같은 형태입니다. YOLO는 같은 `--set`으로 `export_yolo.py`를 먼저 한 번 실행합니다.

**순서**: ① 아래 블록으로 변수를 지정한 뒤 ② 표의 `실행 방법` 칸을 그대로 복사해 **같은 터미널 창**에 붙여 넣습니다. 표의 명령은 이 블록의 변수(`$BASE`, `$P`, `$D1` …)를 쓰므로, 창을 새로 열었다면 ①부터 다시 합니다. 모델을 바꾸려면 `$BASE`와 `$P` 두 줄만 바꿉니다.

**▶ PowerShell 터미널** — ① 변수 지정 (창을 열 때마다 한 번, 통째로 붙여 넣기)

```powershell
# 모델 (여기 두 줄만 내 것으로 바꾼다)
$BASE = "configs/yolo11s.yaml"          # 내 기준 모델 설정 파일
$P    = "1030_modelA_yolo11s"           # 실험 이름 앞부분 {MMDD}_{담당}_{모델}
$PR   = "1030_modelB_frcnn"             # R-CNN 실험 이름 앞부분 (R-CNN 담당만)

# 데이터 실험 묶음 (--set 뒤에 붙일 "설정키=값" 목록)
$D1  = @("data.repeat_threshold=0.1")
$D2  = @("data.version=v2", "data.train_images=data/synth/v2/images", "data.train_annotations=data/annotations/v2.json")
$D3  = @("yolo_args.fliplr=0.5", "yolo_args.flipud=0.5", "yolo_args.degrees=0.0", "yolo_args.hsv_h=0.0",
         "yolo_args.hsv_s=0.4", "yolo_args.hsv_v=0.3", "yolo_args.scale=0.3", "yolo_args.translate=0.1",
         "yolo_args.mosaic=1.0", "yolo_args.close_mosaic=10")
$D4  = @("data.img_size=1024", "train.batch_size=8")
$D5  = @("data.version=v1", "data.train_annotations=data/annotations/v1.json")
$D6  = @("data.version=v3", "data.train_images=data/external/v3/images", "data.train_annotations=data/annotations/v3.json")
$D6A = @("data.version=v3all", "data.train_images=data/external/v3all/images",
         "data.train_annotations=data/annotations/v3all.json", "data.class_map=data/class_map_v3all.json")
$D8  = @("data.version=v4pseudo", "data.train_images=data/pseudo/v4pseudo/images", "data.train_annotations=data/annotations/v4pseudo.json")
$D9  = $D3 + @("yolo_args.mixup=0.1", "yolo_args.scale=0.5")
$D1R = @("train.balance_power=0.5")
$D3R = @("augment.name=basic", "augment.hflip=0.5", "augment.vflip=0.5", "augment.rot90=0.5",
         "augment.brightness=0.2", "augment.sat_val=0.5", "augment.blur=0.2")

# 확인: 아래처럼 나오면 준비 끝
$BASE; $P; $D1
# configs/yolo11s.yaml
# 1030_modelA_yolo11s
# data.repeat_threshold=0.1
```

예를 들어 표의 D1 줄 명령은, 변수를 풀어 쓰면 아래와 같습니다.

```text
uv run python scripts/export_yolo.py --config configs/yolo11s.yaml --set data.repeat_threshold=0.1
uv run python scripts/run_train.py --config configs/yolo11s.yaml --exp-name 1030_modelA_yolo11s_v0_rfs01 --set data.repeat_threshold=0.1
```

**▶ Bash 터미널** — 같은 내용. 「준비 ③」의 bash 블록(변수 + `run` 함수)을 `my_sets.sh`로 저장해 두고 불러온 뒤 실행

```bash
source my_sets.sh            # BASE, P, D1~D9, run 함수 지정 (없으면 「준비 ③」 bash 블록을 붙여 넣기)

uv run python scripts/export_yolo.py --config "$BASE"
uv run python scripts/run_train.py   --config "$BASE" --exp-name "${P}_v0_base"   # 기준
run v0_rfs01     "${D1[@]}"                  # D1
run v2_copypaste "${D2[@]}"                  # D2
run v0_augsafe   "${D3[@]}"                  # D3
run v0_img1024   "${D4[@]}"                  # D4
run v1_clean     "${D5[@]}"                  # D5·D10
run v3_aihub     "${D6[@]}"                  # D6
run v3all_aihub  "${D6A[@]}"                 # D6 v3all
run v0_augmixup  "${D9[@]}"                  # D9
```

**▶ 실험 기록표** — `실행 방법` 칸은 PowerShell 명령입니다. 위 ①을 실행한 같은 창에 칸 내용을 통째로 붙여 넣으면 됩니다(`;`는 명령 구분). bash는 위 bash 블록의 같은 줄을 씁니다.

| ID | 실험 | 실행 방법 (PowerShell, ① 변수 지정 후) | 비교 기준 | mAP@\[0.75:0.95\] | 변화 | 판단 | 담당 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | 기준 (내 모델 그대로) | `uv run python scripts/export_yolo.py --config $BASE; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_base"` | — |  | — | — |  |
| D1 | 드문 클래스 오버샘플링 (YOLO) | `uv run python scripts/export_yolo.py --config $BASE --set $D1; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_rfs01" --set $D1` | D0 |  |  |  |  |
| D1 | 드문 클래스 샘플러 (R-CNN) | `uv run python scripts/run_train.py --config configs/rcnn_baseline.yaml --exp-name "${PR}_v0_balance05" --set $D1R` | R-CNN 기준 |  |  |  |  |
| D2 | 합성 데이터 v2 | `uv run python scripts/export_yolo.py --config $BASE --set $D2; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v2_copypaste" --set $D2` | D0 |  |  |  |  |
| D3 | 안전한 증강 (YOLO) | `uv run python scripts/export_yolo.py --config $BASE --set $D3; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_augsafe" --set $D3` | D0 |  |  |  |  |
| D3 | 안전한 증강 (R-CNN) | `uv run python scripts/run_train.py --config configs/rcnn_baseline.yaml --exp-name "${PR}_v0_aug" --set $D3R` | R-CNN 기준 |  |  |  |  |
| D4 | 입력 해상도 1024 | `uv run python scripts/export_yolo.py --config $BASE --set $D4; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_img1024" --set $D4` | D0 (학습 시간도 기록) |  |  |  |  |
| D5·D10 | 정제 v1 (빼기 + 고치기) | `uv run python scripts/export_yolo.py --config $BASE --set $D5; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v1_clean" --set $D5; uv run python scripts/evaluate.py --exp "${P}_v0_base" --val-config configs/yolo_v1_clean.yaml` | D0을 v1 정답으로 다시 채점한 값 (세 번째 명령, 아래 ①) |  |  |  |  |
| D6 | 외부 데이터 v3 (기존 클래스) | `uv run python scripts/export_yolo.py --config $BASE --set $D6; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v3_aihub" --set $D6` | D0 |  |  |  |  |
| D6 | 외부 데이터 v3all (새 알약) | `uv run python scripts/export_yolo.py --config $BASE --set $D6A; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v3all_aihub" --set $D6A` | Kaggle 점수 (검증은 하락 여부만, ②) |  |  |  |  |
| D9 | MixUp·넓은 크기 변화 (D3 포함) | `uv run python scripts/export_yolo.py --config $BASE --set $D9; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v0_augmixup" --set $D9` | D3 결과 |  |  |  |  |
| D11 | 각인 OCR 재분류 (학습 없음, D2 모델 위에) | `uv sync --extra ocr; uv run python scripts/run_train.py --config configs/ocr_rerank.yaml --exp-name "${P}_v2_copypaste_ocr" --set ($D2 + @("model.detector_exp=${P}_v2_copypaste"))` | 탐지에 쓴 실험 (여기서는 D2) |  |  |  |  |
| D8 | 테스트 의사 라벨 (팀 결정 후) | `uv run python scripts/export_yolo.py --config $BASE --set $D8; uv run python scripts/run_train.py --config $BASE --exp-name "${P}_v4pseudo" --set $D8` | D0 |  |  |  |  |

**판단** 칸에는 “효과 있음(+0.01 이상)”, “불분명”, “나빠짐” 중 하나를 적습니다. 점수는 학습이 끝날 때 화면 마지막 줄과 `outputs/<실험>/metrics.json`의 `map`에 나옵니다.

### 한 번에 돌리기

여러 실험을 밤새 한 번에 돌리려면 위 `run_matrix.py`를 씁니다. `--models`에 내 모델 하나만 주면 이 표의 실험을 차례로 실행합니다.

### 따로 주의할 줄

**① D5·D10 (정답이 바뀜)**: v1은 검증셋 정답도 고칩니다. 기준은 옛 정답, v1 모델은 새 정답으로 채점되면 비교가 어긋나므로, 기준을 v1 정답으로 다시 채점해서 비교합니다(재학습 없음).

**▶ 터미널 (PowerShell·Bash 공통, `$P` 지정 후)** — 기준을 v1 정답으로 다시 채점

```powershell
uv run python scripts/evaluate.py --exp "${P}_v0_base" --val-config configs/yolo_v1_clean.yaml
# → eval_…_valv1.json의 map과 ${P}_v1_clean의 map을 비교
```

**② D6 v3all (클래스 수가 바뀜)**: 검증셋에는 새 알약이 없어 검증 점수로는 효과가 안 보입니다. 검증은 “v3보다 0.01 이상 떨어지지 않았는지”만 보고, Experimentation Lead가 v3와 v3all을 둘 다 제출해 Kaggle Public으로 고릅니다. v3all 모델은 v3all 모델끼리만 앙상블할 수 있습니다.

**③ D11 (OCR 재분류)**: 학습이 없는 후처리라 마지막에 합니다. 이미 학습된 내 최고 실험을 `detector_exp`로 지정하고, 그 실험에 쓴 데이터 묶음(검증·class\_map이 같아야 함)을 같이 넣습니다.

**▶ PowerShell 터미널** — D11 OCR 재분류 (bash는 바로 아래 블록)

```powershell
uv sync --extra ocr                                     # 처음 한 번
$BEST = "${P}_v2_copypaste"                             # 내 최고 탐지 실험 이름
uv run python scripts/run_train.py --config configs/ocr_rerank.yaml --exp-name "${BEST}_ocr" `
  --set ($D2 + @("model.detector_exp=$BEST"))
```

**▶ Bash 터미널** — D11 OCR 재분류

```bash
uv sync --extra ocr
BEST="${P}_v2_copypaste"
uv run python scripts/run_train.py --config configs/ocr_rerank.yaml --exp-name "${BEST}_ocr" \
  --set "${D2[@]}" "model.detector_exp=$BEST"
```

점수가 오르지 않으면 `--set`에 `model.ocr_min_sim=0.9`처럼 값을 바꿔 다시 돌립니다(몇 분). 값의 뜻은 데이터 가이드 D11에 있습니다.

**④ D8 (의사 라벨)**: 대회 규칙이 테스트 이미지 학습을 허용할 때만 합니다. 데이터 담당이 `make_pseudo.py`로 v4pseudo를 만든 뒤 아래 묶음을 씁니다.

**▶ PowerShell 터미널** — D8 묶음 (bash는 바로 아래 블록)

```powershell
$D8 = @("data.version=v4pseudo", "data.train_images=data/pseudo/v4pseudo/images", "data.train_annotations=data/annotations/v4pseudo.json")
```

**▶ Bash 터미널** — D8 묶음

```bash
D8=(data.version=v4pseudo data.train_images=data/pseudo/v4pseudo/images data.train_annotations=data/annotations/v4pseudo.json)
```

## 결합 실험: 효과 있던 것 합치기

**설정 묶음(D1·D3·D4·D9)은 `+`로 더하면 되고, 데이터 묶음(D2·D5·D10·D6·D8)은 한 실험에 하나만 쓸 수 있습니다.** 데이터 묶음은 모두 `data.version`을 바꾸기 때문에, 두 개를 더하면 뒤의 것만 남습니다. 그래서 정제·외부·합성을 함께 쓰려면 **결합 데이터 버전을 한 번 더 만듭니다**.

| 종류 | 실험 | 결합 방법 |
| --- | --- | --- |
| 설정 묶음 (YOLO) | D1, D3 또는 D9, D4 | `$D1 + $D3 + $D4` 처럼 더하기. D3와 D9는 둘 중 하나 |
| 설정 묶음 (R-CNN) | D1R, D3R | `$D1R + $D3R` |
| 데이터 묶음 | D2, D5·D10, D6, D8 | 아래 ①의 순서로 결합 버전을 만들고, 그 묶음 하나만 씁음 |
| 후처리 | D11 | 결합 모델을 다 학습한 뒤 마지막에 (단일 실험 ③과 같은 방법) |

### ① 결합 데이터 버전 만들기 (데이터 담당)

순서는 **정제 → 외부 → 합성**입니다. 틀린 라벨을 먼저 고치고, 외부 이미지를 더한 뒤, 그래도 모자란 드문 클래스만 합성으로 채웁니다. 효과 없던 단계는 건너뜁니다.

**▶ 터미널 (PowerShell·Bash 공통)** — 결합 데이터 버전 만들기 (데이터 담당)

```powershell
# 1) 정제 v1 (D5·D10): manual_drop.csv + manual_fix.csv 적용
uv run python scripts/clean_labels.py --config configs/base.yaml --version v1
# 2) v1 위에 외부 데이터 (D6) → v3c
uv run python scripts/make_external.py --config configs/yolo_v1_clean.yaml --version v3c --classes known
# 3) v3c 위에 합성 (D2) → v3cp  (아래 설정 파일을 먼저 만듦)
uv run python scripts/make_copy_paste.py --config configs/data_v3c.yaml --version v3cp
uv run python scripts/review_labels.py --config configs/data_v3cp.yaml --images synth --only-flagged   # covered 0개 확인
```

`make_copy_paste.py`는 `--set`이 없어서, 읽을 데이터 버전을 작은 설정 파일로 알려줍니다. 두 파일 모두 커밋해서 팀이 같이 씁니다.

**▶ YAML 설정 파일 (새로 만듦)** — `configs/data_v3c.yaml`

```yaml
# configs/data_v3c.yaml — 결합 데이터: 정제 v1 + AI Hub (D5·D10 + D6)
base: base.yaml
data:
  train_images: data/external/v3c/images
  train_annotations: data/annotations/v3c.json
  version: v3c
```

**▶ YAML 설정 파일 (새로 만듦)** — `configs/data_v3cp.yaml`

```yaml
# configs/data_v3cp.yaml — 결합 데이터: 정제 v1 + AI Hub + 합성 (D5·D10 + D6 + D2)
base: base.yaml
data:
  train_images: data/synth/v3cp/images
  train_annotations: data/annotations/v3cp.json
  version: v3cp
```

외부 데이터로 드문 클래스가 이미 늘어난 상태라, 3)에서 붙이는 알약 수는 v2보다 적게 나옵니다. 두 단계만 합칠 때의 명령입니다.

| 합칠 것 | 만드는 명령 | 묶음에 쓸 버전 |
| --- | --- | --- |
| D5·D10 + D2 | `make_copy_paste.py --config configs/yolo_v1_clean.yaml --version v2c` | `v2c`, `data/synth/v2c/images` |
| D5·D10 + D6 | 1), 2)만 | `v3c`, `data/external/v3c/images` |
| D6 + D2 | `make_copy_paste.py --config configs/yolo_v3_aihub.yaml --version v3p` | `v3p`, `data/synth/v3p/images` |

### ② 결합 학습 (모델 담당)

결합 데이터 묶음을 하나 만들고, 효과 있던 설정 묶음을 더합니다. 아래는 D1·D2·D3·D4·D5·D6·D10이 모두 효과 있었다고 가정한 예입니다.

**▶ PowerShell 터미널** — 결합 학습 (「준비 ②③」의 변수를 먼저 지정, bash는 바로 아래 블록)

```powershell
$DATA = @("data.version=v3cp", "data.train_images=data/synth/v3cp/images", "data.train_annotations=data/annotations/v3cp.json")
$ALL  = $DATA + $D1 + $D3 + $D4
uv run python scripts/export_yolo.py --config $BASE --set $ALL
uv run python scripts/run_train.py   --config $BASE --exp-name "${P}_v3cp_rfs01_augsafe_1024" --set $ALL
```

**▶ Bash 터미널** — 결합 학습

```bash
DATA=(data.version=v3cp data.train_images=data/synth/v3cp/images data.train_annotations=data/annotations/v3cp.json)
run v3cp_rfs01_augsafe_1024 "${DATA[@]}" "${D1[@]}" "${D3[@]}" "${D4[@]}"
```

정제(v1)가 들어간 결합은 검증 정답도 바뀌므로, 비교 기준은 단일 실험 ①처럼 v1 정답으로 다시 채점한 기준 점수입니다.

### ③ 결합이 기대보다 낮을 때: 하나씩 빼기

따로는 좋았는데 합쳤을 때 떨어지는 일은 흔합니다. 결합 점수가 **가장 좋은 단일 실험보다 낮으면**, 하나씩 빼 보고 점수가 오르는 쪽을 남깁니다.

| 실험 이름 끝 | 묶음 | 빼 것 |
| --- | --- | --- |
| `_noD1` | `$DATA + $D3 + $D4` | 오버샘플링 |
| `_noD3` | `$DATA + $D1 + $D4` | 증강 |
| `_noD4` | `$DATA + $D1 + $D3` | 해상도 (학습 시간 절반) |
| `_noD2` | `v3c` 묶음 `+ $D1 + $D3 + $D4` | 합성 |

자주 부딪히는 조합입니다.

- **D1 + D2**: 둘 다 드문 알약을 더 자주 보여줍니다. 같은 몇 장에 과적합되기 쉬워, 합칠 때 `data.repeat_threshold=0.05`처럼 반복을 줄여 봅니다.
- **D6 + 에폭**: 학습 이미지가 몇 배로 늘면 같은 에폭에서도 학습량이 늘어 시간이 길어집니다. 다음 장의 곡선을 보고 에폭을 조절합니다.
- **D4 + 큰 모델**: GPU 메모리가 모자라면 `train.batch_size=4`를 뒤에 더합니다(뒤에 쓴 값이 이김).

### ④ 최종 조합이 정해지면

최종 묶음을 실험 이슈에 적고 Experimentation Lead에게 넘깁니다. 모델 가이드의 T4(시드 앙상블, `--set seed=7`)와 P5(전체 데이터 학습)를 같은 묶음으로 진행하고, D11은 그 결과 모델 위에 마지막으로 얹습니다.

## 과적합·과소적합 확인

**학습 곡선(에폭별 loss와 검증 점수)을 보고, 필요하면 같은 지표로 학습셋 점수와 검증 점수의 차이를 쟽니다.** 점수 하나만 보면 “덜 배웠는지”와 “외웠는지”를 구분할 수 없기 때문입니다.

| 상태 | 곡선에서 보이는 것 | 조치 (한 번에 하나씩) |
| --- | --- | --- |
| **과소적합** (덜 배움) | 마지막 에폭까지 train·val loss가 함께 내려가고, 검증 점수도 아직 오르는 중. 학습셋 점수도 낮음 | 에폭 늘리기(`train.epochs=100`), 큰 모델, 해상도 키우기(D4), 증강 약하게 |
| **적당함** | 두 loss가 내려가다 평평해지고, 검증 점수가 끝부분에서 제자리 | 유지. 이 설정으로 결합 실험 |
| **과적합** (외움) | train loss는 계속 내려가는데 val loss가 어느 에폭부터 다시 오르고, 검증 점수가 중간에 최고를 찍고 떨어짐. 학습셋 점수만 높음 | 증강 추가(D3), 데이터 추가(D2·D6), 에폭 줄이기·조기 종료(`yolo_args.patience=15`), 작은 모델, D1 반복 줄이기 |

검증셋이 47장뿐이라 검증 곡선은 에폭마다 흔들립니다. 한두 에폭의 오르내림이 아니라 **5에폭 이상 이어지는 방향**으로 판단합니다.

### ① 학습 곡선 보기

학습이 끝나면 곡선 자료가 저장됩니다.

| 모델 | 파일 | 들어 있는 것 |
| --- | --- | --- |
| YOLO | `outputs/<실험>/yolo_run/results.csv` | 에폭별 `train/*_loss`, `val/*_loss`, `metrics/mAP50-95(B)` |
| R-CNN | `outputs/<실험>/train_history.json` | 에폭별 `train_loss`. `train.select_best=true`로 학습하면 `val_map`도 |

YOLO 곡선은 개인 노트북(`notebooks/scratch/<이름>_curves.ipynb`)에 아래 셀을 붙여 그립니다. 기준과 실험을 같은 그림에 겹쳐 볼 수 있습니다.

**▶ 노트북 셀 (.ipynb)** — `notebooks/scratch/<이름>_curves.ipynb`의 코드 셀 (YOLO 곡선)

```python
import matplotlib.pyplot as plt
import pandas as pd

from src.interfaces import OUTPUT_DIR
from src.visualize import set_korean_font

set_korean_font()
EXPS = ["1030_modelA_yolo11s_v0_base", "1030_modelA_yolo11s_v0_augsafe"]  # 기준을 맨 앞에

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
for i, exp in enumerate(EXPS):
    df = pd.read_csv(OUTPUT_DIR / exp / "yolo_run" / "results.csv")
    df.columns = df.columns.str.strip()
    # 모델마다 loss 종류가 다름 (YOLO26은 dfl_loss 없음) → 있는 *_loss 칸을 모두 더함
    loss = [c for c in df.columns if c.endswith("_loss")]
    train = df[[c for c in loss if c.startswith("train/")]].sum(axis=1)
    val = df[[c for c in loss if c.startswith("val/")]].sum(axis=1)
    m = df["metrics/mAP50-95(B)"]
    c = f"C{i}"
    ax1.plot(df["epoch"], train, c, label=f"{exp} train")
    ax1.plot(df["epoch"], val, c, ls="--", label=f"{exp} val")
    ax2.plot(df["epoch"], m, c, label=exp)
    best = int(df.loc[m.idxmax(), "epoch"])
    print(
        f"{exp}: 최고 mAP50-95 {m.max():.3f} (epoch {best}/{len(df)}), 마지막 {m.iloc[-1]:.3f}, "
        f"val loss 최저 epoch {int(df.loc[val.idxmin(), 'epoch'])}"
    )
ax1.set(title="loss (실선 train, 점선 val)", xlabel="epoch")
ax2.set(title="검증 mAP50-95 (ultralytics, 추세용)", xlabel="epoch")
ax1.legend(fontsize=7)
ax2.legend(fontsize=7)
plt.tight_layout()
```

**숫자로 읽는 법** (출력 한 줄이 실험 하나):

- `최고 epoch`이 **마지막 근처**이고 val loss도 마지막에 최저 → 아직 덜 배움. 에폭을 늘려 봅니다. 합성 연습 데이터 30에폭 학습이 이 모양이었습니다(최고 epoch 29/30).
- `최고 epoch`이 **중간**이고 `마지막` 점수가 최고보다 뚜렷하게 낮음 → 과적합. YOLO는 최고 에폭을 `best.pt`로 저장하므로 점수는 지켜지지만, 남은 에폭은 시간 낭비입니다.
- ultralytics의 `mAP50-95`는 IoU 0.5\~0.95 기준이라 대회 지표와 값이 다릅니다. **오르내리는 추세만** 보고, 실험 간 비교는 `metrics.json`의 `map`으로 합니다.
- 그림 파일이 필요하면 `--set yolo_args.plots=true`로 학습해 `yolo_run/results.png`를 볼 수도 있습니다. 같이 생기는 `train_batch*.jpg`는 대회 이미지라 올리지 않습니다.

R-CNN은 같은 노트북의 위 셀 다음에 아래처럼 봅니다.

**▶ 노트북 셀 (.ipynb)** — 같은 노트북의 다음 셀 (R-CNN 곡선)

```python
import json

h = pd.DataFrame(
    json.load(
        open(OUTPUT_DIR / "1030_modelB_frcnn_v0_base" / "train_history.json", encoding="utf-8")
    )
)
h.plot(x="epoch", y=[c for c in ["train_loss", "val_map"] if c in h], subplots=True, figsize=(8, 5))
```

### ② 학습셋 점수와 검증 점수 비교 (같은 대회 지표로)

과적합이 의심되면, 같은 모델을 학습 이미지로도 채점해 봅니다. 학습 목록을 검증 목록 자리에 복사한 분할을 하나 만들고 `--val-config`로 채점하는 방법입니다(재학습 없음, 합성 데이터로 동작 확인).

**▶ PowerShell 터미널** — 학습셋 점수 확인용 분할 만들기 + 채점 (bash는 바로 아래 블록)

```powershell
# 처음 한 번: 학습 목록을 val.txt로 둔 점검용 분할 (커밋하지 않음)
New-Item -ItemType Directory -Force data\splits\v0_trainchk | Out-Null
Copy-Item data\splits\v0\train.txt data\splits\v0_trainchk\val.txt
@"
base: base.yaml
data:
  version: v0_trainchk
"@ | Set-Content -Encoding ascii configs\check_train.yaml   # utf8은 PowerShell 5에서 BOM이 붙어 설정을 못 읽음

# 실험마다: 학습셋(185장) 점수
uv run python scripts/evaluate.py --exp "${P}_v0_base" --val-config configs/check_train.yaml
```

**▶ Bash 터미널** — 같은 내용

```bash
mkdir -p data/splits/v0_trainchk
cp data/splits/v0/train.txt data/splits/v0_trainchk/val.txt
printf 'base: base.yaml\ndata:\n  version: v0_trainchk\n' > configs/check_train.yaml

uv run python scripts/evaluate.py --exp "${P}_v0_base" --val-config configs/check_train.yaml
```

결과는 `eval_…_valv0_trainchk.json`에 저장됩니다. `metrics.json`의 검증 `map`과 나란히 실험 기록표 비고에 적습니다.

| 학습셋 map | 검증 map | 읽기 |
| --- | --- | --- |
| 낮음 | 낮음 (비슷) | 과소적합 |
| 높음 | 조금 낮음 | 정상. 어느 정도 차이는 항상 있음 |
| 매우 높음 | 훨씬 낮음 | 과적합. 차이가 기준보다 커졌다면 그 실험이 과적합을 키운 것 |

절대값보다 **기준 대비 차이가 커졌는지**를 봅니다. 예를 들어 D1·D2를 합친 뒤 학습셋 map은 올랐는데 검증 map이 그대로면, 드문 알약 몇 장을 외운 것입니다.

## 평가 지표 보는 법

**판단은 `map` 하나로 하고, 나머지 지표는 “왜 오르거나 떨어졌는지”를 찾는 데 씁니다.** 모든 값은 학습 끝에 `outputs/<실험>/metrics.json`에 저장되고, `evaluate.py`로 다시 채점하면 `eval_*.json`에 같은 형식으로 남습니다.

### ① metrics.json의 값

| 키 | 뜻 | 이렇게 읽음 |
| --- | --- | --- |
| `map` | **대회 지표** mAP@\[0.75:0.95\]. IoU 0.75·0.80·0.85·0.90·0.95 다섯 기준 AP의 평균 | 실험 비교는 이 값만. +0.01 이상이면 효과 있음 |
| `map_75` | IoU 0.75 한 기준의 AP (가장 너그러운 기준) | `map_75`는 높은데 `map`이 훨씬 낮으면 → 알약은 맞히지만 **박스가 덜 정확** (D4, 회전 없는 D3, `yolo_args.box`) |
| `mar_1`, `mar_10`, `mar_100` | 재현율: 정답 알약 중 찾은 비율 (이미지당 예측 1·10·100개까지 볼 때) | `max_det`가 4라 `mar_10`과 `mar_100`은 같은 값. `mar`는 높은데 `map`이 낮으면 → 찾기는 하나 **이름을 틀림** (D11 OCR, 모델 가이드 M4) |
| `map_small`, `map_medium`, `map_large` | 박스 크기별 AP (COCO 기준 32²·96² 픽셀) | 해당 크기 박스가 없으면 빠짐. 작은 알약만 낮으면 D4 |
| `map_per_class`, `classes` | 클래스별 AP와 그 label\_idx (`--class-metrics`일 때만) | 드문 알약이 올랐는지 (D1·D2·D6) |
| `train_min` | 학습 시간(분) | 점수가 같으면 빠른 쪽. D4·큰 모델은 시간도 같이 기록 |

`map_50`은 없습니다. 채점 기준이 IoU 0.75부터라 계산 대상이 아니기 때문입니다.

### ② 실험 나란히 보기

**▶ 터미널 (PowerShell·Bash 공통)** — 실험 비교 리포트

```powershell
uv run python scripts/make_reports.py 03      # reports/03_experiment_compare.html (모든 실험 표 + 그래프)
```

`outputs/experiment_log.jsonl`에 쌓인 모든 학습 결과가 `map` 순서로 나옵니다. 내 `$P`로 시작하는 줄만 보면 단일·결합 실험이 한 표에 있습니다. HTML은 공개 레포에 올리지 않고 디스코드로 공유합니다.

### ③ 클래스별 AP 비교

드문 알약을 노린 실험(D1·D2·D6)은 전체 `map`보다 **그 알약들의 AP가 올랐는지**를 봐야 합니다. 두 실험을 다시 채점한 뒤(1\~2분), 노트북에서 category\_id로 맞춰 비교합니다. v3all처럼 label\_idx가 다른 실험도 같이 비교됩니다.

**▶ 터미널 (PowerShell·Bash 공통, `$P` 지정 후)** — 클래스별 AP로 다시 채점

```powershell
uv run python scripts/evaluate.py --exp "${P}_v0_base"      --class-metrics
uv run python scripts/evaluate.py --exp "${P}_v2_copypaste" --class-metrics
```

**▶ 노트북 셀 (.ipynb)** — 클래스별 AP 비교표 (`notebooks/scratch/<이름>_class_ap.ipynb` 등)

```python
import json

import pandas as pd

from src.interfaces import OUTPUT_DIR
from src.utils import load_class_map, load_config

EVAL = "eval_conf0.001_nms0.5_det4.json"
EXPS = {"base": "1030_modelA_yolo11s_v0_base", "D2": "1030_modelA_yolo11s_v2_copypaste"}


def class_ap(exp):
    """category_id → AP. 실험마다 자기 class_map으로 label_idx를 바꾼다."""
    ev = json.load(open(OUTPUT_DIR / exp / EVAL, encoding="utf-8"))
    cfg = load_config(OUTPUT_DIR / exp / "config.yaml")
    l2c = {c["label_idx"]: c["category_id"] for c in load_class_map(cfg["data"]["class_map"])}
    return pd.Series({l2c[lbl]: ap for lbl, ap in zip(ev["classes"], ev["map_per_class"])})


names = {c["category_id"]: c["name"] for c in load_class_map("data/class_map.json")}
df = pd.DataFrame({k: class_ap(v) for k, v in EXPS.items()})
df["변화"] = df["D2"] - df["base"]
df.insert(0, "약 이름", [names.get(i, i) for i in df.index])
df.sort_values("변화").round(3)  # 위 = 나빠진 클래스, 아래 = 좋아진 클래스
```

`classes` 키가 없다는 오류가 나면 develop의 `src/metrics.py`를 받은 뒤 위 명령을 다시 실행합니다. 어떤 알약을 어떤 알약으로 틀렸는지는 `notebooks/04_error_analysis.ipynb`에서 `EXP`만 바꿔 봅니다.

### ④ 점수 차이가 진짜인지 확인하기

검증셋이 47장이라 같은 설정도 시드만 바뀌면 점수가 흔들립니다. 차이가 0.01 근처면 한 번 더 확인합니다.

1. **시드 바꿔 다시**: 기준과 후보를 `--set seed=7`을 더해 한 번씩 더 학습합니다. 두 시드 모두에서 이겨야 효과 있음으로 봅니다. 기준의 두 시드 차이가 곧 우연으로 흔들리는 크기입니다.
2. **K-fold (D7, 시간이 있을 때)**: 데이터 담당이 `make_folds.py`로 만든 `v0_f1`\~`v0_f5`에서 `--set data.version=v0_f1`처럼 5번 학습하고 평균으로 비교합니다. v0 데이터를 쓰는 실험(D1·D3·D4·D9)만 가능합니다.
3. **Kaggle**: 최종 후보 2\~3개는 Experimentation Lead가 제출해 Public 점수로 확인합니다. Public도 테스트 일부라, 검증과 Public이 같은 방향일 때 가장 믿을 만합니다.

### ⑤ 재학습 없이 점수 올리기 (evaluate.py)

최종 모델이 정해지면 예측 설정만 바꿔 다시 채점해 봅니다. 파일 이름에 설정이 들어가 결과가 덮어쓰지 않습니다.

| 옵션 | 바꾸는 것 | 언제 |
| --- | --- | --- |
| `--nms-iou 0.6`, `--conf 0.01` | 겹친 박스 정리 기준, 낮은 점수 버리기 | 같은 알약에 박스가 격쳐 나올 때 (모델 가이드 P1) |
| `--tta` | 뒤집기·크기 바꾼 예측을 합침 (YOLO) | 최종 후보 (P4) |
| `--ensemble <다른 실험> --method wbf` | 여러 모델 합치기 | 시드·모델이 다른 상위 실험 (P2). 같은 class\_map끼리만 |

좋은 값을 찾으면 설정의 `eval:` 블록에 반영해 제출에 씁니다. 이 값들은 학습과 무관하므로 데이터 실험 비교는 모두 같은 기본값(`conf 0.001`, `nms 0.5`, `max_det 4`)으로 합니다.

## 기록·PR·커밋

**모델 담당자는 코드를 고치지 않으므로, 남길 것은 점수 기록과 최종 묶음입니다.** `--set`으로 돌린 실험은 `outputs/<실험>/config.yaml`에 실제 값이 남아서, 실험 이름과 묶음만 적어 두면 누구나 다시 돌릴 수 있습니다.

1. **실험 이슈 하나** (담당자별): 위 기록표를 복사해 점수·변화·판단을 채우고, 결합 실험과 빼기 실험 줄도 더합니다. 과적합 판단(최고 epoch, 학습셋 map)은 비고에 한 줄.
2. **최종 묶음**은 설정 파일로 굳혀 PR로 올립니다. `--set`으로 쓴 값을 그대로 옮기면 됩니다. Experimentation Lead가 이 파일로 시드 앙상블·전체 학습·제출을 합니다.

**▶ YAML 설정 파일 (새로 만듦, 예시)** — `configs/yolo11s_final.yaml`

```yaml
# configs/yolo11s_final.yaml — 내 기준 + 효과 있던 데이터 실험 (예)
base: yolo11s.yaml
exp_name: 1103_modelA_yolo11s_v3cp_rfs01_augsafe_1024
owner: modelA
note: 결합 D1+D2+D3+D4+D5+D6+D10 (검증 map 0.000, 기준 대비 +0.000)
data:
  train_images: data/synth/v3cp/images
  train_annotations: data/annotations/v3cp.json
  version: v3cp
  repeat_threshold: 0.1
  img_size: 1024
train:
  batch_size: 8
yolo_args:
  fliplr: 0.5
  flipud: 0.5
  degrees: 0.0
  hsv_h: 0.0
  hsv_s: 0.4
  hsv_v: 0.3
  scale: 0.3
  translate: 0.1
  mosaic: 1.0
  close_mosaic: 10
```

3. **PR 전 확인**: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`. 브랜치는 개인 이름 브랜치 → develop으로 PR, 병합은 “Create a merge commit”.

| 커밋함 | 커밋하지 않음 (공개 레포, 대회 데이터 재배포 금지) |
| --- | --- |
| 최종 설정 파일, 내 계획 파일 configs/matrix\_<이름>.yaml, data/annotations/external\_drop.csv, 결합 데이터 설정(`configs/data_v3c.yaml` 등), 결합 버전의 `data/splits/<버전>/`와 `*_copy_paste.csv`·`*_external.csv` (데이터 담당) | `outputs/` 전부(가중치, 곡선, eval json), `data/` 이미지·`annotations/*.json`, `reports/*.html`, `train_batch*.jpg`, 점검용 `data/splits/v0_trainchk/`와 `configs/check_train.yaml` |

점수가 오르지 않은 실험도 기록표에 남깁니다. “D1과 D2는 따로는 +0.02였지만 합치면 과적합됐다” 같은 결과가 보고서의 좋은 재료입니다.


## 빠른 시작: 처음부터 끝까지 (이 순서대로)

**develop 동기화 → 데이터 버전 만들기 → 준비 점검 → 1 epoch 동작 확인 → 본 실행 → 결과 확인** 순서입니다. 여기서는 실행 순서만 보여 주고, 각 단계의 자세한 내용은 아래 장들에 있습니다. **모두가 돌릴 필요는 없습니다.** GPU PC 한 대에서 한 사람이 돌리거나, 모델 담당자끼리 `--models`로 모델을 나눠 돌립니다(사람마다 계획 파일을 복사해 `prefix`를 다르게).

**▶ 터미널 (PowerShell·Bash 공통)** — 전체 순서

```bash
# 1) develop과 동기화 + 패키지 (데모·OCR은 쓸 때만)
git switch <내이름>
git fetch origin
git merge origin/develop
uv sync --extra demo                 # OCR(D11)까지 쓰려면 --extra ocr 추가 (develop에 D11이 들어온 뒤)

# 2) 데이터 버전 만들기 (처음 한 번, 대회 데이터가 data/raw/ 에 있어야 함)
uv run python scripts/export_yolo.py --config configs/base.yaml                          # v0 (기준)
uv run python scripts/make_copy_paste.py --config configs/base.yaml --version v2          # v2 합성 (D2)
uv run python scripts/clean_labels.py --config configs/base.yaml --version v1             # v1 정제 (D5·D10, manual_*.csv 반영)
uv run python scripts/make_external.py --config configs/base.yaml --version v3 --classes known   # v3 (D6, AI Hub를 받은 경우만)

# 3) 준비 점검: 필수 항목이 모두 OK인지 (OCR 2줄은 '없음'이어도 됨)
uv run python scripts/check_pipeline_ready.py
uv run --locked pytest -q

# 4) 동작 확인: 모든 단계를 1 epoch으로 (점수는 0에 가까워도 정상). 끝에 "최종 1등"과 "보고서"가 나오면 통과
uv run python scripts/run_pipeline.py --epochs 1 --tune-iterations 2 --tune-epochs 1

# 5) 본 실행 (하루 이상 걸릴 수 있음). 멈추면 같은 명령을 다시 → 끝난 실험은 건너뜀
uv run python scripts/run_pipeline.py --demo
```

| 단계 | 통과 기준 | 안 될 때 |
| --- | --- | --- |
| 1 동기화 | merge가 충돌 없이 끝남 | 충돌 파일을 Experimentation Lead와 해결 |
| 2 데이터 | 각 명령 끝에 “저장: …” | 필요한 것만 만들어도 됨. 없는 버전의 실험은 5)에서 “데이터 없음”으로 건너뜀. v1은 `manual_drop.csv`·`manual_fix.csv`를 먼저 채운 뒤(데이터 가이드 D5·D10), v3는 「준비 ①」의 AI Hub 내려받기와 점검 후 |
| 3 점검 | “모두 준비됨” 또는 OCR 2줄만 없음, `N passed` | “없음/부족” 줄의 파일을 정답본 zip과 비교해 가져옴 |
| 4 동작 확인 | `최종 1등: …`, `보고서: outputs/pipeline_…_e1/report.html` | 오류 메시지와 함께 Experimentation Lead에게 |
| 5 본 실행 | 같은 두 줄 + 데모가 http://127.0.0.1:7860 에서 열림 (끝내기 Ctrl+C) | `--stages plot report`로 그래프·보고서만 다시 만들 수 있음 |

**6) 결과 확인과 다음 일**

- `outputs/pipeline_<prefix>/report.html`을 브라우저로 열어 최종 1등과 그 근거(점수표·그래프)를 확인합니다. 대회 이미지가 없어 디스코드·PR에 공유해도 됩니다.
- `submissions/<최종 1등>.csv`는 Experimentation Lead가 Kaggle에 제출합니다. 검증 점수 차이가 작으면 상위 2\~3개를 제출해 Public 점수로 고릅니다.
- 사람 판단이 필요한 것(v3all 채택, D7 K-fold, D8 의사 라벨, 전체 데이터 최종 학습 P5)은 보고서를 보고 팀이 정합니다.
- 커밋하지 않습니다: `outputs/`, `submissions/`, `data/` 아래 생성물. 커밋 전에는 `ruff check`·`ruff format`·`pytest`·`check_repo_policy.py`를 통과시킵니다.

### 결과를 어디까지 믿을까: 한계와 제출 순서

**`run_pipeline.py`의 최종 1등은 “이번에 시도한 조합 중 검증 점수가 가장 높은 것”입니다.** 가능한 모든 조합 중 최고는 아니고, 가이드의 단일 실험을 손으로 하지 않았어도 이 스크립트가 그 단계들을 대신합니다. `check_pipeline_ready.py`는 **코드만** 확인하므로, 데이터 버전(위 2)을 만들지 않으면 D2·D5·D6 실험은 건너뛰고 v0 실험(D1·D3·D4·D9)만 비교됩니다.

| 한계 | 이유 | 보완 |
| --- | --- | --- |
| 판단 기준이 검증셋 47장 | 0.01 안팎 차이는 우연일 수 있음 | 보고서 점수표 상위 2\~3개를 모두 제출해 Public 점수로 결정 |
| 시도 범위가 정해져 있음 | 모델 4개, 계획 파일의 실험, 결합 규칙(데이터 버전은 하나만), 튜닝 20회 | 아래 “더 개선하려면” |
| v3all·D7·D8 제외 | 팀 판단이 필요해 자동으로 하지 않음 | 보고서를 보고 따로 실험 |
| 최종 모델이 학습셋(185장)으로만 학습됨 | 검증셋 47장은 비교용으로 남겨 둠 | 고른 설정으로 전체 데이터 재학습(아래 ①) 후 제출하면 보통 더 오름 |

**제출까지 권장 순서**

1. 위 1)\~4): 동기화, 데이터 버전, 점검, 1 epoch 동작 확인
2. `run_pipeline.py --demo` 본 실행
3. `report.html`에서 1등과 근소한 차이의 상위 후보 확인
4. 1등 CSV(`submissions/<실험>.csv`) 제출. 차이가 작으면 2\~3등도 `uv run python scripts/make_submission.py --exp <실험>`으로 만들어 제출하고 Public 점수로 고름
5. (시간이 되면) 고른 설정으로 전체 데이터 재학습 후 최종 제출

### 더 개선하려면: 추가로 시도해 볼 것

**보고서를 본 뒤 기대 효과가 크고 비용이 작은 순서(①부터)로 해 봅니다.** 대부분 모델 가이드(v2)·데이터 가이드(v3)에 코드가 있고, develop에 아직 없는 것은 정답본 zip과 같은 파일을 비교해 가져옵니다.

| # | 시도 | 기대 효과 | 방법 (관련 단계) | 비용 |
| --- | --- | --- | --- | --- |
| ① | 전체 데이터 재학습 | 학습 이미지 185 → 232장, 거의 항상 상승 | `make_folds.py`로 `v0_full` 분할 → `configs/yolo_full.yaml`의 base를 최종 설정으로 (모델 P5, 데이터 D7). 검증 점수는 의미 없으므로 Kaggle로만 확인 | 학습 1회 |
| ② | 예측 설정 조정 | 재학습 없이 +α | `evaluate.py --exp <1등> --nms-iou 0.6`, `--conf 0.01` 등을 바꿔 가며 비교 → 좋은 값을 설정 `eval:`에 (모델 P1). YOLO26은 NMS가 없는 구조라 효과가 작을 수 있음 | 몇 분 |
| ③ | TTA | 뒤집기·크기 변환 예측을 합쳐 박스 안정 | `evaluate.py --exp <1등> --tta`로 확인 후 `make_submission.py --exp <1등> --tta` (모델 P4) | 예측 시간 2\~3배 |
| ④ | 시드·모델 앙상블 | 우연 오차가 줄고 보통 +0.01\~ | 같은 설정을 `--set seed=7`로 한 번 더 학습(모델 T4) 또는 보고서 상위 모델들을 `make_submission.py --exp A B --method wbf`(모델 P2). 같은 class\_map끼리만 | 학습 1회 + 예측 |
| ⑤ | 데이터 버전 연결 | 정제·외부·합성 효과를 한 번에 | 자동 결합은 데이터 버전을 하나만 씁니다. 둘 이상이 효과 있었다면 「결합 실험 ①」의 v1 → v3c → v3cp를 만들고 계획 파일에 `V3CP` 실험 추가 | 데이터 생성 + 학습 |
| ⑥ | 보고서 하위 10개 클래스 집중 | 클래스별 평균이라 낮은 클래스가 오르면 크게 오름 | 그 알약을 `make_copy_paste.py --min-boxes 20`으로 더 합성하거나 AI Hub `TL_3`\~ 더 받기. 어떤 알약과 헷갈리는지 `04_error_analysis.ipynb`에서 확인 | 데이터 + 학습 |
| ⑦ | 더 큰 모델·해상도 | 작은 각인 구분 | `configs/yolo11m_1024.yaml`(모델 M5)을 계획 파일 `models`에 추가. GPU 메모리가 모자라면 batch 4 → 2 | 학습 4\~6배 |
| ⑧ | 튜닝 더 길게 | 아직 찾지 못한 설정 | `run_pipeline.py --stages tune report --tune-iterations 50` (기본 20회) | 반나절\~하루 |
| ⑨ | 학습 곡선 보고 에폭 조절 | 덜 배웠으면 큰 폭 상승 | 보고서 곡선에서 1등의 최고 epoch이 끝 근처면 `--set train.epochs=100`으로 재학습 (「과적합·과소적합 확인」) | 학습 2배 |
| ⑩ | 2단계 분류기 | 모양·색이 비슷한 알약 구분 | `configs/two_stage.yaml`의 `detector_exp`를 1등으로 (모델 M4). ⑥에서 헷갈리는 쌍이 많을 때 | 분류기 학습 |
| ⑪ | D11 OCR | 각인으로 이름 보정 | develop에 D11(데이터 가이드 v3)이 들어오면 파이프라인이 자동으로 포함 | 몇 분 |
| ⑫ | v3all·의사 라벨 | 테스트에만 있는 알약, 데이터 양 | 팀 결정 후 `D6A`·`D8` 실험. 검증으로는 효과가 안 보여 Kaggle로 판단 | 데이터 + 학습 |

어떤 것을 하든 **한 번에 하나씩** 바꾸고, 실험 기록표에 기준 대비 변화를 남깁니다. ①·③·④은 최종 모델이 정해진 뒤 마지막에 하고, ①을 한 모델은 검증 점수로 비교할 수 없으므로 Kaggle 점수로만 판단합니다.
