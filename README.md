# 💊 경구약제 이미지 객체 탐지 (Pill Detection)

> 헬스케어 스타트업 **헬스잇(Health Eat)** AI 엔지니어링 팀 프로젝트
> 사진 속 최대 4개 알약의 **이름(클래스)과 위치(바운딩 박스)** 를 검출하는 모델 개발

| 항목 | 내용 |
| --- | --- |
| 기간 | 2026.10.22 (목) – 2026.11.06 (금) |
| 과제 | 객체 탐지 (Object Detection) |
| 평가 | Kaggle Private Competition — 평가지표: **mAP@[0.75:0.95]** |
| 데이터 | 학습 이미지 232장 (PNG, COCO 형식 JSON) · 테스트 이미지 842장 · 테스트에는 학습에 없는 클래스 포함 |
| 최종 점수 | Public `TODO` / Private `TODO` |

## 📄 결과물

- 보고서: [`reports/final_report.pdf`](reports/final_report.pdf) `TODO: 11/5 19:00 전 업로드`
- 발표자료: [`reports/presentation.pdf`](reports/presentation.pdf)
- 간트차트 · R&R · Source 작성 표준: `TODO: 링크`

## 👥 팀 구성과 역할

| 이름 | 역할 | 담당 | 협업일지 |
| --- | --- | --- | --- |
| `TODO` | PM(X) | 일정·회의, Kaggle 제출, 보고서·발표, 데모 | [링크](TODO) |
| `TODO` | Experimentation Lead | 환경·공통 코드, 학습·평가·제출 파이프라인, 실험 관리 | [링크](TODO) |
| `TODO` | Data Engineer A | EDA, 시각화, 오분류 분석 | [링크](TODO) |
| `TODO` | Data Engineer B | 데이터셋, 정제·분할, 증강 | [링크](TODO) |
| `TODO` | Model Architect A | YOLO 베이스라인·튜닝 | [링크](TODO) |
| `TODO` | Model Architect B | Faster R-CNN, 2단계 모델, 앙상블 | [링크](TODO) |

## 🗓️ 주요 일정

| 날짜 | 마일스톤 |
| --- | --- |
| 10/22 (목) | OT, 프로젝트 소개, R&R 확정, 환경설정 |
| 10/26 (월) | 워킹 스켈레톤 완성, 코드 워크스루 |
| 10/27 (화) | 데이터 분할 확정 (검증셋 고정) |
| 10/28 (수) | 첫 Kaggle 제출 |
| 10/30 (금) | 중간 점검 |
| 11/04 (수) | 코드 동결, 최종 모델 선정 |
| **11/05 (목) 19:00** | **GitHub 레포 · 보고서 PDF 제출 마감** |
| 11/06 (금) | 발표, 협업일지 제출 (23:50) |

---

## ⚙️ 환경설정

### 1. uv 설치 (최초 1회)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 다시 열고 `uv --version`으로 확인합니다.

### 2. 레포 클론과 패키지 설치

```bash
git clone https://github.com/TODO_ORG/pill-detection.git
cd pill-detection
uv sync                      # Python 3.11과 모든 패키지를 uv.lock 버전 그대로 설치
uv run pre-commit install    # 커밋 시 자동 검사 연결 (최초 1회, 아래 "커밋할 때 자동 검사")
uv run python scripts/check_env.py   # 설치 확인
```

- `uv sync`가 `.venv/` 가상환경을 자동으로 만듭니다. 직접 `activate`할 필요 없이 `uv run`으로 실행합니다.
- 데모 담당은 `uv sync --extra demo`로 Gradio를 추가 설치합니다.
- VS Code에서는 인터프리터로 `.venv`를 선택하고, Jupyter 커널도 `.venv`를 선택합니다.

### 3. 데이터 준비

Kaggle 대회 [Data](https://www.kaggle.com/competitions/ai15-level1-project/data) 탭에서 **Download All**로 받은 뒤, 압축 안의 `sprint_ai_project1_data/` 폴더 속 세 폴더를 `data/raw/`로 옮깁니다.

```
data/raw/
├── train_images/        # 232장
├── train_annotations/   # 조합 폴더 114개 → 알약 폴더 → JSON
└── test_images/         # 842장 (1.png, 10.png, ...)
```

자세한 구조는 [`data/README.md`](data/README.md)를 참고하세요.

> ⚠️ AI Hub 경구약제조합 5000종의 `TL_2_조합.zip`, `TS_2_조합.zip`은 대회 데이터 원본이므로 **사용 금지**입니다.

### 패키지를 추가하고 싶을 때

각자 `uv add`를 하면 `uv.lock` 충돌이 납니다. **Experimentation Lead에게 요청**하거나, 패키지 추가만 담은 PR을 올립니다.

```bash
uv add 패키지명            # pyproject.toml과 uv.lock이 함께 바뀜 → 둘 다 커밋
uv add --dev 패키지명      # 개발 도구
```

다른 사람이 패키지를 추가한 뒤에는 `git pull` 후 `uv sync`만 다시 실행하면 됩니다.

---

## 🚀 실행 방법

### 0. 데이터 없이 스켈레톤 동작 확인 (CPU로 약 5분)

합성 알약 데이터로 전체 흐름이 도는지 먼저 확인합니다.

```bash
uv run python scripts/make_sample_data.py
uv run python scripts/prepare_data.py --config configs/sample_yolo.yaml
uv run python scripts/split_data.py   --config configs/sample_yolo.yaml
uv run python scripts/export_yolo.py  --config configs/sample_yolo.yaml
uv run python scripts/run_train.py    --config configs/sample_yolo.yaml
uv run python scripts/make_submission.py --exp sample_yolo
uv run pytest                         # 팀 표준 형식 검사
```

### 1. 실제 데이터로 학습부터 제출까지

```bash
uv run python scripts/check_data.py                                     # data/raw 배치·공용 파일 확인 (처음 한 번)
uv run python scripts/prepare_data.py --config configs/base.yaml        # class_map.json, records.json
uv run python scripts/split_data.py   --config configs/base.yaml        # data/splits/v0/train.txt, val.txt
uv run python scripts/export_yolo.py  --config configs/yolo_baseline.yaml   # YOLO용 데이터 (YOLO만)

uv run python scripts/run_train.py --config configs/yolo_baseline.yaml     # 모델 A
uv run python scripts/run_train.py --config configs/rcnn_baseline.yaml     # 모델 B
uv run python scripts/run_train.py --config configs/yolo_baseline.yaml --epochs 3   # 빠른 확인

uv run python scripts/make_submission.py --exp 1028_modelA_yolov8s_v0_baseline        # 제출 CSV
uv run python scripts/make_submission.py --exp <실험A> <실험B>                         # 앙상블
```

학습 결과는 `outputs/{실험 이름}/`에 `config.yaml`, `best.pt`, `metrics.json`, `preds_val.json`으로 저장되고, `outputs/experiment_log.jsonl`에 한 줄씩 쌓입니다.

역할별 도구 (자세한 사용법은 역할별 실행 가이드):

| 스크립트 | 하는 일 | 주 사용자 |
| --- | --- | --- |
| `scripts/clean_labels.py --version v1` | 라벨 정제본 `data/annotations/v1.json` + 보고서 + `splits/v1` | 데이터 |
| `scripts/make_reports.py 01 04` | 노트북을 실행해 그래프가 든 `reports/*.html` 생성 | 데이터·모델 |
| `scripts/evaluate.py --exp <실험> --nms-iou 0.6` | 재학습 없이 예측 설정만 바꿔 검증 점수 재계산 | 모델 |
| `scripts/build_drug_info.py` | 데모 결과 표용 약 정보 `data/drug_info.json` (커밋 안 함) | 데모 |

### 2. 분석 노트북

| 노트북 | 내용 | 담당 |
| --- | --- | --- |
| `01_eda` | 클래스·크기·박스 분포, 라벨 오류 의심 항목 | 데이터 A |
| `02_data_check` | Dataset 출력 형식, 증강 결과, 분할 확인 | 데이터 B |
| `03_experiment_compare` | 전체 실험 점수 비교표와 그래프 | Experimentation Lead |
| `04_error_analysis` | 틀린 예측, 헷갈리는 클래스 쌍 | 데이터 A |
| `05_report_figures` | 보고서용 그림 저장 | PM(X) |

### 3. 데모

```bash
uv sync --extra demo
uv run python app/demo.py --exp 1028_modelA_yolov8s_v0_baseline
# 브라우저에서 http://127.0.0.1:7860 , 발표 때 외부 링크가 필요하면 --share
```

### 최고 점수 재현 (11/5 제출 전 확정)

```bash
# TODO: 최종 모델의 config와 명령을 여기에 고정
```

## 📁 폴더 구조

```
pill-detection/
├── configs/        # 실험 설정 (base.yaml 복사해서 사용)
├── data/           # 데이터 (git 제외)
├── notebooks/      # 01_eda, 02_data_check, 03_experiment_compare, 04_error_analysis, 05_report_figures
│   └── scratch/    # 개인 실험 노트북 ({이름}_{주제}.ipynb)
├── src/            # 공통 코드 (interfaces, utils, dataset, transforms, eda, visualize, metrics, models/, train, infer, ensemble)
├── scripts/        # 실행 진입점 (check_env, check_data, prepare·split·export, run_train, evaluate, make_submission, clean_labels, make_reports, build_drug_info)
├── app/            # 데모 (Gradio)
├── outputs/        # 실험 결과 (git 제외)
├── tests/          # 형식 검사 테스트
├── reports/        # make_reports.py 결과·그림 (공개 레포라 커밋하지 않음)
└── docs/           # 회의록, 실험 기록
```

## 📊 실험 결과

| 실험 | 모델 | 데이터 | 주요 변경 | mAP (검증) | Kaggle Public |
| --- | --- | --- | --- | --- | --- |
| 스켈레톤 | YOLOv8s | v0 | - | `TODO` | `TODO` |

---

## 🤝 협업 규칙

### 브랜치 전략: 사람별 브랜치 + PR

```
[GitHub]  main ◀── PR 병합 ──  kim · lee · park ...   (각자 이름 브랜치)
            │ pull                  ▲ push
[내 PC]   main ── merge ──▶  kim                       (실제 작업하는 곳)
```

- `main`: 보호 브랜치. 직접 push 금지, PR과 리뷰 승인으로만 병합합니다.
- 각자 clone 후 **영문 소문자 이름으로 브랜치 하나**를 만들어 2주 동안 씁니다 (`kim`, `lee`, ...).
- PR은 `base: main ← compare: 내 이름`으로, **하루 이틀 안에 병합할 크기**로 자주 올립니다.
- 병합 방식은 **Create a merge commit**, 병합 후 **Delete branch는 누르지 않습니다**.
- PR이 열려 있는 동안 다른 작업을 해야 하면 `kim-rotate`처럼 작업용 브랜치를 하나 더 만듭니다.

### 처음 한 번 (내 PC)

```bash
git config --global user.name "홍길동"
git config --global user.email "github가입@이메일.com"
cd pill-detection
git switch main && git pull
git switch -c kim                        # 내 이름 브랜치 생성
git push -u origin kim                   # GitHub에도 생성
```

### 매일 작업 시작 전 (내 PC)

```bash
git switch main && git pull              # GitHub main 최신 내용 받기
git switch kim && git merge main         # 내 브랜치에 합치기
uv sync                                  # 패키지 변경 반영
```

### 소스 수정 후 (내 PC → GitHub)

```bash
uv run ruff check . && uv run pytest     # 검사
git status                               # 바뀐 파일 확인
git add src/dataset.py                   # 바뀐 파일만
git commit -m "feat: 라벨 정제 추가"
git push                                 # GitHub의 kim 브랜치로
# → GitHub에서 New pull request (base: main ← compare: kim), 승인 후 Create a merge commit
```

### 커밋할 때 자동 검사 (pre-commit)

`uv run pre-commit install`을 한 번 해 두면 `git commit` 때마다 아래 검사가 자동으로 돕니다. 설정은 `.pre-commit-config.yaml`에 있습니다.

| 검사 | 하는 일 | 걸렸을 때 |
| --- | --- | --- |
| no-commit-to-branch | `main`에서 직접 커밋 차단 | `git switch <내 이름>` 후 다시 커밋 |
| check-added-large-files | 5MB 넘는 파일(가중치 등) 차단 | 파일을 커밋에서 빼기 (`git restore --staged 파일`) |
| check-merge-conflict | 충돌 표시(`<<<<<<<`)가 남은 파일 차단 | 충돌 부분 정리 |
| check-yaml / toml / json | 설정 파일 문법 오류 | 메시지의 줄 번호를 보고 수정 |
| detect-private-key | 개인 키 커밋 차단 | 파일 삭제 |
| trailing-whitespace, end-of-file-fixer | 줄 끝 공백, 파일 끝 줄바꿈 **자동 수정** | 다시 `git add` → `git commit` |
| nbstripout | 노트북 출력 **자동 제거** | 다시 `git add` → `git commit` |
| ruff-check | 린트, 고칠 수 있는 것은 **자동 수정** (import 정리 등) | 자동 수정분은 다시 add, 나머지(예: F821 정의 안 된 이름)는 직접 수정 |
| ruff-format | 코드 포맷 **자동 수정** | 다시 `git add` → `git commit` |

```bash
git commit -m "feat: ..."     # Failed가 보이면 커밋이 취소된 것
git add -A                    # 자동으로 고쳐진 파일 다시 담기 (또는 고친 파일만)
git commit -m "feat: ..."     # 모두 Passed면 커밋 완료
uv run pre-commit run --all-files   # 커밋 없이 전체 검사만
```

### 커밋 메시지

`feat:` 기능 · `fix:` 버그 · `exp:` 실험 · `docs:` 문서 · `refactor:` 정리 · `chore:` 설정

### 리뷰 규칙

- 파일 담당자(`.github/CODEOWNERS`)가 자동으로 리뷰어로 지정되며, 승인 1명 이상 필요합니다.
- 리뷰는 24시간 안에 합니다. 담당자 부재 시 백업 담당자가 승인합니다.
- **11/4 코드 동결 이후**에는 버그 수정만 병합합니다.

### 커밋하지 않는 것 (이 레포는 공개입니다)

데이터, 모델 가중치(`.pt`), `outputs/`, 제출 CSV, `kaggle.json`·API 키, 개인 PC 절대 경로,
`reports/*.html`·`reports/figures/` (대회 이미지가 들어감), 약 정보 `data/drug_info.json`, 정제 어노테이션 `data/annotations/*.json`

대회 데이터는 재배포 금지입니다. 공개 레포의 PR 본문·이슈·커밋에 **대회 이미지(원본, 예측 박스 그림, 실제 데이터로 띄운 데모 캡처)를 올리지 않습니다.**
PR에는 숫자 표와 그래프(막대·히스토그램)만 넣고, 이미지가 들어간 결과는 팀 디스코드로 공유합니다.
