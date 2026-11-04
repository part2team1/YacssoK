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

## 🚀 빠른 시작

```bash
uv sync                                  # 패키지 설치 (uv 설치는 DEVELOPMENT.md)
uv run python scripts/check_env.py       # 설치 확인
uv run python scripts/run_train.py --config configs/yolo_baseline.yaml   # 학습
uv run python scripts/make_submission.py --exp <실험 이름>                # 제출 CSV
```

데이터 준비, 전체 실행 순서, 협업 규칙은 **[DEVELOPMENT.md](DEVELOPMENT.md)** 에 있습니다.

## 🏆 최고 점수 재현 (11/5 제출 전 확정)

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

## 📚 문서

| 문서 | 내용 |
| --- | --- |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 환경설정, 데이터 준비, 실행 방법, 분석 노트북, 협업 규칙(브랜치·PR·pre-commit) |
| [data/README.md](data/README.md) | 데이터 폴더 구조 |
| [docs/web_demo.md](docs/web_demo.md) | 웹 데모 실행·공유 |
| [docs/guides/](docs/guides/) | 역할별 실행 가이드 |
| [docs/experiments/](docs/experiments/) | 실험 기록 |

> ⚠️ 공개 레포입니다. 데이터·가중치(`.pt`)·`outputs/`·제출 CSV·API 키·대회 이미지는 올리지 않습니다. 자세한 목록은 [DEVELOPMENT.md](DEVELOPMENT.md#커밋하지-않는-것-이-레포는-공개입니다)를 참고하세요.
