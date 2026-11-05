# YacssoK 사진으로 찾아보는 💊알약 도우미💊

> 헬스케어 스타트업 **헬스잇(Health Eat)** AI 엔지니어링 팀 프로젝트
> 사진 속 최대 4개 알약의 **이름(클래스)과 위치(바운딩 박스)** 를 검출하는 모델 개발

| 항목 | 내용 |
| --- | --- |
| 기간 | 2026.10.22 (목) – 2026.11.06 (금) |
| 과제 | 객체 탐지 (Object Detection) |
| 평가 | Kaggle Private Competition — 평가지표: **mAP@[0.75:0.95]** |
| 원본 데이터 | 학습 이미지 232장 · 라벨 JSON 763개 · 정답 박스 763개 · 알약 56종 · 테스트 이미지 842장 |
| 최종 선정 모델 | YOLO26s |
| 제출 결과 | 최종 제출 파일과 Public·Private 점수 확인 후 업데이트 |

## 서비스 소개

YacssoK은 알약 사진을 업로드하면 위치와 예상 이름을 보여주는 식별 보조 서비스입니다.
공식 알약 사진과 정보 링크를 함께 제공해 사용자가 예측 결과를 대조할 수 있도록 하고, 정보를 찾지 못한 항목은 미조회로 안내합니다.

사진 기반 예측 결과이므로 복용 여부나 용량은 약사·의사에게 확인해야 합니다. 모델의 신뢰도 점수는 복용 안전성을 의미하지 않습니다.

## 📄 결과물

- 보고서 PDF와 발표자료 PDF: 최종 파일 업로드 후 링크 추가
- 간트차트·팀별 R&R·협업일지: 공유 경로 확인 후 링크 추가
- 실행 방법과 협업 규칙: [DEVELOPMENT.md](DEVELOPMENT.md)

## 👥 팀 구성과 역할

5인 팀이 아래 업무를 분담하고 필요에 따라 공동으로 수행했습니다. 구성원별 최종 담당과 협업일지 링크는 팀 확인 후 추가합니다.

| 업무 | 범위 |
| --- | --- |
| 프로젝트 운영·서비스 | 일정·회의, Kaggle 제출, 보고서·발표, 웹 데모 |
| 공통 실행 환경 | 환경설정, 학습·평가·제출 파이프라인, 실험 기록 |
| 데이터 분석·개선 | EDA, 시각화, 라벨 검수, 분할, 증강, 외부 데이터 |
| YOLO 모델 실험 | 버전·크기·데이터 조건 비교, 튜닝 |
| 대안 모델·결과 검증 | Faster R-CNN 비교, K-fold, 앙상블 검토 |

## 🗓️ 주요 일정

| 날짜 | 마일스톤 |
| --- | --- |
| 10/22 (목) | OT, 프로젝트 소개, R&R 확정, 환경설정 |
| 10/23 (금) | EDA |
| 10/24 (토) | EDA 및 데모 가이드 실시 - 1 |
| 10/25 (일) | EDA 및 데모 가이드 실시 - 2 |
| 10/26 (월) | 워킹 스켈레톤 완성, 코드 워크스루 |
| 10/27 (화) | 데이터 분할 확정 (검증셋 고정) |
| 10/28 (수) | 첫 Kaggle 제출 |
| 10/30 (금) | 중간 점검 |
| 11/02 (월) | 프로젝트 소개·문제 정의·R&R·환경설정, uv·GitHub 환경설정 (전원 설치 지원) |
| 11/03 (화) | 최종 모델 선정 |
| 11/04 (수) | 보고서 정리 완료 |
| **11/05 (목) 19:00** | **GitHub 레포 · 보고서 PDF 제출 마감** |
| **11/06 (금)** | **13:00 발표 · 협업일지 23:50 제출 마감** |

---

## 🚀 빠른 시작

```bash
uv sync --locked
uv run --locked python scripts/check_env.py
uv run --locked python scripts/check_data.py --config configs/yolo_baseline.yaml
uv run --locked python scripts/export_yolo.py --config configs/yolo_baseline.yaml
uv run --locked python scripts/run_train.py --config configs/yolo_baseline.yaml
```

위 명령은 대회 데이터를 `data/raw/`에 준비한 뒤 실행하는 **YOLOv8s 기준 실험**입니다. uv 설치와 데이터 준비는 [DEVELOPMENT.md](DEVELOPMENT.md)를 참고하세요.

웹 데모는 가중치와 해당 클래스 맵을 준비하고 `configs/web_demo.yaml`의 경로·모델 표시명을 맞춘 뒤 실행합니다.

```bash
uv sync --locked --extra demo
uv run --locked python -m app.demo_web --config configs/web_demo.yaml
```

모델 연결과 임시 공유 방법은 [웹 데모 가이드](docs/web_demo.md)를 참고하세요.

## 🏆 최종 모델과 재현 조건

최종 선정 모델은 **YOLO26s**입니다. 데이터 개선 조합은 정제 라벨(v1), AI Hub 외부 데이터, Copy-Paste 합성, RFS(`repeat_threshold: 0.1`)입니다.

공유받은 5-fold 실험은 입력 크기 640, 50 epoch, 배치 16, seed 42 조건이며, 각 fold의 학습 데이터로 합성을 새로 생성하고 검증에는 대회 이미지만 사용했습니다.

**최종 제출 재현 절차는 클래스 매핑 대조 후 확정합니다.** 전달받은 단일 모델 가중치에는 클래스 이름 91개가 저장되어 있으나 동봉 맵은 76개였습니다. 실제 학습에 사용한 YOLO 데이터 설정과 클래스 순서·`category_id` 변환을 확인한 뒤, 최종 설정 파일명·가중치 확보 방법·제출 명령을 추가합니다.

공통 제출 진입점은 `scripts/make_submission.py --exp <실험 이름>`입니다. 실행 전 `outputs/<실험 이름>/`에 `config.yaml`과 `best.pt`를 준비하고, 설정이 학습 당시의 클래스 맵을 가리키는지 확인합니다. 생성한 CSV는 대회 제출 예시의 열 순서·이미지 ID·박스 형식과 대조합니다.

## 📁 폴더 구조

```
YacssoK/
├── configs/        # YAML 상속을 사용하는 실험 설정
├── data/           # 원본·합성 이미지는 로컬 보관, 맵·분할·정제 CSV는 공유
├── notebooks/      # 01_eda, 02_data_check, 03_experiment_compare, 04_error_analysis, 05_report_figures
│   └── scratch/    # 개인 실험 노트북 ({이름}_{주제}.ipynb)
├── src/            # 공통 코드 (interfaces, utils, dataset, transforms, eda, visualize, metrics, models/, train, infer, ensemble)
├── scripts/        # 실행 진입점 (check_env, check_data, prepare·split·export, run_train, evaluate, make_submission, clean_labels, make_reports, build_drug_info)
├── app/            # 데모 (Gradio)
├── outputs/        # 실험 결과 (git 제외)
├── tests/          # 형식 검사 테스트
├── reports/        # 보고서·발표자료, 이미지 포함 HTML·그림은 공개 범위 검토
└── docs/           # 회의록, 실험 기록
```

## 📊 실험 결과

| 실험 | 모델 | 데이터 조건 | mAP@[0.75:0.95] | 비고 |
| --- | --- | --- | --- | --- |
| 조합 단위 5-fold | YOLO26s | 정제 라벨 + AI Hub + Copy-Paste + RFS 0.1 | 0.9428 ± 0.0126 | 5-fold 평균 ± 표준편차. 공유받은 결과이며 클래스 매핑 대조가 남아 있음 |

위 수치는 내부 검증 결과입니다. 단일 모델 검증값 및 Kaggle Public·Private 점수와 구분해 기록하며, 최종 제출 결과는 확인 후 추가합니다.

## 📚 문서

| 문서 | 내용 |
| --- | --- |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 환경설정, 데이터 준비, 실행 방법, 분석 노트북, 협업 규칙(브랜치·PR·pre-commit) |
| [data/README.md](data/README.md) | 데이터 폴더 구조 |
| [docs/web_demo.md](docs/web_demo.md) | 웹 데모 실행·공유 |
| [docs/guides/](docs/guides/) | 역할별 실행 가이드 |
| [docs/experiments/](docs/experiments/) | 실험 기록 |

> ⚠️ 공개 레포입니다. 원본·합성 이미지, 실제 대회 이미지가 포함된 예측 결과·데모 캡처, 가중치(`.pt`), `outputs/`, 제출 CSV, API 키는 올리지 않습니다. 클래스 맵·분할 목록·정제 규칙 CSV는 팀 공용 파일로 공유합니다. 보고서·발표자료도 공개 전에 이미지 포함 여부를 확인합니다. 자세한 기준은 [DEVELOPMENT.md](DEVELOPMENT.md)를 참고하세요.
