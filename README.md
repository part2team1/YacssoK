# 💊 경구약제 이미지 객체 탐지 (Pill Detection)

> 헬스케어 스타트업 **헬스잇(Health Eat)** AI 엔지니어링 팀 프로젝트
> 사진 속 최대 4개 알약의 **이름(클래스)과 위치(바운딩 박스)** 를 검출하는 모델 개발

| 항목 | 내용 |
| --- | --- |
| 기간 | 2026.10.22 (목) – 2026.11.06 (금) |
| 과제 | 객체 탐지 (Object Detection) |
| 평가 | Kaggle Private Competition — 평가지표: **mAP@[0.75:0.95]** |
| 데이터 | 학습 이미지 232장 (PNG, COCO 형식 JSON) · 테스트 이미지 842장 · 테스트에는 학습에 없는 클래스 포함 |
| 최종 점수 | 최종 모델·Public 점수 팀 확인 후 기입 · Private 점수는 대회 결과 발표 후 기입 |

## 📄 결과물

- [보고서 공동편집 원본](https://docs.google.com/document/d/1ZL_c3rD0sL2Lslt9RWjanWtDJ8rvv4DfaBbPmcRgebg/edit) — 최종 PDF는 확정 후 저장소에 추가합니다.
- [발표자료 공동편집 원본](https://docs.google.com/presentation/d/1YsQXBAhZhR6VCR0wd_vWxnZuKSFb_5jVuiN48IO4LPo/edit) — PDF와 구분합니다.
- 간트차트·협업일지·역할 세부 내용은 팀 공유 Notion에서 관리합니다.

## 👥 팀 구성과 역할

| 담당 | 팀원 |
| --- | --- |
| PM | 권순우 · 김이연 |
| 모델 | 박이결 · 배선영 |
| 데이터 | 이충은 · 방두성 |

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

## 🏆 최종 모델 재현 안내 — 확정 대기

최종 실험명·설정 파일·가중치 전달 위치는 팀 확정 후 기록합니다.
위의 빠른 시작은 베이스라인 실행 방법입니다.
모델 비교·결과 재사용 시에는 [검증 기준과 오류 방지 안내](docs/review_fixes.md)를 따릅니다.

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

실험별 설정과 결과는 `outputs/<실험 이름>/config.yaml`, `metrics.json`에 저장합니다.
모델을 비교할 때는 동일한 사진·정답·클래스 번호표·IoU로 평가한 점수를 사용합니다.
최종 제출 모델과 점수는 팀 확인 후 이 문서에 반영합니다.

## 📚 문서

| 문서 | 내용 |
| --- | --- |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 환경설정, 데이터 준비, 실행 방법, 분석 노트북, 협업 규칙(브랜치·PR·pre-commit) |
| [data/README.md](data/README.md) | 데이터 폴더 구조 |
| [docs/web_portal.md](docs/web_portal.md) | 앱 데모·테스트 통합 웹 실행·공유 |
| [docs/web_demo.md](docs/web_demo.md) | 기존 웹 데모 실행 |
| [docs/review_fixes.md](docs/review_fixes.md) | 분할 재생성·점수 비교·기존 결과 재사용·공유 인증 |
| [docs/guides/](docs/guides/) | 역할별 실행 가이드 |
| [docs/experiments/](docs/experiments/) | 실험 기록 |

> ⚠️ 공개 레포입니다. 데이터·가중치(`.pt`)·`outputs/`·제출 CSV·API 키·대회 이미지는 올리지 않습니다. 자세한 목록은 [DEVELOPMENT.md](DEVELOPMENT.md#커밋하지-않는-것-이-레포는-공개입니다)를 참고하세요.
