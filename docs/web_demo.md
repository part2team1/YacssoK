# YacssoK 웹 데모

사진 업로드 → 인식하기 → 알약 위치·이름·개수·점수 확인 → CSV 다운로드 순서로 사용합니다.
기존 `app/demo.py`는 유지하며, 새 화면은 `app/demo_web.py`로 실행합니다.

## 실행

저장소 루트에서 실행합니다. Python 3.11과 uv가 필요합니다.

```bash
uv sync --locked --extra demo
uv run python -m app.demo_web --config configs/web_demo.yaml
```

브라우저에서 http://127.0.0.1:7862 로 접속합니다. 종료는 Ctrl+C입니다.
포트가 사용 중이면 `--port 7865`를 추가하세요.

## 모델 연결

`configs/web_demo.yaml`의 `weights`를 전달받은 팀 YOLO `best.pt` 경로로 바꿉니다.
기본 경로는 `1028_dataB_yolov8s_v1_clean` 실험을 기준으로 한 예시이며, 가중치는 포함하지 않습니다.
`class_map`은 해당 모델과 같은 클래스 순서·이름의 파일을 사용해야 합니다.
모델 파일이 없거나 클래스가 다르면 준비 실패 안내가 표시됩니다.
다른 YOLO 모델로 교체할 때는 `model_label`도 실제 모델명에 맞춰 바꾸세요.
현재 CPU 추론이며 Hugging Face ZeroGPU 연결은 포함하지 않습니다.

## 예시 사진 (선택)

대회 원본·예측 사진은 공개 저장소에 올리지 않습니다.
허용된 로컬 사진을 `data/demo_examples/`에 아래 이름으로 준비하면 예시가 표시됩니다.

- `01_pills.png`, `verified_01_pills.png`: 첫 번째 원본·인식 결과 쌍
- `02_pills.png`, `verified_02_pills.png`: 두 번째 원본·인식 결과 쌍

예시 쌍이 없어도 사진 업로드·인식은 사용할 수 있습니다. 상단 전후 비교는 미리 만든 결과를 보여주며 새 추론을 실행하지 않습니다.
최소 신뢰도를 바꾸면 인식하기를 다시 눌러 결과를 확인합니다.
배너는 생성한 디자인 이미지로 실제 검증 결과가 아닙니다.

## 비밀번호 보호 임시 공유 (선택)

`YACSSOK_DEMO_USER`와 `YACSSOK_DEMO_PASSWORD`를 본인 환경에 설정한 뒤 실행합니다.
실제 비밀번호는 코드·README·PR에 적지 않습니다.

```bash
uv run python -m app.demo_web --config configs/web_demo.yaml --share
```

명령은 외부 접속 가능한 임시 링크를 생성합니다. 링크와 로그인 정보가 있는 사람이 접속할 수 있습니다.
PC와 서버가 켜져 있어야 하며 재시작하면 주소가 바뀔 수 있습니다.

현재 검토 링크: https://722816cf11fe922328.gradio.live/
이 링크는 임시 주소이며 실행자가 서버를 종료하면 사용할 수 없습니다. 로그인 정보는 별도 전달합니다.

## 확인 범위

로컬 Mac에서 팀 YOLOv8s 체크포인트로 실제 인식·CSV 생성·예시 전환을 확인했습니다.
이 PR은 웹 연결과 화면 변경이며 모델 성능 향상이나 검증 점수 변화를 주장하지 않습니다.
UI 확인 항목: 예시 없는 초기 실행, 모델 미확보 안내, 이미지 미선택 안내, 추론 오류 안내, CSV 생성, 예시 전환.
