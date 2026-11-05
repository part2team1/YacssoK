# 약속 웹 실행 안내

로그인 화면에서 **앱 데모** 또는 **테스트용**을 선택합니다. 로그인 후에도 상단에서 전환할 수 있으며 두 화면은 같은 모델을 사용합니다. 로그아웃은 해당 브라우저의 로그인만 해제합니다.

- 앱 데모: 사진 찍기·선택 → 사진 확인 → 알약 인식 → 쉬운 용도와 공식 사진 확인
- 테스트용: 사진 업로드, 최소 신뢰도 조절, 인식 위치·표·CSV 확인, 공식 정보 검색

## 설치와 실행

저장소 루트에서 Python 3.11 환경을 사용합니다.

```bash
uv sync --locked --extra demo
```

팀에서 전달받은 신뢰할 수 있는 체크포인트를 `configs/web_portal.yaml`의 `weights` 위치에 둡니다. `.pt`는 Git에 올리지 않습니다. 기본 설정은 기존 YOLOv8s **56종** 모델입니다. 91종 공식 정보 목록이 있다고 모델도 91종을 인식하는 것은 아닙니다.

계정은 실행 환경에서만 설정합니다. 아래 값은 예시이므로 실제 공유 계정으로 바꾸고 저장소에 기록하지 마세요.

macOS·Linux:

```bash
export PILL_USERNAME='your-account'
read -s PILL_PASSWORD
export PILL_PASSWORD
uv run --locked python -m app.demo_portal --config configs/web_portal.yaml
```

PowerShell:

```powershell
$env:PILL_USERNAME = 'your-account'
$secret = Read-Host '앱 비밀번호' -AsSecureString
$env:PILL_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
uv run --locked python -m app.demo_portal --config configs/web_portal.yaml
```

브라우저에서 `http://127.0.0.1:7871`을 엽니다. 계정·비밀번호 또는 모델 파일이 없거나 클래스 목록이 맞지 않으면 시작을 중단합니다. 이 명령은 **로컬 CPU**에서 실행하며 외부 공유 링크를 만들지 않습니다. 공유 서버에서는 HTTPS와 서버 환경변수로 계정을 설정해야 합니다.

## 최종 모델 교체

26s 최종 모델은 아직 이 변경에 포함되지 않았습니다. 파일을 받으면 YAML의 아래 항목을 함께 바꾸고 서버를 다시 시작합니다.

| 설정 | 넣을 값 |
| --- | --- |
| `weights` | 전달받은 체크포인트 경로 |
| `class_map` | 그 체크포인트와 같은 순서의 클래스 JSON 경로 |
| `expected_classes` | 실제 클래스 수 |
| `id`, `label` | 실제 모델 식별자와 표시 이름 |
| `imgsz` | 팀에서 정한 추론 입력 크기 |

클래스 JSON은 기존 `data/class_map.json`과 같은 `[{"label_idx": 0, "name": "약 이름"}, ...]` 형식입니다. 번호는 0부터 연속이어야 하며 체크포인트에 저장된 이름·순서와 일치해야 합니다. 파일 이름만 26s로 바꾸거나 이름이 다른 클래스 목록을 연결하면 안 됩니다.

경로는 저장소 루트를 기준으로 해석합니다. 개인 PC 절대 경로는 로컬 설정에서만 사용하세요. 신뢰할 수 없는 `.pt`는 로드하지 마세요.

## 공식 정보와 사진

식약처 공식 제품정보와 식별사진을 `app/assets/drug_catalog/`에 둡니다. 학습 사진·대회 이미지·인식 결과 캡처는 포함하지 않습니다.

- 공식 품목 목록 91종: 제품정보 조회 84종, 미조회 7종은 상태·이전 공식 기록과 링크 표시
- 공식 식별사진 83장: 없는 사진은 없는 상태로 안내
- 쉬운 대표 용도는 공식 원문과 구분하며, 삭제된 효능은 현재 용도로 안내하지 않음
- 각 카드에 품목코드와 공식 출처 링크 표시
- API 인증키 없이 저장된 자료를 표시함. 실시간 조회가 아니므로 품목별 조회일을 확인

파일별 SHA-256과 공식 출처는 [자료 목록](web_catalog_manifest.json)에 있습니다. 소프트웨어·자료 출처는 [외부 자료 고지](web_third_party_notices.md)를 참고하세요. 사진과 제품정보의 이용 조건은 소프트웨어 라이선스와 별개입니다.

## 확인 방법

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv run --locked python scripts/check_repo_policy.py
```

웹 테스트는 `--extra demo`가 설치된 환경에서 실행합니다. 모델 없는 CI에서는 설정·공식 자료·로그인 세션 검사를 하고, 실제 인식은 로컬의 신뢰할 수 있는 팀 모델로 별도 확인합니다. 모델 파일·실제 학습 사진은 테스트용으로도 커밋하지 않습니다.

기존 `app.demo`와 `app.demo_web` 실행 방식은 유지합니다. 이 변경은 `develop` 검토용이며, 실제 병합은 팀의 코드 동결·리뷰 기준에 따라 결정합니다.
