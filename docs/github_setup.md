# GitHub 레포 초기 설정 (PM·실험리드 Y, 10/22)

팀원이 클론하기 전에 레포 관리자가 한 번만 진행합니다.

## 1. 레포 만들고 초기 파일 올리기

GitHub에서 `pill-detection` 레포를 만듭니다. README, .gitignore 추가 옵션은 **체크하지 않습니다** (이 폴더에 이미 있음).

```bash
cd pill-detection
# pyproject.toml의 CUDA 버전을 팀 GPU에 맞게 먼저 확인 (아래 3번)
uv lock                          # uv.lock 생성
uv sync
uv run python scripts/check_env.py

git init -b main
git add .
git commit -m "chore: 초기 레포 구조와 uv 환경 설정"
git remote add origin https://github.com/<계정 또는 조직>/pill-detection.git
git push -u origin main

uv run pre-commit install        # 초기 push 후에 연결 (이후 main 직접 커밋은 차단됨)
uv run pre-commit run --all-files   # 전체 파일이 Passed인지 확인 (no-commit-to-branch만 Failed면 정상)
```

> 초기 커밋은 pre-commit을 연결하기 **전에** 합니다. `no-commit-to-branch` 훅이 main 커밋을 막기 때문입니다.
> 훅 버전 갱신은 `uv run pre-commit autoupdate` 후 ruff 버전을 `pyproject.toml`의 `ruff==` 와 같게 맞춰 PR로 올립니다.

## 2. 팀원 초대와 보호 규칙

1. **Settings → Collaborators**: 팀원 5명 초대 (Write 권한)
2. **`.github/CODEOWNERS`**: `@X_ID` 등을 실제 GitHub 아이디로 바꿔 커밋
3. **Settings → Branches → Add branch protection rule** (또는 Rules → Rulesets), 대상 `main`
   - Require a pull request before merging
   - Require approvals: 1
   - Require review from Code Owners
   - Do not allow bypassing the above settings (관리자도 PR로만)
   - Block force pushes, 삭제 금지
4. **Settings → General → Pull Requests**: "Allow merge commits"를 켜고, "Automatically delete head branches"는 **끕니다** (각자 이름 브랜치를 계속 쓰기 때문)
5. **Projects → New project → Board**: To do / Doing / Review / Done 칸반을 만들고 레포와 연결

> 개인 계정의 **비공개(private)** 레포는 무료 플랜에서 브랜치 보호 규칙이 적용되지 않을 수 있습니다. 공개 레포로 하거나, 적용되지 않으면 "main 직접 push 금지"를 팀 규칙으로 지킵니다. 공개로 할 경우 데이터와 `kaggle.json`이 올라가지 않았는지 다시 확인합니다.

## 3. PyTorch CUDA 버전 정하기

각 팀원 PC에서 `nvidia-smi`를 실행해 오른쪽 위 **CUDA Version**을 확인합니다. 이 숫자는 드라이버가 지원하는 **최대** 버전입니다.

| 상황 | `pyproject.toml`의 인덱스 |
| --- | --- |
| 드라이버가 CUDA 12.4 이상 지원 (대부분) | `cu124` (기본값 유지) |
| RTX 50 시리즈 등 최신 GPU | `cu128`로 이름과 URL 변경 |
| 팀원 전원 GPU 없음, Colab 사용 | `[tool.uv.sources]`, `[[tool.uv.index]]` 블록 삭제 |
| macOS | 설정 그대로 (자동으로 PyPI 빌드 설치) |

바꾼 뒤에는 `uv lock`을 다시 실행하고 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다.

## 4. 첫날 연습: 내 이름 브랜치 만들고 1인 1 PR

각자 내 이름 브랜치를 만들고, README의 팀 구성 표에 자기 이름을 넣는 PR을 올립니다.

```bash
git config --global user.name "<이름>"
git config --global user.email "<GitHub 가입 이메일>"
git switch main && git pull
git switch -c <이름>                       # 예: kim (영문 소문자)
git push -u origin <이름>
# README.md 팀 구성 표에서 자기 줄의 TODO를 이름으로 수정
git commit -am "docs: 팀 구성에 <이름> 추가"
git push
```

GitHub에서 `base: main ← compare: <이름>`으로 PR을 만들고, 승인 후 **Create a merge commit**으로 병합합니다 (Delete branch는 누르지 않음).

같은 표를 6명이 고치므로 **충돌 해결까지 연습**할 수 있습니다. 먼저 병합된 사람이 생기면 나머지는 아래로 main을 내 브랜치에 합치고, 충돌을 고친 뒤 다시 push 합니다.

```bash
git switch main && git pull
git switch <이름> && git merge main       # 충돌 파일 정리 → git add → git commit
git push
```
