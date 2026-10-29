"""Git에 등록된 파일에서 노트북 출력과 업로드 금지 파일을 검사한다."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # 로컬 데이터와 가상환경은 건드리지 않고 Git 등록 파일만 검사한다.
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    errors = []

    for name in result.stdout.decode("utf-8").split("\0"):
        if not name:
            continue

        path = Path(name)

        # 키 값은 읽거나 출력하지 않고 파일 이름만 확인한다.
        if (
            path.name == ".env"
            or (
                path.name.startswith(".env.")
                and not path.name.endswith((".example", ".sample", ".template"))
            )
            or path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
            or name.startswith(("data/raw/", ".venv/"))
        ):
            errors.append(f"{name}: 업로드 금지 파일")

        if path.suffix.lower() != ".ipynb":
            continue

        try:
            notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: 노트북 읽기 실패 ({type(exc).__name__})")
            continue

        for index, cell in enumerate(notebook["cells"], start=1):
            if cell.get("outputs") or cell.get("attachments"):
                errors.append(f"{name}: 셀 {index}에 출력 또는 첨부가 있음")

    if errors:
        print("\n".join(errors))
        return 1

    print("저장소 정책 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())