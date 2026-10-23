"""노트북을 실행해 그래프가 포함된 HTML 리포트로 저장한다.

실행:
    uv run python scripts/make_reports.py                      # 01 EDA 리포트
    uv run python scripts/make_reports.py 01 02 03 04          # 여러 개
    uv run python scripts/make_reports.py 03 --exp sample_yolo # 04는 --exp로 분석할 실험 지정

결과: reports/{노트북 이름}.html  (브라우저로 열기, 디스코드로 공유)
레포가 공개이므로 커밋하지 않는다(.gitignore). 01·04 리포트에는 대회 이미지가 들어 있다.

노트북 파일은 바꾸지 않는다. 복사본에서 CONFIG / EXP 값만 바꿔 실행한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.interfaces import PROJECT_ROOT

NOTEBOOKS = {
    "01": "01_eda.ipynb",
    "02": "02_data_check.ipynb",
    "03": "03_experiment_compare.ipynb",
    "04": "04_error_analysis.ipynb",
    "05": "05_report_figures.ipynb",
}


def patch(nb: dict, config: str | None, exp: str | None) -> dict:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if config:
            src = src.replace('CONFIG = "configs/base.yaml"', f'CONFIG = "{config}"')
        if exp:
            src = src.replace('EXP = "sample_yolo"', f'EXP = "{exp}"')
        cell["source"] = src
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", default=["01"], help="노트북 번호 (01~05)")
    ap.add_argument(
        "--config", default=None, help="예: configs/sample_yolo.yaml (합성 데이터로 연습)"
    )
    ap.add_argument("--exp", default=None, help="04 오분류 분석에 쓸 실험 이름")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    for nb_id in args.ids:
        name = NOTEBOOKS[nb_id.zfill(2)]
        nb = json.loads((PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        nb = patch(nb, args.config, args.exp)
        # 노트북은 notebooks/ 기준 상대 경로를 쓰므로 같은 폴더에 임시 복사본을 둔다
        with tempfile.NamedTemporaryFile(
            "w", suffix=".ipynb", dir=PROJECT_ROOT / "notebooks", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(nb, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        try:
            print(f"[{name}] 실행 중...")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "html",
                    "--execute",
                    "--ExecutePreprocessor.timeout=1800",
                    str(tmp_path),
                    "--output-dir",
                    str(out_dir),
                    "--output",
                    Path(name).stem,
                ],
                capture_output=True,
                text=True,
                # 노트북 커널이 어디서 실행되든 src/ 를 import 할 수 있게
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
            if result.returncode != 0:
                print(result.stderr[-2000:])
                print(f"[{name}] 실패 — 위 오류를 확인하세요.")
            else:
                print(f"[{name}] 저장: reports/{Path(name).stem}.html")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
