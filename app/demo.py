"""헬스잇 알약 인식 데모 (PM·Experimentation Lead 담당, 선택 과제).

실행:
    uv sync --extra demo
    uv run python app/demo.py --exp 1028_modelA_yolov8s_v0_baseline
    → 브라우저에서 http://127.0.0.1:7860 접속 (끝내기: Ctrl+C)

파일 구성 (수정할 곳)
  1. 화면 설정   : TITLE, DESCRIPTION, NOTICE, THEME, CSS        → 문구·색·글꼴
  2. 기능(로직)  : make_recognizer()                              → 사진 → 박스 그림 + 결과 표
  3. 화면 배치   : build_app()                                    → 입력·버튼·결과 위치
모델·데이터 코드(src/)는 고치지 않는다. 모델은 --exp 로 고른 실험 폴더(outputs/)에서 읽는다.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from src.infer import load_experiment
from src.utils import load_class_map, load_json, resolve_path
from src.visualize import draw_boxes

# ── 1. 화면 설정 ─────────────────────────────────────────────
TITLE = "헬스잇 알약 인식"
DESCRIPTION = "알약 사진을 올리면 최대 4개 알약의 위치와 이름을 찾아 드립니다."
NOTICE = "확신도가 낮은 결과는 약사에게 확인하세요. 이 데모는 학습 프로젝트용이며 의료 판단에 쓰지 않습니다."
LOW_CONF = 0.5  # 이 값보다 낮으면 "확인 필요"
THEME = gr.themes.Soft(primary_hue="blue")  # gr.themes.Default / Soft / Glass / Monochrome
CSS = """
#result-table { font-size: 15px; }
footer { display: none !important; }
"""
RESULT_COLUMNS = [
    "번호",
    "약 이름",
    "확신도",
    "판정",
]  # 약 정보 열은 drug_info.json이 있으면 자동 추가


# ── 2. 기능 (사진 → 결과) ────────────────────────────────────
def load_drug_info(cfg: dict) -> dict[str, dict]:
    """scripts/build_drug_info.py 가 만든 약 정보표. 없으면 빈 dict (이름만 표시)."""
    path = resolve_path(cfg["data"]["class_map"]).parent / "drug_info.json"
    return load_json(path) if path.exists() else {}


def make_recognizer(exp_name: str):
    """사진 → (박스 그린 이미지, 결과 표) 함수를 만든다. 화면과 분리해 두어 테스트하기 쉽다."""
    cfg, model = load_experiment(exp_name)
    class_map = load_class_map(cfg["data"]["class_map"])
    drug_info = load_drug_info(cfg)

    def recognize(image, conf_thr):
        if image is None:
            return None, pd.DataFrame(columns=RESULT_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            pred = model.predict([path], conf_thr=conf_thr)[0]
        drawn = draw_boxes(image, pred["boxes"], pred["labels"], scores=pred["scores"])
        rows = []
        for label, score in zip(pred["labels"].tolist(), pred["scores"].tolist(), strict=True):
            cls = class_map[label]
            rows.append(
                {
                    "번호": f"#{label}",
                    "약 이름": cls["name"],
                    "확신도": round(score, 3),
                    "판정": "확인 필요" if score < LOW_CONF else "인식",
                    **drug_info.get(str(cls["category_id"]), {}),
                }
            )
        df = pd.DataFrame(rows)
        return drawn, df if len(df) else pd.DataFrame(columns=RESULT_COLUMNS)

    return cfg, recognize


# ── 3. 화면 배치 ─────────────────────────────────────────────
def build_app(exp_name: str) -> gr.Blocks:
    cfg, recognize = make_recognizer(exp_name)

    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(f"## {TITLE}\n{DESCRIPTION}\n\n모델: `{exp_name}`")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(
                    type="numpy", label="알약 사진", sources=["upload", "webcam", "clipboard"]
                )
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="최소 확신도")
                btn = gr.Button("인식하기", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="인식 결과")
                out_tbl = gr.Dataframe(label="인식된 약", elem_id="result-table", wrap=True)
        gr.Markdown(NOTICE)
        btn.click(recognize, [inp, conf], [out_img, out_tbl])

        test_dir = resolve_path(cfg["data"]["test_images"])
        examples = sorted(test_dir.glob("*.png"))[:4] if test_dir.exists() else []
        if examples:
            gr.Examples([[str(p)] for p in examples], inputs=[inp], label="예시 사진")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="outputs/ 아래 실험 이름")
    ap.add_argument("--share", action="store_true", help="외부 공유 링크 생성 (발표용, 72시간)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_app(args.exp).launch(server_port=args.port, share=args.share, theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
