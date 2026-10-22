"""헬스잇 알약 인식 데모 (PM·실험리드 X 담당, 선택 과제).

실행:
    uv sync --extra demo
    uv run python app/demo.py --exp 1028_modelA_yolov8s_v0_baseline
    → 브라우저에서 http://127.0.0.1:7860 접속

사진을 올리면 알약 위치를 표시하고, 약 이름·확신도를 표로 보여준다.
확신도가 낮은 결과는 "확인 필요"로 표시한다 (오인식이 건강 피해로 이어질 수 있기 때문).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from src.infer import load_experiment
from src.utils import load_class_map, resolve_path
from src.visualize import draw_boxes

LOW_CONF = 0.5  # 이 값보다 낮으면 "확인 필요"


def make_recognizer(exp_name: str):
    """사진 → (박스 그린 이미지, 결과 표) 함수를 만든다. 화면과 분리해 두어 테스트하기 쉽다."""
    cfg, model = load_experiment(exp_name)
    class_map = load_class_map(cfg["data"]["class_map"])

    def recognize(image, conf_thr):
        if image is None:
            return None, pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            pred = model.predict([path], conf_thr=conf_thr)[0]
        drawn = draw_boxes(image, pred["boxes"], pred["labels"], scores=pred["scores"])
        rows = []
        for label, score in zip(pred["labels"].tolist(), pred["scores"].tolist(), strict=True):
            info = class_map[label]
            rows.append(
                {
                    "번호": f"#{label}",
                    "약 이름": info["name"],
                    "코드": info["category_id"],
                    "확신도": round(score, 3),
                    "판정": "확인 필요" if score < LOW_CONF else "인식",
                }
            )
        return drawn, pd.DataFrame(rows, columns=["번호", "약 이름", "코드", "확신도", "판정"])

    return cfg, recognize


def build_app(exp_name: str) -> gr.Blocks:
    cfg, recognize = make_recognizer(exp_name)

    with gr.Blocks(title="헬스잇 알약 인식 데모") as app:
        gr.Markdown(
            f"## 헬스잇 알약 인식 데모\n모델: `{exp_name}` · 사진 한 장에 최대 4개 알약을 찾습니다."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="알약 사진")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="최소 확신도")
                btn = gr.Button("인식하기", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="인식 결과")
                out_tbl = gr.Dataframe(label="인식된 약")
        gr.Markdown(
            "확신도가 낮은 결과는 약사에게 확인하세요. 이 데모는 학습 프로젝트용이며 의료 판단에 쓰지 않습니다."
        )
        btn.click(recognize, [inp, conf], [out_img, out_tbl])
        test_dir = resolve_path(cfg["data"]["test_images"])
        examples = sorted(test_dir.glob("*.png"))[:4] if test_dir.exists() else []
        if examples:
            gr.Examples([[str(p)] for p in examples], inputs=[inp])
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="outputs/ 아래 실험 이름")
    ap.add_argument("--share", action="store_true", help="외부 공유 링크 생성 (발표용)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_app(args.exp).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
