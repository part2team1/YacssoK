"""Local Gradio pill demo. Model and class map can be replaced together."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HERO_IMAGE = ROOT / "app/assets/hero-pill-recognition-v1.png"
gr.set_static_paths(paths=[str(HERO_IMAGE)])
COLUMNS = ["번호", "약 이름 / 클래스", "신뢰도 점수", "확인 상태"]
CSS = """
body,.gradio-container {background:#fff!important}
.gradio-container {width:100%!important;max-width:1140px!important;margin:auto!important;--body-background-fill:#fff;--background-fill-primary:#fff;--background-fill-secondary:#fafafa;--block-background-fill:#fff;--border-color-primary:#e7e7e5;--block-border-color:#e7e7e5;--body-text-color:#242727;--body-text-color-subdued:#777b79;--color-accent:#337f73}
#brand-header{padding:0!important;border:0!important;background:transparent!important;margin:0 0 38px!important}
.brand-header{display:flex;align-items:center;justify-content:space-between;padding:14px 0 18px;border-bottom:1px solid #ececea}
.brand-wordmark{display:flex;align-items:center;gap:10px;color:#222827;font-size:22px;font-weight:700;letter-spacing:-.7px}
.brand-mark{display:inline-block;width:22px;height:11px;border:1.8px solid #337f73;border-radius:8px;transform:rotate(-35deg);position:relative}
.brand-mark:after{content:'';position:absolute;left:9px;top:0;height:8px;border-left:1px solid #337f73}
.brand-caption{font-size:13px;color:#8b908d}
#intro{padding:0!important;border:0!important;background:transparent!important;margin-top:40px!important}
.intro-eyebrow{font-size:14px;color:#66746d;margin-bottom:20px}
.intro-copy h1{font-size:46px!important;line-height:1.3!important;font-weight:650!important;letter-spacing:-2px!important;color:#202624!important;margin:0 0 24px!important}
.intro-copy p{font-size:16px!important;color:#777e79!important;line-height:1.8!important;margin:0!important}
#hero-row{gap:45px!important;align-items:center!important;margin-bottom:35px!important}
#copy-column{min-width:290px!important}#comparison-column{min-width:min(360px,100%)!important}#comparison{border:1px solid #e8eae6!important;border-radius:18px!important;overflow:hidden}#compare-caption{font-size:13px!important;color:#737d75!important;text-align:center}#compare-controls{gap:8px!important}#compare-controls button{border:1px solid #e2e5df!important;border-radius:8px!important;background:#fff!important;box-shadow:none!important;font-size:12px!important;min-width:70px!important}#demo-selector{border:0!important;background:transparent!important}#upload-heading{text-align:center!important;border:0!important;background:transparent!important}#upload-heading h2{font-size:23px!important;font-weight:650!important;letter-spacing:-.6px!important;margin-bottom:5px!important}#upload-heading p{color:#7e867f;font-size:14px!important}
#upload-card{max-width:660px!important;width:100%!important;margin:10px auto 20px!important;background:#fff!important;border:1px solid #e6e8e5!important;border-radius:24px!important;padding:25px!important;box-shadow:0 12px 45px rgba(24,47,36,.07);gap:16px!important;min-width:min(360px,100%)!important;box-sizing:border-box}
#photo-input{border:1.5px dashed #cdd5ce!important;border-radius:15px!important;background:#fcfdfb!important;overflow:hidden}
#photo-input .upload-text{color:#727f75!important}
#recognize{background:#283c33!important;border:1px solid #283c33!important;color:white!important;border-radius:28px!important;min-height:52px!important;font-size:16px!important;box-shadow:none!important;transition:background .15s}
#recognize:hover{background:#3d5548!important}
#example-title{padding:0!important;border:0!important;font-size:12px!important;color:#939a94!important;text-align:center;margin-top:5px!important}
#upload-card .examples{justify-content:center}
#options{border:0!important;background:#fff!important;border-top:1px solid #f0f0ed!important;border-radius:0!important;font-size:13px}
#summary{background:transparent!important;border:0!important;padding:6px 0!important;min-height:0!important}
#results{border-top:1px solid #e7e9e6;padding-top:25px!important;margin-top:15px!important}
#results h2{font-size:23px!important;font-weight:650!important;color:#283c33}
#result-image{border:1px solid #e8eae6!important;border-radius:16px!important}
#clear{background:#fff!important;border:1px solid #dfe4dd!important;border-radius:10px!important;box-shadow:none!important}
#model-details{margin-top:28px!important;border:1px solid #eceeeb!important;border-radius:10px!important;background:white!important}
#notice{font-size:12px!important;color:#8a928b!important;text-align:center;padding:20px 0!important;border:0!important;background:transparent!important}
footer{display:none!important}
@media(max-width:760px){#hero-row{gap:25px!important}.intro-copy h1{font-size:35px!important;letter-spacing:-1.4px!important}#intro{margin-top:0!important}.brand-header{padding-top:8px}.brand-caption{font-size:11px}#brand-header{margin-bottom:20px!important}#upload-card{padding:18px!important}.intro-copy p{font-size:14px!important}}

/* Single-card alternative; the original draft stays on port 7861. */
#intro {margin:4px 0 30px!important;text-align:center!important}
.intro-eyebrow{display:none}
.intro-copy h1{font-size:40px!important;letter-spacing:-1.5px!important;margin:0 0 12px!important}
.intro-copy p{font-size:15px!important}
#single-card{padding:22px!important;background:#f6f7f5!important;border:1px solid #e0e4de!important;border-radius:24px!important;box-shadow:0 15px 50px #16282009;margin:0 0 18px!important;gap:24px!important;align-items:stretch!important}
#comparison-column{min-width:min(320px,100%)!important;gap:12px!important}
#comparison{border-radius:15px!important;border:0!important;background:#e9ece7!important}
#upload-card{max-width:none!important;width:100%!important;min-width:min(320px,100%)!important;margin:0!important;border:1.5px dashed #b9c5b7!important;border-radius:15px!important;padding:24px!important;background:#fff!important;box-shadow:none!important;gap:14px!important}
#upload-card h2{font-size:22px!important;font-weight:650!important;letter-spacing:-.6px!important;margin:0 0 3px!important;color:#293c32!important}
#upload-title{border:0!important;padding:0!important;margin:0!important;text-align:left}
#upload-title p{font-size:13px;color:#7c867d}
#photo-input{border:0!important;background:#fafbf9!important;border-radius:10px!important}
#example-title{margin-top:0!important}
#compare-controls button{font-size:11px!important}
#model-details{margin-top:16px!important}
@media(max-width:760px){#single-card{padding:14px!important;gap:18px!important}.intro-copy h1{font-size:29px!important}#intro{margin:0 0 22px!important}#upload-card{padding:18px!important}}

/* Matched preview and upload panels. */
#single-card{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));align-items:stretch!important}
#comparison-column,#upload-card{width:100%!important;min-width:0!important;box-sizing:border-box!important;padding:24px!important;gap:14px!important;border:1.5px solid #dce2d9!important;border-radius:15px!important;background:#fff!important}
#preview-title{border:0!important;padding:0!important;margin:0!important}
#preview-title h2{font-size:22px!important;font-weight:650!important;letter-spacing:-.6px!important;margin:0 0 3px!important;color:#293c32!important}
#preview-title p{font-size:13px;color:#7c867d}
#comparison,#photo-input{height:300px!important;min-height:300px!important;max-height:300px!important}
@media(max-width:760px){#single-card{grid-template-columns:1fr}#comparison-column,#upload-card{padding:18px!important}#comparison,#photo-input{height:260px!important;min-height:260px!important;max-height:260px!important}}

/* Result layout follows the same spacing as the upload card. */
#summary{margin:22px 0 12px!important;padding:0!important}
#results{background:#f6f7f5!important;border:1px solid #e0e4de!important;border-radius:24px!important;padding:24px!important;margin:0 0 24px!important;box-sizing:border-box!important}
#result-heading{background:transparent!important;border:0!important;padding:0!important;margin:0 0 18px!important}
#result-heading h2{margin:0!important;line-height:1.4!important}
#result-row{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px!important;align-items:start!important}
#result-photo-column,#result-data-column{min-width:0!important;gap:16px!important;padding:0!important}
#result-image{height:410px!important;margin:0!important;background:white!important;border-radius:14px!important;overflow:hidden!important}
#result-table,#result-download{margin:0!important;background:#fff!important;border:1px solid #e0e4de!important;border-radius:12px!important;overflow:hidden!important}
#result-table table{table-layout:auto!important}
#clear{margin:0!important;min-height:46px!important;width:100%!important}
@media(max-width:760px){#results{padding:18px!important}#result-row{grid-template-columns:1fr;gap:18px!important}#result-image{height:340px!important}}

/* Spacious result table and compact example selector. */
#found-title{padding:0!important;margin:0 0 2px!important;background:transparent!important;border:0!important}
#found-title h3{font-size:18px!important;font-weight:700!important;line-height:1.5!important;color:#293c32!important;margin:0!important}
#result-table th,#result-table td{padding:13px 12px!important;line-height:1.6!important;vertical-align:middle!important;white-space:normal!important}
#result-table th{font-weight:700!important;background:#f7f9f6!important}
#result-table th:first-child,#result-table td:first-child{min-width:44px!important;text-align:center!important}
#result-table th:nth-child(2),#result-table td:nth-child(2){min-width:150px!important}
#result-table th:nth-child(3),#result-table td:nth-child(3){min-width:82px!important}
#result-table th:nth-child(4),#result-table td:nth-child(4){min-width:84px!important}
#demo-selector{padding:4px 0 0!important;margin:0!important;border:0!important;background:transparent!important;box-shadow:none!important}
#demo-selector fieldset,#demo-selector .wrap{padding:0!important;margin:0!important;background:transparent!important;border:0!important;gap:10px!important}
#demo-selector label{padding:9px 13px!important;margin:0!important;border-radius:9px!important;box-shadow:none!important}
#compare-caption{margin:0!important;padding:0!important}
#compare-controls{margin:0!important}
#result-table th .cell-wrap,#result-table td .cell-wrap{padding:0!important}
#result-table th .header{white-space:nowrap!important;word-break:keep-all!important}
#result-table .cell-menu-button{display:none!important}
#result-table td{height:52px!important;box-sizing:border-box!important}
#comparison-column .form{background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important}
#demo-selector label.selected{background:#337f73!important;color:white!important;border-color:#337f73!important}
#model-note{margin-top:auto!important;padding:18px 0 0!important;border:0!important;border-top:1px solid #e8ece5!important;background:transparent!important}
.model-note h3{font-size:15px;font-weight:700;color:#293c32;margin:0 0 12px}.model-note dl{display:grid;grid-template-columns:1fr 1fr;gap:14px 20px;margin:0}.model-note dl>div{display:flex;flex-direction:column;gap:4px}.model-note dt{font-size:12px;color:#80877f}.model-note dd{font-size:14px;font-weight:600;color:#34463b;margin:0}.model-note p{font-size:12px;color:#8a928a;margin:14px 0 0}

/* Align both result panels from heading to bottom edge. */
#result-row{align-items:stretch!important}
#result-photo-column,#result-data-column{display:flex!important;flex-direction:column!important;gap:16px!important}
#location-title,#found-title{margin:0!important;padding:0!important;border:0!important;background:transparent!important;min-height:27px!important}
#location-title h3,#found-title h3{font-size:18px!important;font-weight:700!important;line-height:27px!important;color:#293c32!important;margin:0!important}
#result-image{height:460px!important;min-height:460px!important;flex:1!important}
#result-data-column{min-height:503px!important}
#result-table{flex:1!important;min-height:0!important}
#result-table td{height:52px!important}
#result-download{padding:12px!important}
#result-download .label-wrap{margin-bottom:8px!important}
#clear{margin-top:auto!important;flex:none!important;min-height:46px!important}
@media(max-width:760px){#result-image{height:360px!important;min-height:360px!important}#result-data-column{min-height:0!important}#result-table{flex:none!important}}

#intro{margin:0 0 28px!important;text-align:left!important;overflow:hidden!important;border-radius:20px!important}
.hero-banner{display:grid;grid-template-columns:1.05fr 1fr;background:#edf2e9;border:1px solid #e0e7dc;border-radius:20px;overflow:hidden;align-items:stretch}
.hero-banner-copy{padding:35px 38px;display:flex;flex-direction:column;justify-content:center}
.hero-kicker{font-size:11px;font-weight:650;letter-spacing:2px;color:#66846b}
.hero-banner h1{font-size:34px!important;line-height:1.35!important;letter-spacing:-1.4px!important;color:#263c2e!important;margin:14px 0!important;font-weight:700!important}
.hero-banner p{font-size:14px!important;line-height:1.8!important;color:#798477!important;margin:0!important}
.hero-banner img{width:100%;height:280px;object-fit:cover;display:block}
@media(max-width:700px){.hero-banner{grid-template-columns:1fr}.hero-banner-copy{padding:26px}.hero-banner h1{font-size:28px!important}.hero-banner img{height:220px}}

/* Compact hero with less vertical whitespace. */
#brand-header{margin-bottom:18px!important}
#intro{margin:0 0 16px!important}
.hero-banner-copy{padding:22px 34px!important}
.hero-banner h1{font-size:30px!important;margin:10px 0!important;line-height:1.3!important}
.hero-banner p{font-size:13px!important;line-height:1.65!important}
.hero-banner img{height:210px!important}
@media(max-width:700px){.hero-banner-copy{padding:20px 24px!important}.hero-banner h1{font-size:26px!important}.hero-banner img{height:170px!important}}

/* Slim hero, second compact pass. */
#brand-header{margin-bottom:8px!important}
#intro{margin:0 0 8px!important}
.hero-banner-copy{padding:16px 30px!important}
.hero-banner h1{font-size:26px!important;line-height:1.25!important;margin:7px 0!important}
.hero-banner p{font-size:12px!important;line-height:1.5!important}
.hero-kicker{font-size:10px!important}
.hero-banner img{height:164px!important}
@media(max-width:700px){.hero-banner-copy{padding:16px 22px!important}.hero-banner h1{font-size:24px!important}.hero-banner img{height:140px!important}}
#intro{padding:0!important;width:100%!important;max-width:none!important;box-sizing:border-box!important;align-self:stretch!important}
#intro>div{width:100%!important;max-width:none!important}
.hero-banner{width:100%!important;box-sizing:border-box!important}
#intro .html-container{padding:0!important;margin:0!important}
"""
HERO = """
<div class="brand-header"><div class="brand-wordmark"><span class="brand-mark" aria-hidden="true"></span>YacssoK</div><span class="brand-caption">알약 사진 인식</span></div>
"""
INTRO = f"""
<div class="hero-banner"><div class="hero-banner-copy"><span class="hero-kicker">YACSSOK</span><h1>사진 속 알약,<br>한눈에 살펴보세요.</h1><p>예시를 확인하거나 사진을 올려<br>알약의 위치와 인식 결과를 살펴보세요.</p></div><img src="/gradio_api/file={HERO_IMAGE}" alt="알약과 인식 표시를 담은 배너 일러스트"></div>
"""


def font(size):
    for path in [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Recognizer:
    def __init__(self, weights, class_map=None, imgsz=640, model=None):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.model = model
        self.names = {}
        self.lock = threading.Lock()
        if class_map:
            entries = json.loads(Path(class_map).read_text())
            self.names = {int(x["label_idx"]): x["name"] for x in entries}

    def load(self):
        if self.model is None:
            if not self.weights.is_file():
                raise FileNotFoundError("model checkpoint missing")
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights))
        model_ids = set(int(i) for i in self.model.names)
        if self.names and set(self.names) != model_ids:
            self.model = None
            raise ValueError("class map does not match model classes")
        if self.names and any(
            str(self.model.names[i]).strip() != str(self.names[i]).strip() for i in model_ids
        ):
            self.model = None
            raise ValueError("class names do not match checkpoint")
        if not self.names:
            self.names = {i: f"클래스 {i}" for i in model_ids}

    def empty(self, message, image=None):
        return image, pd.DataFrame(columns=COLUMNS), message, gr.update(value=None, visible=False)

    def recognize(self, image, threshold=0.25):
        if image is None:
            return self.empty("사진을 먼저 올리거나 아래 예시 사진을 선택해주세요.")
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return self.empty("이미지를 읽지 못했습니다. JPG 또는 PNG 사진을 다시 올려주세요.")
        with self.lock:
            if self.model is None:
                try:
                    self.load()
                except Exception:
                    return self.empty(
                        "모델을 준비하지 못했습니다. 모델 파일과 클래스 목록을 확인해주세요."
                    )
            try:
                start = time.perf_counter()
                result = self.model.predict(
                    source=Image.fromarray(image.astype("uint8")),
                    conf=float(threshold),
                    imgsz=self.imgsz,
                    device="cpu",
                    max_det=100,
                    verbose=False,
                )[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                drawn = Image.fromarray(image.astype("uint8")).convert("RGB")
                draw = ImageDraw.Draw(drawn)
                face = font(max(16, drawn.width // 45))
                rows = []
                for index, (box, label, score) in enumerate(
                    zip(boxes, labels, scores, strict=True), 1
                ):
                    name = self.names.get(int(label), f"클래스 {label}")
                    color = "#dc8c22" if score < 0.5 else "#1679aa"
                    x1, y1, x2, y2 = [float(v) for v in box]
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
                    text = f"{index}. {name} {score:.2f}"
                    bounds = draw.textbbox((0, 0), text, font=face)
                    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
                    tx = max(0, min(x1, drawn.width - tw - 12))
                    ty = max(0, y1 - th - 14)
                    draw.rectangle(
                        (tx, ty, min(drawn.width, tx + tw + 12), ty + th + 12), fill=color
                    )
                    draw.text((tx + 6, ty + 3 - bounds[1]), text, font=face, fill="white")
                    rows.append(
                        [
                            index,
                            name,
                            round(float(score), 3),
                            "확인 필요" if score < 0.5 else "예측 결과",
                        ]
                    )
                elapsed = time.perf_counter() - start
                df = pd.DataFrame(rows, columns=COLUMNS)
                if not rows:
                    return self.empty(
                        f"알약을 찾지 못했습니다. 사진을 밝고 선명하게 다시 찍거나 표시 기준을 낮춰보세요. · 처리 {elapsed:.2f}초",
                        np.asarray(drawn),
                    )
                handle = tempfile.NamedTemporaryFile(
                    prefix="pill_result_", suffix=".csv", delete=False
                )
                csv_path = handle.name
                handle.close()
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                return (
                    np.asarray(drawn),
                    df,
                    f"**찾은 알약 {len(rows)}개** · 처리 **{elapsed:.2f}초**",
                    gr.update(value=csv_path, visible=True),
                )
            except Exception:
                return self.empty("인식 중 오류가 발생했습니다. 다른 사진으로 다시 시도해주세요.")


def build_app(
    weights, class_map, imgsz=640, model_label="기존 학습 모델 · YOLOv8s", examples_dir=None
):
    examples_dir = Path(examples_dir) if examples_dir else ROOT / "data/demo_examples"
    engine = Recognizer(weights, class_map, imgsz)
    try:
        engine.load()
        status = f"{model_label} · {len(engine.names)}개 클래스 · 입력 기준 {imgsz}px"
    except Exception:
        status = "모델 준비 실패 · 파일과 클래스 정보 확인 필요"
    with gr.Blocks(
        title="YacssoK · 한 카드 시안", delete_cache=(3600, 3600), analytics_enabled=False
    ) as app:
        gr.HTML(HERO, elem_id="brand-header")
        preview_pairs = {
            "예시 1 · 알약 3개": (
                str(examples_dir / "01_pills.png"),
                str(examples_dir / "verified_01_pills.png"),
            ),
            "예시 2 · 알약 4개": (
                str(examples_dir / "02_pills.png"),
                str(examples_dir / "verified_02_pills.png"),
            ),
        }
        # Load optional local examples as pixels; returning file paths from callbacks can be blocked.
        preview_pairs = {
            name: tuple(Image.open(path).convert("RGB").copy() for path in pair)
            for name, pair in preview_pairs.items()
            if all(Path(path).is_file() for path in pair)
        }
        first_example = next(iter(preview_pairs), None)
        gr.HTML(INTRO, elem_id="intro")
        with gr.Row(elem_id="single-card"):
            with gr.Column(scale=1, elem_id="comparison-column"):
                gr.Markdown(
                    "## 인식 예시\n사진을 밀어서 인식 전후를 비교해보세요.", elem_id="preview-title"
                )
                comparison = gr.ImageSlider(
                    value=preview_pairs.get(first_example),
                    label="원본 / 인식 결과",
                    show_label=False,
                    interactive=False,
                    visible=bool(preview_pairs),
                    height=300,
                    max_height=300,
                    slider_position=50,
                    buttons=[],
                    elem_id="comparison",
                )
                gr.Markdown("원본　 ← 밀어서 비교 →　 인식 결과", elem_id="compare-caption")
                with gr.Row(elem_id="compare-controls", visible=bool(preview_pairs)):
                    before = gr.Button("원본", size="sm", interactive=bool(preview_pairs))
                    both = gr.Button("전후 비교", size="sm", interactive=bool(preview_pairs))
                    after = gr.Button("인식 결과", size="sm", interactive=bool(preview_pairs))
                selector = gr.Radio(
                    list(preview_pairs),
                    value=first_example,
                    visible=bool(preview_pairs),
                    show_label=False,
                    elem_id="demo-selector",
                )
                gr.HTML(
                    f'<div class="model-note"><h3>모델 정보</h3><dl><div><dt>모델</dt><dd>YOLOv8s</dd></div><div><dt>인식 종류</dt><dd>{len(engine.names)}종</dd></div><div><dt>입력 크기</dt><dd>{imgsz} × {imgsz}px</dd></div><div><dt>실행 환경</dt><dd>CPU</dd></div></dl><p>{status}</p></div>',
                    elem_id="model-note",
                )
                if not preview_pairs:
                    gr.Markdown(
                        "예시 사진은 실행 안내에 따라 로컬에 준비해주세요. 오른쪽에 사진을 올려 바로 인식할 수 있습니다."
                    )
            with gr.Column(scale=1, elem_id="upload-card"):
                gr.Markdown("## 내 사진 넣기\nJPG·PNG 사진을 올려주세요.", elem_id="upload-title")
                image = gr.Image(
                    type="numpy",
                    image_mode="RGB",
                    label="알약 사진",
                    show_label=False,
                    sources=["upload"],
                    height=300,
                    elem_id="photo-input",
                    buttons=["fullscreen"],
                )
                run = gr.Button("인식하기", variant="primary", size="lg", elem_id="recognize")
                examples = sorted(examples_dir.glob("[0-9]*.png"))
                if examples:
                    gr.Markdown("또는 예시 사진을 선택해보세요.", elem_id="example-title")
                    gr.Examples([[str(p)] for p in examples], inputs=[image], label="예시 사진")
                with gr.Accordion("표시 기준 조절", open=False, elem_id="options"):
                    threshold = gr.Slider(
                        0.05,
                        0.95,
                        value=0.25,
                        step=0.05,
                        label="최소 신뢰도",
                        info="낮추면 더 많은 결과가 표시됩니다.",
                    )
        before.click(lambda: gr.update(slider_position=100), [], comparison)
        after.click(lambda: gr.update(slider_position=0), [], comparison)
        both.click(lambda: gr.update(slider_position=50), [], comparison)
        selector.change(
            lambda name: gr.update(value=preview_pairs[name], slider_position=50),
            selector,
            comparison,
        )
        summary = gr.Markdown("", elem_id="summary")
        with gr.Column(visible=False, elem_id="results") as result_group:
            gr.Markdown("## 인식 결과", elem_id="result-heading")
            with gr.Row(elem_id="result-row"):
                with gr.Column(scale=1, elem_id="result-photo-column"):
                    gr.Markdown("### 알약 위치", elem_id="location-title")
                    output = gr.Image(
                        label="알약 위치",
                        show_label=False,
                        height=460,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_id="result-image",
                    )
                with gr.Column(scale=1, elem_id="result-data-column"):
                    gr.Markdown("### 찾은 알약", elem_id="found-title")
                    table = gr.Dataframe(
                        headers=COLUMNS,
                        label="찾은 알약",
                        show_label=False,
                        interactive=False,
                        wrap=True,
                        column_widths=[52, 190, 105, 95],
                        buttons=[],
                        elem_id="result-table",
                    )
                    download = gr.File(
                        label="결과 CSV 내려받기", visible=False, elem_id="result-download"
                    )
                    clear = gr.Button("다시 시작하기", elem_id="clear")
        gr.Markdown("학습용 알약 인식 데모입니다.", elem_id="notice")

        def recognize(image, threshold):
            drawn, df, message, file = engine.recognize(image, threshold)
            return drawn, df, message, file, gr.update(visible=drawn is not None)

        run.click(
            recognize,
            [image, threshold],
            [output, table, summary, download, result_group],
            api_name="recognize",
            concurrency_limit=1,
        )

        def reset():
            return (
                None,
                None,
                pd.DataFrame(columns=COLUMNS),
                "",
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        clear.click(reset, [], [image, output, table, summary, download, result_group])

        def reset_result():
            return (
                None,
                pd.DataFrame(columns=COLUMNS),
                "",
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        image.change(reset_result, [], [output, table, summary, download, result_group])
        threshold.release(reset_result, [], [output, table, summary, download, result_group])
    return app, engine


def main():
    """Run locally by default; sharing requires an explicit flag and environment credentials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_demo.yaml")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    def local_path(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    auth = None
    if args.share:
        user, password = os.getenv("YACSSOK_DEMO_USER"), os.getenv("YACSSOK_DEMO_PASSWORD")
        if not user or not password:
            parser.error("--share requires YACSSOK_DEMO_USER and YACSSOK_DEMO_PASSWORD")
        auth = (user, password)
    app, _ = build_app(
        local_path(cfg["weights"]),
        local_path(cfg["class_map"]),
        cfg.get("imgsz", 640),
        cfg.get("model_label", "YOLOv8s"),
        local_path(cfg.get("examples_dir", "data/demo_examples")),
    )
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="127.0.0.1",
        server_port=args.port or cfg.get("port", 7862),
        share=args.share,
        auth=auth,
        auth_message="YacssoK 웹 데모" if auth else None,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
