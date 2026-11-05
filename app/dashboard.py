"""Authenticated local preview: simple user flow and team review share one model."""

import html
from pathlib import Path

import gradio as gr

from .demo_adobe import ROOT, build_app
from .drug_info import DATA, configure_catalog
from .simple_drug_info import render_simple
from .web_config import load_model_config

TEAM_LOGO = ROOT / "app/assets/team-logo.jpg"
gr.set_static_paths(paths=[str(TEAM_LOGO)])

SENIOR_CSS = Path(__file__).with_name("guide.css").read_text()


def build_dashboard(config_path):
    config = load_model_config(config_path)
    configure_catalog(config["catalog_dir"])
    review, engine = build_app(
        config["weights"], config["class_map"], config["imgsz"], config["label"]
    )
    if len(engine.names) != config["expected_classes"]:
        raise ValueError("Checkpoint class count differs from active model config")
    catalog_names = {name.strip(): name for name in DATA}

    # Korean fallback copy kept separate from official data and model class names.
    def korean_info(name):
        canonical = catalog_names.get((name or "").strip())
        if canonical:
            return render_simple(canonical)
        if name:
            return "<p>이 알약의 공식 정보는 아직 연결되지 않았습니다. 약사에게 확인해주세요.</p>"
        return "<p>위에서 알약 이름을 누르면 자세한 정보를 볼 수 있습니다.</p>"

    def compute_result(image):
        if image is None:
            return (
                None,
                "사진을 먼저 올려주세요.",
                gr.update(choices=[], value=None),
                korean_info(None),
                gr.update(visible=False),
            )
        drawn, table, message, _ = engine.recognize(image, 0.25, create_csv=False)
        if table.empty:
            no_pills = "알약을 찾지 못했습니다" in message
            text = (
                "알약을 찾지 못했어요. 밝은 곳에서 알약이 크게 보이도록 다시 찍어주세요."
                if no_pills
                else "사진을 확인하지 못했어요. 다른 사진을 올려 다시 시도해주세요."
            )
            return (
                drawn,
                text,
                gr.update(choices=[], value=None),
                korean_info(None),
                gr.update(visible=False),
            )
        grouped = {}
        for _, row in table.iterrows():
            item = grouped.setdefault(row.iloc[1], {"numbers": [], "low": False})
            item["numbers"].append(str(int(row.iloc[0])))
            item["low"] |= row.iloc[3] == "확인 필요"
        options = [
            (
                ", ".join(item["numbers"])
                + "번 · "
                + name
                + (" · 추가 확인 필요" if item["low"] else ""),
                name,
            )
            for name, item in grouped.items()
        ]
        first = options[0][1]
        return (
            drawn,
            f"알약 {len(table)}개를 찾았어요. 사진의 번호와 예상 이름을 함께 확인해주세요.",
            gr.update(choices=options, value=first),
            korean_info(first),
            gr.update(visible=True),
        )

    def recognize(image, flow_state):
        # The dict is private to this Gradio session and shared with reset/upload callbacks.
        flow_state["busy"] = True
        generation = flow_state["generation"]
        try:
            yield (
                gr.skip(),
                "사진을 확인하고 있어요. 잠시만 기다려 주세요.",
                *[gr.skip() for _ in range(5)],
            )
            values = compute_result(image)
            if generation != flow_state["generation"]:
                yield tuple(gr.skip() for _ in range(7))
                return
            found = bool(values[2].get("value"))
            # Update the result and screen together; no late callback can reopen a step.
            yield (
                *values,
                gr.update(visible=image is None),
                gr.update(visible=image is not None and not found),
            )
        finally:
            flow_state["busy"] = False

    def reset_result():
        return (
            None,
            "",
            gr.update(choices=[], value=None),
            korean_info(None),
            gr.update(visible=False),
        )

    def select_login_view(view):
        return gr.update(selected="review" if view == "test" else "simple")

    with gr.Blocks(
        title="약속 · 알약 사진 확인", analytics_enabled=False, delete_cache=(3600, 3600)
    ) as app:
        gr.HTML(
            f'<div class="portal-brand"><div class="brand-name"><div class="team-logo"><img src="/gradio_api/file={html.escape(str(TEAM_LOGO), quote=True)}" alt=""></div>약속</div><div class="portal-actions"><form method="post" action="/session/logout"><button class="portal-logout" type="submit">로그아웃</button></form></div></div>',
            elem_id="portal-brand",
        )
        login_view = gr.Textbox(value="demo", visible=False)
        flow = gr.State({"generation": 0, "busy": False})
        with gr.Tabs(selected="simple", elem_id="portal-modes") as modes:
            with gr.Tab("앱 데모", id="simple"):
                with gr.Column(elem_id="elder-page"):
                    status = gr.Markdown("", elem_id="elder-status")
                    with gr.Column(elem_id="elder-home") as home:
                        gr.HTML(
                            '<div class="guide-progress"><div class="guide-step"><b>1</b>사진 준비<span>1 / 2</span></div><div class="guide-track"><i class="on"></i><i></i></div></div><h1>알약 사진을<br>찍어 주세요.</h1><div class="guide-viewfinder" role="img" aria-label="알약을 겹치지 않게 놓고 사진을 찍어 주세요"><svg viewBox="0 0 320 180" width="100%" height="180" aria-hidden="true"><g fill="none" stroke="#477465" stroke-width="3" stroke-linecap="round"><path d="M34 62V34h28m196 0h28v28M34 118v28h28m196 0h28v-28"/><rect x="90" y="63" width="84" height="38" rx="19" transform="rotate(-25 132 82)"/><path d="m124 65 16 34"/><circle cx="222" cy="112" r="23"/><path d="M210 112h24"/></g></svg></div><p class="guide-copy">밝은 곳에서 알약이 겹치지 않게<br>놓으면 더 잘 보여요.</p>',
                            elem_id="elder-intro",
                        )
                        camera = gr.UploadButton(
                            "사진 찍기",
                            file_types=["image"],
                            type="filepath",
                            elem_id="guide-camera",
                        )
                        gallery = gr.UploadButton(
                            "이미 찍은 사진이 있어요",
                            file_types=["image"],
                            type="filepath",
                            elem_id="guide-gallery",
                        )
                    with gr.Column(visible=False, elem_id="elder-confirm") as confirm:
                        gr.HTML(
                            '<div class="guide-progress"><div class="guide-step"><b>2</b>사진 확인<span>2 / 2</span></div><div class="guide-track"><i class="on"></i><i class="on"></i></div></div><h1>이 사진으로<br>확인할까요?</h1>'
                        )
                        image = gr.Image(
                            type="numpy",
                            image_mode="RGB",
                            interactive=False,
                            label="선택한 알약 사진",
                            height=320,
                            buttons=["fullscreen"],
                            elem_id="elder-input",
                        )
                        gr.Markdown("알약의 모양과 글자가 잘 보이는지 확인해 주세요.")
                        run = gr.Button("알약 확인하기", variant="primary", elem_id="elder-run")
                        back = gr.Button("다른 사진 선택", elem_id="elder-back")
                    with gr.Column(visible=False, elem_id="elder-result") as result:
                        gr.HTML(
                            '<h1>이 약으로 보여요.</h1><p class="guide-copy">사진의 번호와 예상 이름을 확인해 주세요.</p>'
                        )
                        output = gr.Image(
                            label="번호가 표시된 알약 사진",
                            interactive=False,
                            height=380,
                            buttons=["fullscreen"],
                            elem_id="elder-output",
                        )
                        choices = gr.Radio(
                            choices=[],
                            label="자세히 볼 알약 이름을 눌러 주세요",
                            elem_id="elder-choices",
                        )
                        card = gr.HTML(korean_info(None), elem_id="elder-drug-card")
                        again = gr.Button("다른 알약 확인하기", elem_id="elder-again")
                    gr.Markdown(
                        "사진 인식은 틀릴 수 있어요.\n복용 전 약사에게 확인해 주세요.",
                        elem_id="elder-help",
                    )
                    gr.HTML(
                        f'<div class="guide-coverage">현재 사진으로 확인할 수 있는 알약은 {len(engine.names)}종입니다.</div>'
                    )

                    def prepare_photo(file, flow_state):
                        if flow_state["busy"]:
                            return tuple(gr.skip() for _ in range(8))
                        flow_state["generation"] += 1
                        return (
                            file,
                            *reset_result(),
                            gr.update(visible=not bool(file)),
                            gr.update(visible=bool(file)),
                        )

                    def reset_flow(flow_state):
                        if flow_state["busy"]:
                            return tuple(gr.skip() for _ in range(10))
                        flow_state["generation"] += 1
                        return (
                            None,
                            *reset_result(),
                            gr.update(visible=True),
                            gr.update(visible=False),
                            None,
                            None,
                        )

                    def photo_ready_status(file):
                        return "" if file else "사진을 먼저 선택해 주세요."

                    choices.change(korean_info, choices, card, api_name="simple_info")
                    recognition_event = run.click(
                        recognize,
                        [image, flow],
                        [output, status, choices, card, result, home, confirm],
                        api_name="simple_recognize",
                        js="""(...args) => {
                            const back = document.getElementById('elder-back');
                            if (back) back.disabled = true;
                            const run = document.getElementById('elder-run');
                            if (run) { run.disabled = true; run.setAttribute('aria-busy', 'true'); }
                            return args;
                        }""",
                    )
                    recognition_event.then(
                        None,
                        js="""() => {
                        const back = document.getElementById('elder-back');
                        if (back) back.disabled = false;
                        const run = document.getElementById('elder-run');
                        if (run) { run.disabled = false; run.removeAttribute('aria-busy'); }
                    }""",
                        api_name=False,
                        queue=False,
                    )
                    # Image is read-only and only prepare/reset change it. Keep reset API
                    # separate so a programmatic image.change cannot cancel a new inference.
                    reset_api = gr.Button(visible=False)
                    reset_api.click(
                        reset_result,
                        [],
                        [output, status, choices, card, result],
                        api_name="simple_reset",
                        queue=False,
                    )
                    photo_outputs = [image, output, status, choices, card, result, home, confirm]
                    for uploader in (camera, gallery):
                        event = uploader.upload(
                            prepare_photo,
                            [uploader, flow],
                            photo_outputs,
                            api_name=False,
                            cancels=[recognition_event],
                            queue=False,
                        )
                        event.then(
                            photo_ready_status, uploader, status, api_name=False, queue=False
                        )
                    for button in (back, again):
                        button.click(
                            reset_flow,
                            flow,
                            [*photo_outputs, camera, gallery],
                            api_name="simple_again" if button is again else False,
                            cancels=[recognition_event],
                            queue=False,
                        )
            with gr.Tab("테스트용", id="review"):
                gr.Markdown(
                    f"연결 모델: **{html.escape(config['label'])}** · 모델 식별자: `{config['id']}` · {len(engine.names)}종\n두 화면은 같은 모델을 사용합니다. 확인할 알약 사진을 넣어 실행해주세요."
                )
                review.render()
        app.load(
            select_login_view,
            login_view,
            modes,
            js="() => [new URLSearchParams(window.location.search).get('view') === 'test' ? 'test' : 'demo']",
            api_name=False,
            queue=False,
        )
        app.load(
            None,
            js="""() => {
            const status = document.getElementById('elder-status');
            if (status) { status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); }
            const setCapture = element => {
                // Gradio 6.30 renders the file input immediately before the button.
                const input = element?.previousElementSibling;
                if (!input?.matches('input[type=file]')) return;
                if (element.id === 'guide-camera') input.setAttribute('capture', 'environment');
                else input.removeAttribute('capture');
            };
            setCapture(document.getElementById('guide-camera'));
            setCapture(document.getElementById('guide-gallery'));
            if (!window.__pillGuideCaptureBound) {
                // Columns are recreated when a step changes. Delegate to the current
                // button so the native file chooser always receives the right hint.
                document.addEventListener('click', event => {
                    const element = event.target instanceof Element
                        ? event.target.closest('#guide-camera, #guide-gallery') : null;
                    if (element) setCapture(element);
                }, true);
                window.__pillGuideCaptureBound = true;
            }
        }""",
            api_name=False,
            queue=False,
        )
    return app, engine, config
