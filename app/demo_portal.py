"""Run the authenticated two-mode web app with an external team checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .web_config import ROOT


def credentials():
    """Do not start an unprotected app when environment secrets are missing."""
    username = os.environ.get("PILL_USERNAME", "")
    password = os.environ.get("PILL_PASSWORD", "")
    if not username.strip() or not password.strip():
        raise ValueError("Set PILL_USERNAME and PILL_PASSWORD before starting the web app")
    return username, password


def create_app(config_path):
    # Import the optional demo dependency only when starting the web, so training
    # tools and config tests keep working without installing the demo extra.
    import gradio as gr

    from .dashboard import SENIOR_CSS, build_dashboard
    from .demo_adobe import CSS
    from .portal_auth import portal_app_kwargs

    app, _, config = build_dashboard(config_path)
    return (
        app,
        config,
        {
            "app_kwargs": portal_app_kwargs(),
            "theme": gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
            "css": CSS + SENIOR_CSS,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_portal.yaml")
    args = parser.parse_args()
    auth = credentials()
    # This entry point runs on the local CPU. HF ZeroGPU has a separate deployment
    # adapter and must not be advertised as active merely because an env var exists.
    os.environ["PILL_RUNTIME"] = "cpu"
    app, config, presentation = create_app(args.config)
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=config.get("host", "127.0.0.1"),
        server_port=config.get("port", 7871),
        share=False,
        ssr_mode=False,
        auth=auth,
        auth_message="YacssoK · 알약 사진 확인",
        **presentation,
    )


if __name__ == "__main__":
    main()
