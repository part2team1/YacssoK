"""가짜 추론으로 세션 취소·CSV 수명·공유 인증을 실제 콜백에 대해 확인한다."""

import importlib
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gradio")


@pytest.mark.parametrize("module_name", ["app.demo_adobe", "app.demo_web"])
@pytest.mark.parametrize("action", ["reset", "reset_result"])
def test_changed_input_discards_delayed_inference(module_name, action, monkeypatch, tmp_path):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.Recognizer, "load", lambda self: setattr(self, "names", {0: "pill"}))
    app, engine = module.build_app(tmp_path / "unused.pt", None)
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    start, release = threading.Event(), threading.Event()

    def slow(image, threshold):
        start.set()
        assert release.wait(10)
        return (
            image,
            pd.DataFrame([[1, "pill", 0.9, "예측 결과"]], columns=module.COLUMNS),
            "old result",
            {"value": "old.csv"},
        )

    monkeypatch.setattr(engine, "recognize", slow)
    state = {"generation": 0}
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    callbacks["reset_result"](image, 0.25, state)
    iterator = callbacks["recognize"](image, 0.25, state)
    next(iterator)  # 첫 yield 뒤 실제 추론이 시작된다.
    received = []
    worker = threading.Thread(target=lambda: received.append(next(iterator)))
    worker.start()
    assert start.wait(10)
    if action == "reset":
        callbacks[action](0.25, state)
    else:
        callbacks[action](None, 0.25, state)
    release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert all(value == {"__type__": "update"} for value in received[0])
    assert all(
        f.queue is False
        for f in app.fns.values()
        if f.fn and f.fn.__name__ in ("reset", "reset_result")
    )
    engine.result_files.close()
    app.close()


@pytest.mark.parametrize("module_name", ["app.demo_adobe", "app.demo_web"])
def test_unchanged_input_keeps_results_and_other_session_reset_does_not_cancel(
    module_name, monkeypatch, tmp_path
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.Recognizer, "load", lambda self: setattr(self, "names", {0: "pill"}))
    app, engine = module.build_app(tmp_path / "unused.pt", None)
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    monkeypatch.setattr(
        engine,
        "recognize",
        lambda *a: (
            np.zeros((10, 10, 3), dtype=np.uint8),
            pd.DataFrame([[1, "pill", 0.9, "예測"]], columns=module.COLUMNS),
            "current result",
            {},
        ),
    )
    a, b = {"generation": 0}, {"generation": 0}
    image = np.zeros((10, 10, 3))
    callbacks["reset_result"](image, 0.25, a)
    iterator = callbacks["recognize"](image, 0.25, a)
    next(iterator)
    callbacks["reset_result"](None, 0.25, b)
    assert next(iterator)[2] == "current result"
    app.close()


def test_csv_store_cleans_expired_files_and_removes_folder_on_close(tmp_path):
    from app.result_files import ResultFiles

    store = ResultFiles(ttl_seconds=60)
    frame = pd.DataFrame({"pill": ["test"]})
    old = Path(store.save(frame))
    os.utime(old, (1, 1))
    current = Path(store.save(frame))
    assert not old.exists() and current.exists()
    folder = current.parent
    store.close()
    assert not folder.exists()


def test_legacy_share_requires_credentials(monkeypatch):
    from app.demo import share_credentials

    monkeypatch.delenv("PILL_USERNAME", raising=False)
    monkeypatch.delenv("PILL_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="requires"):
        share_credentials()
    monkeypatch.setenv("PILL_USERNAME", "review-test")
    monkeypatch.setenv("PILL_PASSWORD", "test-password")
    assert share_credentials() == ("review-test", "test-password")
    monkeypatch.setenv("PILL_PASSWORD", "  ")
    with pytest.raises(ValueError):
        share_credentials()


@pytest.mark.parametrize("module_name", ["app.demo_adobe", "app.demo_web"])
def test_queued_old_image_is_discarded_before_inference(module_name, monkeypatch, tmp_path):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.Recognizer, "load", lambda self: setattr(self, "names", {0: "pill"}))
    app, engine = module.build_app(tmp_path / "unused.pt", None)
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    state = {"generation": 0}
    old_image = np.zeros((10, 10, 3), dtype=np.uint8)
    new_image = np.ones((10, 10, 3), dtype=np.uint8)
    callbacks["reset_result"](old_image, 0.25, state)
    iterator = callbacks["recognize"](old_image, 0.25, state)
    callbacks["reset_result"](new_image, 0.25, state)

    def should_not_run(*args):
        pytest.fail("outdated queued inference must not start")

    monkeypatch.setattr(engine, "recognize", should_not_run)
    assert len(list(iterator)) == 1
    assert all(not f.cancels for f in app.fns.values())
    app.close()


@pytest.mark.parametrize("module_name", ["app.demo_adobe", "app.demo_web"])
def test_named_api_keeps_two_arguments_without_ui_initialization(
    module_name, monkeypatch, tmp_path
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module.Recognizer, "load", lambda self: setattr(self, "names", {0: "pill"}))
    app, engine = module.build_app(tmp_path / "unused.pt", None)
    monkeypatch.setattr(
        engine,
        "recognize",
        lambda image, threshold: (image, pd.DataFrame(columns=module.COLUMNS), "api complete", {}),
    )
    endpoint = next(f for f in app.fns.values() if f.api_name == "recognize")
    assert len(endpoint.inputs) == 2
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert endpoint.fn(image, 0.25)[2] == "api complete"
    app.close()
