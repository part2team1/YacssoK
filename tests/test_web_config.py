"""Model replacement must reject a bad map before loading a pickle checkpoint."""

import json

import pytest
import yaml

from app.demo_portal import credentials
from app.web_config import load_model_config


@pytest.fixture
def web_config(tmp_path):
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"not loaded in config tests")
    class_map = tmp_path / "classes.json"
    class_map.write_text(json.dumps([{"label_idx": 0, "name": "test pill"}]))
    config = {
        "id": "test",
        "label": "Test model",
        "weights": str(weights),
        "class_map": str(class_map),
        "catalog_dir": str(tmp_path / "catalog"),
        "expected_classes": 1,
        "imgsz": 640,
    }
    path = tmp_path / "web.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, class_map


def test_valid_config_resolves_paths_without_loading_weights(web_config):
    path, _, _ = web_config
    config = load_model_config(path)
    assert config["weights"].is_file()
    assert config["expected_classes"] == 1


@pytest.mark.parametrize(
    "entries",
    [
        [{"label_idx": 1, "name": "a"}],
        [{"label_idx": True, "name": "a"}],
        [{"label_idx": 0, "name": "a"}, {"label_idx": 1, "name": "a"}],
        [{"label_idx": 0, "name": " "}],
    ],
)
def test_reject_bad_class_map(web_config, entries):
    path, _, class_map = web_config
    class_map.write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        load_model_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("expected_classes", 91), ("imgsz", 0), ("imgsz", True), ("port", 70000)],
)
def test_reject_wrong_model_metadata(web_config, key, value):
    path, config, _ = web_config
    config[key] = value
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_model_config(path)


def test_missing_model_does_not_start(web_config):
    path, config, _ = web_config
    config["weights"] += ".missing"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="weights"):
        load_model_config(path)


@pytest.mark.parametrize("variable", ["PILL_USERNAME", "PILL_PASSWORD"])
def test_missing_or_blank_login_fails_closed(monkeypatch, variable):
    monkeypatch.setenv("PILL_USERNAME", "test-account")
    monkeypatch.setenv("PILL_PASSWORD", "test-only-password")
    monkeypatch.setenv(variable, " ")
    with pytest.raises(ValueError):
        credentials()
