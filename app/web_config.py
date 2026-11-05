"""Validate portable web settings before loading any model or serving files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_model_config(path):
    """Resolve repository-relative paths and reject a mismatched class map early."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Web config must be a YAML mapping")
    for key in ("id", "label", "weights", "class_map", "imgsz", "expected_classes", "catalog_dir"):
        if key not in config:
            raise ValueError(f"Web config is missing {key}")
    for key in ("id", "label"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Invalid {key}")
    for key in ("imgsz", "expected_classes"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("weights", "class_map", "catalog_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Invalid {key} path")
        value = Path(config[key]).expanduser()
        config[key] = (value if value.is_absolute() else ROOT / value).resolve()
        if key != "catalog_dir" and not config[key].is_file():
            raise ValueError(f"Web config file not found: {key}")
    entries = json.loads(config["class_map"].read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Class map must be a nonempty list")
    ids = [item["label_idx"] for item in entries]
    names = [item["name"].strip() for item in entries]
    if (
        any(type(index) is not int for index in ids)
        or sorted(ids) != list(range(len(entries)))
        or any(not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Class map must have unique names and consecutive IDs from 0")
    if len(entries) != config["expected_classes"]:
        raise ValueError("Expected class count differs from class map")
    if type(config.get("port", 7871)) is not int or not 1 <= config.get("port", 7871) <= 65535:
        raise ValueError("Invalid server port")
    if not isinstance(config.get("host", "127.0.0.1"), str):
        raise ValueError("Invalid server host")
    return config
