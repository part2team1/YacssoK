"""모델 학습 없이 파일 재생성·클래스·점수 비교·캐시 회귀를 확인한다."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts import export_yolo, make_submission, run_matrix, run_pipeline
from src.provenance import evaluation_signature, experiment_signature, file_digest
from src.utils import save_json, save_yaml


@pytest.fixture
def tiny_data(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (32, 32), "white").save(photos / name)
    annotations = tmp_path / "annotations.json"
    save_json(
        {
            "images": [
                {"id": i, "file_name": name, "width": 32, "height": 32}
                for i, name in enumerate(("a.png", "b.png"), 1)
            ],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 1900, "bbox": [2, 2, 10, 10]}
                for i in (1, 2)
            ],
            "categories": [{"id": 1900, "name": "test pill"}],
        },
        annotations,
    )
    classes = tmp_path / "classes.json"
    save_json([{"label_idx": 0, "category_id": 1900, "name": "test pill"}], classes)
    splits = tmp_path / "splits" / "v0"
    splits.mkdir(parents=True)
    (splits / "train.txt").write_text("a.png\n")
    (splits / "val.txt").write_text("b.png\n")
    return {
        "exp_name": "tiny",
        "seed": 42,
        "train": {"epochs": 1},
        "eval": {"iou_thresholds": [0.75]},
        "model": {"family": "yolo", "weights": "unused.pt"},
        "data": {
            "version": "v0",
            "train_images": str(photos),
            "train_annotations": str(annotations),
            "class_map": str(classes),
            "split_dir": str(splits.parent),
            "yolo_dir": str(tmp_path / "yolo"),
        },
    }


def test_reexport_replaces_removed_files_and_changed_source(tiny_data, tmp_path):
    cfg = tiny_data
    root = export_yolo.export_dataset(cfg)
    (root / "labels/train.cache").write_text("old")
    split = tmp_path / "splits/v0"
    (split / "train.txt").write_text("b.png\n")
    (split / "val.txt").write_text("a.png\n")
    new_photos = tmp_path / "new-photos"
    new_photos.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (32, 32), "blue").save(new_photos / name)
    cfg["data"]["train_images"] = str(new_photos)
    export_yolo.export_dataset(cfg)
    assert [p.name for p in (root / "images/train").iterdir()] == ["b.png"]
    assert [p.name for p in (root / "images/val").iterdir()] == ["a.png"]
    assert not (root / "labels/train.cache").exists()
    assert (root / "images/train/b.png").resolve() == new_photos / "b.png"
    assert str(root) in (root / "data.yaml").read_text()


def test_export_rejects_overlap_and_preserves_previous_on_failure(tiny_data, tmp_path, monkeypatch):
    root = export_yolo.export_dataset(tiny_data)
    original = (root / "data.yaml").read_bytes()
    val = tmp_path / "splits/v0/val.txt"
    val.write_text("a.png\n")
    with pytest.raises(ValueError, match="같은 사진"):
        export_yolo.export_dataset(tiny_data)
    val.write_text("b.png\n")

    def fail(*args):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(export_yolo, "_write_export", fail)
    with pytest.raises(OSError):
        export_yolo.export_dataset(tiny_data)
    assert (root / "data.yaml").read_bytes() == original
    assert (root / "images/train/a.png").is_file()


def test_mixed_class_maps_fail_before_model_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(make_submission, "OUTPUT_DIR", tmp_path)
    for exp, category in (("a", 1900), ("b", 250)):
        classes = tmp_path / f"{exp}.json"
        save_json([{"label_idx": 0, "category_id": category, "name": "pill"}], classes)
        save_yaml({"data": {"class_map": str(classes)}}, tmp_path / exp / "config.yaml")
    with pytest.raises(ValueError, match="번호표"):
        make_submission.submission_configs(["a", "b"])
    assert len(make_submission.submission_configs(["a", "a"])) == 2


def test_wrong_prediction_labels_or_photo_sets_are_rejected():
    a = {"file_name": "1.png", "image_id": 1, "labels": torch.tensor([0])}
    b = {**a, "labels": torch.tensor([1])}
    with pytest.raises(ValueError, match="예측 라벨"):
        make_submission.validate_predictions([[b]], {0: 1900})
    with pytest.raises(ValueError, match="사진 목록"):
        make_submission.validate_predictions([[a], [{**a, "image_id": 2}]], {0: 1900})


def test_completed_cache_checks_configuration_and_annotation_contents(
    tiny_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(run_matrix, "OUTPUT_DIR", tmp_path / "outputs")
    result = tmp_path / "outputs/tiny/metrics.json"
    save_json({"map": 0.9, "provenance": {"experiment": experiment_signature(tiny_data)}}, result)
    assert run_matrix.completed_for_config("tiny", tiny_data)
    changed = deepcopy(tiny_data)
    changed["train"]["epochs"] = 2
    with pytest.raises(ValueError, match="지문"):
        run_matrix.completed_for_config("tiny", changed)
    ann = tmp_path / "annotations.json"
    data = json.loads(ann.read_text())
    data["annotations"][0]["bbox"] = [3, 3, 10, 10]
    save_json(data, ann)
    with pytest.raises(ValueError, match="지문"):
        run_matrix.completed_for_config("tiny", tiny_data)


def test_evaluation_signature_tracks_ground_truth_and_class_order(tiny_data, tmp_path):
    before = evaluation_signature(tiny_data)
    classes = tmp_path / "classes.json"
    data = json.loads(classes.read_text())
    data[0]["category_id"] = 250
    save_json(data, classes)
    assert evaluation_signature(tiny_data) != before


def test_ranking_rejects_different_exam_and_accepts_common_rescoring(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path)
    for exp, score, key in (("base", 0.95, "original"), ("clean", 0.97, "corrected")):
        weights = tmp_path / exp / "best.pt"
        weights.parent.mkdir()
        weights.write_bytes(exp.encode())
        save_yaml({"model": {"family": "yolo"}}, weights.parent / "config.yaml")
        save_json(
            {"map": score, "provenance": {"evaluation": key, "weights": file_digest(weights)}},
            weights.parent / "metrics.json",
        )
    candidates = [("base", "yolo", "base"), ("clean", "yolo", "clean")]
    with pytest.raises(ValueError, match="서로 다른 검증"):
        run_pipeline.comparable_leaderboard(candidates)
    save_json(
        {
            "map": 0.98,
            "provenance": {
                "evaluation": "corrected",
                "weights": file_digest(tmp_path / "base/best.pt"),
            },
        },
        tmp_path / "base/selection_metrics.json",
    )
    assert run_pipeline.comparable_leaderboard(candidates)[0] == (0.98, "yolo", "base")
    (tmp_path / "base/best.pt").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="가중치 지문"):
        run_pipeline.comparable_leaderboard(candidates)


def test_export_protects_annotation_source_inside_target(tiny_data, tmp_path):
    root = tmp_path / "yolo/v0"
    root.mkdir(parents=True)
    annotation = root / "raw.json"
    annotation.write_bytes(Path(tiny_data["data"]["train_annotations"]).read_bytes())
    tiny_data["data"]["train_annotations"] = str(annotation)
    with pytest.raises(ValueError, match="경로는 겹칠"):
        export_yolo.export_dataset(tiny_data)
    assert annotation.is_file()


def test_export_keeps_other_rfs_variants_valid(tiny_data):
    tiny_data["data"]["repeat_threshold"] = 0.1
    root = export_yolo.export_dataset(tiny_data)
    tiny_data["data"]["repeat_threshold"] = 0.2
    export_yolo.export_dataset(tiny_data)
    assert (root / "data_rfs0.1.yaml").is_file()
    assert (root / "data_rfs0.2.yaml").is_file()
    assert (root / "train_rfs0.1.txt").is_file()
    assert (root / "train_rfs0.2.txt").is_file()


def test_incomplete_result_never_automatically_retrains(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "OUTPUT_DIR", tmp_path)
    save_json({"map": None}, tmp_path / "incomplete/metrics.json")
    with pytest.raises(ValueError, match="force"):
        run_matrix.completed_for_config("incomplete", {})
    save_yaml({}, tmp_path / "interrupted/config.yaml")
    with pytest.raises(ValueError, match="force"):
        run_matrix.completed_for_config("interrupted", {})


def test_pipeline_stops_after_matrix_failure_before_using_old_scores(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path / "outputs")
    plan = tmp_path / "plan.yaml"
    save_yaml(
        {"prefix": "qa", "models": {"model": "unused.yaml"}, "bundles": {}, "runs": {"base": []}},
        plan,
    )
    commands = []
    monkeypatch.setattr(run_pipeline, "sh", lambda cmd, log: commands.append(cmd) or False)
    monkeypatch.setattr(
        sys, "argv", ["run_pipeline.py", "--plan", str(plan), "--stages", "data_exp", "submit"]
    )
    with pytest.raises(SystemExit, match="이전 결과"):
        run_pipeline.main()
    assert len(commands) == 1 and "scripts/run_matrix.py" in commands[0]
    assert not (tmp_path / "outputs/pipeline_qa/state.json").exists()


def test_pipeline_rejects_changed_plan_even_when_weights_and_evaluation_match(
    tiny_data, tmp_path, monkeypatch
):
    import sys

    cfg = deepcopy(tiny_data)
    cfg["exp_name"] = "qa_model_base"
    model_config = tmp_path / "model.yaml"
    save_yaml(cfg, model_config)
    plan = tmp_path / "plan.yaml"
    save_yaml(
        {
            "prefix": "qa",
            "models": {"model": str(model_config)},
            "bundles": {},
            "runs": {"base": []},
        },
        plan,
    )
    outputs = tmp_path / "outputs"
    folder = outputs / cfg["exp_name"]
    save_yaml(cfg, folder / "config.yaml")
    (folder / "best.pt").write_bytes(b"unused checkpoint")
    save_json(
        {
            "map": 0.9,
            "provenance": {
                "experiment": experiment_signature(cfg),
                "evaluation": evaluation_signature(cfg),
                "weights": file_digest(folder / "best.pt"),
            },
        },
        folder / "metrics.json",
    )
    changed = deepcopy(cfg)
    changed["train"]["epochs"] = 9
    save_yaml(changed, model_config)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--plan", str(plan), "--stages", "submit"])
    with pytest.raises(SystemExit, match="현재 실험 계획"):
        run_pipeline.main()


def test_ocr_cache_follows_real_detector_and_imprint_table(tiny_data, tmp_path, monkeypatch):
    from src import provenance

    monkeypatch.setattr(provenance, "OUTPUT_DIR", tmp_path / "outputs")
    detector = tmp_path / "outputs/detector"
    save_yaml(tiny_data, detector / "config.yaml")
    (detector / "best.pt").write_bytes(b"detector version 1")
    drug_info = tmp_path / "drug_info.json"
    save_json({}, drug_info)
    cfg = deepcopy(tiny_data)
    cfg["model"] = {"family": "ocr_rerank", "detector_exp": "detector", "drug_info": str(drug_info)}
    wrapper = tmp_path / "outputs/ocr/best.pt"
    save_yaml(cfg, wrapper.parent / "config.yaml")
    save_json({"detector_exp": "detector"}, wrapper)
    before = provenance.checkpoint_signature(cfg, wrapper)
    experiment = experiment_signature(cfg)
    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path / "outputs")
    save_json(
        {"map": 0.9, "provenance": {"evaluation": "same", "weights": before}},
        wrapper.parent / "metrics.json",
    )
    (detector / "best.pt").write_bytes(b"detector version 2")
    assert provenance.checkpoint_signature(cfg, wrapper) != before
    assert experiment_signature(cfg) != experiment
    with pytest.raises(ValueError, match="가중치 지문"):
        run_pipeline.comparable_leaderboard([("ocr", "model", "ocr")])
    before = provenance.checkpoint_signature(cfg, wrapper)
    save_json({"1900": {"식별문자(앞)": "AB"}}, drug_info)
    assert provenance.checkpoint_signature(cfg, wrapper) != before


def test_generated_report_uses_common_scores_and_supplements(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(run_pipeline, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(run_pipeline, "PROJECT_ROOT", tmp_path)
    for run, raw, common in (("base", 0.95, 0.98), ("clean", 0.97, 0.97)):
        folder = tmp_path / "outputs" / f"qa_m_{run}"
        save_json(
            {"map": raw, "map_75": 0.02, "mar_100": 0.03, "train_min": 4}, folder / "metrics.json"
        )
        save_json(
            {
                "map": common,
                "map_75": 0.99,
                "mar_100": 0.88,
                "provenance": {"evaluation": "common"},
            },
            folder / "selection_metrics.json",
        )
    out = tmp_path / "outputs/pipeline_qa"
    out.mkdir()
    plan = {"prefix": "qa", "runs": {"base": [], "clean": []}}
    state = {"best": {"exp": "qa_m_base", "model": "m", "run": "base", "map": 0.98}}
    args = SimpleNamespace(
        epochs=None,
        plan="test.yaml",
        min_share=0.5,
        min_gain=0.01,
        tune_iterations=2,
        tune_epochs=1,
    )
    figures = tmp_path / "outputs/compare_qa"
    figures.mkdir()
    for filename in ("scores.png", "metrics.png", "curves_m.png"):
        Image.new("RGB", (5, 5)).save(figures / filename)
    run_pipeline.write_report(
        out,
        plan,
        state,
        ["m"],
        lambda m, r: f"qa_{m}_{r}",
        tmp_path / "submissions/result.csv",
        [],
        args,
    )
    report = (out / "report.html").read_text()
    assert "0.9800" in report and "0.9900" in report and "0.8800" in report
    assert "0.9500" not in report and "0.0200" not in report
    assert "(-0.0100)" in report
    assert "data:image/png;base64" not in report
