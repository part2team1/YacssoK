"""YOLO 계열 탐지기 어댑터 (모델 A 담당).

ultralytics는 자체 데이터 형식(이미지 폴더 + txt 라벨 + data.yaml)을 쓰므로,
학습 전에 scripts/export_yolo.py로 같은 분할을 YOLO 형식으로 내보낸다.
YOLO의 클래스 번호 = 팀 표준 label_idx 이므로 번호 변환은 필요 없다.

고도화 아이디어 (실험 계획 M1, M2, D6, T1): 모델 크기 s/m/l, YOLO11·RT-DETR, imgsz 1024, 학습률 튜닝.
라이선스: ultralytics는 AGPL-3.0 (상용화 시 별도 라이선스 필요).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.interfaces import MAX_PILLS_PER_IMAGE, Prediction
from src.models.base import BaseDetector

#from src.utils import get_device, image_id_from_name, resolve_path
from src.utils import get_device, image_id_from_name, yolo_data_yaml  #v2-v3 D1.5


class YoloDetector(BaseDetector):
    name = "yolo"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        self.model = None

    def _device_arg(self):
        dev = get_device(self.cfg["train"].get("device"))
        return dev.index if dev.type == "cuda" else dev.type

    def fit(self, out_dir: Path) -> Path:
        from ultralytics import YOLO

        m, t = self.cfg["model"], self.cfg["train"]
        data_yaml = yolo_data_yaml(self.cfg)  # repeat_threshold를 켜면 data_rfs*.yaml  #v2.D1.5

        if not data_yaml.exists():
            raise FileNotFoundError(
                f"{data_yaml} 이 없습니다. 먼저 실행: uv run python scripts/export_yolo.py --config <config>"
            )
        # weights가 .pt면 사전학습 가중치에서, .yaml이면 처음부터 학습
        self.model = YOLO(m["weights"])
        # self.model.train(
        #     data=str(data_yaml),
        #     epochs=t["epochs"],
        #     imgsz=self.cfg["data"]["img_size"],
        #     batch=t["batch_size"],
        #     lr0=t["lr"],
        #     optimizer=t.get("optimizer", "auto"),
        #     workers=t.get("num_workers", 4),
        #     device=self._device_arg(),
        #     seed=self.cfg.get("seed", 42),
        #     project=str(out_dir),
        #     name="yolo_run",
        #     exist_ok=True,
        #     plots=False,
        #     verbose=False,
        #     **self.cfg.get("yolo_args", {}),  # ultralytics 고유 옵션은 configs의 yolo_args에
        # )
        args = {
            "data": str(data_yaml),
            "epochs": t["epochs"],
            "imgsz": self.cfg["data"]["img_size"],
            "batch": t["batch_size"],
            "lr0": t["lr"],
            "optimizer": t.get("optimizer", "auto"),
            "workers": t.get("num_workers", 4),
            "device": self._device_arg(),
            "seed": self.cfg.get("seed", 42),
            "project": str(out_dir),
            "name": "yolo_run",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
        # ultralytics 고유 옵션은 configs의 yolo_args에. 같은 키(plots 등)는 yolo_args 값이 이긴다
        args.update(self.cfg.get("yolo_args") or {})
        self.model.train(**args)

        best = out_dir / "yolo_run" / "weights" / "best.pt"
        if not best.exists():
            best = out_dir / "yolo_run" / "weights" / "last.pt"
        shutil.copy(best, out_dir / "best.pt")
        self.load(out_dir / "best.pt")
        return out_dir / "best.pt"

    def load(self, weights: str | Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        ev = self.cfg["eval"]
        conf = ev["conf_thr"] if conf_thr is None else conf_thr
        preds: list[Prediction] = []
        batch = self.cfg["train"].get("batch_size", 16)
        for i in range(0, len(image_paths), batch):
            chunk = [str(p) for p in image_paths[i : i + batch]]
            results = self.model.predict(
                chunk,
                conf=conf,
                iou=ev.get("nms_iou", 0.5),
                max_det=ev.get("max_det", MAX_PILLS_PER_IMAGE),
                imgsz=self.cfg["data"]["img_size"],
                device=self._device_arg(),
                verbose=False,
            )
            for path, r in zip(chunk, results, strict=True):
                b = r.boxes
                preds.append(
                    Prediction(
                        image_id=image_id_from_name(Path(path).name),
                        file_name=Path(path).name,
                        boxes=b.xyxy.detach().cpu().float(),  # 원본 이미지 좌표 xyxy
                        scores=b.conf.detach().cpu().float(),
                        labels=b.cls.detach().cpu().long(),
                    )
                )
        return preds
