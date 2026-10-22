"""Faster R-CNN 탐지기 (모델 B 담당). torchvision 모델을 PyTorch 학습 루프로 직접 학습한다.

주의: torchvision 탐지 모델은 0번 클래스를 '배경'으로 쓴다.
      팀 표준 label_idx에 1을 더해 넣고, 예측을 내보낼 때 1을 뺀다. 이 변환은 이 파일 안에서만 한다.

고도화 아이디어 (실험 계획 M3, T2, M4): 백본 비교, 클래스 불균형 대응, 탐지 + 분류 2단계 방식.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PillDataset, collate_fn, load_records, load_split, read_image_rgb
from src.interfaces import MAX_PILLS_PER_IMAGE, Prediction
from src.models.base import BaseDetector
from src.transforms import get_transforms
from src.utils import get_device, image_id_from_name, load_class_map, save_json

BACKBONES = {
    "fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
}


def _build_torchvision_model(arch: str, num_classes: int, pretrained: bool):
    from torchvision.models import detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    builder = getattr(detection, BACKBONES[arch])
    # 사전학습 가중치는 COCO(91 클래스)용이므로 불러온 뒤 분류 헤드만 우리 클래스 수로 바꾼다
    model = builder(weights="DEFAULT" if pretrained else None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 = 배경
    return model


class RcnnDetector(BaseDetector):
    name = "rcnn"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        self.device = get_device(cfg["train"].get("device"))
        self.model = _build_torchvision_model(
            cfg["model"].get("arch", "fasterrcnn_resnet50_fpn_v2"),
            num_classes,
            cfg["model"].get("pretrained", True),
        ).to(self.device)
        self.model.roi_heads.detections_per_img = cfg["eval"].get("max_det", MAX_PILLS_PER_IMAGE)

    def _loader(self, split: str, train: bool) -> DataLoader:
        d, t = self.cfg["data"], self.cfg["train"]
        records, _ = load_records(d["train_annotations"])
        records = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
        ds = PillDataset(
            records,
            d["train_images"],
            load_class_map(d["class_map"]),
            transforms=get_transforms(self.cfg.get("augment"), train=train),
        )
        return DataLoader(
            ds,
            batch_size=t["batch_size"],
            shuffle=train,
            num_workers=t.get("num_workers", 2),
            collate_fn=collate_fn,
        )

    def fit(self, out_dir: Path) -> Path:
        t = self.cfg["train"]
        loader = self._loader("train", train=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if t.get("optimizer", "sgd").lower() == "adamw":
            optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t["epochs"]))

        history = []
        for epoch in range(1, t["epochs"] + 1):
            self.model.train()
            total = 0.0
            for images, targets in tqdm(loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
                images = [img.to(self.device) for img in images]
                tv_targets = [
                    {
                        "boxes": tg["boxes"].to(self.device),
                        "labels": (tg["labels"] + 1).to(self.device),
                    }
                    for tg in targets
                ]
                losses = self.model(images, tv_targets)
                loss = sum(losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
            scheduler.step()
            avg = total / max(1, len(loader))
            history.append({"epoch": epoch, "train_loss": round(avg, 5)})
            print(f"[rcnn] epoch {epoch}: train_loss={avg:.4f}")

        # 스켈레톤은 마지막 epoch을 best로 저장. 고도화: epoch마다 검증 mAP로 best 선택
        best = out_dir / "best.pt"
        torch.save(self.model.state_dict(), best)
        save_json(history, out_dir / "train_history.json")
        return best

    def load(self, weights: str | Path) -> None:
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        ev = self.cfg["eval"]
        conf = ev["conf_thr"] if conf_thr is None else conf_thr
        self.model.eval()
        preds: list[Prediction] = []
        batch = self.cfg["train"].get("batch_size", 4)
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i : i + batch]
            images = [
                torch.from_numpy(read_image_rgb(p))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .to(self.device)
                for p in chunk
            ]
            outputs = self.model(images)
            for path, out in zip(chunk, outputs, strict=True):
                keep = out["scores"] >= conf
                preds.append(
                    Prediction(
                        image_id=image_id_from_name(Path(path).name),
                        file_name=Path(path).name,
                        boxes=out["boxes"][keep].cpu().float(),
                        scores=out["scores"][keep].cpu().float(),
                        labels=(out["labels"][keep] - 1).cpu().long(),  # 배경 +1 되돌리기
                    )
                )
        return preds
