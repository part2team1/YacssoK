"""Faster R-CNN 탐지기 (모델 B 담당). torchvision 모델을 PyTorch 학습 루프로 직접 학습한다.

주의: torchvision 탐지 모델은 0번 클래스를 '배경'으로 쓴다.
      팀 표준 label_idx에 1을 더해 넣고, 예측을 내보낼 때 1을 뺀다. 이 변환은 이 파일 안에서만 한다.

고도화 아이디어 (실험 계획 M3, T2, M4): 백본 비교, 클래스 불균형 대응, 탐지 + 분류 2단계 방식.

[변경 이력] 검증 mAP 기준 best 저장 (고도화)
  - 이전: 학습이 끝나면 '마지막 epoch' 가중치를 best.pt로 저장했다.
          마지막 epoch이 가장 좋다는 보장이 없다 (과적합되면 오히려 나빠진다).
  - 이후: epoch마다 val 데이터로 mAP를 재고, 가장 높았던 epoch의 가중치만 best.pt로 남긴다.
  - fit() / load() / predict()의 이름과 입출력 형식은 그대로다 (팀 규칙).
    fit()은 여전히 out_dir / "best.pt" 경로를 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PillDataset, collate_fn, load_records, load_split, read_image_rgb
from src.interfaces import MAX_PILLS_PER_IMAGE, Prediction
# [추가] 검증 mAP를 계산하는 팀 공용 함수. best 선택 기준을 loss가 아니라
#        실제 평가 지표(mAP)로 삼기 위해 가져온다.
from src.metrics import compute_map
from src.models.base import BaseDetector
from src.transforms import get_transforms
# [추가] resolve_path: 설정 파일의 상대 경로를 실제 경로로 바꿔 준다.
#        predict()는 이미지 '경로 목록'을 받으므로, val 이미지의 전체 경로를 만드는 데 쓴다.
#        새 줄을 만들지 않고 기존 src.utils import 줄에 이어 붙였다.
from src.utils import get_device, image_id_from_name, load_class_map, resolve_path, save_json

# 설정(cfg["model"]["arch"])에 쓸 이름 → torchvision 생성 함수 이름.
# 백본 비교 실험(M3)을 할 때 여기에 항목만 추가하면 된다.
BACKBONES = {
    "fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
}


def _build_torchvision_model(arch: str, num_classes: int, pretrained: bool):
    """torchvision Faster R-CNN을 만들고, 분류 헤드를 우리 클래스 수에 맞게 교체한다."""
    from torchvision.models import detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    builder = getattr(detection, BACKBONES[arch])
    # 사전학습 가중치는 COCO(91 클래스)용이므로 불러온 뒤 분류 헤드만 우리 클래스 수로 바꾼다
    # weights_backbone=None: 전체 모델 가중치(weights)를 이미 불러오므로 백본만 따로 받을 필요 없다
    model = builder(weights="DEFAULT" if pretrained else None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 = 배경
    return model


class RcnnDetector(BaseDetector):
    """팀 공통 인터페이스(BaseDetector)를 따르는 Faster R-CNN 래퍼.

    외부(학습/평가 스크립트)는 fit / load / predict 세 함수만 호출한다.
    """

    name = "rcnn"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        self.device = get_device(cfg["train"].get("device"))
        self.model = _build_torchvision_model(
            cfg["model"].get("arch", "fasterrcnn_resnet50_fpn_v2"),
            num_classes,
            cfg["model"].get("pretrained", True),
        ).to(self.device)
        # 이미지 한 장당 최대 박스 수. 알약은 한 장에 몇 개뿐이므로 팀 상한값으로 자른다
        self.model.roi_heads.detections_per_img = cfg["eval"].get("max_det", MAX_PILLS_PER_IMAGE)

    def _loader(self, split: str, train: bool) -> DataLoader:
        """split 파일(train.txt 등)에 적힌 이미지만 골라 DataLoader를 만든다."""
        d, t = self.cfg["data"], self.cfg["train"]
        records, _ = load_records(d["train_annotations"])
        records = load_split(records, f"{d['split_dir']}/{d['version']}/{split}.txt")
        ds = PillDataset(
            records,
            d["train_images"],
            load_class_map(d["class_map"]),
            transforms=get_transforms(self.cfg.get("augment"), train=train),  # train일 때만 증강
        )
        return DataLoader(
            ds,
            batch_size=t["batch_size"],
            shuffle=train,
            num_workers=t.get("num_workers", 2),
            # 탐지 데이터는 이미지마다 박스 개수가 달라 기본 collate로 묶을 수 없어 전용 함수를 쓴다
            collate_fn=collate_fn,
        )

    # [추가] 검증 데이터 준비 함수
    #   왜 필요한가: epoch마다 검증 mAP를 재려면 (1) predict()에 넣을 val 이미지 경로와
    #               (2) compute_map()에 넣을 정답 박스/라벨이 필요하다.
    #   왜 DataLoader를 안 쓰나: predict()가 이미지 '경로'를 받아 직접 읽기 때문이다.
    #               그래서 경로 목록과 정답 목록만 따로 만든다.
    #   증강(transforms)을 주지 않은 이유: 검증은 원본 이미지 그대로 평가해야 공정하다.
    #   정답 라벨은 팀 표준 label_idx 그대로다 (+1 안 함). predict()도 -1 해서 같은 기준으로 내보내므로 맞는다.
    #   이름이 _로 시작: 외부 인터페이스가 아닌 내부용 함수라는 뜻이다.
    def _val_data(self):
        d = self.cfg["data"]
        records, _ = load_records(d["train_annotations"])
        val = load_split(records, f"{d['split_dir']}/{d['version']}/val.txt")
        ds = PillDataset(val, d["train_images"], load_class_map(d["class_map"]))
        paths = [resolve_path(d["train_images"]) / r["file_name"] for r in val]
        # ds[i]는 (이미지, 정답) 튜플이므로 [1]로 정답만 모은다. paths와 순서가 같다
        return paths, [ds[i][1] for i in range(len(ds))]

    def fit(self, out_dir: Path) -> Path:
        """학습하고 best 가중치 파일 경로를 반환한다. (이름/입출력 형식 변경 없음)"""
        t = self.cfg["train"]
        loader = self._loader("train", train=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if t.get("optimizer", "sgd").lower() == "adamw":
            optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(params, lr=t["lr"], momentum=0.9, weight_decay=1e-4)
        # 학습률을 코사인 곡선으로 epoch마다 서서히 줄인다
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t["epochs"]))

        # [추가] 검증 데이터는 epoch마다 같으므로 루프 전에 한 번만 준비한다 (매번 읽으면 느리다).
        #        이름에 val_ 를 붙인 이유: 아래 학습 루프가 이미 targets 변수를 쓰고 있어서,
        #        같은 이름을 쓰면 학습 루프가 돌 때 검증 정답이 덮어써진다.
        val_paths, val_targets = self._val_data()
        # [추가] 지금까지 가장 좋은 mAP. mAP는 0 이상이므로 -1로 시작하면
        #        첫 epoch은 무조건 저장된다 → best.pt가 항상 만들어진다.
        best_score = -1.0

        history = []
        for epoch in range(1, t["epochs"] + 1):
            # 아래 검증 단계에서 predict()가 모델을 eval 모드로 바꿔 놓는다.
            # 매 epoch 시작에 train 모드로 되돌려야 loss 계산(학습)이 된다.
            self.model.train()
            total = 0.0
            for images, targets in tqdm(loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
                images = [img.to(self.device) for img in images]
                tv_targets = [
                    {
                        "boxes": tg["boxes"].to(self.device),
                        "labels": (tg["labels"] + 1).to(self.device),  # 0번은 배경이라 +1
                    }
                    for tg in targets
                ]
                # train 모드의 torchvision 탐지 모델은 예측 대신 loss 딕셔너리를 돌려준다
                losses = self.model(images, tv_targets)
                loss = sum(losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
            scheduler.step()
            avg = total / max(1, len(loader))
            history.append({"epoch": epoch, "train_loss": round(avg, 5)})

            # [추가] epoch이 끝날 때마다 검증 mAP를 재서 best를 고른다
            #   왜 바꿨나: 원래 주석의 지시("고도화: epoch마다 검증 mAP로 best 선택")를 구현한 것.
            #             train_loss가 낮아도 val 성능은 떨어질 수 있으므로(과적합),
            #             대회/평가 지표와 같은 mAP로 판단하는 게 정확하다.
            #   predict()를 그대로 재사용: 실제 추론과 똑같은 경로(conf_thr, 라벨 -1 변환)로 평가된다.
            score = compute_map(self.predict(val_paths), val_targets,
                                self.cfg["eval"].get("iou_thresholds"))["map"]
            if score > best_score:  # 기록을 넘었을 때만 저장 → best.pt에는 항상 최고 epoch만 남는다
                best_score = score
                torch.save(self.model.state_dict(), out_dir / "best.pt")
            history[-1]["val_map"] = score  # 방금 추가한 이번 epoch 기록에 val_map을 덧붙인다
            # [변경] 로그에 val_map도 함께 출력 (원래는 train_loss만 출력)
            print(f"[rcnn] epoch {epoch}: train_loss={avg:.4f} val_map={score:.4f}")

        # [삭제] 원래 여기 있던 두 줄:
        #          best = out_dir / "best.pt"
        #          torch.save(self.model.state_dict(), best)
        #        이 코드는 '마지막 epoch'을 best.pt에 덮어쓴다. 남겨 두면 위에서 고른 최고 epoch이
        #        사라지므로 삭제했다.
        save_json(history, out_dir / "train_history.json")  # epoch별 train_loss, val_map 기록
        # [변경] 원래는 'return best'였다. 삭제한 줄이 best 변수를 만들고 있었으므로
        #        같은 경로를 직접 반환한다. 반환값(경로)은 이전과 완전히 같다.
        # 주의: fit()이 끝난 뒤 self.model에는 '마지막 epoch' 가중치가 들어 있다.
        #       학습 직후 바로 예측하려면 먼저 load(반환된 경로)를 호출해야 best로 예측된다.
        return out_dir / "best.pt"

    def load(self, weights: str | Path) -> None:
        """저장된 가중치를 불러와 추론 준비(eval 모드)를 한다. (변경 없음)"""
        # weights_only=True: 가중치 텐서만 불러온다 (임의 코드 실행 방지)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()  # 추론에서는 기울기가 필요 없으므로 메모리/속도를 아낀다
    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        """이미지 경로 목록을 받아 팀 표준 Prediction 목록을 반환한다. (변경 없음)

        fit()의 검증 단계에서도 이 함수를 그대로 호출한다.
        """
        ev = self.cfg["eval"]
        conf = ev["conf_thr"] if conf_thr is None else conf_thr
        self.model.eval()  # 이 줄 때문에 fit()의 매 epoch 시작에 train()으로 되돌린다
        preds: list[Prediction] = []
        # 참고: 학습용 batch_size를 그대로 쓴다. eval은 메모리를 덜 써서 보통 괜찮지만,
        #       검증 중 메모리 부족이 나면 이 값을 따로 설정하는 걸 고려한다.
        batch = self.cfg["train"].get("batch_size", 4)
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i : i + batch]
            # HWC uint8(0~255) → CHW float(0~1): torchvision 모델의 입력 형식
            images = [
                torch.from_numpy(read_image_rgb(p))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .to(self.device)
                for p in chunk
            ]
            outputs = self.model(images)  # eval 모드에서는 이미지별 boxes/scores/labels를 돌려준다
            for path, out in zip(chunk, outputs, strict=True):
                keep = out["scores"] >= conf  # 신뢰도 임계값 미만 박스는 버린다
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





# **넣은 위치**
# 1. 상단 import에 `from src.metrics import compute_map`를 넣었고, `resolve_path`는 원래 있던 `from src.utils import ...` 줄에 같이 넣었어요.
# 2. `_val_data()` 메서드는 `_loader()` 바로 다음에 넣었어요.
# 3. `fit()`의 `history = []` 위에서 `val_paths, val_targets`와 `best_score`를 준비해요.
# 4. 각 epoch의 `history.append(...)` 다음에서 검증 mAP를 계산하고, 최고 기록일 때만 `best.pt`를 저장해요. epoch마다 출력하는 print에도 `val_map`이 같이 나오게 했어요.
# 5. 루프 뒤에 있던 "마지막 epoch 저장" 두 줄은 지웠어요. 그런데 그 줄이 `return best`에서 쓰는 `best` 변수도 만들고 있었기 때문에, `return out_dir / "best.pt"`로 바꾸고 `train_history.json` 저장은 그대로 남겨 두었어요.

# **알아두면 좋은 점**
# • `predict()`가 모델을 eval 모드로 바꿔 놓지만, 매 epoch 시작마다 `self.model.train()`을 다시 호출하니까 학습에는 문제가 없어요.
# • 검증할 때 `predict()`가 **학습용 `batch_size`**를 그대로 써요. eval 모드는 메모리를 덜 쓰니까 보통은 괜찮은데, 메모리가 부족하면 `eval`에 따로 배치 크기 설정을 두는 게 좋아요.
# • `fit()`이 끝난 뒤 모델 안에 남아 있는 건 마지막 epoch의 가중치이고, 저장된 best가 아니에요. 학습 직후 바로 예측까지 돌린다면 그 전에 `load(best경로)`를 꼭 호출하세요.
# • 검증 mAP를 매 epoch 계산하니까 val 데이터가 크면 학습이 그만큼 느려져요.