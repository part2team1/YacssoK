"""탐지 + 각인 OCR로 알약 이름 다시 고르기 (실험 D11, 데이터 A·모델 B).

1단계: 이미 학습이 끝난 실험(model.detector_exp: YOLO, R-CNN, two_stage 모두 가능)이 박스를 찾는다.
2단계: 확신도가 낮은 박스(score < ocr_max_score)만 잘라 각인을 읽고, 약 정보표의 식별문자와 비교한다.
  - 읽은 글자가 어떤 알약과 ocr_min_sim 이상 비슷하면 그 알약으로 이름을 정한다.
  - 점수 = (1 - ocr_weight) x 탐지 점수 + ocr_weight x 각인 비슷한 정도
  - 글자를 못 읽었거나 비슷한 알약이 없으면 1단계 결과를 그대로 둔다.
박스 위치는 바꾸지 않는다. 학습은 하지 않으므로 run_train.py는 설정 저장과 검증 채점만 한다.

필요: uv sync --extra ocr, data/drug_info.json (scripts/build_drug_info.py)
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.dataset import read_image_rgb
from src.interfaces import Prediction
from src.models.base import BaseDetector
from src.utils import load_class_map, resolve_path, save_json


class OcrRerankDetector(BaseDetector):
    name = "ocr_rerank"

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__(cfg, num_classes)
        self.detector = None
        self.ocr = None
        self.table: dict[int, list[str]] = {}

    def fit(self, out_dir: Path) -> Path:
        # 학습할 것이 없다. 어떤 탐지 실험과 묶였는지만 best.pt 자리에 기록한다
        best = out_dir / "best.pt"
        save_json({"detector_exp": self.cfg["model"]["detector_exp"]}, best)
        return best

    def load(self, weights: str | Path) -> None:
        from src.infer import load_experiment  # 실험 폴더에서 1단계 모델을 읽는다
        from src.ocr import PillOCR, imprint_table

        m, d = self.cfg["model"], self.cfg["data"]
        _, self.detector = load_experiment(m["detector_exp"])
        if self.detector.num_classes != self.num_classes:
            raise ValueError(
                "탐지 실험과 클래스 수가 다릅니다. 같은 class_map으로 학습한 실험을 쓰세요."
            )
        self.ocr = PillOCR(min_conf=float(m.get("ocr_min_conf", 0.5)))
        drug_info = m.get("drug_info") or resolve_path(d["class_map"]).parent / "drug_info.json"
        self.table = imprint_table(load_class_map(d["class_map"]), drug_info)

    @torch.no_grad()
    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        from src.ocr import match

        m = self.cfg["model"]
        max_score = float(m.get("ocr_max_score", 0.6))
        min_sim = float(m.get("ocr_min_sim", 0.8))
        w = float(m.get("ocr_weight", 0.5))
        self.detector.cfg["eval"].update(self.cfg["eval"])  # --tta 등을 1단계에도 적용
        out = []
        for path, p in zip(image_paths, self.detector.predict(image_paths, conf_thr), strict=True):
            todo = (p["scores"] < max_score).nonzero().flatten().tolist()
            if not todo:
                out.append(p)
                continue
            img = read_image_rgb(path)
            scores, labels = p["scores"].clone(), p["labels"].clone()
            for i in todo:
                text, _ = self.ocr.read(img, p["boxes"][i].tolist())
                ranked = match(text, self.table)
                if ranked and ranked[0][1] >= min_sim:
                    labels[i] = ranked[0][0]
                    scores[i] = (1 - w) * scores[i] + w * ranked[0][1]
            out.append(Prediction(**{**p, "scores": scores, "labels": labels}))
        return out
