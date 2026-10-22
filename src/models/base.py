"""모든 모델의 공통 부모 클래스 (PM·실험리드 Y 담당).

모델 A·B는 이 클래스를 상속해 fit / predict / load 세 가지만 구현한다.
평가·앙상블·제출·데모 코드는 이 세 함수만 부르므로 모델 종류를 몰라도 된다.
"""

from __future__ import annotations

from pathlib import Path

from src.interfaces import Prediction


class BaseDetector:
    name: str = "base"

    def __init__(self, cfg: dict, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes

    def fit(self, out_dir: Path) -> Path:
        """cfg의 데이터로 학습하고 가장 좋은 가중치 경로(out_dir/best.pt)를 돌려준다."""
        raise NotImplementedError

    def load(self, weights: str | Path) -> None:
        """학습된 가중치를 읽어 predict 할 준비를 한다."""
        raise NotImplementedError

    def predict(self, image_paths: list[Path], conf_thr: float | None = None) -> list[Prediction]:
        """이미지마다 Prediction 하나. 반드시 팀 표준 형식(xyxy, label_idx)으로 돌려준다."""
        raise NotImplementedError
