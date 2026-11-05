"""알약 각인(식별문자) OCR 도우미 (실험 D11, 데이터 A).

알약 사진을 잘라 글자를 읽고, 약 정보표(data/drug_info.json)의 식별문자(앞/뒤)와 비교해
어느 알약과 가장 비슷한지 점수를 매긴다. 라벨 점검(find_label_issues.py --ocr)과
예측 이름 다시 고르기(src/models/ocr_rerank.py)가 함께 쓴다.

OCR 엔진: RapidOCR (pip 패키지 안에 모델이 들어 있어 따로 내려받지 않음)
    설치: uv sync --extra ocr
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from src.interfaces import ClassInfo
from src.utils import load_json, resolve_path

FRONT, BACK = "식별문자(앞)", "식별문자(뒤)"  # build_drug_info.py가 만든 항목 이름


def normalize(text: str | None) -> str:
    """대소문자·공백·기호를 무시하고 영문 대문자와 숫자만 남긴다 (예: 'ty 500' → 'TY500')."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def imprint_table(class_map: list[ClassInfo], drug_info_path) -> dict[int, list[str]]:
    """label_idx → 비교할 각인 문자열 목록 (앞, 뒤, 앞+뒤, 뒤+앞). 각인이 없는 알약은 빠진다."""
    path = resolve_path(drug_info_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 이 없습니다. 먼저 scripts/build_drug_info.py를 실행하세요."
        )
    info = load_json(path)
    table: dict[int, list[str]] = {}
    for c in class_map:
        row = info.get(str(c["category_id"]), {})
        front, back = normalize(row.get(FRONT)), normalize(row.get(BACK))
        options = {s for s in (front, back, front + back, back + front) if len(s) >= 2}
        if options:
            table[c["label_idx"]] = sorted(options)
    return table


def match(text: str, table: dict[int, list[str]]) -> list[tuple[int, float]]:
    """읽은 글자와 각 알약 각인의 비슷한 정도(0~1)를 높은 순서로."""
    if len(text) < 2:
        return []
    scores = [
        (label, max(SequenceMatcher(None, text, s).ratio() for s in options))
        for label, options in table.items()
    ]
    return sorted(scores, key=lambda x: -x[1])


class PillOCR:
    """알약 하나를 잘라 4방향으로 돌려 읽고, 가장 확신하는 결과의 글자를 돌려준다."""

    def __init__(self, min_conf: float = 0.5, min_side: int = 160):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError("OCR 패키지가 없습니다. 실행: uv sync --extra ocr") from e
        self.engine = RapidOCR()
        self.min_conf = min_conf
        self.min_side = min_side

    def crop(self, img: np.ndarray, box, pad: float = 0.05) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(x2 + pad * bw))
        y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(y2 + pad * bh))
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            return c
        scale = max(1.0, self.min_side / min(c.shape[:2]))  # 작은 알약은 키워서 읽는다
        return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def read(self, img: np.ndarray, box) -> tuple[str, float]:
        """(정리된 글자, 확신도). 못 읽으면 ('', 0)."""
        c = self.crop(img, box)
        if c.size == 0:
            return "", 0.0
        best = ("", 0.0)
        for k in range(4):
            result, _ = self.engine(np.ascontiguousarray(np.rot90(c, k)))
            if not result:
                continue
            parts = [(normalize(r[1]), float(r[2])) for r in result if float(r[2]) >= self.min_conf]
            text = "".join(t for t, _ in parts)
            conf = sum(s for _, s in parts) / len(parts) if parts else 0.0
            if len(text) >= 2 and conf > best[1]:
                best = (text, conf)
        return best
