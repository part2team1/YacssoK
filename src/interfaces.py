"""팀 공통 형식 정의 (Source 작성 표준 4장).

이 파일은 "데이터를 어떤 모양으로 주고받는가"만 정의한다. 함수는 utils.py에 둔다.
변경할 때는 PR + 전원 공지. 가장 아래 계층이므로 프로젝트의 다른 모듈을 import 하지 않는다.

팀 표준
- 박스: xyxy, 원본 이미지 기준 절대 픽셀, float32
- 클래스: label_idx (0부터 연속 정수). 대회 원본 코드(category_id)는 class_map.json으로만 변환
- 이미지: Dataset은 RGB, float32, 0~1, [3, H, W] 텐서를 반환. mean/std 정규화는 모델 쪽에서
- 모델별 형식(YOLO의 정규화 cxcywh, Faster R-CNN의 배경 클래스 +1)은 각 모델 파일 안에서만 변환
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch

# ── 고정 상수 ────────────────────────────────────────────────
SEED = 42
BOX_FORMAT = "xyxy"
MAX_PILLS_PER_IMAGE = 4  # 과제 조건: 사진 한 장에 최대 4개
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ── 어노테이션 레코드 (prepare_data.py가 만드는 이미지 1장의 정보) ──
class Record(TypedDict):
    image_id: int
    file_name: str  # 이미지 파일 이름 (폴더 제외)
    width: int
    height: int
    boxes: list[list[float]]  # [[x1, y1, x2, y2], ...] 팀 표준 xyxy
    category_ids: list[int]  # 대회 원본 코드, boxes와 같은 순서


# ── 클래스 매핑 (data/class_map.json의 한 항목) ───────────────
class ClassInfo(TypedDict):
    label_idx: int  # 학습에 쓰는 번호 (0부터)
    category_id: int  # 대회 원본 코드 (제출할 때 이 값으로 되돌림)
    name: str  # 약 이름 (데모·시각화용)


# ── Dataset 출력 (데이터 B 담당 dataset.py) ──────────────────
class Target(TypedDict):
    boxes: torch.Tensor  # float32 [N, 4], xyxy, 절대 픽셀
    labels: torch.Tensor  # int64 [N], label_idx
    image_id: int
    orig_size: tuple[int, int]  # (H, W)


# ── 모델 예측 출력 (모델 A·B 공통) ───────────────────────────
class Prediction(TypedDict):
    image_id: int
    file_name: str
    boxes: torch.Tensor  # float32 [K, 4], xyxy, 원본 이미지 기준 절대 픽셀
    scores: torch.Tensor  # float32 [K], 0~1
    labels: torch.Tensor  # int64 [K], label_idx
