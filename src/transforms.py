"""전처리·증강 (데이터 B 담당).

모든 변환은 (img, boxes, labels)를 받아 같은 모양으로 돌려준다.
  img: uint8 RGB [H, W, 3] / boxes: float32 [N, 4] xyxy 절대 픽셀 / labels: int64 [N]
박스를 함께 바꾸지 않는 증강(예: 이미지만 회전)은 정답이 틀어지므로 금지.

고도화 아이디어 (실험 계획 D2~D5): 회전·원근 변환, 약한 색상 변화, Copy-paste.
albumentations를 쓰고 싶으면 Y에게 패키지 추가를 요청하고 이 파일 안에서만 감싼다.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            w = img.shape[1]
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img, boxes, labels


class VerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            h = img.shape[0]
            img = img[::-1]
            boxes = boxes.copy()
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return img, boxes, labels


class BrightnessContrast:
    """밝기·대비만 바꾼다. 색상(hue)은 알약 구분 단서라 건드리지 않는다."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness, self.contrast, self.p = brightness, contrast, p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta = 255 * random.uniform(-self.brightness, self.brightness)
            img = cv2.convertScaleAbs(np.ascontiguousarray(img), alpha=alpha, beta=beta)
        return img, boxes, labels


def get_transforms(cfg_augment: dict | None, train: bool):
    """configs의 augment 블록으로 변환을 만든다. 검증·추론에는 증강을 쓰지 않는다."""
    if not train or not cfg_augment or cfg_augment.get("name", "none") == "none":
        return None
    ops = []
    if cfg_augment.get("hflip", 0) > 0:
        ops.append(HorizontalFlip(cfg_augment["hflip"]))
    if cfg_augment.get("vflip", 0) > 0:
        ops.append(VerticalFlip(cfg_augment["vflip"]))
    if cfg_augment.get("brightness", 0) > 0:
        ops.append(BrightnessContrast(cfg_augment["brightness"], cfg_augment.get("contrast", 0.2)))
    return Compose(ops) if ops else None
