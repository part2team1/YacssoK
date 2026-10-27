"""전처리·증강 (데이터 B 담당).

모든 변환은 (img, boxes, labels)를 받아 같은 모양으로 돌려준다.
  img: uint8 RGB [H, W, 3] / boxes: float32 [N, 4] xyxy 절대 픽셀 / labels: int64 [N]
박스를 함께 바꾸지 않는 증강(예: 이미지만 회전)은 정답이 틀어지므로 금지.

고도화 아이디어 (실험 계획 D2~D5): 회전·원근 변환, 약한 색상 변화, Copy-paste.
albumentations를 쓰고 싶으면 Experimentation Lead에게 패키지 추가를 요청하고 이 파일 안에서만 감싼다.
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


class Rotate90:
    """90도 단위 회전 (실험 D3). 박스가 축에 맞춰 그대로 돌아가서 박스가 헐거워지지 않는다.

    임의 각도 회전은 회전된 박스를 감싸는 더 큰 박스가 되어 정답이 헐거워진다.
    mAP@[0.75:0.95]처럼 박스 정확도를 보는 지표에서는 90도 단위가 안전하다.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90, 180, 270도 (시계 반대 방향)
            for _ in range(k):
                w = img.shape[1]
                img = np.rot90(img)  # 반시계 90도: (x, y) → (y, w - x)
                b = boxes.copy()
                boxes = np.stack([b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]], axis=1).astype(
                    np.float32
                )
        return img, boxes, labels


class SaturationValue:
    """채도·명도만 조금 바꾼다 (실험 D3). 색상(hue)은 알약 구분 단서라 그대로 둔다."""

    def __init__(self, saturation: float = 0.3, value: float = 0.2, p: float = 0.5):
        self.saturation, self.value, self.p = saturation, value, p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv[..., 2] *= 1.0 + random.uniform(-self.value, self.value)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img, boxes, labels


class GaussianBlur:
    """약한 흐림 (실험 D3). 초점이 덜 맞은 사진에 대비한다. 각인이 사라질 만큼 세게 하지 않는다."""

    def __init__(self, max_kernel: int = 5, p: float = 0.3):
        self.max_kernel, self.p = max_kernel, p

    def __call__(self, img, boxes, labels):
        if random.random() < self.p:
            k = random.choice([k for k in range(3, self.max_kernel + 1, 2)] or [3])
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), 0)
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
    if cfg_augment.get("rot90", 0) > 0:                       # ← 여기부터 추가
        ops.append(Rotate90(cfg_augment["rot90"]))
    if cfg_augment.get("sat_val", 0) > 0:
        ops.append(SaturationValue(p=cfg_augment["sat_val"]))
    if cfg_augment.get("blur", 0) > 0:
        ops.append(GaussianBlur(p=cfg_augment["blur"]))
    return Compose(ops) if ops else None
