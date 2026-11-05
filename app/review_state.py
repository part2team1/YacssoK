"""Gradio 취소 API 없이 세션의 현재 사진·기준과 일치하는 결과만 허용한다."""

import hashlib

import numpy as np


def input_key(image, threshold):
    if image is None:
        return None, float(threshold)
    array = np.asarray(image)
    content = hashlib.sha256(array.tobytes()).hexdigest()
    return (array.shape, str(array.dtype), content), float(threshold)


def invalidate(state, image, threshold):
    state["generation"] += 1
    state["input_key"] = input_key(image, threshold)


def current(state, generation, image, threshold):
    return state["generation"] == generation and state.get("input_key") == input_key(
        image, threshold
    )
