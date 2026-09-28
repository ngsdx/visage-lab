"""Sliding-window search with the learned AdaBoost face classifier."""

from __future__ import annotations

import numpy as np

from visage.adaboost import StrongClassifier
from visage.geometry import Detection, nms
from visage.haar import HaarFeature, window_responses
from visage.integral import as_gray_float, integral_image


def sliding_window_detect(
    image: np.ndarray,
    model: StrongClassifier,
    features: list[HaarFeature],
    *,
    win: int = 24,
    step: int = 4,
    threshold: float = 0.0,
    nms_iou: float = 0.3,
) -> list[Detection]:
    """Single-scale scan. The window matches the size the stumps were trained at."""
    gray = as_gray_float(image)
    if gray.shape[0] < win or gray.shape[1] < win:
        return []
    ii = integral_image(gray)
    ii2 = integral_image(gray * gray)
    found: list[Detection] = []
    h, w = gray.shape
    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            column = window_responses(ii, ii2, features, x, y, win)
            score = float(model.decision_function(column)[0])
            if score >= threshold:
                found.append(
                    Detection(x=x, y=y, w=win, h=win, score=score, source="adaboost")
                )
    return nms(found, nms_iou)
