"""Face-crop quality: blur, exposure, size, and in-plane roll.

The composite is a heuristic for ranking detections, not a calibrated
probability. Blur is the variance of the Laplacian (Pech-Pacheco et al.).
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.integral import as_gray_float


def assess(image: np.ndarray, roll: float | None = None) -> dict:
    gray = as_gray_float(image)
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lap = float(cv2.Laplacian(u8, cv2.CV_64F).var())
    mean = float(u8.mean())
    blur_score = float(np.clip(np.log1p(lap) / np.log1p(500.0), 0.0, 1.0))
    exposure_score = float(1.0 - min(abs(mean - 128.0) / 128.0, 1.0))
    short_side = float(min(u8.shape[:2]))
    size_score = float(np.clip(short_side / 120.0, 0.0, 1.0))
    if roll is None:
        roll_score = 1.0
        roll_value = None
    else:
        roll_value = float(roll)
        roll_score = float(np.clip(1.0 - abs(roll_value) / 40.0, 0.0, 1.0))
    score = 0.40 * blur_score + 0.25 * exposure_score + 0.20 * size_score + 0.15 * roll_score
    return {
        "blur": round(lap, 2),
        "blur_score": round(blur_score, 4),
        "exposure": round(mean, 2),
        "exposure_score": round(exposure_score, 4),
        "size_score": round(size_score, 4),
        "roll": None if roll_value is None else round(roll_value, 2),
        "score": round(float(score), 4),
    }
