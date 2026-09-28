"""SFace embeddings (Zhong et al., 2021) via OpenCV's FaceRecognizerSF.

Used only as a verification baseline. The classical recognizers do not
depend on these weights.
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.models import sface_path

# Cosine threshold published with the OpenCV Zoo demo.
COSINE_THRESHOLD = 0.363

_RECOGNIZER = None


def recognizer():
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = cv2.FaceRecognizerSF_create(str(sface_path()), "")
    return _RECOGNIZER


def embed(bgr: np.ndarray, yunet_row: list[float] | np.ndarray) -> np.ndarray:
    rec = recognizer()
    row = np.asarray(yunet_row, dtype=np.float32).reshape(-1)
    aligned = rec.alignCrop(bgr, row)
    feat = rec.feature(aligned)
    return np.asarray(feat, dtype=np.float32).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    rec = recognizer()
    return float(rec.match(a.reshape(1, -1), b.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
