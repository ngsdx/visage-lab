"""Production detectors.

* Viola–Jones cascades shipped with OpenCV (frontal default, alt2, and
  profile, including a horizontal flip so the profile cascade sees both
  directions).
* YuNet (Wu, Peng, Yu), the millisecond OpenCV DNN detector, which also
  returns five landmarks.
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.geometry import Detection
from visage.models import yunet_path

_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _CASCADES:
        path = cv2.data.haarcascades + name
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise RuntimeError(f"failed to load cascade {path}")
        _CASCADES[name] = clf
    return _CASCADES[name]


def _from_rects(rects, weights, source: str) -> list[Detection]:
    found = []
    if rects is None or len(rects) == 0:
        return found
    for rect, weight in zip(rects, weights):
        x, y, w, h = (int(v) for v in rect)
        found.append(
            Detection(
                x=x,
                y=y,
                w=w,
                h=h,
                # Cascade "level weights" are not probabilities. Squash them
                # into (0, 1) so fusion can compare them with YuNet scores.
                score=float(weight) / (float(weight) + 2.0) if weight > 0 else 0.0,
                source=source,
            )
        )
    return found


def _run_cascade(gray: np.ndarray, name: str, source: str, min_size: int) -> list[Detection]:
    clf = _cascade(name)
    rects, _rejects, weights = clf.detectMultiScale3(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_size, min_size),
        outputRejectLevels=True,
    )
    return _from_rects(rects, weights, source)


def detect_haar(bgr: np.ndarray, min_size: int = 48) -> list[Detection]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    found: list[Detection] = []
    found += _run_cascade(gray, "haarcascade_frontalface_default.xml", "haar", min_size)
    found += _run_cascade(gray, "haarcascade_frontalface_alt2.xml", "haar-alt2", min_size)
    found += _run_cascade(gray, "haarcascade_profileface.xml", "profile", min_size)
    flipped = cv2.flip(gray, 1)
    width = gray.shape[1]
    for det in _run_cascade(flipped, "haarcascade_profileface.xml", "profile-flip", min_size):
        det.x = width - det.x - det.w
        found.append(det)
    return found


def detect_eyes(gray_face: np.ndarray) -> list[tuple[int, int, int, int]]:
    if gray_face.size == 0:
        return []
    clf = _cascade("haarcascade_eye.xml")
    rects = clf.detectMultiScale(
        gray_face,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(8, 8),
    )
    if rects is None or len(rects) == 0:
        return []
    return [tuple(int(v) for v in r) for r in rects]


_YUNET = None
_YUNET_SIZE: tuple[int, int] | None = None


def detect_yunet(bgr: np.ndarray, score_threshold: float = 0.6) -> list[Detection]:
    global _YUNET, _YUNET_SIZE
    height, width = bgr.shape[:2]
    if _YUNET is None:
        _YUNET = cv2.FaceDetectorYN_create(
            str(yunet_path()),
            "",
            (width, height),
            score_threshold,
            0.3,
            5000,
        )
        _YUNET_SIZE = (width, height)
    if _YUNET_SIZE != (width, height):
        _YUNET.setInputSize((width, height))
        _YUNET_SIZE = (width, height)
    _YUNET.setScoreThreshold(score_threshold)
    _ok, faces = _YUNET.detect(bgr)
    if faces is None:
        return []
    found = []
    for row in faces:
        x, y, w, h = (float(v) for v in row[:4])
        re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y = (
            float(v) for v in row[4:14]
        )
        # YuNet order: right eye, left eye, nose, right mouth, left mouth
        # (the subject's right, which is image-left on a frontal face).
        found.append(
            Detection(
                x=int(round(x)),
                y=int(round(y)),
                w=max(1, int(round(w))),
                h=max(1, int(round(h))),
                score=float(row[14]),
                source="yunet",
                landmarks={
                    "right_eye": (re_x, re_y),
                    "left_eye": (le_x, le_y),
                    "nose": (nt_x, nt_y),
                    "mouth_right": (rcm_x, rcm_y),
                    "mouth_left": (lcm_x, lcm_y),
                },
                extra={"yunet_row": [float(v) for v in row[:15]]},
            )
        )
    return found
