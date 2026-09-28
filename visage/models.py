"""Fetch the OpenCV Zoo weights used as the modern baseline.

YuNet (face_detection_yunet_2023mar.onnx) and SFace
(face_recognition_sface_2021dec.onnx). They are not redistributed in this
repository; the first call downloads them into ``models/``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def yunet_path() -> Path:
    return _fetch(YUNET_URL, MODEL_DIR / "face_detection_yunet_2023mar.onnx")


def sface_path() -> Path:
    return _fetch(SFACE_URL, MODEL_DIR / "face_recognition_sface_2021dec.onnx")
