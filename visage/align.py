"""Similarity alignment from a pair of eye centers.

Maps the inter-ocular segment onto a canonical pair so a later subspace
model sees faces in one frame. This is a 4-DOF similarity (scale, rotation,
translation), not a full projective rectification.
"""

from __future__ import annotations

import cv2
import numpy as np


def similarity_matrix(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    out: int = 112,
    dst_left: tuple[float, float] | None = None,
    dst_right: tuple[float, float] | None = None,
) -> np.ndarray:
    """2×3 matrix taking image points to the aligned crop."""
    if dst_left is None:
        dst_left = (out * 0.30, out * 0.38)
    if dst_right is None:
        dst_right = (out * 0.70, out * 0.38)
    sl = np.asarray(left_eye, dtype=np.float64)
    sr = np.asarray(right_eye, dtype=np.float64)
    dl = np.asarray(dst_left, dtype=np.float64)
    dr = np.asarray(dst_right, dtype=np.float64)
    vs = sr - sl
    vd = dr - dl
    src_norm = float(np.linalg.norm(vs))
    if src_norm < 1e-6:
        raise ValueError("eye centers coincide")
    scale = float(np.linalg.norm(vd)) / src_norm
    ang = float(np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0]))
    ca, sa = np.cos(ang), np.sin(ang)
    linear = scale * np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    trans = dl - linear @ sl
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = trans
    return matrix


def roll_degrees(left_eye: tuple[float, float], right_eye: tuple[float, float]) -> float:
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def warp_face(
    image: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    out: int = 112,
) -> np.ndarray:
    matrix = similarity_matrix(left_eye, right_eye, out)
    return cv2.warpAffine(
        image,
        matrix,
        (out, out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def order_eyes(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (image-left, image-right)."""
    if a[0] <= b[0]:
        return a, b
    return b, a
