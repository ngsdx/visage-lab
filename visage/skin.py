"""Skin-color field in RGB and YCrCb.

The rules are the conjunction of the Kovac / Peer RGB predicate and a
YCrCb box used throughout the face-detection literature (Cb 77–127,
Cr 133–173, Y > 80). They are a proposal cue and a false-positive
filter, not a detector. They fail under strong color cast and they do
not cover every skin tone equally — see the limitations in THEORY.md.
"""

from __future__ import annotations

import cv2
import numpy as np


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 mask, 255 on skin, after a small open/close."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("skin_mask expects a BGR image")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    b, g, r = cv2.split(bgr)
    r_i = r.astype(np.int16)
    g_i = g.astype(np.int16)
    mask_rgb = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r_i - g_i) > 15)
    mask_ycc = (y > 80) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    mask = (mask_rgb & mask_ycc).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def skin_ratio(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    height, width = mask.shape[:2]
    x = max(0, int(x))
    y = max(0, int(y))
    x2 = min(width, x + int(w))
    y2 = min(height, y + int(h))
    roi = mask[y:y2, x:x2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


def skin_proposals(
    mask: np.ndarray,
    *,
    min_area: int = 900,
    aspect: tuple[float, float] = (0.7, 1.45),
) -> list[tuple[int, int, int, int, float]]:
    """Connected components that could be a face. Score is the fill ratio.

    Returned tuples are (x, y, w, h, score). Callers should treat these as
    weak proposals: hands and arms satisfy the same color rules.
    """
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    proposals = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or h == 0:
            continue
        ratio = w / float(h)
        if not (aspect[0] <= ratio <= aspect[1]):
            continue
        fill = area / float(w * h)
        if fill < 0.45:
            continue
        proposals.append((x, y, w, h, float(fill)))
    proposals.sort(key=lambda p: p[4], reverse=True)
    return proposals
