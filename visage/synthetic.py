"""Procedural faces for the classical learner.

These are not a substitute for a real training corpus. They exist so the
AdaBoost, HOG–SVM, and subspace experiments are reproducible without
downloading a face dataset, and so unit tests have known ground truth.
Identity is a deterministic function of an integer id (structure + speckles).
Sample-level jitter is illumination, noise, and a one-pixel shift.
"""

from __future__ import annotations

import cv2
import numpy as np


def _identity_rng(identity: int) -> np.random.RandomState:
    return np.random.RandomState(10_000 + int(identity) * 97)


def render_face(
    size: int,
    identity: int,
    rng: np.random.Generator,
    *,
    jitter: bool = True,
) -> np.ndarray:
    """Return a float32 image in [0, 1], shape (size, size)."""
    rs = _identity_rng(identity)
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 0.18, np.float32)
    skin = 0.62 + 0.2 * rs.rand()
    cx = size * (0.50 + 0.04 * (rs.rand() - 0.5))
    cy = size * (0.48 + 0.04 * (rs.rand() - 0.5))
    ax = size * (0.30 + 0.05 * rs.rand())
    ay = size * (0.38 + 0.04 * rs.rand())
    eye_dx = size * (0.11 + 0.04 * rs.rand())
    eye_y = cy - size * (0.07 + 0.03 * rs.rand())
    eye_r = size * (0.040 + 0.012 * rs.rand())
    mouth_w = size * (0.09 + 0.04 * rs.rand())
    mouth_y = cy + size * (0.14 + 0.04 * rs.rand())
    if jitter:
        cx += float(rng.uniform(-1.2, 1.2))
        cy += float(rng.uniform(-1.2, 1.2))
        eye_y += float(rng.uniform(-0.6, 0.6))
    ellipse = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
    img[ellipse] = skin
    light = 0.82 + 0.28 * (xx / max(size - 1, 1))
    if jitter:
        light = light * float(rng.uniform(0.85, 1.1))
        light = light + float(rng.uniform(-0.05, 0.05))
    img[ellipse] *= light[ellipse]
    for sign in (-1.0, 1.0):
        ex = cx + sign * eye_dx
        eye = (xx - ex) ** 2 + (yy - eye_y) ** 2 <= eye_r ** 2
        img[eye] = 0.07
    mouth = (np.abs(yy - mouth_y) <= max(size * 0.018, 1)) & (np.abs(xx - cx) <= mouth_w)
    img[mouth] = 0.10
    speckles = (rs.rand(size, size) > 0.78).astype(np.float32) * 0.18
    img[ellipse] = np.clip(img[ellipse] - speckles[ellipse], 0, 1)
    # Mid-frequency identity texture. Eigenfaces can live on shading alone;
    # uniform LBP cannot, because flat skin collapses to one code.
    tex = rs.randn(size, size).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.9)
    tex /= float(np.std(tex) + 1e-6)
    img[ellipse] = np.clip(img[ellipse] + 0.07 * tex[ellipse], 0, 1)
    if jitter:
        img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def render_negative(size: int, rng: np.random.Generator) -> np.ndarray:
    img = rng.uniform(0.15, 0.85, (size, size)).astype(np.float32)
    sigma = float(rng.uniform(0.6, 2.2))
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    for _ in range(int(rng.integers(2, 7))):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        w = int(rng.integers(2, max(3, size // 2)))
        h = int(rng.integers(2, max(3, size // 2)))
        img[y : y + h, x : x + w] = float(rng.uniform(0, 1))
    return np.clip(img, 0, 1).astype(np.float32)


def classification_set(
    n_pos: int,
    n_neg: int,
    size: int,
    seed: int,
    n_identities: int = 12,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels = []
    for i in range(n_pos):
        images.append(render_face(size, int(rng.integers(0, n_identities)), rng))
        labels.append(1.0)
    for _ in range(n_neg):
        images.append(render_negative(size, rng))
        labels.append(-1.0)
    return images, np.asarray(labels, dtype=np.float64)


def recognition_set(
    n_identities: int,
    n_each: int,
    size: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels = []
    for ident in range(n_identities):
        for _ in range(n_each):
            images.append(render_face(size, ident, rng, jitter=True))
            labels.append(ident)
    return images, np.asarray(labels, dtype=np.int32)


def detection_canvas(
    rng: np.random.Generator,
    *,
    canvas: int = 96,
    win: int = 24,
    origin: tuple[int, int] = (36, 28),
    identity: int = 3,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Plant one training-distribution face on a flat field. Box is ground truth."""
    image = np.full((canvas, canvas), 0.18, np.float32)
    image += rng.normal(0, 0.01, image.shape).astype(np.float32)
    face = render_face(win, identity, rng, jitter=False)
    x, y = origin
    image[y : y + win, x : x + win] = face
    image = np.clip(image, 0, 1).astype(np.float32)
    return image, (x, y, win, win)
