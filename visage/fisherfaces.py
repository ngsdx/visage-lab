"""Fisherfaces (Belhumeur, Hespanha, Kriegman, PAMI 1997).

PCA first reduces the pixel space to N − C dimensions so the within-class
scatter is full rank, then LDA solves the generalized eigenproblem
S_b v = λ S_w v and keeps C − 1 directions. Nearest neighbor classifies.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import linalg

from visage.integral import as_gray_float


def _prepare(image: np.ndarray, size: int) -> np.ndarray:
    gray = as_gray_float(image)
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float64).ravel()


class Fisherfaces:
    def __init__(self, size: int = 32):
        self.size = size
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.proj_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, images: list[np.ndarray], labels: list | np.ndarray) -> "Fisherfaces":
        x = np.stack([_prepare(im, self.size) for im in images])
        y = np.asarray(labels)
        classes = np.unique(y)
        n, _d = x.shape
        c = len(classes)
        if c < 2:
            raise ValueError("fisherfaces needs at least two classes")
        if n <= c:
            raise ValueError("fisherfaces needs more images than classes")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n - c, vt.shape[0])
        k = max(k, 1)
        wpca = vt[:k].T
        xp = centered @ wpca
        overall = xp.mean(axis=0)
        sw = np.zeros((k, k), dtype=np.float64)
        sb = np.zeros((k, k), dtype=np.float64)
        for cls in classes:
            xi = xp[y == cls]
            if len(xi) == 0:
                continue
            mean_i = xi.mean(axis=0)
            diff = xi - mean_i
            sw += diff.T @ diff
            delta = (mean_i - overall).reshape(-1, 1)
            sb += len(xi) * (delta @ delta.T)
        sw += 1e-3 * np.eye(k)
        evals, evecs = linalg.eigh(sb, sw)
        order = np.argsort(evals)[::-1]
        m = max(1, min(c - 1, k))
        wlda = np.real(evecs[:, order[:m]])
        self.W_ = wpca @ wlda
        self.proj_ = centered @ self.W_
        self.labels_ = y
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.W_ is None:
            raise RuntimeError("model is not fit")
        vec = _prepare(image, self.size)
        return (vec - self.mean_) @ self.W_

    def predict(self, image: np.ndarray) -> tuple[object, float]:
        if self.proj_ is None or self.labels_ is None:
            raise RuntimeError("model is not fit")
        z = self.transform(image)
        dist = np.linalg.norm(self.proj_ - z, axis=1)
        i = int(np.argmin(dist))
        return self.labels_[i], float(dist[i])
