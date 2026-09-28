"""Discrete AdaBoost over decision stumps (Freund & Schapire; Viola & Jones).

Each stump is one Haar feature, one threshold, and a polarity:
``h(x) = p if f(x) < θ else -p``, with ``p ∈ {+1, -1}``.
The strong classifier is ``sign(Σ α_t h_t)``.

This is the learner, not a full attentional cascade. Viola–Jones then
bootstraps hard negatives into successive stages; that outer loop needs
a large real-image corpus, so it is not reproduced here. See THEORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: float
    alpha: float

    def apply(self, column: np.ndarray) -> np.ndarray:
        return np.where(column < self.threshold, self.polarity, -self.polarity)


@dataclass
class StrongClassifier:
    stumps: list[Stump]

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        score = np.zeros(x.shape[0], dtype=np.float64)
        for stump in self.stumps:
            score += stump.alpha * stump.apply(x[:, stump.feature])
        return score

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(x) >= 0.0, 1.0, -1.0)


def _best_stump(values: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Return threshold, polarity, weighted error.

    Polarity +1 means the left side of the threshold (smaller feature) is
    predicted positive. Error is the sum of weights on mistakes; weights
    are assumed to sum to 1.
    """
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    yy = y[order]
    ww = w[order]
    pos = yy > 0
    w_pos = np.cumsum(np.where(pos, ww, 0.0))
    w_neg = np.cumsum(np.where(~pos, ww, 0.0))
    total_pos = float(w_pos[-1])
    total_neg = float(w_neg[-1])
    err_left_pos = w_neg + (total_pos - w_pos)
    err_left_neg = w_pos + (total_neg - w_neg)
    valid = np.empty(len(v), dtype=bool)
    valid[-1] = False
    if len(v) > 1:
        valid[:-1] = v[:-1] < v[1:]
    err_left_pos = np.where(valid, err_left_pos, 2.0)
    err_left_neg = np.where(valid, err_left_neg, 2.0)
    i_pos = int(np.argmin(err_left_pos))
    i_neg = int(np.argmin(err_left_neg))
    if err_left_pos[i_pos] <= err_left_neg[i_neg]:
        i, polarity, err = i_pos, 1.0, float(err_left_pos[i_pos])
    else:
        i, polarity, err = i_neg, -1.0, float(err_left_neg[i_neg])
    if i >= len(v) - 1:
        thresh = float(v[-1] + 1e-6)
    else:
        thresh = float(0.5 * (v[i] + v[i + 1]))
    return thresh, polarity, err


def train_adaboost(x: np.ndarray, y: np.ndarray, rounds: int = 12) -> StrongClassifier:
    """``y`` must be in {+1, -1}. ``x`` is (n, f)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if set(np.unique(y).tolist()) - {-1.0, 1.0}:
        raise ValueError("labels must be +1 or -1")
    n, f = x.shape
    w = np.full(n, 1.0 / n)
    stumps: list[Stump] = []
    for _ in range(rounds):
        best: tuple[float, int, float, float] | None = None
        for j in range(f):
            thresh, polarity, err = _best_stump(x[:, j], y, w)
            if best is None or err < best[0]:
                best = (err, j, thresh, polarity)
        assert best is not None
        err, j, thresh, polarity = best
        if err >= 0.5:
            break
        err = float(np.clip(err, 1e-10, 1.0 - 1e-10))
        alpha = 0.5 * float(np.log((1.0 - err) / err))
        stump = Stump(j, thresh, polarity, alpha)
        pred = stump.apply(x[:, j])
        w *= np.exp(-alpha * y * pred)
        w /= w.sum()
        stumps.append(stump)
        if err <= 1e-8:
            break
    if not stumps:
        raise RuntimeError("AdaBoost found no stump better than chance")
    return StrongClassifier(stumps)
