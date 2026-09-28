"""Detection scores: IoU matching and VOC-style average precision."""

from __future__ import annotations

import numpy as np

from visage.geometry import iou


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point interpolated average precision (VOC 2010 onwards)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def average_precision(
    predictions: list[tuple[float, tuple[int, int, int, int]]],
    ground_truth: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.5,
) -> dict:
    """Single-image AP. Predictions are (score, box), best score first or not."""
    preds = sorted(predictions, key=lambda p: p[0], reverse=True)
    if not ground_truth:
        return {"ap": 0.0 if preds else 1.0, "precision": [], "recall": [], "tp": 0, "fp": len(preds)}
    matched = np.zeros(len(ground_truth), dtype=bool)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (_score, box) in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(ground_truth):
            if matched[j]:
                continue
            overlap = iou(box, gt)
            if overlap > best_iou:
                best_iou, best_j = overlap, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched[best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    if len(preds) == 0:
        return {"ap": 0.0, "precision": [], "recall": [], "tp": 0, "fp": 0}
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / float(len(ground_truth))
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return {
        "ap": voc_ap(recall, precision),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
    }


def rank1_accuracy(pairs: list[tuple[object, object]]) -> float:
    if not pairs:
        return 0.0
    correct = sum(int(a == b) for a, b in pairs)
    return correct / float(len(pairs))
