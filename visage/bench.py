"""Reproducible experiment log for the classical and modern stacks.

Run ``python -m visage.bench``. Writes ``results/bench.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from visage.adaboost import train_adaboost
from visage.eigenfaces import Eigenfaces
from visage.evaluate import average_precision, rank1_accuracy
from visage.fisherfaces import Fisherfaces
from visage.geometry import Detection, iou
from visage.haar import feature_matrix, make_feature_bank
from visage.hog import hog_descriptor, hog_matrix
from visage.lbph import LBPH, uniform_lut
from visage.pipeline import analyze, draw
from visage.scan import sliding_window_detect
from visage.sface import cosine, embed
from visage.synthetic import classification_set, detection_canvas, recognition_set
from visage.tracker import SortTracker

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SAMPLES = ROOT / "samples"


def _acc(model_predict, images, labels) -> float:
    pairs = []
    for image, label in zip(images, labels):
        pred, _dist = model_predict(image)
        pairs.append((pred, label))
    return rank1_accuracy(pairs)


def _split_ids(labels, seed, test_size=0.3):
    idx = np.arange(len(labels))
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)


def classical_detection() -> dict:
    rng_seed = 7
    features = make_feature_bank(64, win=24, seed=rng_seed)
    images, labels = classification_set(160, 160, 24, seed=rng_seed, n_identities=8)
    x = feature_matrix(images, features)
    train_i, test_i = _split_ids(np.where(labels > 0, 1, 0), seed=rng_seed)
    model = train_adaboost(x[train_i], labels[train_i], rounds=10)
    pred = model.predict(x[test_i])
    acc = float((pred == labels[test_i]).mean())
    # Localization: one planted face, single scale.
    rng = np.random.default_rng(rng_seed)
    canvas, gt = detection_canvas(rng)
    dets = sliding_window_detect(canvas, model, features, win=24, step=4, threshold=0.0)
    best_iou = 0.0
    if dets:
        best_iou = max(iou(d.box, gt) for d in dets)
    # Score of the exact window must beat a negative window.
    exact = feature_matrix([canvas[gt[1] : gt[1] + 24, gt[0] : gt[0] + 24]], features)
    neg = feature_matrix([np.full((24, 24), 0.18, np.float32)], features)
    margin = float(model.decision_function(exact)[0] - model.decision_function(neg)[0])
    preds = [(d.score, d.box) for d in dets]
    ap = average_precision(preds, [gt], 0.5)["ap"]
    return {
        "features": len(features),
        "rounds": len(model.stumps),
        "heldout_accuracy": round(acc, 4),
        "localization_iou": round(best_iou, 4),
        "localization_ap50": round(ap, 4),
        "face_minus_blank_margin": round(margin, 4),
        "detections": len(dets),
    }


def hog_svm() -> dict:
    images, labels = classification_set(80, 80, 64, seed=11, n_identities=8)
    y = (labels > 0).astype(np.int32)
    x = hog_matrix(images, 64)
    train_i, test_i = _split_ids(y, seed=11)
    clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=4000, C=1.0))
    clf.fit(x[train_i], y[train_i])
    acc = float(clf.score(x[test_i], y[test_i]))
    # Orientation sanity: a vertical step edge loads the 0° bin.
    edge = np.zeros((64, 64), np.float32)
    edge[:, 32:] = 1.0
    _desc, hist = hog_descriptor(edge)
    mean_hist = hist.mean(axis=(0, 1))
    return {
        "heldout_accuracy": round(acc, 4),
        "descriptor_length": int(x.shape[1]),
        "vertical_edge_peak_bin": int(np.argmax(mean_hist)),
    }


def subspace() -> dict:
    images, labels = recognition_set(6, 8, 40, seed=3)
    train_i, test_i = _split_ids(labels, seed=3, test_size=0.25)
    train_im = [images[i] for i in train_i]
    test_im = [images[i] for i in test_i]
    y_train, y_test = labels[train_i], labels[test_i]
    eigen = Eigenfaces(n_components=15, size=40).fit(train_im, y_train)
    fisher = Fisherfaces(size=40).fit(train_im, y_train)
    lbph = LBPH(size=48, grid=4).fit(train_im, y_train)
    _lut, bins = uniform_lut()
    return {
        "identities": 6,
        "train": int(len(train_i)),
        "test": int(len(test_i)),
        "uniform_lbp_bins": int(bins),
        "eigenfaces_rank1": round(_acc(eigen.predict, test_im, y_test), 4),
        "fisherfaces_rank1": round(_acc(fisher.predict, test_im, y_test), 4),
        "lbph_rank1": round(_acc(lbph.predict, test_im, y_test), 4),
    }


def tracking() -> dict:
    tracker = SortTracker()
    ids = []
    for t in range(20):
        det = Detection(x=10 + 3 * t, y=20, w=30, h=40, score=0.9, source="synth")
        tracks = tracker.update([det])
        ids.append(tracks[0].id if tracks else None)
    # A one-frame dropout should keep the id.
    tracker.update([])
    tracks = tracker.update([Detection(x=10 + 3 * 22, y=20, w=30, h=40, score=0.9, source="synth")])
    return {
        "stable_id": len(set(ids)) == 1,
        "id_after_gap": tracks[0].id if tracks else None,
        "frames": 20,
    }


def _variants(bgr: np.ndarray) -> list[np.ndarray]:
    out = []
    gamma = 0.7
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    out.append(cv2.LUT(bgr, table))
    gamma = 1.5
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    out.append(cv2.LUT(bgr, table))
    out.append(cv2.GaussianBlur(bgr, (0, 0), 1.2))
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    if ok:
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is not None:
            out.append(decoded)
    return out


def _with_row(faces: list[dict]) -> list[dict]:
    return [f for f in faces if f.get("extra", {}).get("yunet_row")]


def _box(face: dict) -> tuple[int, int, int, int]:
    return (int(face["x"]), int(face["y"]), int(face["w"]), int(face["h"]))


def real_plates() -> dict:
    plates = sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))
    if not plates:
        return {"skipped": True, "reason": "no images in samples/"}
    reports = []
    embeddings: dict[str, list] = {}
    genuine: list[float] = []
    impostor: list[float] = []
    for path in plates:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        t0 = time.perf_counter()
        result = analyze(bgr)
        elapsed = time.perf_counter() - t0
        faces = result["faces"]
        reports.append(
            {
                "file": path.name,
                "width": result["width"],
                "height": result["height"],
                "raw_count": len(result["raw"]),
                "faces": len(faces),
                "seconds": round(elapsed, 3),
                "raw": [
                    {
                        "x": d["x"],
                        "y": d["y"],
                        "w": d["w"],
                        "h": d["h"],
                        "score": d["score"],
                        "source": d["source"],
                    }
                    for d in result["raw"]
                ],
                "detections": [
                    {
                        "source": f["source"],
                        "score": f["score"],
                        "box": [f["x"], f["y"], f["w"], f["h"]],
                        "skin_ratio": f["skin_ratio"],
                        "quality": f["quality"],
                        "aligned": f["has_alignment"],
                        "cluster": f.get("extra", {}).get("cluster"),
                        "sources": f.get("extra", {}).get("sources", [f["source"]]),
                        "landmarks": f.get("landmarks"),
                    }
                    for f in faces
                ],
            }
        )
        # Verification pairs. A photometric copy is a genuine pair only when
        # the detection matches the same face (IoU). Other faces are impostors.
        # The group plate is what makes the impostor set non-trivial.
        base_faces = _with_row(faces)
        base_feats = []
        for face in base_faces:
            try:
                base_feats.append(embed(bgr, face["extra"]["yunet_row"]))
            except cv2.error:
                base_feats.append(None)
        embeddings[path.name] = base_feats
        for variant in _variants(bgr):
            variant_faces = _with_row(analyze(variant, use_haar=False)["faces"])
            if not variant_faces:
                continue
            var_feats = []
            for face in variant_faces:
                try:
                    var_feats.append(embed(variant, face["extra"]["yunet_row"]))
                except cv2.error:
                    var_feats.append(None)
            for i, base_feat in enumerate(base_feats):
                if base_feat is None:
                    continue
                order = sorted(
                    range(len(variant_faces)),
                    key=lambda j: iou(_box(base_faces[i]), _box(variant_faces[j])),
                    reverse=True,
                )
                if not order:
                    continue
                j = order[0]
                if var_feats[j] is None or iou(_box(base_faces[i]), _box(variant_faces[j])) < 0.3:
                    continue
                genuine.append(cosine(base_feat, var_feats[j]))
        for i in range(len(base_feats)):
            for j in range(i + 1, len(base_feats)):
                if base_feats[i] is None or base_feats[j] is None:
                    continue
                impostor.append(cosine(base_feats[i], base_feats[j]))
        annotated = draw(bgr, result)
        cv2.imwrite(str(RESULTS / f"annotated-{path.stem}.jpg"), annotated)
    names = list(embeddings)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for fa in embeddings[a]:
                for fb in embeddings[b]:
                    if fa is None or fb is None:
                        continue
                    impostor.append(cosine(fa, fb))
    return {
        "skipped": False,
        "plates": reports,
        "sface_genuine_cosine": [round(float(v), 4) for v in genuine],
        "sface_impostor_cosine": [round(float(v), 4) for v in impostor],
        "sface_genuine_cosine_mean": None if not genuine else round(float(np.mean(genuine)), 4),
        "sface_impostor_cosine_mean": None if not impostor else round(float(np.mean(impostor)), 4),
        "sface_genuine_n": len(genuine),
        "sface_impostor_n": len(impostor),
        "sface_threshold": 0.363,
    }


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = {
        "adaboost_haar": classical_detection(),
        "hog_svm": hog_svm(),
        "recognition": subspace(),
        "sort": tracking(),
        "plates": real_plates(),
    }
    report["seconds"] = round(time.perf_counter() - t0, 2)
    dest = RESULTS / "bench.json"
    dest.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
