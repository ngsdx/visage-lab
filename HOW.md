# How this works

VISAGE is a still-image pipeline plus a separate tracker. The web instrument shows the pipeline’s measurements. It does not re-run OpenCV.

## One pass

`analyze` in `visage/pipeline.py` is the whole still-image system.

1. **Haar cascades.** Frontal default, frontal alt2, and the profile cascade, the last of those also on a horizontal flip with boxes mapped back. Histogram equalization first. A cascade level weight is not a probability. It is squashed with `w / (w + 2)` so it lives on the same 0–1 scale as YuNet (`detectors.py`).

2. **YuNet.** A published CNN. Each hit is a box, five landmarks, and a score. This repository does not train it. Weights download into `models/` on first use.

3. **Skin ratio, classical boxes only.** A pixel is skin when Kovac’s RGB predicate and a YCrCb box both hold (`Y > 80`, `77 < Cb < 127`, `133 < Cr < 173`), after a 5×5 elliptical open and close (`skin.py`). A cascade box under a skin ratio of 0.12 is dropped. YuNet is not subjected to that test. A profile or a dark exposure can be a face the color model rejects, and the color model is uneven across skin tones.

4. **Weighted box fusion.** Boxes with IoU at least 0.45 are one cluster (`geometry.py`). Corners are averaged by score. The reported score is the max in the cluster. If any member is YuNet, that row and its landmarks are kept even when a cascade scored higher, because alignment needs the eyes.

5. **Lone weak cascade.** A cluster of one whose source is a cascade and whose score is under 0.55 is discarded. On `samples/three.jpg` this removes a profile hit on hair. The cost is a missed lone profile.

6. **Eyes, then a similarity.** Eye centers come from YuNet, or from an eye cascade in the upper 60% of the box. They map to `(0.30 W, 0.38 W)` and `(0.70 W, 0.38 W)` in a 112×112 crop (`align.py`): scale, rotation, translation. No eyes, no warp. Landmarks are not invented by placing eyes at fixed fractions of the box.

7. **Quality.** A weighted sum of Laplacian variance, distance of the mean gray value from 128, short side versus 120 px, and a penalty on roll (`quality.py`). It ranks crops. It is not a probability of identity.

SFace, when asked, consumes the YuNet row through `cv2.FaceRecognizerSF.alignCrop`. Cosine similarity is `match` with `FR_COSINE`. The OpenCV Zoo threshold is 0.363.

## What is reimplemented

| Piece | Where | What it actually is |
| --- | --- | --- |
| Integral image, rectangle sum, window σ | `integral.py` | Four lookups for any axis-aligned sum. `window_std` is the Viola–Jones normalizer. |
| Haar-like features | `haar.py` | Two-, three-, and four-rectangle features, area-normalized, divided by window σ. |
| Discrete AdaBoost | `adaboost.py` | Each round picks the stump with the smallest weighted error. Both polarities. |
| Sliding window | `scan.py` | One scale, the window the stumps were trained on. Not an attentional cascade. |
| HOG | `hog.py` | 9 bins, 2×2 blocks, L2-Hys. The benchmark puts a linear SVM on it. |
| Skin proposals | `skin.py` | Connected components on the mask. They are not added to the detector. |
| Eigenfaces | `eigenfaces.py` | Mean removed, economy SVD, nearest neighbor. |
| Fisherfaces | `fisherfaces.py` | PCA to `N − C`, then LDA, `C − 1` directions. `Sw` is ridged by `1e-3 I`. |
| Uniform LBPH | `lbph.py` | 8-neighbor LBP, 59 bins, 4×4 grid, chi-square. |
| SORT | `tracker.py` | Constant-velocity Kalman state, Hungarian assignment on `1 − IoU`, gate 0.3, coast 8 frames. No appearance. |
| Average precision | `evaluate.py` | VOC all-point interpolation. A hit is the first unmatched prediction at IoU ≥ 0.5. |

The Haar bank is a seeded sample of legal rectangles in a 24×24 window, not the exhaustive pool of about 160,000. AdaBoost here learns stumps. It does not build the bootstrapped cascade. The derivation and the refusals are in [THEORY.md](THEORY.md).

The unit tests in `tests/test_core.py` do not read the photographs and do not download weights. Synthetic rank-1 of 1.0 means the estimators separate the identities the renderer drew. It is not a recognition benchmark.

## What the viewer recomputes

The plate page ships the library’s raw boxes and the fused faces. Three controls are computed in the browser from the photograph:

- **Greedy NMS** sorts the raw boxes by score and drops overlaps above the slider. It is not weighted box fusion. NMS keeps a winner. Fusion averages corners.
- **Skin field** is the same predicate, before morphology.
- **HOG cells** are the dominant unsigned bin of each cell. The library SVM uses block normalization. The overlay does not.

**Enroll** stores a 32×32 gray crop in that browser and matches by nearest neighbor after subtracting the mean face. That is the eigenfaces objective with every component kept.

**Camera** is a different front end: MediaPipe Face Landmarker, then an IoU tracker without a Kalman filter. Crossing people swap ids.

## Plates

`samples/` holds three fictional studio photographs. `python -m visage.bench` globs only that directory, runs the fusion, and builds the SFace pair table from photometric copies (gamma 0.7, gamma 1.5, blur, JPEG q=35) matched back by IoU.

`samples/photographs/` holds twelve photographs. They are not in that pair table. Detect them by path. Counts in the README are detector output, not ground truth. Plate `sun-hat.jpg` fuses to nothing: a profile under a brim and sunglasses. `nasa-class.jpg` keeps a cascade box on the table. `market-shade.jpg` keeps a box on foliage. Credits are in [samples/photographs/ATTRIBUTION.md](samples/photographs/ATTRIBUTION.md).

## Run

Python 3.10+. Pin OpenCV 4.10. The 5.0 wheel drops `cv2.CascadeClassifier`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest tests.test_core -v
python -m visage.bench
python -m visage detect samples/three.jpg -o results/three.jpg
python -m visage detect samples/photographs/sun-hat.jpg
python -m visage track clip.mp4 -o results/tracked.mp4
```

## Limits

Nothing in this repository measures demographic error. The cascades and the skin box have that error anyway. Do not deploy this as an attendance system, a door lock, or a way to identify people who did not agree to it.
