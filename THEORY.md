# Theory notes

A short account of what the code actually computes, and where it stops being the paper.

## 1. Integral image

For a gray image \(i\), the summed-area table is

\[
I(x, y) = \sum_{x' < x} \sum_{y' < y} i(x', y')
\]

with a zero border so the indices line up. The sum of any axis-aligned rectangle \((x, y, w, h)\) is four lookups:

\[
I(x+w, y+h) - I(x, y+h) - I(x+w, y) + I(x, y).
\]

The squared table \(I_2\) gives the window variance the same way:

\[
\sigma^2 = \frac{1}{N} \sum i^2 - \left(\frac{1}{N} \sum i\right)^2.
\]

Viola and Jones divide each feature by \(\sigma\) so a gain change does not flip the stump. `window_std` implements that. Degenerate windows are floored at \(10^{-4}\) rather than dividing by zero.

## 2. Haar-like features

A feature is a small set of rectangles with weights, evaluated as a difference of means (the weights include \(1/\text{area}\)). The bank has four kinds:

- two-rectangle, horizontal and vertical
- three-rectangle (the center twice the weight of each side)
- four-rectangle checkerboard

The bank is a seeded random sample of legal rectangles inside a 24×24 window, not the exhaustive Viola–Jones pool of about 160,000. Exhaustive search is how you train a real cascade. Here it would dominate the runtime without changing what the assignment is testing: that a stump can be chosen by a weighted error scan, and that the scan localizes a face drawn from the same distribution.

## 3. AdaBoost

Labels are \(\{+1, -1\}\). Weights start uniform. Each round picks the feature whose best threshold has the smallest weighted error. For a column sorted by feature value, the best threshold is a single pass over the cumulative positive and negative weight. Two polarities are tried:

\[
h(x) = p \;\text{if}\; f(x) < \theta, \quad -p \;\text{otherwise}, \quad p \in \{+1,-1\}.
\]

The round weight and the update are the usual ones:

\[
\alpha = \tfrac12 \ln \frac{1-e}{e}, \qquad w_i \leftarrow w_i \exp(-\alpha y_i h(x_i)).
\]

The strong classifier is \(\mathrm{sign}(\sum \alpha_t h_t)\). Training stops early if a stump is perfect or no stump beats chance.

What this is not: the attentional cascade. Viola–Jones train a stage, lower its threshold until the detection rate is almost 1, then bootstrap false positives as the next stage's negatives. That outer loop wants a large negative set mined from real images. We do not have one, and we do not pretend a 10-stump committee trained on ellipses is that cascade. Real images go through OpenCV's published cascades instead. The learned committee is tested by a sliding window on a canvas with one planted face and a known box.

## 4. HOG

Unsigned orientation in \([0, 180^\circ)\), 9 bins of 20°. Each pixel votes into the two neighboring bins with linear interpolation, weighted by gradient magnitude. Cells are 8×8. Blocks are 2×2 cells, L2-normalized, clipped at 0.2, L2-normalized again (L2-Hys). A 64×64 window yields \(7 \times 7 \times 36 = 1764\) dimensions. A linear SVM (`LinearSVC`) classifies the descriptor.

Spatial interpolation across cells, which Dalal and Triggs also do, is omitted. Block overlap still couples neighbors. The unit test only checks a fact that should survive that omission: a vertical intensity step has a horizontal gradient, angle 0, and the mean cell histogram peaks in bin 0.

## 5. Skin

A pixel passes only if both rules pass.

RGB, after Kovac, Peer, Solina:

\[
R>95,\; G>40,\; B>20,\; R>G,\; R>B,\; |R-G|>15.
\]

YCrCb, the box used throughout this literature:

\[
Y>80,\; 77 < C_b < 127,\; 133 < C_r < 173.
\]

A 5×5 elliptical open and close follows. Connected components with a face-like aspect ratio are available as **proposals**. They are not added to the detector. Hands, wood, and some walls satisfy the same inequalities. The pipeline uses the mask only as a ratio inside a classical box, and only to reject boxes that are almost devoid of skin. YuNet is not subjected to that test: a profile or a dark exposure can be a face with a low skin ratio, and the color model is the wrong instrument to overrule a trained detector.

The color model is also uneven across skin tones. That is not a footnote. Gender Shades (Buolamwini and Gebru) is the reason a coursework detector should not grow a "skin = face" heuristic and ship it.

## 6. Fusion

Haar `detectMultiScale3` returns a level weight, not a probability. It is squashed with \(s = w/(w+2)\) so it lives on the same 0–1 scale as YuNet's score. The profile cascade is run on the image and on its horizontal flip, with boxes mapped back.

Weighted box fusion clusters every pair with IoU at least 0.45 (union-find) and averages the corners, weighted by score. The reported score is the max in the cluster. If any member carries a YuNet row, that row and its five landmarks are kept even when a cascade scored higher: alignment and SFace need the landmarks, not the winning name.

A cluster of one whose source is a cascade and whose score is below 0.55 is discarded. On `samples/three.jpg` this removes a profile hit on hair. The cost is a missed lone profile. The report prefers that miss to a fourth face.

## 7. Alignment and quality

Given image-left and image-right eye centers, a similarity (scale, rotation, translation) maps them to \((0.30 W, 0.38 W)\) and \((0.70 W, 0.38 W)\) in a 112×112 crop. Roll is \(\mathrm{atan2}(\Delta y, \Delta x)\) of that same pair. If the eyes are missing, the face is not aligned; we do not invent landmarks by putting eyes at fixed fractions of the box.

Quality is a weighted sum of four heuristics: log variance of the Laplacian (blur), distance of the mean gray value from 128 (exposure), short-side versus 120 px (size), and a linear penalty on roll. It ranks crops. It is not a probability of identity.

## 8. Subspace recognition

**Eigenfaces.** Faces are rows of a matrix, the mean face is removed, and the economy SVD supplies the principal axes. The default rank is about half the number of training images, capped at \(N-1\). Classification is nearest neighbor in that subspace.

**Fisherfaces.** PCA first cuts the pixel space down to \(N - C\) so the within-class scatter can be inverted. LDA then solves \(S_b v = \lambda S_w v\) and keeps \(C - 1\) directions. \(S_w\) is ridged by \(10^{-3} I\) because a class with little variation still makes it ill-conditioned. Nearest neighbor again.

**LBPH.** 8-neighbor LBP at radius 1, bit 0 at the east pixel, clockwise. Codes with more than two circular transitions fall into one non-uniform bin: 58 uniform patterns + 1 = 59. A 4×4 grid of \(L_1\)-normalized histograms is compared with chi-square. Flat skin has almost no LBP signal, which is why the synthetic faces carry a stable mid-frequency texture. On a real, smooth, motion-blurred crop this representation is weak, and the code will still return a nearest neighbor. Distance is part of the answer; the label alone is not.

The rank-1 scores on the synthetic set are 1.0 because within-identity jitter (a pixel of shift, a lighting ramp, small noise) is smaller than the identity gap we drew. That is a unit test of the estimators. It is not a recognition benchmark.

## 9. SORT

State is \((c_x, c_y, w, h, \dot c_x, \dot c_y, \dot w, \dot h)\). The motion model is constant velocity with \(\Delta t = 1\) frame. Association is the Hungarian algorithm on \(1 - \mathrm{IoU}\), gated at IoU 0.3. Unmatched tracks are predicted for up to 8 frames. There is no appearance embedding, so two people who cross will swap ids. DeepSORT exists because of that failure. We stop at Bewley et al. and say so.

## 10. Average precision

Predictions on one image are sorted by score. A prediction is a true positive if it is the first to match an unmatched ground-truth box at IoU ≥ 0.5. Precision and recall are the cumulative counts. AP is the all-point VOC integral: precision is made monotonically decreasing from the right, then the area under the resulting step function is taken. The unit test checks the two extremes (a perfect hit, a total miss). The planted-face experiment is the non-trivial call.

## 11. YuNet and SFace

YuNet is a small CNN face detector distributed with OpenCV. We do not train it. Each detection is a box, five landmarks, and a score. SFace is a hypersphere embedding. `cv2.FaceRecognizerSF.alignCrop` consumes the YuNet row; `feature` produces the vector; cosine similarity is `match` with `FR_COSINE`.

The Zoo demo uses a cosine threshold of 0.363. On our plates, genuine pairs (a face against a gamma, blur, or JPEG copy of itself, matched by IoU) all sit above 0.91. Impostor pairs reach 0.55. At 0.363 the false-accept rate on these 10 impostors is 0.60. The gap between the worst genuine (0.91) and the worst impostor (0.55) is real for this handful of images and is not a license to publish a new threshold. The faces are fictional, few, and from one rendering process, which tends to pack different identities closer together than a diverse photograph collection would.

## 12. What a camera should not be used for

Nothing in this repository measures demographic error. The cascades and the skin box have that error anyway. Do not deploy this as an attendance system, a door lock, or a way to identify people who did not agree to it. The gallery in any live demo should be the operator's own face, stored locally, and thrown away.
