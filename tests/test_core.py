"""Unit tests for the classical stack. No network, no sample photos."""

from __future__ import annotations

import unittest

import numpy as np

from visage.adaboost import train_adaboost
from visage.align import roll_degrees, similarity_matrix
from visage.eigenfaces import Eigenfaces
from visage.evaluate import average_precision
from visage.fisherfaces import Fisherfaces
from visage.geometry import Detection, iou, nms, weighted_box_fusion
from visage.haar import feature_matrix, make_feature_bank
from visage.hog import hog_descriptor
from visage.integral import integral_image, rect_sum, window_std
from visage.lbph import LBPH, uniform_lut
from visage.scan import sliding_window_detect
from visage.skin import skin_mask, skin_ratio
from visage.synthetic import classification_set, detection_canvas, recognition_set
from visage.tracker import SortTracker


class IntegralTests(unittest.TestCase):
    def test_rect_sum_matches_numpy(self):
        rng = np.random.default_rng(0)
        gray = rng.random((17, 13))
        ii = integral_image(gray)
        total = rect_sum(ii, 2, 3, 5, 4)
        self.assertAlmostEqual(total, float(gray[3:7, 2:7].sum()), places=6)

    def test_window_std(self):
        gray = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        ii = integral_image(gray)
        ii2 = integral_image(gray * gray)
        sigma = window_std(ii, ii2, 0, 0, 2, 2)
        self.assertAlmostEqual(sigma, float(gray.std()), places=6)


class AdaBoostTests(unittest.TestCase):
    def test_perfectly_separated_stump(self):
        x = np.array([[0.1], [0.2], [0.8], [0.9]])
        y = np.array([-1.0, -1.0, 1.0, 1.0])
        model = train_adaboost(x, y, rounds=1)
        self.assertTrue(np.all(model.predict(x) == y))

    def test_synthetic_faces_are_learnable(self):
        features = make_feature_bank(40, win=24, seed=1)
        images, labels = classification_set(80, 80, 24, seed=1, n_identities=6)
        x = feature_matrix(images, features)
        model = train_adaboost(x, labels, rounds=8)
        acc = float((model.predict(x) == labels).mean())
        self.assertGreaterEqual(acc, 0.9)

    def test_sliding_window_finds_planted_face(self):
        features = make_feature_bank(48, win=24, seed=4)
        images, labels = classification_set(100, 100, 24, seed=4, n_identities=6)
        model = train_adaboost(feature_matrix(images, features), labels, rounds=8)
        canvas, gt = detection_canvas(np.random.default_rng(4))
        dets = sliding_window_detect(canvas, model, features, step=4)
        self.assertTrue(dets, "detector returned nothing")
        best = max(iou(d.box, gt) for d in dets)
        self.assertGreaterEqual(best, 0.5)


class HogTests(unittest.TestCase):
    def test_vertical_edge_peaks_at_zero_degrees(self):
        edge = np.zeros((64, 64), np.float32)
        edge[:, 32:] = 1.0
        desc, hist = hog_descriptor(edge)
        self.assertGreater(desc.size, 0)
        peak = int(np.argmax(hist.mean(axis=(0, 1))))
        self.assertEqual(peak, 0)


class GeometryTests(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_nms_keeps_highest(self):
        dets = [
            Detection(0, 0, 10, 10, 0.2, "a"),
            Detection(1, 1, 10, 10, 0.9, "b"),
            Detection(40, 40, 10, 10, 0.5, "c"),
        ]
        kept = nms(dets, 0.4)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].source, "b")

    def test_wbf_averages_cluster(self):
        dets = [
            Detection(0, 0, 10, 10, 1.0, "haar"),
            Detection(2, 0, 10, 10, 1.0, "yunet", extra={"yunet_row": [1, 2, 3]}),
        ]
        fused = weighted_box_fusion(dets, 0.3)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].source, "fusion")
        self.assertEqual(fused[0].x, 1)
        self.assertEqual(fused[0].extra["yunet_row"], [1, 2, 3])


class SkinTests(unittest.TestCase):
    def test_skin_accepted_and_blue_rejected(self):
        image = np.zeros((60, 80, 3), np.uint8)
        image[:, :40] = (120, 140, 180)  # BGR of a light skin RGB
        image[:, 40:] = (200, 40, 40)  # blue
        mask = skin_mask(image)
        skin_side = float((mask[:, :40] > 0).mean())
        blue_side = float((mask[:, 40:] > 0).mean())
        self.assertGreater(skin_side, 0.8)
        self.assertLess(blue_side, 0.05)
        self.assertGreater(skin_ratio(mask, (0, 0, 40, 60)), 0.8)


class AlignTests(unittest.TestCase):
    def test_similarity_lands_on_canonical_eyes(self):
        matrix = similarity_matrix((10.0, 20.0), (50.0, 28.0), out=100)

        def apply(point):
            v = matrix @ np.array([point[0], point[1], 1.0])
            return v

        np.testing.assert_allclose(apply((10.0, 20.0)), (30.0, 38.0), atol=1e-6)
        np.testing.assert_allclose(apply((50.0, 28.0)), (70.0, 38.0), atol=1e-6)
        self.assertAlmostEqual(roll_degrees((0, 0), (10, 0)), 0.0, places=5)


class RecognizerTests(unittest.TestCase):
    def test_uniform_lbp_has_59_bins(self):
        _lut, bins = uniform_lut()
        self.assertEqual(bins, 59)

    def test_subspace_and_lbph_rank1(self):
        images, labels = recognition_set(4, 8, 48, seed=5)
        # Hold out the last two samples of each identity.
        train, test, y_train, y_test = [], [], [], []
        for ident in range(4):
            group = [images[i] for i in range(len(labels)) if labels[i] == ident]
            train.extend(group[:6])
            test.extend(group[6:])
            y_train.extend([ident] * 6)
            y_test.extend([ident] * 2)
        eigen = Eigenfaces(n_components=8, size=32).fit(train, y_train)
        fisher = Fisherfaces(size=32).fit(train, y_train)
        lbph = LBPH(size=48).fit(train, y_train)
        for model in (eigen, fisher, lbph):
            correct = 0
            for image, label in zip(test, y_test):
                pred, _dist = model.predict(image)
                correct += int(pred == label)
            self.assertGreaterEqual(correct / len(y_test), 0.75, msg=type(model).__name__)


class TrackerTests(unittest.TestCase):
    def test_constant_velocity_keeps_one_id(self):
        tracker = SortTracker()
        ids = []
        for t in range(15):
            tracks = tracker.update(
                [Detection(x=5 + 4 * t, y=8, w=28, h=36, score=0.8, source="t")]
            )
            ids.append(tracks[0].id)
        self.assertEqual(set(ids), {1})

    def test_two_faces_get_two_ids(self):
        tracker = SortTracker()
        tracks = tracker.update(
            [
                Detection(0, 0, 20, 20, 0.9, "a"),
                Detection(80, 10, 20, 24, 0.8, "b"),
            ]
        )
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len({t.id for t in tracks}), 2)


class APTests(unittest.TestCase):
    def test_perfect_and_empty(self):
        box = (0, 0, 10, 10)
        perfect = average_precision([(0.9, box)], [box])
        self.assertAlmostEqual(perfect["ap"], 1.0)
        miss = average_precision([(0.9, (50, 50, 10, 10))], [box])
        self.assertAlmostEqual(miss["ap"], 0.0)


if __name__ == "__main__":
    unittest.main()
