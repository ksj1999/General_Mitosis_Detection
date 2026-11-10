import sys
import unittest
import csv
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch


DEMO2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO2_DIR))

from prepare_demo2_data import centered_crop_geometry, resolution_to_mpp  # noqa: E402
from foundation_models import extract_foundation_features  # noqa: E402
from stain_augmentation import hed_jitter  # noqa: E402
from run_adversarial_probe_validation import gradient_reverse  # noqa: E402
from combine_embedding_caches import combine_caches  # noqa: E402
from evaluate_scanner_probe import evaluate_feature_set  # noqa: E402


class ResolutionTests(unittest.TestCase):
    def test_pixels_per_inch_to_microns_per_pixel(self):
        self.assertAlmostEqual(resolution_to_mpp(101600, 2), 0.25)

    def test_pixels_per_centimeter_to_microns_per_pixel(self):
        self.assertAlmostEqual(resolution_to_mpp(40000, 3), 0.25)


class CropGeometryTests(unittest.TestCase):
    def test_20x_field_of_view_uses_mpp(self):
        crop = centered_crop_geometry(500, 500, 1000, 1000, 0.25, 0.25, 0.50)
        self.assertEqual(crop["width"], 448)
        self.assertEqual(crop["height"], 448)
        self.assertFalse(crop["touches_edge"])

    def test_edge_crop_is_flagged_without_shifting_center(self):
        crop = centered_crop_geometry(10, 10, 1000, 1000, 0.25, 0.25, 0.50)
        self.assertLess(crop["left"], 0)
        self.assertLess(crop["top"], 0)
        self.assertTrue(crop["touches_edge"])


class GradientReversalTests(unittest.TestCase):
    def test_gradient_is_reversed_and_scaled(self):
        values = torch.tensor([1.0, -2.0], requires_grad=True)
        gradient_reverse(values, 0.25).sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([-0.25, -0.25]))


class MultiScaleFusionTests(unittest.TestCase):
    def test_aligned_features_are_concatenated(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for scale, offset in (("20x", 0), ("40x", 10)):
                path = Path(directory) / f"{scale}.npz"
                np.savez(
                    path,
                    features=np.asarray([[offset + 1, offset + 2], [offset + 3, offset + 4]], dtype=np.float32),
                    candidate_ids=np.asarray([1, 2]),
                    labels=np.asarray([0, 1]),
                    image_ids=np.asarray([10, 20]),
                    scanners=np.asarray(["a", "b"]),
                    scale=np.asarray([scale]),
                    model_id=np.asarray(["model"]),
                    model_key=np.asarray(["key"]),
                    pooling=np.asarray(["pool"]),
                )
                paths.append(path)
            combined, scales, dimensions = combine_caches(paths)
            self.assertEqual(combined["features"].shape, (2, 4))
            self.assertEqual(scales, ["20x", "40x"])
            self.assertEqual(dimensions, [2, 2])


class GeneratedFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = DEMO2_DIR / "data"
        with (data_dir / "fold_images.csv").open(newline="", encoding="utf-8") as handle:
            cls.image_rows = list(csv.DictReader(handle))
        with (data_dir / "fold_candidates.csv").open(newline="", encoding="utf-8") as handle:
            cls.candidate_rows = list(csv.DictReader(handle))

    def test_each_fold_is_image_disjoint_and_complete(self):
        folds = {row["fold"] for row in self.image_rows}
        self.assertEqual(len(folds), 3)
        for fold in folds:
            role_ids = {
                role: {row["image_id"] for row in self.image_rows if row["fold"] == fold and row["role"] == role}
                for role in ("train", "val", "test")
            }
            self.assertFalse(role_ids["train"] & role_ids["val"])
            self.assertFalse(role_ids["train"] & role_ids["test"])
            self.assertFalse(role_ids["val"] & role_ids["test"])
            self.assertEqual(len(set.union(*role_ids.values())), 150)

    def test_each_candidate_has_one_role_per_fold(self):
        folds = {row["fold"] for row in self.candidate_rows}
        for fold in folds:
            candidate_ids = [row["candidate_id"] for row in self.candidate_rows if row["fold"] == fold]
            self.assertEqual(len(candidate_ids), 4435)
            self.assertEqual(len(set(candidate_ids)), 4435)


class FoundationPoolingTests(unittest.TestCase):
    def test_transformers_pooler_output(self):
        expected = torch.tensor([[1.0, 2.0]])
        outputs = SimpleNamespace(pooler_output=expected)
        model = lambda pixel_values: outputs
        actual = extract_foundation_features(
            model,
            torch.zeros(1, 3, 224, 224),
            {"backend": "transformers", "pooling": "pooler_output"},
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_virchow_pooling_skips_four_register_tokens(self):
        tokens = torch.zeros(1, 7, 2)
        tokens[:, 0] = torch.tensor([1.0, 2.0])
        tokens[:, 1:5] = 100.0
        tokens[:, 5] = torch.tensor([3.0, 5.0])
        tokens[:, 6] = torch.tensor([7.0, 9.0])
        model = lambda pixels: tokens
        actual = extract_foundation_features(
            model,
            torch.zeros(1, 3, 224, 224),
            {"backend": "timm_virchow2", "pooling": "cls_plus_mean_patch_skip_4_registers"},
        )
        self.assertTrue(torch.equal(actual, torch.tensor([[1.0, 2.0, 5.0, 7.0]])))


class StainAugmentationTests(unittest.TestCase):
    def test_output_remains_valid_rgb(self):
        images = torch.rand(2, 3, 16, 16).clamp_min(1e-4)
        actual = hed_jitter(images, sigma=0.05)
        self.assertEqual(actual.shape, images.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    def test_seed_makes_augmented_images_reproducible(self):
        images = torch.rand(2, 3, 16, 16)
        torch.manual_seed(7)
        first = hed_jitter(images, sigma=0.05, mode="both")
        torch.manual_seed(7)
        second = hed_jitter(images, sigma=0.05, mode="both")
        self.assertTrue(torch.equal(first, second))


class ScannerProbeTests(unittest.TestCase):
    def test_probe_ignores_held_out_scanner_rows(self):
        candidate_ids = np.arange(1, 11, dtype=np.int64)
        features = np.asarray([[value, value % 2] for value in candidate_ids], dtype=np.float32)
        scanners = ["source_a", "source_b"] * 4 + ["held_out", "held_out"]
        labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
        manifest = {
            candidate_id: {"scanner": scanner, "label": str(label)}
            for candidate_id, scanner, label in zip(candidate_ids, scanners, labels)
        }
        roles = {
            "train": {1, 2, 3, 4},
            "val": {5, 6, 7, 8},
            "test": {9, 10},
        }
        rows = evaluate_feature_set(
            features, candidate_ids, manifest, roles, "fold", 0, 42, "method", 1.0
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["train_candidates"], 4)
        self.assertEqual(rows[0]["validation_candidates"], 4)


if __name__ == "__main__":
    unittest.main()
