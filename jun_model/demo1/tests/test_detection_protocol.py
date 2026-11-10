import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


DEMO1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO1_DIR))

from demo1_build_detection_tiles import build_grid_split  # noqa: E402
from demo1_evaluate_detection_frcnn import match_image  # noqa: E402


class MatcherTests(unittest.TestCase):
    def test_prediction_can_match_second_best_unmatched_ground_truth(self):
        ground_truth = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]],
            dtype=torch.float32,
        )
        predictions = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

        self.assertEqual(match_image(predictions, scores, ground_truth, 0.5, 0.3), (2, 0, 0))


class GridProtocolTests(unittest.TestCase):
    def test_grid_covers_negative_slides_and_does_not_duplicate_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            Image.new("RGB", (700, 600), "white").save(temp_dir / "positive.png")
            Image.new("RGB", (700, 600), "white").save(temp_dir / "negative.png")
            coco = {
                "images": [
                    {"id": 1, "file_name": "positive.png", "width": 700, "height": 600},
                    {"id": 2, "file_name": "negative.png", "width": 700, "height": 600},
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [500, 500, 40, 40],
                        "area": 1600,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "mitotic figure"}],
            }
            coco_path = temp_dir / "test_coco.json"
            coco_path.write_text(json.dumps(coco), encoding="utf-8")
            output_dir = temp_dir / "tiles"

            summary = build_grid_split(coco_path, output_dir, "test", 512)
            tiled = json.loads((output_dir / "test_tiles_coco.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["source_images"], 2)
            self.assertEqual({image["source_image_id"] for image in tiled["images"]}, {1, 2})
            self.assertEqual(len(tiled["annotations"]), 1)
            self.assertEqual(tiled["annotations"][0]["source_annotation_id"], 10)


if __name__ == "__main__":
    unittest.main()

