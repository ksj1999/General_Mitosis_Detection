import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


JITTER_SETTINGS = {
    "none": None,
    "mild": {"brightness": 0.08, "contrast": 0.08, "saturation": 0.05, "hue": 0.01},
    "strong": {"brightness": 0.18, "contrast": 0.18, "saturation": 0.12, "hue": 0.03},
}

RGB_NORMALIZATION_SETTINGS = {
    "none": None,
    # Fixed, gentle Reinhard-style RGB target. This is intentionally simple so
    # it can be tested as a first stain-normalization baseline.
    "reinhard_rgb": {
        "mean": np.array([185.0, 135.0, 175.0], dtype=np.float32),
        "std": np.array([45.0, 35.0, 40.0], dtype=np.float32),
    },
}


def normalize_rgb_reinhard_like(image, target_mean, target_std):
    array = np.asarray(image).astype(np.float32)
    flat = array.reshape(-1, 3)
    source_mean = flat.mean(axis=0)
    source_std = flat.std(axis=0)
    normalized = (array - source_mean) / np.maximum(source_std, 1.0)
    normalized = normalized * target_std + target_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="RGB")


class CocoTileDetectionDataset(Dataset):
    def __init__(self, coco_path, color_jitter=False, augmentation="none", normalization="none"):
        self.coco_path = Path(coco_path)
        with self.coco_path.open(encoding="utf-8") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations_by_image = defaultdict(list)
        for annotation in self.coco["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

        if color_jitter:
            augmentation = "strong"
        if augmentation not in JITTER_SETTINGS:
            raise ValueError(f"Unknown augmentation: {augmentation}")
        if normalization not in RGB_NORMALIZATION_SETTINGS:
            raise ValueError(f"Unknown normalization: {normalization}")
        self.normalization = normalization

        transform_steps = []
        jitter = JITTER_SETTINGS[augmentation]
        if jitter:
            transform_steps.append(transforms.ColorJitter(**jitter))
        transform_steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = self.coco_path.parent / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        normalization = RGB_NORMALIZATION_SETTINGS[self.normalization]
        if normalization:
            image = normalize_rgb_reinhard_like(image, normalization["mean"], normalization["std"])
        image_tensor = self.transform(image)

        annotations = self.annotations_by_image.get(image_info["id"], [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(annotation["category_id"])
            areas.append(annotation["area"])
            iscrowd.append(annotation.get("iscrowd", 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
