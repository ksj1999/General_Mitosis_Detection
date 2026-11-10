import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from foundation_models import extract_foundation_features, load_foundation_model


class PatchDataset(Dataset):
    def __init__(self, manifest, patch_dir, scale, max_candidates=None):
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if max_candidates is not None:
            self.rows = self.rows[:max_candidates]
        self.patch_dir = patch_dir
        self.scale = scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.patch_dir / self.scale / f"{int(row['candidate_id']):05d}.jpg"
        with Image.open(path) as image:
            patch = image.convert("RGB")
        return patch, int(row["candidate_id"]), int(row["label"]), int(row["image_id"]), row["scanner"]


def make_collate(processor):
    def collate(batch):
        images, candidate_ids, labels, image_ids, scanners = zip(*batch)
        pixels = processor(list(images))
        return pixels, candidate_ids, labels, image_ids, scanners

    return collate


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen pathology foundation-model embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("demo2/data/candidates.csv"))
    parser.add_argument("--patch-dir", type=Path, default=Path("demo2/patches"))
    parser.add_argument("--scale", choices=("20x", "40x"), required=True)
    parser.add_argument("--model-key", choices=("phikon_v2", "hibou_b", "virchow2"), default="phikon_v2")
    parser.add_argument(
        "--registry", type=Path, default=Path("demo2/configs/foundation_models.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, help="Override the registry's model-specific default.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("demo2/.hf_cache"))
    parser.add_argument("--max-candidates", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dataset = PatchDataset(args.manifest, args.patch_dir, args.scale, args.max_candidates)
    missing = [
        row["candidate_id"]
        for row in dataset.rows
        if not (args.patch_dir / args.scale / f"{int(row['candidate_id']):05d}.jpg").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} patches; run extract_candidate_patches.py first")

    bundle = load_foundation_model(args.model_key, args.registry, args.cache_dir)
    processor = bundle.processor
    model = bundle.model.to(device).eval()
    batch_size = args.batch_size or int(bundle.spec["default_batch_size"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(processor),
        pin_memory=device.type == "cuda",
    )
    feature_batches = []
    candidate_ids = []
    labels = []
    image_ids = []
    scanners = []
    with torch.inference_mode():
        for pixels, batch_candidate_ids, batch_labels, batch_image_ids, batch_scanners in tqdm(
            loader, desc=f"{bundle.spec['display_name']} {args.scale}", unit="batch"
        ):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                features = extract_foundation_features(model, pixels, bundle.spec)
            feature_batches.append(features.float().cpu().numpy())
            candidate_ids.extend(batch_candidate_ids)
            labels.extend(batch_labels)
            image_ids.extend(batch_image_ids)
            scanners.extend(batch_scanners)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(feature_batches),
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        scanners=np.asarray(scanners),
        scale=np.asarray([args.scale]),
        model_id=np.asarray([bundle.spec["repository"]]),
        model_key=np.asarray([args.model_key]),
        pooling=np.asarray([bundle.spec["pooling"]]),
    )
    metadata = {
        "model_key": args.model_key,
        "model_id": bundle.spec["repository"],
        "model_revision": bundle.revision,
        "backend": bundle.spec["backend"],
        "scale": args.scale,
        "pooling": bundle.spec["pooling"],
        "candidates": len(dataset),
        "feature_dimension": int(feature_batches[0].shape[1]),
        "batch_size": batch_size,
        "processor": processor.metadata(),
    }
    expected_dimension = int(bundle.spec["expected_feature_dimension"])
    if metadata["feature_dimension"] != expected_dimension:
        raise RuntimeError(
            f"Expected {expected_dimension} features for {args.model_key}, got {metadata['feature_dimension']}"
        )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
