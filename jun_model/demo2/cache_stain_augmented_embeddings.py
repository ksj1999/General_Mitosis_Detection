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
from stain_augmentation import STAIN_MODES, hed_jitter


SOURCE_REPOSITORY = "https://github.com/ksj1999/General_Mitosis_Detection"
SOURCE_REVISION = "30e9ee9bd947dfdda513264194dd25ec3d873e25"
SOURCE_FILE = "aaron_model/scripts/train_patch.py"


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
            patch = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return patch, int(row["candidate_id"]), int(row["label"]), int(row["image_id"]), row["scanner"]


def collate_raw(batch):
    images, candidate_ids, labels, image_ids, scanners = zip(*batch)
    pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
    return pixels, candidate_ids, labels, image_ids, scanners


def augmented_pil_images(pixels, sigma, mode):
    augmented = hed_jitter(pixels, sigma, mode)
    arrays = (
        augmented.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache frozen embeddings for deterministic HED-jittered patch variants."
    )
    parser.add_argument("--manifest", type=Path, default=Path("jun_model/demo2/data/candidates.csv"))
    parser.add_argument("--patch-dir", type=Path, default=Path("jun_model/demo2/patches"))
    parser.add_argument("--scale", choices=("20x", "40x"), required=True)
    parser.add_argument(
        "--model-key", choices=("phikon_v2", "hibou_b", "virchow2"), required=True
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("jun_model/demo2/configs/foundation_models.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--stain-mode", choices=STAIN_MODES, default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("jun_model/demo2/.hf_cache"))
    parser.add_argument("--max-candidates", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.variants < 1:
        raise ValueError("--variants must be at least 1")
    if not 0 <= args.sigma <= 1:
        raise ValueError("--sigma must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    dataset = PatchDataset(args.manifest, args.patch_dir, args.scale, args.max_candidates)
    bundle = load_foundation_model(args.model_key, args.registry, args.cache_dir)
    model = bundle.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch_size = args.batch_size or int(bundle.spec["default_batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_raw,
        pin_memory=device.type == "cuda",
    )

    torch.manual_seed(args.seed)
    variant_batches = [[] for _ in range(args.variants)]
    candidate_ids = []
    labels = []
    image_ids = []
    scanners = []
    with torch.inference_mode():
        for raw, batch_candidate_ids, batch_labels, batch_image_ids, batch_scanners in tqdm(
            loader, desc=f"{bundle.spec['display_name']} HED {args.scale}", unit="batch"
        ):
            for variant in range(args.variants):
                images = augmented_pil_images(raw, args.sigma, args.stain_mode)
                pixels = bundle.processor(images).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    features = extract_foundation_features(model, pixels, bundle.spec)
                variant_batches[variant].append(features.float().cpu().numpy())
            candidate_ids.extend(batch_candidate_ids)
            labels.extend(batch_labels)
            image_ids.extend(batch_image_ids)
            scanners.extend(batch_scanners)

    features = np.stack([np.concatenate(batches) for batches in variant_batches])
    expected_dimension = int(bundle.spec["expected_feature_dimension"])
    if features.shape != (args.variants, len(dataset), expected_dimension):
        raise RuntimeError(f"Unexpected augmented feature shape {features.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        scanners=np.asarray(scanners),
        scale=np.asarray([args.scale]),
        model_id=np.asarray([bundle.spec["repository"]]),
        model_key=np.asarray([args.model_key]),
        pooling=np.asarray([bundle.spec["pooling"]]),
        augmentation=np.asarray(["hed_jitter"]),
        stain_mode=np.asarray([args.stain_mode]),
        sigma=np.asarray([args.sigma], dtype=np.float32),
        seed=np.asarray([args.seed], dtype=np.int64),
    )
    metadata = {
        "model_key": args.model_key,
        "model_id": bundle.spec["repository"],
        "model_revision": bundle.revision,
        "scale": args.scale,
        "pooling": bundle.spec["pooling"],
        "candidates": len(dataset),
        "variants": args.variants,
        "feature_dimension": expected_dimension,
        "batch_size": batch_size,
        "augmentation": {
            "name": "hed_jitter",
            "mode": args.stain_mode,
            "sigma": args.sigma,
            "seed": args.seed,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_file": SOURCE_FILE,
        },
        "use_policy": "training candidates only; validation and test use plain embeddings",
        "processor": bundle.processor.metadata(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
