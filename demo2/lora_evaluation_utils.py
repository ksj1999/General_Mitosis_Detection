import csv
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from foundation_models import load_foundation_model
from run_lora_probe_validation import LoRAProbe


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["candidate_id"]): row for row in csv.DictReader(handle)}


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_paths(directory):
    paths = sorted(Path(directory).glob("*_seed*_adapter.pt"))
    if not paths:
        raise FileNotFoundError(f"No LoRA adapter checkpoints found in {directory}")
    return paths


def restore_lora_probe(checkpoint_path, registry, cache_dir, device):
    checkpoint = load_checkpoint(checkpoint_path)
    bundle = load_foundation_model(checkpoint["model_key"], registry, cache_dir)
    if bundle.spec["repository"] != checkpoint["model_id"]:
        raise ValueError(
            f"Checkpoint expects {checkpoint['model_id']}, got {bundle.spec['repository']}"
        )
    if bundle.revision != checkpoint["model_revision"]:
        raise ValueError(
            f"Checkpoint revision {checkpoint['model_revision']} does not match {bundle.revision}"
        )

    config = LoraConfig(
        r=int(checkpoint["lora_rank"]),
        lora_alpha=int(checkpoint["lora_alpha"]),
        lora_dropout=float(checkpoint["lora_dropout"]),
        target_modules=list(checkpoint["target_modules"]),
        bias="none",
    )
    encoder = get_peft_model(bundle.model, config)
    scanner_classes = 2 if float(checkpoint.get("grl_strength", 0.0)) > 0 else 0
    model = LoRAProbe(
        encoder,
        bundle.spec,
        int(bundle.spec["expected_feature_dimension"]),
        float(checkpoint["head_dropout"]),
        scanner_classes=scanner_classes,
        scanner_hidden_dim=int(checkpoint.get("scanner_hidden_dim", 256)),
    )
    current = model.state_dict()
    unknown = sorted(set(checkpoint["state_dict"]) - set(current))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown parameters: {unknown[:5]}")
    current.update(checkpoint["state_dict"])
    model.load_state_dict(current)
    model = model.to(device).eval()
    return model, bundle, checkpoint


def method_name(checkpoint):
    stain = bool(checkpoint.get("stain_augmentation", False))
    grl = float(checkpoint.get("grl_strength", 0.0)) > 0
    if stain and grl:
        return "lora_stain_grl"
    if stain:
        return "lora_stain"
    if grl:
        return "lora_grl"
    return "lora"


def validate_checkpoint_set(paths):
    seen = set()
    records = []
    for path in paths:
        checkpoint = load_checkpoint(path)
        key = (str(checkpoint["fold"]), int(checkpoint["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate checkpoint for fold={key[0]}, seed={key[1]}")
        seen.add(key)
        records.append((path, checkpoint))
    return records


def safe_mean(values):
    return float(np.mean(values)) if values else None
