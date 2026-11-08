#!/usr/bin/env python3
"""
Shared helpers for the add-on study (see ADDONS_PROTOCOL.md).

  run_config(fold, tag, seed)  -> dict: the "config" block of results/{fold}_{tag}_seed{seed}.json,
                                  filled with the defaults of runs made before the block existed
  load_patches(path)           -> uint8 tensor (N,128,128,3), cached per path
  load_model(fold, tag, seed, device) -> Net with the checkpoint loaded, in eval mode
  target_halves(image_ids)     -> boolean mask, True for half A (sorted case ids [::2])
  get_device()
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patch import ROOT, Net  # noqa: E402

FOLDS = ["CS2", "S360", "XR"]
DEFAULTS = {"patches": "data/patches_128.npz", "n_domains": 2, "mixstyle": False,
            "stain_mode": "both", "sigma": 0.05, "ema": 0.0, "soft_labels": False, "uda_half": None}
_PATCH_CACHE = {}


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def run_config(fold, tag, seed):
    res = json.loads((ROOT / "results" / f"{fold}_{tag}_seed{seed}.json").read_text())
    cfg = dict(DEFAULTS)
    cfg.update({"variant": res.get("variant", tag), "tag": tag})
    cfg.update(res.get("config", {}))
    cfg["threshold"] = res.get("threshold")
    return cfg


def load_patches(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    key = str(p)
    if key not in _PATCH_CACHE:
        _PATCH_CACHE[key] = torch.from_numpy(np.load(p)["X"].copy())
    return _PATCH_CACHE[key]


def load_model(fold, tag, seed, device, cfg=None):
    cfg = cfg or run_config(fold, tag, seed)
    model = Net(n_domains=int(cfg["n_domains"])).to(device)     # MixStyle is a no-op at eval; keys unchanged
    state = torch.load(ROOT / "checkpoints" / f"{fold}_{tag}_seed{seed}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def target_halves(image_ids):
    """True for rows whose case belongs to half A (sorted unique case ids, every other one from the first)."""
    image_ids = np.asarray(image_ids)
    cases = np.sort(np.unique(image_ids))
    return np.isin(image_ids, cases[::2])
