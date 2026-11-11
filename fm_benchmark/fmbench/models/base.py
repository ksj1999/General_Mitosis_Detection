"""Encoder interface and model specs.

A model is described by a YAML file in ``configs/models/``. The ``adapter`` key picks the code
that builds the encoder; ``params`` go to that adapter. Every encoder returns a dict of feature
tensors per batch:

    cls         class token                         (ViT encoders)
    patch_mean  mean of the patch tokens, registers excluded  (ViT encoders)
    pooled      global-average-pooled feature map   (CNN encoders)

Embedding modes used by the probe are built from these keys (see ``MODES``).
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from PIL import Image

from .. import paths

# mode name -> feature keys concatenated in this order
MODES: dict[str, tuple[str, ...]] = {
    "cls": ("cls",),
    "cls+mean": ("cls", "patch_mean"),
    "mean": ("patch_mean",),
    "pooled": ("pooled",),
}
# Which of our modes fills each column group of the Virchow2 tables.
TABLE_GROUP = {"cls+mean": "CLS+Mean", "cls": "CLS-Only", "mean": "CLS+Mean", "pooled": "CLS+Mean"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelSpec:
    name: str
    adapter: str
    display_name: str = ""
    params: dict = field(default_factory=dict)
    modes: list = field(default_factory=lambda: ["cls", "cls+mean"])
    batch_size: int = 64
    precision: str = "fp16"           # fp16 | bf16 | fp32 (autocast on CUDA only)
    resize_to: int = 224              # every tile is resized to this square size
    trained_on: list = field(default_factory=list)  # dataset families seen in training (leakage flags)
    paper_name: Optional[str] = None  # row name in the Virchow2 tables, for reproduction checks
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelSpec":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.setdefault("name", Path(path).stem)
        raw.setdefault("display_name", raw["name"])
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(**raw)


def list_specs() -> dict[str, ModelSpec]:
    specs = {}
    for p in sorted(paths.MODEL_CONFIG_DIR.glob("*.yaml")):
        s = ModelSpec.from_yaml(p)
        specs[s.name] = s
    return specs


def load_spec(name_or_path: str) -> ModelSpec:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return ModelSpec.from_yaml(p)
    specs = list_specs()
    if name_or_path not in specs:
        raise KeyError(f"unknown model {name_or_path!r}; known: {', '.join(specs)}")
    return specs[name_or_path]


def resolve_path(value: str) -> Path:
    """Expand ~ and env vars; relative paths are relative to the repository root."""
    import os

    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (paths.REPO_DIR / p)


def file_fingerprint(path: Path, head_bytes: int = 64 << 20) -> str:
    """sha256 of the size plus the first ``head_bytes`` bytes (fast for multi-GB checkpoints)."""
    path = Path(path)
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()[:16]


class TileTransform:
    """PIL tile -> normalised tensor: bicubic resize to ``size``, then the model's mean/std.

    A module-level class (not a closure) so DataLoader workers can pickle it under spawn/forkserver.
    """

    def __init__(self, size: int, mean, std):
        self.size = int(size)
        self.mean = torch.tensor([float(v) for v in mean]).view(3, 1, 1)
        self.std = torch.tensor([float(v) for v in std]).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255)
        return (arr - self.mean) / self.std


class Encoder(abc.ABC):
    """Frozen feature extractor. Subclasses set ``self.model``, ``self.mean`` and ``self.std``."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.mean: tuple = IMAGENET_MEAN
        self.std: tuple = IMAGENET_STD
        self.model: torch.nn.Module

    # -- preprocessing: one uniform rule for all models -----------------------------
    def transform(self) -> Callable[[Image.Image], torch.Tensor]:
        return TileTransform(self.spec.resize_to, self.mean, self.std)

    @abc.abstractmethod
    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return feature tensors for a normalised batch (B, 3, H, W)."""

    def fingerprint(self) -> dict:
        return {}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(self.device, non_blocking=True)
        prec = self.spec.precision
        use_amp = self.device.type == "cuda" and prec in ("fp16", "bf16")
        dtype = torch.float16 if prec == "fp16" else torch.bfloat16
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            out = self.features(x)
        return {k: v.float() for k, v in out.items()}


def vit_token_features(tokens: torch.Tensor, num_prefix_tokens: int) -> dict[str, torch.Tensor]:
    """Split ViT output tokens (B, N, D) into the class token and the mean of the patch tokens.

    Token order is [CLS, register_1..register_R, patch_1..patch_P]; ``num_prefix_tokens`` = 1 + R.
    """
    if tokens.ndim != 3 or tokens.shape[1] <= num_prefix_tokens:
        raise RuntimeError(f"unexpected token tensor shape {tuple(tokens.shape)}")
    return {"cls": tokens[:, 0], "patch_mean": tokens[:, num_prefix_tokens:].mean(dim=1)}
