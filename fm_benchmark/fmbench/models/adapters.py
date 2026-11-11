"""Adapters that turn a ModelSpec into an Encoder.

    timm             any timm model, including Hugging Face hub models ("hf-hub:org/name")
    transformers     Hugging Face AutoModel image encoders (Phikon, Phikon-v2, Hibou)
    torchvision      torchvision CNNs, ImageNet weights or a fine-tuned checkpoint
                     (for example the ResNet-18 of aaron_model/)
    demo2_lora       LoRA adapters trained by jun_model/demo2 (Virchow2, Phikon-v2, Hibou-B)

Add a new adapter by subclassing Encoder and registering it in ADAPTERS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .base import Encoder, ModelSpec, file_fingerprint, resolve_path, vit_token_features


class CTransPathConvStem(torch.nn.Module):
    """Convolutional patch embedding of CTransPath (Wang et al. 2022), for the timm Swin-T port
    ``hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath``. Returns NHWC like timm's PatchEmbed."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert patch_size == 4 and embed_dim % 8 == 0
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        stem, cin, cout = [], in_chans, embed_dim // 8
        for _ in range(2):
            stem += [torch.nn.Conv2d(cin, cout, 3, 2, 1, bias=False), torch.nn.BatchNorm2d(cout), torch.nn.ReLU(inplace=True)]
            cin, cout = cout, cout * 2
        stem.append(torch.nn.Conv2d(cin, embed_dim, 1))
        self.proj = torch.nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


LOCAL_LAYERS = {"CTransPathConvStem": CTransPathConvStem}


def _resolve_timm_kwargs(kwargs: dict) -> dict:
    """Map string values such as "SwiGLUPacked" or "SiLU" to the timm / torch classes."""
    import timm.layers

    out = {}
    for k, v in (kwargs or {}).items():
        if isinstance(v, str) and k.endswith("_layer"):
            obj = LOCAL_LAYERS.get(v) or getattr(timm.layers, v, None) or getattr(torch.nn, v, None)
            if obj is None:
                raise ValueError(f"cannot resolve {k}={v!r} in timm.layers or torch.nn")
            out[k] = obj
        else:
            out[k] = v
    return out


class TimmEncoder(Encoder):
    """params: model (timm name or "hf-hub:org/repo"), timm_kwargs, num_prefix_tokens (optional),
    pretrained (default true), checkpoint (optional local weights), pool ("tokens", "cnn" for NCHW maps,
    "nhwc" for Swin token maps), mean/std (override the pretrained config)."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        import timm

        p = spec.params
        kwargs = _resolve_timm_kwargs(p.get("timm_kwargs", {}))
        self.model = timm.create_model(p["model"], pretrained=bool(p.get("pretrained", True)), **kwargs)
        if p.get("checkpoint"):
            sd = torch.load(resolve_path(p["checkpoint"]), map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if unexpected:
                raise ValueError(f"unexpected keys in checkpoint: {unexpected[:5]}")
        self.model.eval().to(device)
        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        self.mean = tuple(p.get("mean", cfg.get("mean", self.mean)))
        self.std = tuple(p.get("std", cfg.get("std", self.std)))
        self.num_prefix = int(p.get("num_prefix_tokens", getattr(self.model, "num_prefix_tokens", 1)))
        self.pool = p.get("pool", "tokens")

    def features(self, x):
        out = self.model.forward_features(x)
        if self.pool == "nhwc":  # Swin-style token map (B, H, W, C): mean of all tokens
            return {"pooled": out.mean(dim=(1, 2))}
        if self.pool == "cnn" or out.ndim == 4:  # feature map: CNNs and ConvNeXt give (B, C, H, W); timm models
            # that set output_fmt = "NHWC" (Swin, Swin-v2, MambaOut, ...) give (B, H, W, C)
            fmt = "NCHW" if self.pool == "cnn" else getattr(self.model, "output_fmt", "NCHW")
            fmt = str(getattr(fmt, "value", fmt)).upper()  # a plain string, or timm's Format enum (str() gives "Format.NHWC")
            return {"pooled": out.mean(dim=(1, 2) if fmt == "NHWC" else (2, 3))}
        return vit_token_features(out, self.num_prefix)

    def fingerprint(self):
        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        fp = {"timm_model": self.spec.params["model"], "hf_hub_id": cfg.get("hf_hub_id"),
              "num_prefix": self.num_prefix, "pool": self.pool}
        if self.spec.params.get("checkpoint"):
            fp["checkpoint_sha"] = file_fingerprint(resolve_path(self.spec.params["checkpoint"]))
        return fp


class TransformersEncoder(Encoder):
    """params: model (HF repo id or local dir), num_prefix_tokens (1 + registers),
    trust_remote_code (default false), revision (optional)."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        from transformers import AutoImageProcessor, AutoModel

        p = spec.params
        kw = {"trust_remote_code": bool(p.get("trust_remote_code", False))}
        if p.get("revision"):
            kw["revision"] = p["revision"]
        self.model = AutoModel.from_pretrained(p["model"], **kw).eval().to(device)
        mean, std = self.mean, self.std
        try:
            proc = AutoImageProcessor.from_pretrained(p["model"], **kw)
            mean, std = getattr(proc, "image_mean", mean), getattr(proc, "image_std", std)
        except Exception:  # noqa: BLE001 - some repos ship no processor; fall back to ImageNet stats
            pass
        self.mean = tuple(p.get("mean", mean))  # YAML overrides apply even without a processor
        self.std = tuple(p.get("std", std))
        self.num_prefix = int(p.get("num_prefix_tokens", 1))

    def features(self, x):
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return vit_token_features(tokens, self.num_prefix)

    def fingerprint(self):
        return {"hf_model": self.spec.params["model"],
                "revision": getattr(self.model.config, "_commit_hash", None), "num_prefix": self.num_prefix}


class TorchvisionEncoder(Encoder):
    """params: arch (resnet18, resnet50, ...), weights ("imagenet" or "none"),
    checkpoint (optional path), state_dict_prefix (prefix of the encoder keys in the checkpoint,
    "encoder." for aaron_model/scripts/train_patch.py)."""

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        import torchvision

        p = spec.params
        arch = p.get("arch", "resnet18")
        weights = "DEFAULT" if p.get("weights", "imagenet") == "imagenet" else None
        net = getattr(torchvision.models, arch)(weights=weights)
        net.fc = torch.nn.Identity()
        if p.get("checkpoint"):
            ckpt = resolve_path(p["checkpoint"])
            sd = torch.load(ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            prefix = p.get("state_dict_prefix", "encoder.")
            enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not enc:
                raise ValueError(f"{ckpt}: no keys start with {prefix!r}")
            missing, unexpected = net.load_state_dict(enc, strict=False)
            missing = [k for k in missing if not k.startswith("fc.")]
            if missing or unexpected:
                raise ValueError(f"{ckpt}: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model = net.eval().to(device)

    def features(self, x):
        return {"pooled": self.model(x)}

    def fingerprint(self):
        p = self.spec.params
        fp = {"arch": p.get("arch", "resnet18"), "weights": p.get("weights", "imagenet")}
        if p.get("checkpoint"):
            fp["checkpoint_sha"] = file_fingerprint(resolve_path(p["checkpoint"]))
        return fp


class Demo2LoRAEncoder(Encoder):
    """LoRA-adapted encoder from jun_model/demo2 (run_lora_probe_validation.py checkpoints).

    params: checkpoint (path to <fold>_seed<k>_adapter.pt), demo2_dir (default jun_model/demo2),
    registry (default <demo2_dir>/configs/foundation_models.json),
    hf_cache_dir (default <demo2_dir>/.hf_cache), num_prefix_tokens (default from the backend).
    Only the encoder is used; the mitosis head and the scanner head are ignored.
    """

    PREFIX = {"timm_virchow2": 5, "transformers": 1}
    PREFIX_BY_MODEL = {"virchow2": 5, "hibou_b": 5, "phikon_v2": 1}  # CLS + registers

    def __init__(self, spec: ModelSpec, device: torch.device):
        super().__init__(spec, device)
        p = spec.params
        demo2 = resolve_path(p.get("demo2_dir", "jun_model/demo2"))
        if str(demo2) not in sys.path:
            sys.path.insert(0, str(demo2))
        from lora_evaluation_utils import restore_lora_probe  # noqa: E402 (demo2 module)

        registry = resolve_path(p["registry"]) if p.get("registry") else demo2 / "configs" / "foundation_models.json"
        cache = resolve_path(p["hf_cache_dir"]) if p.get("hf_cache_dir") else demo2 / ".hf_cache"
        self.ckpt_path = resolve_path(p["checkpoint"])
        probe, bundle, ckpt = restore_lora_probe(self.ckpt_path, registry, cache, device)
        self.model = probe.encoder.eval()
        self.backend = bundle.spec["backend"]
        self.revision = bundle.revision
        self.ckpt_meta = {k: ckpt.get(k) for k in ("model_key", "fold", "seed", "lora_rank", "stain_augmentation", "grl_strength")}
        meta = bundle.processor.metadata() if hasattr(bundle.processor, "metadata") else {}
        self.mean = tuple(p.get("mean", meta.get("image_mean", meta.get("mean", self.mean))))
        self.std = tuple(p.get("std", meta.get("image_std", meta.get("std", self.std))))
        default_prefix = self.PREFIX_BY_MODEL.get(ckpt.get("model_key"), self.PREFIX.get(self.backend, 1))
        self.num_prefix = int(p.get("num_prefix_tokens", default_prefix))

    def features(self, x):
        if self.backend == "timm_virchow2":
            tokens = self.model(x)
        else:
            out = self.model(pixel_values=x)
            tokens = out.last_hidden_state
        return vit_token_features(tokens, self.num_prefix)

    def fingerprint(self):
        return {"checkpoint_sha": file_fingerprint(self.ckpt_path), "base_revision": self.revision,
                "num_prefix": self.num_prefix, **self.ckpt_meta}


ADAPTERS = {
    "timm": TimmEncoder,
    "transformers": TransformersEncoder,
    "torchvision": TorchvisionEncoder,
    "demo2_lora": Demo2LoRAEncoder,
}


def build_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    if spec.adapter not in ADAPTERS:
        raise KeyError(f"unknown adapter {spec.adapter!r}; known: {', '.join(ADAPTERS)}")
    return ADAPTERS[spec.adapter](spec, device)
