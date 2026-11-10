import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class FoundationModelBundle:
    key: str
    spec: dict
    model: torch.nn.Module
    processor: object
    revision: str


class HuggingFaceProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def metadata(self):
        return self.processor.to_dict()


class TimmProcessor:
    def __init__(self, transform, pretrained_cfg):
        self.transform = transform
        self.pretrained_cfg = pretrained_cfg

    def __call__(self, images):
        return torch.stack([self.transform(image) for image in images])

    def metadata(self):
        return dict(self.pretrained_cfg)


def load_registry(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_foundation_model(model_key, registry_path, cache_dir):
    registry = load_registry(registry_path)
    if model_key not in registry:
        raise KeyError(f"Unknown model key {model_key!r}; choose from {', '.join(sorted(registry))}")
    spec = registry[model_key]
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    if spec["backend"] == "transformers":
        from huggingface_hub import HfApi, get_token, hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoImageProcessor, AutoModel

        token = get_token()
        source = spec["repository"]
        if spec["access"] != "public":
            repo_cache = cache_dir / f"models--{source.replace('/', '--')}" / "snapshots"
            cached_snapshot = next(
                (
                    path
                    for path in sorted(repo_cache.glob("*"), reverse=True)
                    if (path / "config.json").exists()
                    and any(path.glob("*.safetensors"))
                    and (path / "preprocessor_config.json").exists()
                ),
                None,
            )
            if cached_snapshot is not None:
                source = str(cached_snapshot)
            else:
                files = HfApi().list_repo_files(source, token=token)
                required_files = [
                    name for name in files if Path(name).suffix in {".json", ".py", ".safetensors"}
                ]
                downloaded = []
                for name in required_files:
                    for attempt in range(5):
                        try:
                            path = hf_hub_download(source, name, cache_dir=cache_dir, token=token)
                            downloaded.append(path)
                            break
                        except GatedRepoError:
                            if attempt == 4:
                                raise
                            time.sleep(2**attempt)
                source = str(Path(downloaded[0]).parent)
        kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": bool(spec["trust_remote_code"]),
            "local_files_only": spec["access"] != "public",
        }
        if token is not None and spec["access"] == "public":
            kwargs["token"] = token
        processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        model = AutoModel.from_pretrained(source, **kwargs)
        revision = str(
            getattr(model.config, "_commit_hash", None)
            or (Path(source).name if spec["access"] != "public" else "unresolved")
        )
        return FoundationModelBundle(model_key, spec, model, HuggingFaceProcessor(processor), revision)

    if spec["backend"] == "timm_virchow2":
        from huggingface_hub import get_token, hf_hub_download
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked
        from timm.models import load_checkpoint

        token = get_token()
        repo_cache = cache_dir / f"models--{spec['repository'].replace('/', '--')}" / "snapshots"
        cached_configs = sorted(repo_cache.glob("*/config.json"), reverse=True)
        cached_snapshot = next(
            (path.parent for path in cached_configs if (path.parent / "model.safetensors").exists()),
            None,
        )
        if cached_snapshot is None:
            config_path = hf_hub_download(
                spec["repository"], "config.json", cache_dir=cache_dir, token=token
            )
            weights_path = hf_hub_download(
                spec["repository"], "model.safetensors", cache_dir=cache_dir, token=token
            )
        else:
            config_path = cached_snapshot / "config.json"
            weights_path = cached_snapshot / "model.safetensors"
        model_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        model = timm.create_model(
            model_config["architecture"],
            pretrained=False,
            pretrained_cfg=model_config["pretrained_cfg"],
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            **model_config["model_args"],
        )
        load_checkpoint(model, weights_path, use_ema=False)
        data_config = resolve_data_config(model.pretrained_cfg, model=model)
        processor = TimmProcessor(create_transform(**data_config), model.pretrained_cfg)
        revision = Path(config_path).parent.name
        return FoundationModelBundle(model_key, spec, model, processor, revision)

    raise ValueError(f"Unsupported backend {spec['backend']!r}")


def extract_foundation_features(model, pixels, spec):
    backend = spec["backend"]
    if backend == "transformers":
        outputs = model(pixel_values=pixels)
        pooling = spec["pooling"]
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooling == "pooler_output":
            if pooler_output is None:
                raise RuntimeError("Model registry requires pooler_output, but the model did not return it")
            return pooler_output
        if pooling == "pooler_or_cls":
            if pooler_output is not None:
                return pooler_output
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                if torch.is_tensor(outputs) and outputs.ndim == 2:
                    return outputs
                raise RuntimeError("Could not obtain pooled or CLS features from Transformers output")
            return last_hidden_state[:, 0]

    if backend == "timm_virchow2":
        tokens = model(pixels)
        if tokens.ndim != 3 or tokens.shape[1] < 6:
            raise RuntimeError(f"Unexpected Virchow2 output shape {tuple(tokens.shape)}")
        class_token = tokens[:, 0]
        patch_tokens = tokens[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

    raise ValueError(f"Unsupported feature extraction backend {backend!r}")
