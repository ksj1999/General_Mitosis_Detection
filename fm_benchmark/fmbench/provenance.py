"""What produced a result: a hash of the fmbench sources, library versions and the weights revision.

Recorded for information in embedding meta.json and in runs.jsonl. None of it is part of the embedding
cache key, so recording it never invalidates a cache. The git commit alone is not enough: on TGI the
benchmark is an rsynced copy that git does not track.

``cli.main`` calls ``code_sha()`` before running a command, so the hash describes the sources at process
start even if the checkout is updated while the process runs. Modules imported lazily later (task
modules, for example) can still differ from the hashed tree.
"""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path


@functools.lru_cache(maxsize=1)
def code_sha() -> str:
    """sha1 over the fmbench Python sources (path and content), first 12 hex digits."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha1()
    try:
        for p in sorted(root.rglob("*.py")):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    except OSError:  # a file replaced while hashing; provenance must never break a run
        return "unknown"
    return h.hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def library_versions() -> dict:
    out = {}
    for name in ("torch", "torchvision", "timm", "transformers", "huggingface_hub", "peft", "numpy", "PIL", "sklearn"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 - optional dependency
            continue
    return out


def weights_revision(encoder) -> str | None:
    """The Hugging Face snapshot the encoder's weights came from, when that can be told."""
    try:
        model = getattr(encoder, "model", None)
        rev = getattr(getattr(model, "config", None), "_commit_hash", None)  # transformers
        if rev:
            return str(rev)
        cfg = getattr(model, "pretrained_cfg", {}) or {}  # timm models from the hub
        if cfg.get("hf_hub_id"):
            from huggingface_hub import try_to_load_from_cache

            for name in ("model.safetensors", "pytorch_model.bin"):
                hit = try_to_load_from_cache(cfg["hf_hub_id"], name)
                if isinstance(hit, str):
                    return Path(hit).parent.name
        rev = getattr(encoder, "revision", None)  # demo2 LoRA adapters record their base revision
        return str(rev) if rev else None
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None


def embedding_provenance(encoder) -> dict:
    return {"code_sha": code_sha(), "versions": library_versions(), "weights_revision": weights_revision(encoder)}
