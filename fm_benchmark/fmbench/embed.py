"""Extract and cache frozen embeddings for one (model, task) pair.

Layout: <cache>/embeddings/<model>/<task>[__limitN]/<split>/
    <key>.npy      float16 features per key returned by the encoder (cls, patch_mean, pooled)
    labels.npy     int64 labels
    ids.txt        one sample id per line
    meta.json      model fingerprint, split fingerprint, counts, completion flag
    progress.json  rows written so far (lets an interrupted run resume)

A cache is reused only if everything that changes the features matches: split, model fingerprint
(weights, prefix tokens, pooling), resize, mean/std, precision, adapter and the config's params.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import paths
from .models.base import MODES, Encoder, ModelSpec
from .tasks.base import SplitSource, TileTask, limit_indices


class _TileDataset(Dataset):
    def __init__(self, split: SplitSource, transform, start: int = 0):
        self.split, self.transform, self.start = split, transform, start

    def __len__(self):
        return len(self.split) - self.start

    def __getitem__(self, i):
        return self.transform(self.split.image(self.start + i))


def cache_dir(model: str, task: str, limit: int = 0) -> Path:
    name = task if not limit else f"{task}__limit{limit}"
    return paths.embeddings_dir(model, name)


def task_splits(task: TileTask, limit: int = 0) -> dict[str, SplitSource]:
    sp = task.splits()
    if limit:
        sp = {k: v.subset(limit_indices(v.labels, limit)) for k, v in sp.items()}
    return sp


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


CACHE_KEYS = ("split_fingerprint", "n", "model_fingerprint", "resize_to", "mean", "std", "precision", "adapter",
              "spec_params")


def _jsonable(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def is_complete(model: str, task: str, split: str, limit: int = 0) -> bool:
    meta = cache_dir(model, task, limit) / split / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("complete", False)


def embed_split(encoder: Encoder, spec: ModelSpec, task: TileTask, split_name: str, split: SplitSource,
                out_dir: Path, model_fp: dict, num_workers: int = 8, batch_size: int | None = None,
                overwrite: bool = False, log=print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path, prog_path = out_dir / "meta.json", out_dir / "progress.json"
    split_fp = split.fingerprint()
    want = _jsonable({"model": spec.name, "model_fingerprint": model_fp, "task": task.name, "split": split_name,
                      "split_fingerprint": split_fp, "n": len(split), "resize_to": spec.resize_to,
                      "mean": [float(v) for v in encoder.mean], "std": [float(v) for v in encoder.std],
                      "precision": spec.precision, "adapter": spec.adapter, "spec_params": spec.params})

    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        diff = [k for k in CACHE_KEYS if meta.get(k) != want[k]]
        if diff:
            raise RuntimeError(f"{out_dir}: cached embeddings differ from the current setup in {diff}; "
                               f"rerun with --overwrite")
        if meta.get("complete"):
            log(f"  {split_name}: cached ({meta['n']} tiles)")
            return out_dir
    else:
        for f in out_dir.glob("*"):
            f.unlink()
        # labels and ids first, meta.json last: a run killed before meta.json exists starts fresh next time
        with open(out_dir / "labels.npy.tmp", "wb") as fh:
            np.save(fh, split.labels.astype(np.int64))
        os.replace(out_dir / "labels.npy.tmp", out_dir / "labels.npy")
        (out_dir / "ids.txt.tmp").write_text("\n".join(map(str, split.ids.tolist())) + "\n")
        os.replace(out_dir / "ids.txt.tmp", out_dir / "ids.txt")
        _write_json(meta_path, {**want, "complete": False})

    done = json.loads(prog_path.read_text())["done"] if prog_path.exists() else 0
    n = len(split)
    loader = DataLoader(_TileDataset(split, encoder.transform(), start=done), batch_size=batch_size or spec.batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=encoder.device.type == "cuda",
                        persistent_workers=False)
    arrays: dict[str, np.memmap] = {}
    for key_file in out_dir.glob("*.npy"):  # feature arrays already started by an interrupted run
        if key_file.stem != "labels":
            arrays[key_file.stem] = np.load(key_file, mmap_mode="r+")
    t0, pos, last_flush = time.time(), done, time.time()
    for batch in loader:
        feats = encoder(batch)
        for k, v in feats.items():
            if k not in arrays:
                arrays[k] = np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=np.float16,
                                                      shape=(n, v.shape[1]))
            arrays[k][pos:pos + v.shape[0]] = v.cpu().numpy().astype(np.float16)
        pos += batch.shape[0]
        if time.time() - last_flush > 60 or pos == n:
            for a in arrays.values():
                a.flush()
            _write_json(prog_path, {"done": pos})
            rate = (pos - done) / max(1e-6, time.time() - t0)
            log(f"  {split_name}: {pos}/{n} tiles ({rate:.0f}/s)")
            last_flush = time.time()
    for a in arrays.values():
        a.flush()
    if pos != n:
        raise RuntimeError(f"{split_name}: wrote {pos} of {n} rows")
    dims = {k: int(a.shape[1]) for k, a in arrays.items()}
    from .provenance import embedding_provenance

    _write_json(meta_path, {**want, "complete": True, "dims": dims, "seconds": round(time.time() - t0, 1),
                            "provenance": embedding_provenance(encoder)})  # information only, not a cache key
    prog_path.unlink(missing_ok=True)
    return out_dir


def embed_task(spec: ModelSpec, encoder: Encoder, task: TileTask, limit: int = 0, num_workers: int = 8,
               batch_size: int | None = None, overwrite: bool = False, log=print) -> Path:
    root = cache_dir(spec.name, task.name, limit)
    fp = encoder.fingerprint()
    log(f"[embed] {spec.name} x {task.name} -> {root}")
    for split_name, split in task_splits(task, limit).items():
        embed_split(encoder, spec, task, split_name, split, root / split_name, fp, num_workers, batch_size, overwrite, log)
    return root


def load_features(model: str, task: str, split: str, mode: str, limit: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    d = cache_dir(model, task, limit) / split
    meta = json.loads((d / "meta.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError(f"{d}: embeddings are incomplete; run `fmbench embed` first")
    keys = MODES[mode]
    missing = [k for k in keys if not (d / f"{k}.npy").exists()]
    if missing:
        raise KeyError(f"{d}: mode {mode!r} needs {missing}; this encoder gives {sorted(meta.get('dims', {}))}")
    x = np.concatenate([np.load(d / f"{k}.npy").astype(np.float32) for k in keys], axis=1)
    y = np.load(d / "labels.npy")
    return x, y, meta


def available_modes(model: str, task: str, limit: int = 0) -> list[str]:
    d = cache_dir(model, task, limit) / "train"
    if not (d / "meta.json").exists():
        return []
    have = set(json.loads((d / "meta.json").read_text()).get("dims", {}))
    return [m for m, keys in MODES.items() if set(keys) <= have]


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
