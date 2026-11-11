"""End-to-end test of embed -> probe -> leaderboard on a tiny synthetic task (CPU, no downloads)."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from fmbench.tasks.base import FileListSplit, TileTask


class ColourTask(TileTask):
    """Two classes of noisy tiles: reddish vs bluish. Uses the PCam column of the paper table."""

    name = "pcam"
    display_name = "PCam (synthetic)"
    paper_column = "PCam"
    classes = ("blue", "red")
    official_val = False

    def official_splits(self):
        out = {}
        for split, n in (("train", 60), ("test", 30)):
            files, labels = [], []
            d = self.raw_dir / split
            d.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(0 if split == "train" else 1)
            for i in range(n):
                y = i % 2
                f = d / f"{i:03d}.png"
                if not f.exists():
                    base = np.array([200, 60, 60] if y else [60, 60, 200], np.float32)
                    arr = np.clip(base + rng.normal(0, 25, (96, 96, 3)), 0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(f)
                files.append(f)
                labels.append(y)
            out[split] = FileListSplit(files, labels, root=self.raw_dir)
        return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


def test_end_to_end(env):
    from fmbench.embed import available_modes, embed_task, load_features
    from fmbench.leaderboard import build
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec
    from fmbench.probe import ProbeConfig
    from fmbench.runner import read_runs, run_probe

    task = ColourTask()
    sp = task.splits()
    assert {k: len(v) for k, v in sp.items()} == {"train": 54, "val": 6, "test": 30}

    spec = ModelSpec(name="tiny_resnet", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=64)
    torch.manual_seed(0)
    enc = build_encoder(spec, torch.device("cpu"))
    embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)
    assert available_modes("tiny_resnet", "pcam") == ["pooled"]
    x, y, meta = load_features("tiny_resnet", "pcam", "train", "pooled")
    assert x.shape == (54, 512) and meta["complete"]

    # second call reuses the cache
    embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)

    rec = run_probe(spec, task, "pooled", seed=0, cfg=ProbeConfig(iterations=200, eval_every=20), device="cpu",
                    n_boot=50, log=lambda *a: None)
    assert rec["metrics"]["weighted_f1"] > 0.9
    assert read_runs()[-1]["run_id"] == rec["run_id"]

    md = build()
    text = md.read_text()
    assert "tiny_resnet" in text and "Virchow2" in text
    assert "(" in text  # partial average for 1/8 tasks


def test_cache_refuses_changed_data(env):
    from fmbench.embed import embed_task
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec

    task = ColourTask()
    spec = ModelSpec(name="tiny", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    enc = build_encoder(spec, torch.device("cpu"))
    root = embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)
    meta = root / "test" / "meta.json"
    m = json.loads(meta.read_text())
    m["split_fingerprint"] = "different"
    meta.write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)
