"""Download resume and skip rules, tar extraction without the ``filter`` argument, smoke runs in the
leaderboard, DataLoader workers under spawn, and the embedding cache check."""
import hashlib
import http.server
import json
import multiprocessing as mp
import pickle
import tarfile
import threading

import numpy as np
import pytest
import torch

from test_pipeline import ColourTask

PAYLOAD = bytes(range(256)) * 64  # 16 KiB


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


@pytest.fixture()
def s3_like_server():
    """Refuses HEAD (like figshare's S3 redirect target) and answers 416 with no Content-Range at EOF."""
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self.send_response(403)
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            hits.append(rng)
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = PAYLOAD[start:]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.bin", hits
    srv.shutdown()


def test_fetch_accepts_complete_part_after_416(env, s3_like_server):
    from fmbench.download import fetch
    from fmbench.tasks.base import DownloadSpec

    url, hits = s3_like_server
    d = env / "dl"
    d.mkdir()
    (d / "f.bin.part").write_bytes(PAYLOAD)  # run was killed during the checksum of a finished download
    spec = DownloadSpec(url, "f.bin", md5=hashlib.md5(PAYLOAD).hexdigest())
    out = fetch(spec, d, retries=3, log=lambda *a: None)
    assert out.read_bytes() == PAYLOAD and hits == [f"bytes={len(PAYLOAD)}-"]


def test_fetch_restarts_after_416_without_checksum(env, s3_like_server):
    from fmbench.download import fetch
    from fmbench.tasks.base import DownloadSpec

    url, hits = s3_like_server
    d = env / "dl"
    d.mkdir()
    (d / "f.bin.part").write_bytes(PAYLOAD)
    out = fetch(DownloadSpec(url, "f.bin"), d, retries=3, log=lambda *a: None)
    assert out.read_bytes() == PAYLOAD and hits == [f"bytes={len(PAYLOAD)}-", None]


def test_download_skips_extracted_or_ready_tasks(env, monkeypatch):
    import fmbench.download
    from fmbench.cli import main
    from fmbench.tasks.base import TaskStatus
    from fmbench.tasks.tils import TilsTask

    def no_fetch(*a, **k):
        raise AssertionError("fetch should not be called")

    monkeypatch.setattr(fmbench.download, "fetch", no_fetch)
    task = TilsTask()
    task.raw_dir.mkdir(parents=True)
    (task.raw_dir / ".extracted-TCGA-TILs.tar.gz").write_text("ok\n")  # archive deleted after prepare
    main(["download", "tils", "--no-prepare"])
    (task.raw_dir / ".extracted-TCGA-TILs.tar.gz").unlink()
    monkeypatch.setattr(TilsTask, "status", lambda self: TaskStatus(True, "ready"))  # data placed by hand
    main(["download", "tils", "--no-prepare"])


def test_extract_tar_without_filter_argument(env, monkeypatch):
    from fmbench.download import extract

    monkeypatch.delattr(tarfile, "data_filter", raising=False)  # Python < 3.10.12 / 3.11.4
    src = env / "src"
    (src / "keep").mkdir(parents=True)
    (src / "keep" / "a.txt").write_text("a")
    (src / "skip.txt").write_text("b")
    with tarfile.open(env / "ok.tar", "w") as tf:
        tf.add(src / "keep", arcname="keep")
        tf.add(src / "skip.txt", arcname="skip.txt")
    extract(env / "ok.tar", env / "out", members=("keep/",), log=lambda *a: None)
    assert (env / "out" / "keep" / "a.txt").read_text() == "a" and not (env / "out" / "skip.txt").exists()
    with tarfile.open(env / "bad.tar", "w") as tf:
        tf.add(src / "skip.txt", arcname="../evil.txt")
    with pytest.raises(RuntimeError):
        extract(env / "bad.tar", env / "out2", log=lambda *a: None)
    assert not (env / "evil.txt").exists()


def test_smoke_runs_never_share_a_cell_with_full_runs(env):
    from fmbench.leaderboard import build, our_rows
    from fmbench.runner import append_run

    base = {"time_utc": "2026-01-01T00:00:00+00:00", "seed": 0, "mode": "cls", "model": "virchow2",
            "model_display": "Virchow2 (ours)", "paper_name": "Virchow2", "task": "pcam", "paper_column": "PCam",
            "leakage": []}
    append_run({**base, "limit": 0, "smoke": False, "metrics": {"weighted_f1": 0.935}})
    append_run({**base, "limit": 256, "smoke": True, "metrics": {"weighted_f1": 0.60}})
    rows = {r["mode"]: r for r in our_rows(include_smoke=True)}
    assert rows["cls"]["values"]["PCam"] == pytest.approx(93.5) and rows["cls"]["seeds"]["PCam"] == 1
    assert rows["cls [limit 256]"]["values"]["PCam"] == pytest.approx(60.0)
    text = build().read_text()
    assert "93.5" in text and "60.0" not in text and "limit" not in text
    smoke = build(include_smoke=True, out_dir=env / "results" / "smoke").read_text()
    repro = smoke.split("Reproduction check")[1].split("##")[0]
    assert "limit" not in repro  # smoke rows are never compared with the paper


def _tiny_spec(**kw):
    from fmbench.models.base import ModelSpec

    args = dict(name="tiny", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    args.update(kw)
    return ModelSpec(**args)


def test_embedding_workers_under_spawn(env):
    from fmbench.embed import _TileDataset, embed_task, load_features
    from fmbench.models.adapters import build_encoder

    task = ColourTask()
    spec = _tiny_spec()
    torch.manual_seed(0)
    enc = build_encoder(spec, torch.device("cpu"))
    pickle.dumps(_TileDataset(task.splits()["test"], enc.transform()))  # what spawn/forkserver workers need
    old = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        embed_task(spec, enc, task, num_workers=2, log=lambda *a: None)
    finally:
        mp.set_start_method(old, force=True)
    x, y, _ = load_features("tiny", "pcam", "test", "pooled")
    assert x.shape == (30, 512) and np.isfinite(x).all()


@pytest.mark.parametrize("change", ["mean", "precision", "num_prefix_tokens"])
def test_cache_refuses_changed_preprocessing(env, change):
    from fmbench.embed import embed_task
    from fmbench.models.adapters import build_encoder

    task = ColourTask()
    spec = _tiny_spec()
    enc = build_encoder(spec, torch.device("cpu"))
    root = embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)
    meta = json.loads((root / "test" / "meta.json").read_text())
    assert meta["complete"] and (root / "test" / "labels.npy").exists() and (root / "test" / "ids.txt").exists()
    if change == "mean":
        enc.mean = (0.7, 0.5, 0.7)
    elif change == "precision":
        spec.precision = "fp16"
    else:
        spec.params["num_prefix_tokens"] = 5
    with pytest.raises(RuntimeError, match="differ"):
        embed_task(spec, enc, task, num_workers=0, log=lambda *a: None)
    embed_task(spec, enc, task, num_workers=0, overwrite=True, log=lambda *a: None)  # --overwrite rebuilds
