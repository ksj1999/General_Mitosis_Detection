"""Train the linear probe on cached embeddings and record the results.

Every probe run appends one JSON line to <results>/runs.jsonl and saves its test predictions
to <results>/predictions/<model>/<task>/<mode>/seed<k>.npz, so that later runs can be paired
with it (McNemar) without recomputing anything.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from . import PROTOCOL_VERSION, paths
from .embed import cache_dir, load_features
from .metrics import bootstrap_ci, classification_metrics, cluster_bootstrap_ci, majority_weighted_f1
from .models.base import ModelSpec
from .probe import ProbeConfig, train_linear_probe
from .provenance import code_sha, library_versions
from .tasks.base import TileTask

LEAKAGE_RULES = {
    # task data family -> model trained_on tags that make the task in-distribution / contaminated
    "midogpp": {"midogpp", "midogpp_breast"},
    "midogpp_nonbreast": {"midogpp"},
}


def git_state() -> dict:
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=paths.REPO_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"commit": _run("rev-parse", "--short", "HEAD"), "dirty": bool(_run("status", "--porcelain", "--", "fm_benchmark"))}


def leakage_flags(spec: ModelSpec, task: TileTask) -> list[str]:
    flags = []
    for family in task.data_sources:
        overlap = LEAKAGE_RULES.get(family, {family}) & set(spec.trained_on)
        if overlap:
            flags.append(f"model trained on {', '.join(sorted(overlap))}; task uses {family}")
    return flags


def runs_path() -> Path:
    return paths.results_root() / "runs.jsonl"


def append_run(record: dict) -> None:
    p = runs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, default=str) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        # Jobs on different nodes append to this file. On NFS, O_APPEND is emulated by the client and two
        # appends can land at the same offset, so hold a lock (NLM on NFS); the lock also makes the client
        # re-read the file size. Without lock support, write anyway: losing the record would be worse.
        locked = False
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
            locked = True
        except OSError:
            pass
        try:
            os.write(fd, line)  # one write() per record
        finally:
            if locked:
                try:
                    fcntl.lockf(fd, fcntl.LOCK_UN)
                except OSError:
                    pass  # the record is written; os.close releases the lock anyway
    finally:
        os.close(fd)


def read_runs() -> list[dict]:
    p = runs_path()
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:  # two jobs on different nodes append to the same file over NFS; skip a torn line, do not crash
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: {p}:{i} is not valid JSON; skipped", file=sys.stderr)
    return out


def test_groups(task: TileTask, model: str, limit: int, n: int) -> np.ndarray | None:
    """Group (patient, slide) of each cached test tile, for tasks that set ``cluster_unit``; else None."""
    if not getattr(task, "cluster_unit", None):
        return None
    try:
        split = task.splits()["test"]
        if split.groups is None:
            return None
        group_of = dict(zip(map(str, split.ids.tolist()), split.groups.tolist()))
        ids = (cache_dir(model, task.name, limit) / "test" / "ids.txt").read_text().splitlines()
        if len(ids) != n:
            return None
        return np.array([group_of[i] for i in ids], dtype=object)
    except Exception:  # noqa: BLE001 - the cluster CI is extra information; never fail the run for it
        return None


def predictions_path(model: str, task: str, mode: str, seed: int, limit: int = 0) -> Path:
    t = task if not limit else f"{task}__limit{limit}"
    return paths.results_root() / "predictions" / model / t / mode / f"seed{seed}.npz"


def run_probe(spec: ModelSpec, task: TileTask, mode: str, seed: int = 0, cfg: ProbeConfig | None = None,
              limit: int = 0, device: str | None = None, n_boot: int = 1000, log=print) -> dict:
    cfg = cfg or ProbeConfig()
    cfg.seed = seed
    xtr, ytr, mtr = load_features(spec.name, task.name, "train", mode, limit)
    xva, yva, mva = load_features(spec.name, task.name, "val", mode, limit)
    xte, yte, mte = load_features(spec.name, task.name, "test", mode, limit)
    res = train_linear_probe(xtr, ytr, xva, yva, xte, task.num_classes, cfg, device)
    metrics = classification_metrics(yte, res.test_probs)
    metrics["majority_weighted_f1"] = majority_weighted_f1(ytr, yte)
    lo, hi = bootstrap_ci(yte, res.test_pred, n_boot=n_boot, seed=seed) if n_boot else (float("nan"), float("nan"))
    groups = test_groups(task, spec.name, limit, len(yte)) if n_boot else None
    cluster_ci = None
    if groups is not None and len(set(groups.tolist())) >= 2:  # one group: nothing to resample
        cluster_ci = list(cluster_bootstrap_ci(yte, res.test_pred, groups, n_boot=n_boot, seed=seed))

    pred_file = predictions_path(spec.name, task.name, mode, seed, limit)
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_file, y_true=yte, pred=res.test_pred.astype(np.int16),
                        probs=res.test_probs.astype(np.float16))

    record = {
        "run_id": uuid.uuid4().hex[:12],
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": PROTOCOL_VERSION,
        "git": git_state(),
        "code_sha": code_sha(),
        "versions": library_versions(),
        "embedding_provenance": mtr.get("provenance"),
        "python": platform.python_version(),
        "model": spec.name,
        "model_display": spec.display_name,
        "adapter": spec.adapter,
        "model_fingerprint": mtr.get("model_fingerprint"),
        "paper_name": spec.paper_name,
        "task": task.name,
        "task_display": task.display_name,
        "paper_column": task.paper_column,
        "mode": mode,
        "seed": seed,
        "limit": limit,
        "smoke": bool(limit),
        "n": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "split_fingerprints": {"train": mtr["split_fingerprint"], "val": mva["split_fingerprint"], "test": mte["split_fingerprint"]},
        "feature_dim": int(xtr.shape[1]),
        "probe": cfg.to_dict(),
        "best_iteration": res.best_iteration,
        "best_val_loss": res.best_val_loss,
        "metrics": metrics,
        "weighted_f1_ci95": [lo, hi],
        "weighted_f1_ci95_cluster": cluster_ci,
        "cluster_unit": getattr(task, "cluster_unit", None) if cluster_ci else None,
        "n_test_clusters": int(len(set(groups.tolist()))) if groups is not None else None,
        "leakage": leakage_flags(spec, task),
        "predictions": str(pred_file.relative_to(paths.results_root())),
    }
    append_run(record)
    log(f"[probe] {spec.name} {task.name} {mode} seed{seed}: weighted F1 {metrics['weighted_f1']:.4f} "
        f"[{lo:.4f}, {hi:.4f}]" + (f" by {record['cluster_unit']} [{cluster_ci[0]:.4f}, {cluster_ci[1]:.4f}]" if cluster_ci else "")
        + f" (best it {res.best_iteration}; majority class {metrics['majority_weighted_f1']:.4f})")
    return record
