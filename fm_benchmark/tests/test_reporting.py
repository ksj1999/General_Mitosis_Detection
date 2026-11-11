"""What a run records and how the leaderboard reports it: the group-level CI, the majority-class baseline,
provenance, DLBCL kept out of the reproduction check, and NHWC feature maps of timm models."""
import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from fmbench.tasks.base import FileListSplit
from test_pipeline import ColourTask


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


# --------------------------------------------------------------------------- metrics
def test_cluster_ci_resamples_whole_groups():
    from fmbench.metrics import bootstrap_ci, cluster_bootstrap_ci

    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i}" for i in range(10)], 50)
    y = rng.integers(0, 2, len(groups))
    wrong = np.isin(groups, ["p0", "p1", "p2", "p3"])  # errors come in whole patients
    pred = np.where(wrong, 1 - y, y)
    tile = bootstrap_ci(y, pred, n_boot=200, seed=0)
    cluster = cluster_bootstrap_ci(y, pred, groups, n_boot=200, seed=0)
    assert cluster == cluster_bootstrap_ci(y, pred, groups, n_boot=200, seed=0)  # deterministic for a seed
    assert cluster[1] - cluster[0] > 3 * (tile[1] - tile[0])  # correlated tiles: the honest interval is much wider
    assert all(np.isnan(cluster_bootstrap_ci(y, pred, np.array(["p0"] * len(y)), n_boot=20)))  # nothing to resample


def test_majority_baseline():
    from fmbench.metrics import majority_weighted_f1

    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([0, 0, 1, 1, 1])
    assert majority_weighted_f1(y_train, y_test) == pytest.approx(f1_score(y_test, np.zeros(5), average="weighted"))
    assert majority_weighted_f1(y_train, np.array([1, 1, 2])) == 0.0  # majority class absent from test


# --------------------------------------------------------------------------- runner
class PatientTask(ColourTask):
    """ColourTask with 5 test patients of 6 tiles each, so the probe reports a by-patient CI."""

    cluster_unit = "patient"

    def official_splits(self):
        sp = super().official_splits()
        for name, s in sp.items():
            sp[name] = FileListSplit(s.files, s.labels, ids=s.ids, groups=[f"{name}{i // 6}" for i in range(len(s))])
        return sp


@pytest.mark.parametrize("limit", [0, 20])
def test_run_records_cluster_ci_majority_and_provenance(env, limit):
    from fmbench import runner
    from fmbench.embed import embed_task
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec
    from fmbench.probe import ProbeConfig

    task = PatientTask()
    spec = ModelSpec(name="tiny_grouped", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    torch.manual_seed(0)
    embed_task(spec, build_encoder(spec, torch.device("cpu")), task, limit=limit, num_workers=0, log=lambda *a: None)
    rec = runner.run_probe(spec, task, "pooled", seed=0, cfg=ProbeConfig(iterations=60, eval_every=20), limit=limit,
                           device="cpu", n_boot=30, log=lambda *a: None)
    lo, hi = rec["weighted_f1_ci95_cluster"]
    assert 0 <= lo <= hi <= 1
    assert rec["cluster_unit"] == "patient"
    assert 2 <= rec["n_test_clusters"] <= 5
    assert 0 <= rec["metrics"]["majority_weighted_f1"] <= 1
    assert len(rec["code_sha"]) == 12 and "torch" in rec["versions"]
    assert rec["embedding_provenance"]["code_sha"] == rec["code_sha"]

    rec0 = runner.run_probe(spec, task, "pooled", seed=0, cfg=ProbeConfig(iterations=20, eval_every=20), limit=limit,
                            device="cpu", n_boot=0, log=lambda *a: None)
    assert rec0["weighted_f1_ci95_cluster"] is None and rec0["cluster_unit"] is None


def test_single_test_patient_gives_no_cluster_ci(env):
    from fmbench import runner
    from fmbench.embed import embed_task
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec
    from fmbench.probe import ProbeConfig

    class OnePatientTask(PatientTask):
        def official_splits(self):
            sp = super().official_splits()
            t = sp["test"]
            sp["test"] = FileListSplit(t.files, t.labels, ids=t.ids, groups=["p0"] * len(t))
            return sp

    task = OnePatientTask()
    spec = ModelSpec(name="tiny_onepatient", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    embed_task(spec, build_encoder(spec, torch.device("cpu")), task, num_workers=0, log=lambda *a: None)
    rec = runner.run_probe(spec, task, "pooled", seed=0, cfg=ProbeConfig(iterations=20, eval_every=20), device="cpu",
                           n_boot=20, log=lambda *a: None)
    assert rec["weighted_f1_ci95_cluster"] is None and rec["cluster_unit"] is None and rec["n_test_clusters"] == 1


def test_code_sha_is_taken_before_the_command_runs(env, monkeypatch):
    import fmbench.cli as cli
    import fmbench.leaderboard as lb
    from fmbench.provenance import code_sha

    seen = []
    monkeypatch.setattr(lb, "build", lambda *a, **k: seen.append(code_sha.cache_info().currsize) or env / "lb.md")
    code_sha.cache_clear()
    cli.main(["leaderboard"])
    assert seen == [1]  # already hashed when the command started


def test_code_sha_never_raises(monkeypatch):
    from pathlib import Path

    from fmbench.provenance import code_sha

    def boom(self):
        raise OSError("file replaced during rsync")

    code_sha.cache_clear()
    monkeypatch.setattr(Path, "read_bytes", boom)
    try:
        assert code_sha() == "unknown"
    finally:
        code_sha.cache_clear()


def test_append_run_locks_and_never_loses_a_record(env, monkeypatch):
    from fmbench import runner

    calls = []
    real = runner.fcntl.lockf

    def spy(fd, op):
        calls.append(op)
        return real(fd, op)

    monkeypatch.setattr(runner.fcntl, "lockf", spy)
    runner.append_run({"run_id": "locked", "time_utc": "t"})
    assert calls == [runner.fcntl.LOCK_EX, runner.fcntl.LOCK_UN]

    def no_locks(fd, op):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(runner.fcntl, "lockf", no_locks)  # no lock service
    runner.append_run({"run_id": "unlocked", "time_utc": "t"})

    def unlock_fails(fd, op):
        if op == runner.fcntl.LOCK_UN:
            raise OSError(37, "No locks available")
        return real(fd, op)

    monkeypatch.setattr(runner.fcntl, "lockf", unlock_fails)  # the record is written, then the unlock fails
    runner.append_run({"run_id": "unlock_failed", "time_utc": "t"})
    assert [r["run_id"] for r in runner.read_runs()] == ["locked", "unlocked", "unlock_failed"]


# --------------------------------------------------------------------------- leaderboard
def _record(model, task, column, mode, f1, t, **extra):
    return {"run_id": f"{model}-{task}-{t}", "time_utc": f"2026-01-01T00:00:{t:02d}", "model": model,
            "model_display": model, "paper_name": "Phikon", "task": task, "task_display": task,
            "paper_column": column, "mode": mode, "seed": 0, "limit": 0, "smoke": False,
            "metrics": {"weighted_f1": f1}, "leakage": [], **extra}


def test_leaderboard_keeps_dlbcl_out_of_the_reproduction_check(env):
    from fmbench.leaderboard import build
    from fmbench.runner import append_run

    # records written before the class-balance fields existed, as on TGI
    append_run(_record("phikon", "pcam", "PCam", "cls", 0.925, 1))     # paper 90.5 -> +2.0
    append_run(_record("phikon", "wilds", "WILDS", "cls", 0.962, 2))   # paper 97.2 -> -1.0
    append_run(_record("phikon", "dlbcl", "DLBCL", "cls", 0.711, 3))   # paper 54.4 -> +16.7, not comparable
    # a newer WILDS run without a by-slide CI replaces an older one with it
    append_run(_record("phikon", "wilds", "WILDS", "cls+mean", 0.90, 4, weighted_f1_ci95_cluster=[0.8, 0.97],
                       cluster_unit="slide", n_test_clusters=10))
    append_run(_record("phikon", "wilds", "WILDS", "cls+mean", 0.96, 5, weighted_f1_ci95_cluster=None))
    text = build().read_text()

    repro = next(line for line in text.splitlines() if line.startswith("| phikon | cls |"))
    assert "(16.7) †" in repro
    assert repro.rstrip().endswith("| 1.5 (+0.5), 2/7 |")  # abs. mean 1.5; the signed mean alone would hide errors

    balance = text[text.index("## Class balance"):]
    rows = [line for line in balance.splitlines() if line.startswith("| dlbcl") or line.startswith("| wilds")]
    assert [r.split("|")[1].strip() for r in rows] == ["dlbcl"]  # the older WILDS run is not shown
    assert rows[0].split("|")[5].strip() == ""  # no majority baseline in old records: empty, not an error
    assert "| 96.0 |" in text  # the main table shows the newer WILDS run


# --------------------------------------------------------------------------- adapters
@pytest.mark.parametrize("fmt", ["NHWC", "nhwc_enum", "NCHW", None])
def test_timm_feature_map_layout(fmt):
    from fmbench.models.adapters import TimmEncoder

    nchw = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    want = nchw.mean(dim=(2, 3))

    class Stub(torch.nn.Module):
        def forward_features(self, x):
            return nchw.permute(0, 2, 3, 1) if fmt in ("NHWC", "nhwc_enum") else nchw

    model = Stub()
    if fmt == "nhwc_enum":
        timm_layers = pytest.importorskip("timm.layers")
        model.output_fmt = timm_layers.Format.NHWC
    elif fmt:
        model.output_fmt = fmt
    enc = object.__new__(TimmEncoder)
    enc.model, enc.pool, enc.num_prefix = model, "tokens", 0
    pooled = enc.features(torch.zeros(2, 3, 8, 8))["pooled"]
    assert torch.allclose(pooled, want)
