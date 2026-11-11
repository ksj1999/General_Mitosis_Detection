"""HEST runner on a tiny fake hest-bench layout, the leaderboard's extra tables, and model configs."""
import csv
import json

import numpy as np
import pytest
import torch


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


def _fake_hest_task(task_dir, samples=("S1", "S2", "S3"), n=40, genes=("GA", "GB", "GC")):
    import anndata
    import h5py

    (task_dir / "patches").mkdir(parents=True)
    (task_dir / "adata").mkdir()
    (task_dir / "splits").mkdir()
    (task_dir / "var_50genes.json").write_text(json.dumps({"genes": list(genes)}))
    rng = np.random.default_rng(0)
    for s in samples:
        bright = rng.uniform(0, 1, n)
        img = np.clip(bright[:, None, None, None] * 255 + rng.normal(0, 10, (n, 224, 224, 3)), 0, 255).astype(np.uint8)
        bcs = np.array([f"{s}_bc{i}".encode() for i in range(n)]).reshape(-1, 1)
        with h5py.File(task_dir / "patches" / f"{s}.h5", "w") as f:
            f["img"] = img
            f["barcode"] = bcs
            f["coords"] = np.zeros((n, 2))
        counts = np.stack([bright * 50, (1 - bright) * 30, rng.uniform(0, 5, n)], 1)
        extra = rng.uniform(0, 5, (n, 2))  # genes not in the task list
        # rows stored in reverse barcode order, so the loader must align targets by barcode
        a = anndata.AnnData(X=np.concatenate([counts, extra], 1)[::-1].astype(np.float32))
        a.obs_names = [b[0].decode() for b in bcs][::-1]
        a.var_names = [*genes, "X1", "X2"]
        a.write_h5ad(task_dir / "adata" / f"{s}.h5ad")
    folds = [(samples[:2], samples[2:]), (samples[1:], samples[:1])]
    for k, (tr, te) in enumerate(folds):
        for name, ss in (("train", tr), ("test", te)):
            with (task_dir / "splits" / f"{name}_{k}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "patches_path", "expr_path"])
                for s in ss:
                    w.writerow([s, f"patches/{s}.h5", f"adata/{s}.h5ad"])


def test_hest_end_to_end(env):
    pytest.importorskip("anndata")
    from fmbench import hest
    from fmbench.leaderboard import build
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec
    from fmbench.runner import read_runs

    _fake_hest_task(hest.root() / "IDC")
    spec = ModelSpec(name="tiny", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    torch.manual_seed(0)
    enc = build_encoder(spec, torch.device("cpu"))
    hest.embed(spec, enc, "IDC", num_workers=0, log=lambda *a: None)
    rec = hest.evaluate(spec, "IDC", "pooled", method="ridge", log=lambda *a: None)
    assert len(rec["metrics"]["folds"]) == 2 and set(rec["metrics"]["per_gene"]) == {"GA", "GB", "GC"}
    assert -1 <= rec["metrics"]["pearson"] <= 1
    assert len(rec["code_sha"]) == 12 and "numpy" in rec["versions"]  # provenance: git alone is not enough on TGI
    rec_rf = hest.evaluate(spec, "IDC", "pooled", method="rf", log=lambda *a: None)
    assert rec_rf["metrics"]["pearson"] > 0.3  # brightness drives GA and GB
    assert [r["kind"] for r in read_runs()] == ["hest", "hest"]
    text = build().read_text()
    assert "HEST-Benchmark" in text and "0.539" in text  # paper row for Virchow2 IDC


TINY_YAML = ("adapter: torchvision\nparams: {arch: resnet18, weights: none}\nmodes: [pooled]\n"
             "batch_size: 16\nprecision: fp32\nresize_to: 32\n")


def test_hest_stages_embed_on_gpu_regress_without_encoder(env, monkeypatch):
    pytest.importorskip("anndata")
    import fmbench.cli as cli
    from fmbench import hest
    from fmbench.runner import read_runs

    _fake_hest_task(hest.root() / "IDC", n=20)
    cfg = env / "tiny_hest.yaml"
    cfg.write_text(TINY_YAML)
    torch.manual_seed(0)
    cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "embed", "--device", "cpu",
              "--workers", "0"])
    assert read_runs() == []  # embedding only

    def no_encoder(args):
        raise AssertionError("the regression stage must not load the model")

    monkeypatch.setattr(cli, "_load_model", no_encoder)
    cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "regress", "--method", "ridge"])
    (rec,) = read_runs()
    assert rec["model"] == "tiny_hest" and rec["mode"] == "pooled" and rec["task"] == "hest:IDC"
    assert rec["regressor"] == {"backend": "sklearn", "params": {}}

    with pytest.raises(SystemExit):  # --overwrite would need the encoder
        cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "regress", "--overwrite"])

    cfg.write_text(TINY_YAML.replace("resize_to: 32", "resize_to: 48"))
    with pytest.raises(RuntimeError, match="resize_to"):  # embeddings from another setup are not used
        cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "regress", "--method", "ridge"])


def test_hest_regress_stage_needs_embeddings(env):
    pytest.importorskip("anndata")
    import fmbench.cli as cli
    from fmbench import hest

    _fake_hest_task(hest.root() / "IDC", n=5)
    cfg = env / "tiny_none.yaml"
    cfg.write_text(TINY_YAML)
    with pytest.raises(RuntimeError, match="--stage embed"):
        cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "regress"])


def test_sklearn_forest_copies_cuml_defaults(monkeypatch):
    import sys

    import sklearn.ensemble

    from fmbench import hest

    monkeypatch.setitem(sys.modules, "cuml", None)  # no cuML: `import cuml` raises ImportError
    want = {"n_estimators": 70, "random_state": 42, "max_depth": 16, "max_features": "sqrt"}  # cuML 24.06+
    assert hest.regressor_info("rf") == {"backend": "sklearn", "params": want}
    assert hest.regressor_info("rf", 1.0)["params"]["max_features"] == 1.0  # the ablation

    seen = []
    real = sklearn.ensemble.RandomForestRegressor

    def spy(**kw):
        seen.append(kw)
        return real(**{**kw, "n_estimators": 2})

    monkeypatch.setattr(sklearn.ensemble, "RandomForestRegressor", spy)
    rng = np.random.default_rng(0)
    pred = hest.regress(rng.normal(size=(30, 4)), rng.normal(size=(30, 2)), rng.normal(size=(5, 4)), "rf",
                        log=lambda *a: None)
    assert pred.shape == (5, 2)
    assert seen == [{**want, "n_jobs": -1}] * 2  # one forest per gene


def test_cuml_gets_the_same_parameters(monkeypatch):
    import sys
    import types

    from fmbench import hest

    seen = []

    class FakeRF:
        def __init__(self, **kw):
            seen.append(kw)

        def fit(self, x, y):
            self.m = float(np.mean(y))

        def predict(self, x):
            return np.full(len(x), self.m)

    cuml = types.ModuleType("cuml")
    cuml.__version__ = "24.06.00"
    cuml.ensemble = types.SimpleNamespace(RandomForestRegressor=FakeRF)
    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cuml.ensemble", cuml.ensemble)
    info = hest.regressor_info("rf")
    assert info["backend"] == "cuml" and info["version"] == "24.06.00"
    hest.regress(np.zeros((4, 3)), np.ones((4, 2)), np.zeros((2, 3)), "rf", log=lambda *a: None)
    # explicit, so a cuML version with other defaults (25.08+ changed them) fits the same forest
    assert seen == [{"n_estimators": 70, "random_state": 42, "max_depth": 16, "max_features": "sqrt"}] * 2


def test_hest_skip_done_and_ablation_table(env, monkeypatch):
    pytest.importorskip("anndata")
    import fmbench.cli as cli
    from fmbench import hest
    from fmbench.leaderboard import build
    from fmbench.runner import read_runs

    _fake_hest_task(hest.root() / "IDC", n=12)
    cfg = env / "tiny_skip.yaml"
    cfg.write_text(TINY_YAML)
    cli.main(["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "embed", "--device", "cpu",
              "--workers", "0"])
    calls = []
    real = hest.regress
    monkeypatch.setattr(hest, "regress", lambda *a, **k: calls.append(k.get("max_features")) or real(*a, **k))
    regress = ["hest", "--model", str(cfg), "--hest-task", "IDC", "--stage", "regress", "--skip-done"]
    cli.main(regress)
    n = len(calls)
    assert n == 2 and set(calls) == {"sqrt"}  # one call per fold
    cli.main(regress)
    assert len(calls) == n  # already recorded: skipped
    cli.main(regress + ["--rf-max-features", "1.0"])
    assert calls[n:] == [1.0, 1.0]  # a different forest is not "done"
    recs = read_runs()
    assert [r["regressor"]["params"]["max_features"] for r in recs] == ["sqrt", 1.0]
    text = build().read_text()
    main = text[text.index("## HEST-Benchmark"):text.index("Ablation: random forest")]
    ablation = text[text.index("Ablation: random forest"):]
    sqrt_r, full_r = (f"{r['metrics']['pearson']:.3f}" for r in recs)
    assert "| ours | tiny_skip | pooled | " + sqrt_r in main and "| tiny_skip | pooled | " + full_r in ablation
    assert hest.is_done(hest.ModelSpec.from_yaml(cfg), "IDC", "pooled")
    assert not hest.is_done(hest.ModelSpec.from_yaml(cfg), "IDC", "pooled", method="ridge")

    from fmbench.runner import append_run

    append_run({**recs[0], "run_id": "other-rev", "time_utc": "2999-01-01T00:00:00+00:00", "hest_revision": "main",
                "metrics": {**recs[0]["metrics"], "pearson": 0.777}})  # newer, but another dataset revision
    assert [r["values"]["IDC"] for r in hest.our_rows()] == [recs[0]["metrics"]["pearson"]]


def test_hest_targets_follow_patch_barcodes(env):
    pytest.importorskip("anndata")
    from fmbench import hest

    tdir = hest.root() / "IDC"
    _fake_hest_task(tdir, samples=("S1", "S2"), n=5)
    with __import__("h5py").File(tdir / "patches" / "S1.h5") as f:
        bcs = [b.decode() for b in f["barcode"][:].reshape(-1)]
    y = hest._targets(tdir, {"expr_path": "adata/S1.h5ad"}, bcs, ["GA", "GB", "GC"])
    import anndata

    a = anndata.read_h5ad(tdir / "adata" / "S1.h5ad")
    assert np.allclose(y[0], np.log1p(a[bcs[0]][:, ["GA", "GB", "GC"]].X.ravel()))


def test_pearson_handles_constant_prediction():
    from fmbench.hest import _pearson

    y = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]])
    p = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    r = _pearson(y, p)
    assert r[0] == pytest.approx(1.0) and r[1] == 0.0


def test_leaderboard_extra_tasks(env):
    from fmbench.leaderboard import build
    from fmbench.runner import append_run

    base = {"time_utc": "2026-01-01T00:00:00+00:00", "seed": 0, "limit": 0, "smoke": False, "mode": "cls+mean",
            "model": "m", "model_display": "M", "paper_name": None}
    append_run({**base, "task": "midog_nobreast", "paper_column": None, "metrics": {"weighted_f1": 0.81},
                "leakage": ["model trained on midogpp_breast"]})
    append_run({**base, "task": "midog", "paper_column": "MIDOG", "metrics": {"weighted_f1": 0.9}, "leakage": []})
    text = build().read_text()
    assert "Tasks outside Virchow2 Table 2" in text and "81.0 ‡" in text and "(90.0, 1/8)" in text


def test_leaderboard_csv_keeps_leakage_marks(env):
    from fmbench.leaderboard import build
    from fmbench.runner import append_run

    base = {"time_utc": "2026-01-01T00:00:00+00:00", "seed": 0, "limit": 0, "smoke": False, "mode": "cls+mean",
            "model": "m", "model_display": "M", "paper_name": None}
    append_run({**base, "task": "midog", "paper_column": "MIDOG", "metrics": {"weighted_f1": 0.9},
                "leakage": ["model trained on midogpp_breast; task uses midogpp_breast"]})
    append_run({**base, "task": "pcam", "paper_column": "PCam", "metrics": {"weighted_f1": 0.8}, "leakage": []})
    md = build()
    assert "90.0 ‡" in md.read_text()
    with (md.parent / "leaderboard.csv").open() as f:
        rows = {(r["source"], r["model"]): r for r in csv.DictReader(f)}
    assert rows[("ours", "M")]["leakage"] == "MIDOG"
    assert rows[("paper", "Virchow2")]["leakage"] == ""


def test_hest_reproduction_check_and_grouping(env):
    from fmbench import hest
    from fmbench.runner import append_run

    paper = {(p["model"], p["mode"]): p["values"] for p in hest.paper_rows()}
    task_of = {v: k for k, v in hest.PAPER_COLUMN.items()}

    def rec(model, paper_name, mode, col, value, mf="sqrt"):
        append_run({"kind": "hest", "method": "rf", "hest_revision": hest.VIRCHOW2_REVISION,
                    "time_utc": "2026-01-01T00:00:00+00:00", "regressor": {"backend": "sklearn",
                                                                           "params": {"max_features": mf}},
                    "model": model, "model_display": model.upper(), "paper_name": paper_name,
                    "task": f"hest:{task_of[col]}", "paper_column": col, "mode": mode, "seed": 0, "smoke": False,
                    "metrics": {"pearson": value}, "leakage": []})

    for c in hest.COLUMNS:
        rec("ctp", "CTransPath", "pooled", c, paper[("CTransPath", "mean")][c] + 0.01)  # paper mode "mean"
        rec("phk", "Phikon", "cls", c, paper[("Phikon", "cls")][c] - 0.002)
        rec("other", None, "cls", c, 0.1)  # no Table 3 row: not in the check
    rec("ctp", "CTransPath", "pooled", "SKCM", paper[("CTransPath", "mean")]["SKCM"] - 0.03, mf=1.0)
    text = "\n".join(hest.markdown())
    main, rest = text.split("Reproduction check")
    # our rows sit in their embedding group, next to the paper rows of that group
    assert main.index("| ours | CTP | pooled |") < main.index("| paper | Virchow2G | cls |") < main.index("| ours | PHK | cls |")
    assert "| CTP | pooled | 'sqrt' | " + " | ".join(["+0.010"] * 10) + " | 0.010 (+0.010) |" in rest
    assert "| PHK | cls | 'sqrt' | " + " | ".join(["-0.002"] * 10) + " | 0.002 (-0.002) |" in rest
    assert "| CTP | pooled | 1.0 | " in rest and "| -0.030 |" in rest and "| 0.030 (-0.030), 1/10 |" in rest
    assert "| OTHER |" not in rest.split("Ablation")[0]


def test_ctranspath_stem_and_nhwc_pool():
    timm = pytest.importorskip("timm")
    from fmbench.models.adapters import CTransPathConvStem

    m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0,
                          embed_layer=CTransPathConvStem).eval()
    with torch.no_grad():
        tok = m.forward_features(torch.zeros(2, 3, 224, 224))
    assert tuple(tok.shape) == (2, 7, 7, 768)
    assert tuple(tok.mean(dim=(1, 2)).shape) == (2, 768)


def test_all_model_configs_parse():
    from fmbench.models.base import MODES, list_specs

    specs = list_specs()
    assert {"virchow2", "virchow", "uni", "h_optimus_0", "prov_gigapath", "phikon", "ctranspath"} <= set(specs)
    for name, s in specs.items():
        assert all(m in MODES for m in s.modes), name
    from fmbench.leaderboard import paper_table2

    paper_models = {r["model"] for r in paper_table2()}
    for name, s in specs.items():
        if s.paper_name:
            assert s.paper_name in paper_models, (name, s.paper_name)


def test_example_configs_parse():
    import yaml

    from fmbench import paths
    from fmbench.models.base import ModelSpec

    for p in sorted(paths.MODEL_CONFIG_DIR.glob("*.yaml.example")):
        d = yaml.safe_load(p.read_text())
        ModelSpec(name=p.name.split(".")[0], **d)
