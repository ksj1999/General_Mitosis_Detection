"""HEST-Benchmark (Virchow2 Table 3): predict 50 genes from 224 px spatial-transcriptomics spot patches.

Virchow2 sec. 6.5 uses the HEST v1.0.2 protocol with a random forest instead of the linear probe:
for each of the 50 genes of a task, a RandomForestRegressor(n_estimators=70, random_state=42) is fit on
raw embeddings of the training samples and scored by Pearson r on the test samples; a fold's score is
the mean r over genes and a task's score is the mean over folds. Targets are log1p of the counts.

The paper's "74 profiles from 48 patients" matches the hest-bench dataset at revision 78ed54aa (the
old Visium COAD task, replaced on 2024-08-19), so that revision is the default.

HEST uses cuML on GPU. Without cuML we fall back to scikit-learn with cuML's tree depth and features per
split (see rf_params), which runs on CPUs and gives slightly different numbers (exact instead of binned
splits). ``fmbench hest --stage embed`` on a GPU node and then ``--stage regress`` on a CPU node keep the
forests off the GPU. ``--method ridge`` runs the current HEST leaderboard protocol instead
(StandardScaler, PCA to 256, Ridge), which is fast but not comparable to Table 3.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import uuid
from pathlib import Path

import numpy as np

from . import PROTOCOL_VERSION, paths
from .embed import cache_dir, embed_split, load_features
from .models.base import TABLE_GROUP, ModelSpec
from .provenance import code_sha, library_versions
from .runner import append_run, git_state, read_runs
from .tasks.base import H5ArraySplit

REPO_ID = "MahmoodLab/hest-bench"
VIRCHOW2_REVISION = "78ed54aad495c764f8f48c1f53204cd4f9e7743d"
TASKS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC"]
PAPER_COLUMN = {"IDC": "IDC", "PRAD": "PRAD", "PAAD": "PAAD", "SKCM": "SKCM", "COAD": "COAD", "READ": "READ",
                "CCRCC": "CCRCC", "HCC": "HCC", "LUNG": "LUAD", "LYMPH_IDC": "LIDC"}
COLUMNS = list(PAPER_COLUMN.values())


def root(revision: str = VIRCHOW2_REVISION) -> Path:
    tag = "v2paper" if revision == VIRCHOW2_REVISION else revision[:12]
    return paths.data_root() / "hest_bench" / tag


def download(tasks=TASKS, revision: str = VIRCHOW2_REVISION) -> Path:
    from huggingface_hub import snapshot_download

    out = root(revision)
    snapshot_download(REPO_ID, repo_type="dataset", revision=revision, local_dir=out,
                      allow_patterns=[f"{t}/*" for t in tasks], ignore_patterns=["*.png", "fm_v1/*"])
    return out


class _Ref:  # what embed_split needs from a task
    def __init__(self, name: str):
        self.name = name


def _folds(task_dir: Path) -> list[tuple[list[dict], list[dict]]]:
    out, k = [], 0
    while (task_dir / "splits" / f"train_{k}.csv").exists():
        read = lambda s: list(csv.DictReader((task_dir / "splits" / f"{s}_{k}.csv").open()))  # noqa: E731
        out.append((read("train"), read("test")))
        k += 1
    if not out:
        raise FileNotFoundError(f"{task_dir}/splits/train_0.csv (run: python -m fmbench hest-download)")
    return out


def _samples(task_dir: Path) -> dict[str, dict]:
    return {r["sample_id"]: r for tr, te in _folds(task_dir) for r in tr + te}


def _patch_split(task_dir: Path, row: dict) -> H5ArraySplit:
    import h5py

    p = task_dir / row["patches_path"]
    with h5py.File(p, "r") as f:
        bcs = [b.decode() if isinstance(b, bytes) else str(b) for b in np.asarray(f["barcode"][:]).reshape(-1)]
    return H5ArraySplit(p, "img", np.zeros(len(bcs), np.int64), ids=bcs)


def embed(spec: ModelSpec, encoder, task: str, revision: str = VIRCHOW2_REVISION, num_workers: int = 8,
          overwrite: bool = False, log=print) -> None:
    tdir = root(revision) / task
    ref = _Ref(f"hest_{task}")
    fp = {"model": encoder.fingerprint(), "hest_revision": revision}
    for sid, row in sorted(_samples(tdir).items()):
        split = _patch_split(tdir, row)
        embed_split(encoder, spec, ref, sid, split, cache_dir(spec.name, ref.name) / sid, fp,
                    num_workers=num_workers, overwrite=overwrite, log=log)


def cached_dims(spec: ModelSpec, task: str, revision: str = VIRCHOW2_REVISION) -> set[str]:
    """Feature keys cached for every sample of a task, for a regression run without the encoder.

    Checks the cache keys that the model config decides. The model fingerprint (weights, Hub revision) needs
    the loaded encoder, so only an embedding run (`--stage embed` or `all`) checks it."""
    tdir = root(revision) / task
    dims = None
    for sid in sorted(_samples(tdir)):
        meta_path = cache_dir(spec.name, f"hest_{task}") / sid / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not meta.get("complete"):
            raise RuntimeError(f"{meta_path.parent}: no complete HEST embeddings; "
                               f"run `fmbench hest --model {spec.name} --stage embed` first")
        want = {"adapter": spec.adapter, "spec_params": spec.params, "resize_to": spec.resize_to,
                "precision": spec.precision}
        diff = [k for k, v in json.loads(json.dumps(want, sort_keys=True, default=str)).items() if meta.get(k) != v]
        if (meta.get("model_fingerprint") or {}).get("hest_revision") != revision:
            diff.append("hest_revision")
        if diff:
            raise RuntimeError(f"{meta_path.parent}: cached embeddings differ from the current setup in {diff}; "
                               f"re-embed with `fmbench hest --model {spec.name} --stage embed --overwrite`")
        have = set(meta.get("dims", {}))
        dims = have if dims is None else dims & have
    return dims or set()


def _targets(task_dir: Path, row: dict, barcodes: list[str], genes: list[str]) -> np.ndarray:
    import anndata
    import scipy.sparse

    a = anndata.read_h5ad(task_dir / row["expr_path"])
    a = a[barcodes][:, genes]
    x = a.X.toarray() if scipy.sparse.issparse(a.X) else np.asarray(a.X)
    return np.log1p(x.astype(np.float64))


def _pearson(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = y - y.mean(0)
    p = p - p.mean(0)
    den = np.sqrt((y ** 2).sum(0) * (p ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (y * p).sum(0) / den
    return np.nan_to_num(r, nan=0.0)  # a constant prediction scores 0


# HEST v1.0.2 fits cuml.ensemble.RandomForestRegressor(n_estimators=70, random_state=42) and leaves the rest at
# cuML's defaults; its README pins RAPIDS 24.6. In cuML 24.06 to 25.06 the regressor inherits the base class's
# max_features='sqrt' (sqrt(d) features tried per split; its docstring says 1.0, but the code passes nothing),
# while 24.04 and earlier used 'auto', which is 1.0 for regression. cuML also stops trees at depth 16 and searches
# splits over 128 histogram bins per feature. scikit-learn grows trees until the leaves are pure, tries every
# feature (1.0) and every threshold, and has no binning. So both backends get depth 16 and max_features 'sqrt'
# explicitly, which makes the result independent of the installed cuML version; only the binning differs.
# `--rf-max-features 1.0` runs the pre-24.06 behaviour as an ablation; the leaderboard's main HEST table uses
# only the default.
RF_MAX_FEATURES = "sqrt"


def rf_params(max_features=RF_MAX_FEATURES) -> dict:
    return {"n_estimators": 70, "random_state": 42, "max_depth": 16, "max_features": max_features}


def regressor_info(method: str, max_features=RF_MAX_FEATURES) -> dict:
    """The regressor a run uses, for its record."""
    if method != "rf":
        return {"backend": "sklearn", "params": {}}
    try:
        import cuml

        return {"backend": "cuml", "version": getattr(cuml, "__version__", "?"), "params": rf_params(max_features)}
    except ImportError:
        return {"backend": "sklearn", "params": rf_params(max_features)}


def regress(xtr, ytr, xte, method: str = "rf", seed: int = 1, log=print, max_features=RF_MAX_FEATURES) -> np.ndarray:
    if method == "rf":
        info = regressor_info("rf", max_features)
        if info["backend"] == "cuml":
            from cuml.ensemble import RandomForestRegressor as RF

            make = lambda: RF(**info["params"])  # noqa: E731
        else:
            from sklearn.ensemble import RandomForestRegressor as RF

            make = lambda: RF(**info["params"], n_jobs=-1)  # noqa: E731
        log(f"    random forest ({info['backend']}, max_features={max_features}), {ytr.shape[1]} genes, "
            f"{len(xtr)} train spots")
        pred = np.zeros((len(xte), ytr.shape[1]))
        for g in range(ytr.shape[1]):
            m = make()
            m.fit(xtr.astype(np.float32), ytr[:, g].astype(np.float32))
            pred[:, g] = np.asarray(m.predict(xte.astype(np.float32))).reshape(-1)
        return pred
    if method == "ridge":  # current HEST leaderboard protocol
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        k = min(256, xtr.shape[1], len(xtr))
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                             Ridge(solver="lsqr", alpha=100 / (k * ytr.shape[1]), fit_intercept=False,
                                   max_iter=1000, random_state=seed))
        pipe.fit(xtr, ytr)
        return pipe.predict(xte)
    raise ValueError(f"unknown method {method!r}")


def _max_features(record: dict):
    """max_features of a random-forest record. Records written before it was recorded fitted with 1.0."""
    return ((record.get("regressor") or {}).get("params") or {}).get("max_features", 1.0)


def is_done(spec: ModelSpec, task: str, mode: str, method: str = "rf", revision: str = VIRCHOW2_REVISION,
            max_features=RF_MAX_FEATURES) -> bool:
    """Whether runs.jsonl already has this regression, so a resubmitted job can skip it (`--skip-done`)."""
    for r in read_runs():
        if (r.get("kind") == "hest" and r.get("model") == spec.name and r.get("task") == f"hest:{task}"
                and r.get("mode") == mode and r.get("method") == method and r.get("hest_revision") == revision
                and (method != "rf" or _max_features(r) == max_features)):
            return True
    return False


def evaluate(spec: ModelSpec, task: str, mode: str, method: str = "rf", revision: str = VIRCHOW2_REVISION,
             log=print, max_features=RF_MAX_FEATURES) -> dict:
    tdir = root(revision) / task
    genes = json.loads((tdir / "var_50genes.json").read_text())["genes"]
    ref = f"hest_{task}"
    feats, targets = {}, {}
    for sid, row in _samples(tdir).items():
        x, _, _ = load_features(spec.name, ref, sid, mode)
        ids = (cache_dir(spec.name, ref) / sid / "ids.txt").read_text().split()
        feats[sid] = x
        targets[sid] = _targets(tdir, row, ids, genes)
    fold_scores, per_gene = [], []
    for k, (tr, te) in enumerate(_folds(tdir)):
        xtr = np.concatenate([feats[r["sample_id"]] for r in tr])
        ytr = np.concatenate([targets[r["sample_id"]] for r in tr])
        xte = np.concatenate([feats[r["sample_id"]] for r in te])
        yte = np.concatenate([targets[r["sample_id"]] for r in te])
        r = _pearson(yte, regress(xtr, ytr, xte, method, log=log, max_features=max_features))
        fold_scores.append(float(r.mean()))
        per_gene.append(r)
        log(f"    {task} fold {k}: mean Pearson r {r.mean():.4f}")
    record = {
        "run_id": uuid.uuid4().hex[:12],
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "kind": "hest", "protocol": f"hest-v1.0.2-{method}" if method == "rf" else "hest-main-ridge",
        "fmbench_protocol": PROTOCOL_VERSION, "git": git_state(), "code_sha": code_sha(), "versions": library_versions(),
        "model": spec.name, "model_display": spec.display_name, "paper_name": spec.paper_name,
        "task": f"hest:{task}", "paper_column": PAPER_COLUMN[task], "mode": mode, "method": method,
        "regressor": regressor_info(method, max_features), "hest_revision": revision, "seed": 0, "smoke": False,
        "metrics": {"pearson": float(np.mean(fold_scores)), "pearson_std": float(np.std(fold_scores)),
                    "folds": fold_scores,
                    "per_gene": dict(zip(genes, np.mean(per_gene, axis=0).round(4).tolist()))},
        "leakage": [],
    }
    append_run(record)
    label = method if method != "rf" or max_features == RF_MAX_FEATURES else f"{method} max_features={max_features}"
    log(f"[hest] {spec.name} {task} {mode} {label}: Pearson r {record['metrics']['pearson']:.4f}")
    return record


# --------------------------------------------------------------------------- leaderboard section
def paper_rows() -> list[dict]:
    rows = []
    with (paths.REFERENCE_DIR / "virchow2_table3_hest.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({"source": "paper", "group": {"cls+mean": "CLS+Mean", "mean": "CLS+Mean"}.get(r["mode"], "CLS-Only"),
                         "model": r["model"], "mode": r["mode"],
                         "values": {c: float(r[c]) for c in COLUMNS if r.get(c)}, "avg": float(r["Avg."])})
    return rows


def our_rows(method: str = "rf", max_features=RF_MAX_FEATURES, revision: str = VIRCHOW2_REVISION) -> list[dict]:
    latest = {}
    for r in read_runs():
        if r.get("kind") != "hest" or r.get("method") != method:
            continue
        if r.get("hest_revision", VIRCHOW2_REVISION) != revision:  # another dataset revision: not next to the paper
            continue
        if method == "rf" and _max_features(r) != max_features:  # the ablation is not mixed into the main table
            continue
        key = (r["model"], r["task"], r["mode"])
        if key not in latest or r["time_utc"] >= latest[key]["time_utc"]:
            latest[key] = r
    agg: dict = {}
    for (model, _task, mode), r in latest.items():
        a = agg.setdefault((model, mode), {"source": "ours", "group": TABLE_GROUP.get(mode, mode),
                                           "model": r.get("model_display") or model, "mode": mode,
                                           "paper_name": r.get("paper_name"), "values": {}})
        a["values"][r["paper_column"]] = r["metrics"]["pearson"]
    for a in agg.values():
        a["avg"] = float(np.mean([a["values"][c] for c in COLUMNS])) if all(c in a["values"] for c in COLUMNS) else None
    return sorted(agg.values(), key=lambda a: (a["model"], a["mode"]))


def _by_group(paper: list[dict], ours: list[dict]) -> list[dict]:
    """Paper rows, then ours, for each embedding group in the order of the paper's table (CLS+Mean, then CLS-Only)."""
    groups = list(dict.fromkeys([p["group"] for p in paper] + [o["group"] for o in ours]))
    return [r for g in groups for r in [p for p in paper if p["group"] == g] + [o for o in ours if o["group"] == g]]


def _repro_lines(paper: list[dict], forests: list[tuple[str, list[dict]]]) -> list[str]:
    """Ours minus paper for every run of a model that has its own Table 3 row (matched by paper_name and group)."""
    idx = {(p["model"], p["group"]): p for p in paper}
    rows = []
    for label, ours in forests:
        for r in ours:
            p = idx.get((r.get("paper_name"), r["group"]))
            if not p:
                continue
            diffs = {c: r["values"][c] - p["values"][c] for c in COLUMNS if c in r["values"] and c in p["values"]}
            if not diffs:
                continue
            n = len(diffs)
            signed = round(sum(diffs.values()) / n, 3) + 0.0  # + 0.0 turns -0.0 into 0.0
            mean = f"{sum(abs(d) for d in diffs.values()) / n:.3f} ({signed:+.3f})" + ("" if n == len(COLUMNS)
                                                                                      else f", {n}/{len(COLUMNS)}")
            cells = " | ".join("" if c not in diffs else f"{round(diffs[c], 3) + 0.0:+.3f}" for c in COLUMNS)
            rows.append(f"| {r['model']} | {r['mode']} | {label} | {cells} | {mean} |")
    if not rows:
        return []
    return ["", "Reproduction check (ours minus paper), for models with their own Table 3 row. The mean absolute "
            "difference does not let errors of opposite sign cancel; the signed mean shows bias. The `1.0` rows cover "
            "only the tasks of the ablation:", "",
            "| Model | Mode | max_features | " + " | ".join(COLUMNS) + " | Mean abs. difference (signed) |",
            "|---|---|---|" + "---|" * (len(COLUMNS) + 1), *rows]


def markdown() -> list[str]:
    ours = our_rows("rf")
    ablation = our_rows("rf", 1.0)
    ridge = our_rows("ridge")
    if not (ours or ablation or ridge):
        return []
    paper = paper_rows()
    lines = ["## HEST-Benchmark (Pearson r, Virchow2 Table 3 protocol: random forest)", "",
             "| Source | Model | Mode | " + " | ".join(COLUMNS) + " | Avg. |", "|---|---|---|" + "---|" * (len(COLUMNS) + 1)]
    fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    for r in _by_group(paper, ours):
        lines.append(f"| {r['source']} | {r['model']} | {r['mode']} | "
                     + " | ".join(fmt(r["values"].get(c)) for c in COLUMNS) + f" | {fmt(r['avg'])} |")
    lines += _repro_lines(paper, [(f"'{RF_MAX_FEATURES}'", ours), ("1.0", ablation)])
    if ablation:
        lines += ["", "Ablation: random forest with max_features=1.0 (every feature tried per split, as in cuML 24.04 "
                  "and earlier); the rows above use 'sqrt', as in cuML 24.06 and later (see NOTES.md):", "",
                  "| Model | Mode | " + " | ".join(COLUMNS) + " | Avg. |", "|---|---|" + "---|" * (len(COLUMNS) + 1)]
        for r in ablation:
            lines.append(f"| {r['model']} | {r['mode']} | " + " | ".join(fmt(r["values"].get(c)) for c in COLUMNS)
                         + f" | {fmt(r['avg'])} |")
    if ridge:
        lines += ["", "HEST leaderboard protocol (PCA 256 + ridge); not comparable with the paper rows above:", "",
                  "| Model | Mode | " + " | ".join(COLUMNS) + " | Avg. |", "|---|---|" + "---|" * (len(COLUMNS) + 1)]
        for r in ridge:
            lines.append(f"| {r['model']} | {r['mode']} | " + " | ".join(fmt(r["values"].get(c)) for c in COLUMNS)
                         + f" | {fmt(r['avg'])} |")
    return lines + [""]
