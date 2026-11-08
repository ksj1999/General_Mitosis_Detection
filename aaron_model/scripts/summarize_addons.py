#!/usr/bin/env python3
"""
Single summary tool for the add-on study (ADDONS_PROTOCOL.md).

For every requested run tag x inference mode it scores the 3 folds x n seeds on the unseen scanner
(test_out_all = all target candidates) and reports source-validation / in-domain numbers next to them.

Per run (fold, tag, seed)
  results/{fold}_{tag}_seed{s}.json            source val F1/AUROC (res['val']), in-domain test_in F1/AUROC
  results/preds_tta/{fold}_{tag}_seed{s}.csv   row, p_plain, p_tta, p_adabn, p_adabn_tta [, thr_plain, thr_tta]
     merged on 'row' with the optional  _cf.csv   (p_adabncf, p_adabncf_tta)
                                        _tent.csv (p_tent, p_tent_tta, p_tentim, p_tentim_tta)
                                        _rand.csv (p_adabnrand, p_adabnrand_tta)
  results/preds_tta/{fold}_{tag}_seed{s}.json  thr_plain / thr_tta (source val, source BatchNorm)
  Mode m uses probability column p_{m} and threshold thr_tta if m is "tta" or ends with "_tta", else thr_plain.
  Per-row thr_plain / thr_tta columns in the CSVs (merged UDA predictions) take precedence over the JSON.
  Fallbacks: no preds_tta CSV -> mode "plain" from results/preds/{fold}_{tag}_seed{s}.csv (test_out_all rows)
  with the run's own threshold; all other modes of that run are then missing.
Merged UDA tags (e.g. stain_uda: no results/{fold}_stain_uda_seed{s}.json, but {tag}_A and {tag}_B exist)
  source-val / test_in metrics = mean of the two half runs. Target predictions from the merged preds_tta
  files (per-row thresholds); if those do not exist, "plain" is assembled from the half runs' results/preds:
  rows of half B (addon_common.target_halves) from {tag}_A and vice versa, each with its own threshold
  (config.uda_half of {tag}_A / {tag}_B must be A / B).
UDA half-models passed directly (any run whose config.uda_half is set, e.g. stain_uda_A): their training saw
  the unlabelled target images of that half, so the target predictions of those rows are masked; no target
  metric over all target rows exists for them (all their modes are skipped with a note; use the merged tag).
  Their source-val / test_in metrics are still listed in the selection block, flagged as UDA.
A (tag, mode) that is missing for any requested fold x seed is skipped with a printed note.
Source-val selection block: only tags with every requested fold x seed are ranked (protocol selection rule 1);
  incomplete tags are listed unranked after them, flagged with their run count.

Statistics (as in summarize_tta.py / summarize_mixstyle.py)
  Point estimates: per fold the mean over seeds; "mean" = mean over folds of the per-fold seed means.
  Seed SD: per fold the SD over seeds; in the "mean" row the SD over seeds of the per-seed 3-fold mean.
  Paired two-level bootstrap (B iterations, 95 % percentile CI): in each fold the target cases (slides) are
  resampled with replacement, and the seed indices are resampled once per iteration and shared by every
  tag and mode, so both arms of a comparison see the same slides and the same seed indices; the pooled
  difference is the mean over folds of the per-fold bootstrap differences. The draws depend only on
  --rng-seed, the folds, B, the seeds and the cases, so adding tags or modes does not change other CIs.
  Bootstrap F1 is computed from per-case TP/FP/FN counts, bootstrap AUROC from per-case-pair Mann-Whitney
  sums (ties count 1/2); both equal sklearn on the explicitly resampled rows.

Usage
  python scripts/summarize_addons.py --tags baseline stain stain_adv \
      --modes plain tta adabncf adabncf_tta \
      --compare stain@adabncf_tta:stain@plain stain_adv@plain:stain@plain --prefix addons
  (--compare A@modeA:B@modeB = A minus B; a side without "@mode" means @plain.
   Tags / modes named in --compare are added to --tags / --modes.)
Output: results/{prefix}_runs.csv, results/{prefix}_summary.csv, results/{prefix}_pairwise.csv,
        results/{prefix}_tables.md (also printed)
  The prefix must be a plain file-name stem (letters, digits, '_', '-', '.'; no '/' or '..'). An existing output
  file is replaced only if this script wrote it (CSV header / markdown title check); otherwise the run stops
  before any work, unless --overwrite is given (e.g. --prefix tta would otherwise hit summarize_tta.py's files).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from addon_common import FOLDS, ROOT, run_config, target_halves  # noqa: E402

RES = ROOT / "results"
PT = RES / "preds_tta"
EXTRA = ["_cf", "_tent", "_rand"]
DEFAULT_MODES = ["plain", "tta", "adabncf", "adabncf_tta"]
KNOWN_MODES = ["plain", "tta", "adabn", "adabn_tta", "adabncf", "adabncf_tta", "tent", "tent_tta",
               "tentim", "tentim_tta", "adabnrand", "adabnrand_tta"]
THR = ("thr_plain", "thr_tta")
TGT = ("f1", "precision", "recall", "auroc")
RUNLEVEL = ("val_f1", "val_auroc", "test_in_f1", "test_in_auroc", "gap_f1")
RUN_COLS = ["fold", "tag", "seed", "mode", *TGT, "threshold", *RUNLEVEL]
SUMMARY_COLS = ["fold", "tag", "mode", "n_seeds"]
for _k in TGT:
    SUMMARY_COLS += [f"{_k}_mean", f"{_k}_sd"] + ([f"{_k}_ci_lo", f"{_k}_ci_hi"] if _k in ("f1", "auroc") else [])
SUMMARY_COLS += ["auroc_ensemble"] + [f"{_k}_{x}" for _k in RUNLEVEL for x in ("mean", "sd")]
PAIR_COLS = ["metric", "comparison", "tag_a", "mode_a", "tag_b", "mode_b", "fold", "mean_a", "mean_b", "diff",
             "ci_lo", "ci_hi", "significant", "frac_draws_le_0"]
MD_TITLE = "# Add-on study summary ("                       # first line of every tables.md this script writes
PREFIX_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def thr_col(mode):
    """Threshold of a mode: thr_tta for 'tta' and every '*_tta' mode, thr_plain otherwise."""
    return "thr_tta" if mode == "tta" or mode.endswith("_tta") else "thr_plain"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def output_paths(prefix, overwrite):
    """The four output paths. Stops (before any work) on an unsafe prefix, or if an existing output file was
    not written by this script and --overwrite is not given."""
    if not PREFIX_OK.fullmatch(prefix) or ".." in prefix:
        raise SystemExit(f"bad --prefix {prefix!r}: use a plain file-name stem (letters, digits, '_', '-', '.')")
    outs = {"runs": RES / f"{prefix}_runs.csv", "summary": RES / f"{prefix}_summary.csv",
            "pairwise": RES / f"{prefix}_pairwise.csv", "tables": RES / f"{prefix}_tables.md"}
    header = {"runs": ",".join(RUN_COLS), "summary": ",".join(SUMMARY_COLS), "pairwise": ",".join(PAIR_COLS)}
    foreign = []
    for k, p in outs.items():
        if p.exists():
            with p.open() as fh:
                first = fh.readline().rstrip("\r\n")
            if not (first.startswith(MD_TITLE) if k == "tables" else first == header[k]):
                foreign.append(p.relative_to(ROOT))
    if foreign:
        if not overwrite:
            raise SystemExit(f"--prefix {prefix!r} would overwrite file(s) not written by summarize_addons.py: "
                             f"{', '.join(map(str, foreign))}. Pick another prefix (or pass --overwrite).")
        print(f"note: --overwrite: replacing {', '.join(map(str, foreign))}")
    return outs


def is_merged(fold, tag, s):
    return (not (RES / f"{fold}_{tag}_seed{s}.json").exists()
            and all((RES / f"{fold}_{tag}_{h}_seed{s}.json").exists() for h in "AB"))


def uda_half(fold, tag, s):
    """config.uda_half of a run: 'A' / 'B' for a target-aligned DANN half-model, None otherwise (also for old
    runs without a config block, and when the results JSON does not exist)."""
    if not (RES / f"{fold}_{tag}_seed{s}.json").exists():
        return None
    return run_config(fold, tag, s).get("uda_half") or None


def check_halves(fold, tag, s):
    """A merged UDA tag needs {tag}_A trained with target half A and {tag}_B with half B."""
    for h in "AB":
        u = uda_half(fold, f"{tag}_{h}", s)
        if u != h:
            raise SystemExit(f"{fold}_{tag}_{h}_seed{s}: config uda_half is {u!r}, expected {h!r} "
                             f"(needed to evaluate the merged UDA tag {tag})")


def trained_half_mask(img, half):
    """True for target rows whose case is in the half a UDA half-model saw during training."""
    in_a = target_halves(img)
    return in_a if half == "A" else ~in_a


def run_level(fold, tag, s):
    """Source-val and in-domain metrics of a run; mean of the two half runs for a merged UDA tag."""
    r = read_json(RES / f"{fold}_{tag}_seed{s}.json")
    rs = [r] if r is not None else [read_json(RES / f"{fold}_{tag}_{h}_seed{s}.json") for h in "AB"]
    if any(x is None for x in rs):
        return None
    return {"val_f1": np.mean([x["val"]["f1"] for x in rs]), "val_auroc": np.mean([x["val"]["auroc"] for x in rs]),
            "test_in_f1": np.mean([x["test_in"]["f1"] for x in rs]),
            "test_in_auroc": np.mean([x["test_in"]["auroc"] for x in rs]),
            "threshold": rs[0]["threshold"] if len(rs) == 1 else np.nan}


def plain_from_preds(fold, tag, s):
    """p_plain / thr_plain from results/preds (training-time predictions), or None."""
    def one(t):
        f, j = RES / "preds" / f"{fold}_{t}_seed{s}.csv", read_json(RES / f"{fold}_{t}_seed{s}.json")
        if not f.exists() or j is None:
            return None
        d = pd.read_csv(f, usecols=["set", "row", "image_id", "y", "p"])
        d = d[d.set == "test_out_all"].drop(columns="set").rename(columns={"p": "p_plain"})
        d["thr_plain"] = j["threshold"]
        return d
    if not is_merged(fold, tag, s):
        return one(tag)
    halves = {h: one(f"{tag}_{h}") for h in "AB"}
    if any(v is None for v in halves.values()):
        return None
    # the model that saw half A's images is evaluated on half B, and vice versa
    return pd.concat([d[~trained_half_mask(d.image_id.values, h)] for h, d in halves.items()])


def load_target(fold, tag, s, rows, y, img, half=None):
    """DataFrame aligned to `rows` (target rows of patches_meta; case ids `img`) with p_* and per-row thr_*
    columns, or None. For a UDA half-model (`half` = its config.uda_half) the p_* values of the rows of the
    half it trained on are set to NaN, so no mode of it can be scored over all target rows."""
    main = PT / f"{fold}_{tag}_seed{s}.csv"
    src = "preds_tta"
    if main.exists():
        d = pd.read_csv(main)
        d = d[[c for c in d.columns if c in ("row", "y") or c.startswith("p_") or c in THR]]
        for suf in EXTRA:
            f = PT / f"{fold}_{tag}_seed{s}{suf}.csv"
            if f.exists():
                e = pd.read_csv(f)
                cols = [c for c in e.columns if (c.startswith("p_") or c in THR) and c not in d.columns]
                d = d.merge(e[["row"] + cols], on="row", how="left", validate="one_to_one")
        j = read_json(PT / f"{fold}_{tag}_seed{s}.json") or {}
        for t in THR:
            if t not in d.columns:
                d[t] = float(j[t]) if t in j else np.nan
        if d["thr_plain"].isna().all() and not is_merged(fold, tag, s):
            r = read_json(RES / f"{fold}_{tag}_seed{s}.json")
            if r is not None:
                d["thr_plain"] = r["threshold"]
    else:
        d = plain_from_preds(fold, tag, s)
        if d is None:
            return None, None
        d["thr_tta"] = np.nan
        src = "results/preds (plain only)"
    if d.row.duplicated().any():
        raise SystemExit(f"duplicate rows in the predictions of {fold}_{tag}_seed{s}")
    d = d.set_index("row").reindex(rows)
    if "y" in d.columns:
        ok = d.y.notna()
        if not (d.y[ok].values.astype(bool) == y[ok.values]).all():
            raise SystemExit(f"labels of {fold}_{tag}_seed{s} do not match data/patches_meta.csv")
    if half is not None:
        seen = trained_half_mask(img, half)
        for c in [c for c in d.columns if c.startswith("p_")]:
            v = d[c].values.astype(float)
            v[seen] = np.nan
            d[c] = v
    return d, src


def run_arrays(d, mode):
    """(p, thr) arrays for a mode, or None if the run lacks it for any target row."""
    pc, tc = f"p_{mode}", thr_col(mode)
    if d is None or pc not in d.columns or d[pc].isna().any() or d[tc].isna().any():
        return None
    return d[pc].values.astype(float), d[tc].values.astype(float)


class FoldBoot:
    """Case-level bootstrap of one fold: case counts Wc (B x C), seed weights Ws (B x S, rows sum to 1)."""

    def __init__(self, img, y, n_seeds, B, rng):
        self.y = y
        cases, self.cidx = np.unique(img, return_inverse=True)
        self.C = C = len(cases)
        self.Wc = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
        self.Ws = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=B).astype(float) / n_seeds
        npos = np.bincount(self.cidx, weights=y, minlength=C)
        nneg = np.bincount(self.cidx, weights=~y, minlength=C)
        self.PN = (self.Wc @ npos) * (self.Wc @ nneg)
        self.pos_case = self.cidx[y]
        self.neg_of_case = [np.where(~y & (self.cidx == c))[0] for c in range(C)]

    def f1(self, hard):
        y, c, W = self.y, self.cidx, self.Wc
        tp = W @ np.bincount(c, weights=y & hard, minlength=self.C)
        fp = W @ np.bincount(c, weights=~y & hard, minlength=self.C)
        fn = W @ np.bincount(c, weights=y & ~hard, minlength=self.C)
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)

    def auroc(self, p):
        pp = p[self.y]
        A = np.zeros((self.C, self.C))      # A[c, d]: Mann-Whitney sum, positives of case c vs negatives of case d
        for d, ix in enumerate(self.neg_of_case):
            if len(ix):
                sn = np.sort(p[ix])
                lo, hi = np.searchsorted(sn, pp, "left"), np.searchsorted(sn, pp, "right")
                A[:, d] = np.bincount(self.pos_case, weights=lo + 0.5 * (hi - lo), minlength=self.C)
        num = ((self.Wc @ A) * self.Wc).sum(1)
        return np.divide(num, self.PN, out=np.full_like(num, np.nan), where=self.PN > 0)


def ci(a):
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def sd(v):
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def parse_compare(c):
    try:
        a, b = c.split(":")
    except ValueError:
        raise SystemExit(f"bad --compare {c!r}: expected tagA@modeA:tagB@modeB")

    def side(x):
        t, _, m = x.partition("@")
        return t, m or "plain"
    return side(a), side(b)


def fmt(x, signed=False, nd=3):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def main(a):
    t0 = time.time()
    outs = output_paths(a.prefix, a.overwrite)
    folds, seeds, B = a.folds, a.seeds, a.B
    comps = [parse_compare(c) for c in a.compare]
    tags = list(dict.fromkeys(a.tags + [t for c in comps for t, _ in c]))
    modes = list(dict.fromkeys(a.modes + [m for c in comps for _, m in c]))
    for m in modes:
        if m not in KNOWN_MODES:
            print(f"note: mode {m!r} is not a known mode; looking for column p_{m} anyway")
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    rng = np.random.default_rng(a.rng_seed)

    info, data, src, uh = {}, {}, {}, {}
    fb, ys = {}, {}
    for fold in folds:
        rows = np.where(meta.scanner.values == fold)[0]
        img = meta.image_id.values[rows]
        ys[fold] = y = meta.y.values[rows].astype(bool)
        fb[fold] = FoldBoot(img, y, len(seeds), B, rng)
        for tag in tags:
            for s in seeds:
                if is_merged(fold, tag, s):
                    check_halves(fold, tag, s)
                uh[(fold, tag, s)] = uda_half(fold, tag, s)
                info[(fold, tag, s)] = run_level(fold, tag, s)
                data[(fold, tag, s)], src[(fold, tag, s)] = load_target(fold, tag, s, rows, y, img,
                                                                        uh[(fold, tag, s)])

    for tag in tags:
        miss = [f"{f}/seed{s}" for f in folds for s in seeds if info[(f, tag, s)] is None]
        if miss:
            print(f"note: {tag}: no results JSON (source-val / test_in metrics n/a) for {', '.join(miss)}")
        halves = sorted({uh[(f, tag, s)] for f in folds for s in seeds} - {None})
        if halves:
            base = tag[:-2] if tag[-2:] in ("_A", "_B") else "the merged UDA tag"
            print(f"note: {tag}: UDA half-model (config uda_half = {'/'.join(halves)}: trained on the unlabelled "
                  f"target images of that half); target predictions of that half are masked, so no target metric "
                  f"over all target rows is reported for it. Use the merged tag ({base}) for target results.")
        fall = sorted({f"{f}/seed{s}" for f in folds for s in seeds if src[(f, tag, s)] not in (None, "preds_tta")})
        if fall:
            print(f"note: {tag}: no preds_tta CSV, plain predictions taken from results/preds for {', '.join(fall)}")

    # ---- which (tag, mode) are complete
    avail = []
    for tag in tags:
        for m in modes:
            miss = [f"{f}/seed{s}" for f in folds for s in seeds if run_arrays(data[(f, tag, s)], m) is None]
            if miss:
                what = "all runs" if len(miss) == len(folds) * len(seeds) else ", ".join(miss)
                print(f"note: skip {tag}@{m}: p_{m} / {thr_col(m)} missing for {what}")
            else:
                avail.append((tag, m))

    # ---- per-run point metrics and bootstrap arrays
    pt, bt = {}, {}
    for fold in folds:
        y = ys[fold]
        for tag, m in avail:
            for s in seeds:
                p, thr = run_arrays(data[(fold, tag, s)], m)
                hard = p >= thr
                pr, rc, f1, _ = precision_recall_fscore_support(y, hard, average="binary", zero_division=0)
                pt[(fold, tag, s, m)] = {"f1": f1, "precision": pr, "recall": rc, "auroc": roc_auc_score(y, p),
                                         "thr": float(thr[0]) if np.all(thr == thr[0]) else np.nan}
                bt[(fold, tag, s, m)] = {"f1": fb[fold].f1(hard), "auroc": fb[fold].auroc(p)}
        print(f"fold {fold}: bootstrap done ({time.time() - t0:.1f}s)", flush=True)

    def plain_f1(fold, tag, s):
        if (fold, tag, s, "plain") in pt:
            return pt[(fold, tag, s, "plain")]["f1"]
        r = run_arrays(data[(fold, tag, s)], "plain")
        if r is None:
            return np.nan
        return precision_recall_fscore_support(ys[fold], r[0] >= r[1], average="binary", zero_division=0)[2]

    rl = {}
    for fold in folds:
        for tag in tags:
            for s in seeds:
                i = info[(fold, tag, s)] or {}
                rl[(fold, tag, s)] = {k: i.get(k, np.nan) for k in RUNLEVEL[:-1]}
                rl[(fold, tag, s)]["gap_f1"] = i.get("test_in_f1", np.nan) - plain_f1(fold, tag, s)

    # ---- runs.csv
    runs = []
    for fold in folds:
        for tag in tags:
            for s in seeds:
                for m in modes:
                    if (tag, m) in avail:
                        q = pt[(fold, tag, s, m)]
                        runs.append({"fold": fold, "tag": tag, "seed": s, "mode": m,
                                     **{k: q[k] for k in TGT}, "threshold": q["thr"], **rl[(fold, tag, s)]})
    runs = pd.DataFrame(runs, columns=RUN_COLS)
    runs.to_csv(outs["runs"], index=False)

    # ---- summary.csv
    boot = {}                                               # (fold, tag, mode, metric) -> seed-averaged draws
    summ = []
    for tag, m in avail:
        per_seed = {k: {s: [] for s in seeds} for k in TGT + RUNLEVEL}
        fold_rows = []
        for fold in folds:
            row = {"fold": fold, "tag": tag, "mode": m, "n_seeds": len(seeds)}
            for k in TGT + RUNLEVEL:
                v = [pt[(fold, tag, s, m)][k] if k in TGT else rl[(fold, tag, s)][k] for s in seeds]
                for s, x in zip(seeds, v):
                    per_seed[k][s].append(x)
                row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(v)), sd(v)
            for k in ("f1", "auroc"):
                M = np.stack([bt[(fold, tag, s, m)][k] for s in seeds], axis=1)       # B x S
                boot[(fold, tag, m, k)] = (M * fb[fold].Ws).sum(1)
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci(boot[(fold, tag, m, k)])
            row["auroc_ensemble"] = roc_auc_score(ys[fold], np.mean([run_arrays(data[(fold, tag, s)], m)[0]
                                                                   for s in seeds], axis=0))
            fold_rows.append(row)
        mrow = {"fold": "mean", "tag": tag, "mode": m, "n_seeds": len(seeds)}
        for k in TGT + RUNLEVEL:
            mrow[f"{k}_mean"] = float(np.mean([r[f"{k}_mean"] for r in fold_rows]))
            mrow[f"{k}_sd"] = sd([np.mean(per_seed[k][s]) for s in seeds])
        for k in ("f1", "auroc"):
            boot[("mean", tag, m, k)] = np.mean([boot[(f, tag, m, k)] for f in folds], axis=0)
            mrow[f"{k}_ci_lo"], mrow[f"{k}_ci_hi"] = ci(boot[("mean", tag, m, k)])
        mrow["auroc_ensemble"] = float(np.mean([r["auroc_ensemble"] for r in fold_rows]))
        summ += fold_rows + [mrow]
    summ = pd.DataFrame(summ, columns=SUMMARY_COLS)
    summ.to_csv(outs["summary"], index=False)
    S = {(r.fold, r.tag, r.mode): r for r in summ.itertuples(index=False)}

    # ---- pairwise.csv
    pair = []
    for (ta, ma), (tb, mb) in comps:
        name = f"{ta}@{ma} - {tb}@{mb}"
        if (ta, ma) not in avail or (tb, mb) not in avail:
            print(f"note: skip comparison {name}: an arm is not available")
            continue
        for k in ("f1", "auroc"):
            for f in folds + ["mean"]:
                d = boot[(f, ta, ma, k)] - boot[(f, tb, mb, k)]
                va, vb = getattr(S[(f, ta, ma)], f"{k}_mean"), getattr(S[(f, tb, mb)], f"{k}_mean")
                lo, hi = ci(d)
                pair.append({"metric": k, "comparison": name, "tag_a": ta, "mode_a": ma, "tag_b": tb, "mode_b": mb,
                             "fold": f, "mean_a": va, "mean_b": vb, "diff": va - vb, "ci_lo": lo, "ci_hi": hi,
                             "significant": bool(lo > 0 or hi < 0), "frac_draws_le_0": float((d <= 0).mean())})
    pw = pd.DataFrame(pair, columns=PAIR_COLS)
    pw.to_csv(outs["pairwise"], index=False)

    # ---- markdown tables
    md = [f"# Add-on study summary ({a.prefix})", "",
          f"Folds {', '.join(folds)}; seeds {', '.join(map(str, seeds))}; target = all candidates of the held-out "
          f"scanner (test_out_all). Means = mean over folds of the per-fold seed means; 95 % CI from the paired "
          f"two-level bootstrap (B = {B}: slides within fold, seed indices shared by both arms).", ""]
    md += ["## Target scanner", ""]
    hdr = ["tag", "mode", "F1 [95% CI]", "AUROC [95% CI]", "precision", "recall", "seed SD F1"] + \
          [f"F1 {f}" for f in folds] + ["AUROC ens."]
    trows = []
    for tag, m in avail:
        r = S[("mean", tag, m)]
        trows.append([tag, m, f"{fmt(r.f1_mean)} [{fmt(r.f1_ci_lo)}, {fmt(r.f1_ci_hi)}]",
                      f"{fmt(r.auroc_mean)} [{fmt(r.auroc_ci_lo)}, {fmt(r.auroc_ci_hi)}]",
                      fmt(r.precision_mean), fmt(r.recall_mean), fmt(r.f1_sd)] +
                     [fmt(S[(f, tag, m)].f1_mean) for f in folds] + [fmt(r.auroc_ensemble)])
    md += [md_table(hdr, trows), ""]
    if len(pw):
        md += ["## Paired differences (A minus B)", ""]
        prow = []
        for (name, k), g in pw.groupby(["comparison", "metric"], sort=False):
            g = g.set_index("fold")
            cell = lambda f: f"{fmt(g.loc[f, 'diff'], True)} [{fmt(g.loc[f, 'ci_lo'], True)}, {fmt(g.loc[f, 'ci_hi'], True)}]"
            prow.append([name, k] + [cell(f) for f in folds] +
                        [f"**{cell('mean')}**" if g.loc["mean", "significant"] else cell("mean"),
                         fmt(g.loc["mean", "frac_draws_le_0"])])
        md += [md_table(["comparison", "metric"] + folds + ["pooled [95% CI]", "P(draw <= 0)"], prow),
               "", "Bold: pooled 95 % CI excludes 0.", ""]

    # source-val selection block (training-domain validation; target columns are the oracle view)
    sel = []
    for tag in tags:
        def fold_mean(k, t=tag):
            per = [np.nanmean([rl[(f, t, s)][k] for s in seeds]) if any(np.isfinite(rl[(f, t, s)][k]) for s in seeds)
                   else np.nan for f in folds]
            return float(np.mean(per))
        n = sum(info[(f, tag, s)] is not None for f in folds for s in seeds)
        if n == 0:
            continue
        halves = sorted({uh[(f, tag, s)] for f in folds for s in seeds} - {None})
        e = {"tag": tag, "n_runs": n, "complete": n == len(folds) * len(seeds),
             **{k: fold_mean(k) for k in RUNLEVEL},
             "uda": "UDA, uses target images" if any(is_merged(f, tag, s) for f in folds for s in seeds) else
                    f"UDA half {'/'.join(halves)}, uses target images; target n/a" if halves else ""}
        if a.ref in tags:                   # same folds and seeds in both arms (protocol selection rule 2)
            per = []
            for f in folds:
                dd = [rl[(f, tag, s)]["val_f1"] - rl[(f, a.ref, s)]["val_f1"] for s in seeds]
                dd = [x for x in dd if np.isfinite(x)]
                per.append(np.mean(dd) if dd else np.nan)
            e["d_val_f1"] = float(np.mean(per))
        for m in modes:
            e[f"target_f1@{m}"] = S[("mean", tag, m)].f1_mean if (tag, m) in avail else np.nan
        sel.append(e)
    sel = pd.DataFrame(sel)
    if len(sel):
        # protocol selection rule 1: only tags with every fold x seed are ranked; incomplete ones follow, unranked
        sel = sel.sort_values(["complete", "val_f1", "val_auroc"], ascending=False).reset_index(drop=True)
        orank = sel[f"target_f1@{modes[0]}"].where(sel.complete).rank(ascending=False, method="min")
        nfs = len(folds) * len(seeds)
        inc = sel[~sel.complete]
        if len(inc):
            print("note: not ranked in the source-val selection block (runs missing): " +
                  ", ".join(f"{e.tag} ({e.n_runs}/{nfs} runs)" for e in inc.itertuples()))
        hdr = ["rank", "tag", "runs", "val F1", "val AUROC"] + ([f"val F1 - {a.ref}"] if a.ref in tags else []) + \
              ["test_in F1", "test_in AUROC", "gap F1 (in - target, plain)"] + \
              [f"target F1 @{m} (oracle)" for m in modes] + [f"oracle rank ({modes[0]})"]
        srows = []
        for i, e in sel.iterrows():
            flags = ([e.uda] if e.uda else []) + ([] if e.complete else [f"incomplete: {e.n_runs}/{nfs} runs"])
            srows.append([i + 1 if e.complete else "-", e.tag + (f" ({'; '.join(flags)})" if flags else ""),
                          e.n_runs, fmt(e.val_f1), fmt(e.val_auroc)] +
                         ([fmt(e.d_val_f1, True)] if a.ref in tags else []) +
                         [fmt(e.test_in_f1), fmt(e.test_in_auroc), fmt(e.gap_f1, True)] +
                         [fmt(e[f"target_f1@{m}"]) for m in modes] +
                         [int(orank[i]) if np.isfinite(orank[i]) else "n/a"])
        md += ["## Source-val selection (ranked by mean source-val F1, then AUROC)", "",
               md_table(hdr, srows), "",
               f"Only tags with every fold x seed ({nfs} run{'s' if nfs > 1 else ''}) are ranked; incomplete tags "
               "are listed unranked (\"-\"), averaged over the seeds they have (n/a when a fold has none). "
               "Source-val and test_in use source BatchNorm, plain inference, the run's own threshold. "
               "Target columns are the oracle view (not used for selection); the oracle rank is among the ranked "
               "tags only.", ""]
    text = "\n".join(md)
    outs["tables"].write_text(text + "\n")
    print("\n" + text)
    print(f"wrote results/{a.prefix}_runs.csv, _summary.csv, _pairwise.csv, _tables.md "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--compare", nargs="*", default=[], help='"tagA@modeA:tagB@modeB" (A minus B)')
    ap.add_argument("--B", type=int, default=1000, help="bootstrap iterations")
    ap.add_argument("--prefix", default="addons")
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--ref", default="stain", help="reference tag for the source-val delta column")
    ap.add_argument("--rng-seed", type=int, default=2027)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing results/{prefix}_* files even if this script did not write them")
    main(ap.parse_args())
