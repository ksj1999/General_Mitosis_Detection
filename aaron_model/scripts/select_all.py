#!/usr/bin/env python3
"""
Apply the pre-registered selection rule of ADDONS_PROTOCOL.md and write the phase-2 job list.

Uses only source-validation F1 (each run's results JSON "val" block, source BatchNorm, plain inference).
  1. stain config = argmax mean val F1 over {stain, stain_s002, stain_s010, stain_he, stain_scale02}
     (3 folds x 3 seeds each; ties broken by val AUROC)
  2. EMA [stain_ema], soft labels [stain_soft], Macenko [macenko_stain] are kept if their mean val F1 >= that of
     [stain] over the same folds x seeds
  3. "all" = selected stain config + GRL (variant stain_adv) + kept add-ons; TTA + cross-fitted AdaBN at test time

Writes results/addons_selection.json and jobs/phase2.txt: "all", "all minus one training component" and the team
grid (baseline, stain, adv, stain_adv), seeds 0-4 x 3 folds. A configuration identical to an existing tag reuses
that tag, so its finished seeds are not retrained (run_queue.py skips finished jobs).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from addon_common import FOLDS, ROOT, run_config  # noqa: E402

SEEDS_SEL, SEEDS_FINAL = [0, 1, 2], [0, 1, 2, 3, 4]
STAIN_CANDS = ["stain", "stain_s002", "stain_s010", "stain_he", "stain_scale02"]
BINARY = {"ema": "stain_ema", "soft": "stain_soft", "macenko": "macenko_stain"}
MACENKO = "data/patches_128_macenko.npz"
DEF = {"stain_mode": "both", "sigma": 0.05, "ema": 0.0, "soft_labels": False, "patches": "data/patches_128.npz"}
KEYS = ["variant", "stain_mode", "sigma", "ema", "soft_labels", "patches"]
EXISTING = ["baseline", "stain", "adv", "stain_adv"] + STAIN_CANDS[1:] + list(BINARY.values()) + ["macenko"]


def val_stats(tag):
    f1, au = [], []
    for f in FOLDS:
        for s in SEEDS_SEL:
            p = ROOT / "results" / f"{f}_{tag}_seed{s}.json"
            if not p.exists():
                raise SystemExit(f"selection needs every fold x seed: missing {p.relative_to(ROOT)}")
            v = json.loads(p.read_text())["val"]
            f1.append(v["f1"]), au.append(v["auroc"])
    return float(np.mean(f1)), float(np.mean(au))


def has_stain(variant):
    return variant.startswith("stain")


def norm(cfg):
    c = {k: cfg.get(k, DEF.get(k)) for k in KEYS}
    if not has_stain(c["variant"]):
        c["stain_mode"], c["sigma"] = DEF["stain_mode"], DEF["sigma"]
    c["sigma"] = round(float(c["sigma"]), 6)
    c["ema"] = round(float(c["ema"]), 6)
    return c


def flags(c):
    out = ["--variant", c["variant"]]
    if has_stain(c["variant"]):
        if c["sigma"] != DEF["sigma"]:
            out += ["--sigma", f"{c['sigma']:g}"]
        if c["stain_mode"] != DEF["stain_mode"]:
            out += ["--stain-mode", c["stain_mode"]]
    if c["ema"]:
        out += ["--ema", f"{c['ema']:g}"]
    if c["soft_labels"]:
        out += ["--soft-labels"]
    if c["patches"] != DEF["patches"]:
        out += ["--patches", c["patches"]]
    return out


def main():
    stats = {t: val_stats(t) for t in STAIN_CANDS + list(BINARY.values())}
    stain_best = max(STAIN_CANDS, key=lambda t: stats[t])
    ref = stats["stain"][0]
    keep = {k: stats[t][0] >= ref for k, t in BINARY.items()}

    sc = norm(run_config("CS2", stain_best, 0))
    all_cfg = dict(sc, variant="stain_adv", ema=0.995 if keep["ema"] else 0.0, soft_labels=keep["soft"],
                   patches=MACENKO if keep["macenko"] else DEF["patches"])
    all_cfg = norm(all_cfg)

    # configurations to train: all, all minus each training component, team grid
    want = {"all": all_cfg,
            "all_no_stain": norm(dict(all_cfg, variant="adv")),
            "all_no_grl": norm(dict(all_cfg, variant="stain"))}
    if stain_best != "stain":
        want["all_stain_default"] = norm(dict(all_cfg, sigma=DEF["sigma"], stain_mode=DEF["stain_mode"]))
    if keep["ema"]:
        want["all_no_ema"] = norm(dict(all_cfg, ema=0.0))
    if keep["soft"]:
        want["all_no_soft"] = norm(dict(all_cfg, soft_labels=False))
    if keep["macenko"]:
        want["all_no_macenko"] = norm(dict(all_cfg, patches=DEF["patches"]))
    for t in ["baseline", "stain", "adv", "stain_adv"]:
        want[t] = norm({"variant": t})

    # reuse an existing tag with the identical configuration
    existing = {}
    for t in EXISTING:
        if (ROOT / "results" / f"CS2_{t}_seed0.json").exists():
            existing.setdefault(json.dumps(norm(run_config("CS2", t, 0)), sort_keys=True), t)
    tag_of, jobs = {}, []
    for name, c in want.items():
        key = json.dumps(c, sort_keys=True)
        tag = existing.get(key, name)
        if tag in tag_of.values():
            tag_of[name] = tag
            continue
        tag_of[name] = tag
        for s in SEEDS_FINAL:
            for f in FOLDS:
                jobs.append(f"{tag} {f} {s} " + " ".join(flags(c)))

    sel = {"rule": "ADDONS_PROTOCOL.md selection rule (source-val F1, 3 folds x seeds 0-2)",
           "val_f1_auroc": {t: {"val_f1": v[0], "val_auroc": v[1]} for t, v in stats.items()},
           "stain_config": stain_best, "keep": keep, "all_config": all_cfg,
           "tags": tag_of, "configs": want}
    (ROOT / "results" / "addons_selection.json").write_text(json.dumps(sel, indent=2))
    jobs.sort(key=lambda j: (int(j.split()[2]) > 2, j.split()[0] not in (tag_of["all"],)))  # seeds 0-2 first
    (ROOT / "jobs" / "phase2.txt").write_text("# generated by scripts/select_all.py\n" + "\n".join(jobs) + "\n")

    print("source-val F1 / AUROC (3 folds x seeds 0-2):")
    for t, v in stats.items():
        print(f"  {t:14s} {v[0]:.4f} {v[1]:.4f}" + ("  <- stain config" if t == stain_best else "") +
              (f"  keep={keep[[k for k, b in BINARY.items() if b == t][0]]}" if t in BINARY.values() else ""))
    print("all config:", all_cfg)
    print("tags:", tag_of)
    n_new = sum(1 for j in jobs if not (ROOT / "results" / "{1}_{0}_seed{2}.json".format(*j.split()[:3])).exists())
    print(f"wrote jobs/phase2.txt: {len(jobs)} jobs, {n_new} not yet run")


if __name__ == "__main__":
    main()
