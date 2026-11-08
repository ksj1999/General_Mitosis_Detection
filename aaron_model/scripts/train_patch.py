#!/usr/bin/env python3
"""
Demo I model: classify a candidate cell as mitotic figure vs imposter, with
leave-one-scanner-out evaluation and two switches:

  --stain   H&E stain augmentation (HED-space jitter, Tellez et al. 2019)
  --adv     scanner-adversarial branch behind a gradient reversal layer
  --mix     MixStyle (Zhou et al. 2021): mix feature statistics between samples

Variants: baseline (none), stain, adv, stain_adv, mix, stain_mix.

Add-on options (ADDONS_PROTOCOL.md); with none of them the run is identical to the
original script (same RNG calls, outputs named by the variant):
  --stain-mode {both,he_only,scale_only}  he_only: DAB channel untouched; scale_only: no bias term
  --ema DECAY        evaluate/early-stop/threshold/save an EMA of weights and BN buffers
  --soft-labels      BCE target = rater vote fraction (vote_mitotic); evaluation uses hard y
  --patches PATH     patch file (default data/patches_128.npz, same row order as patches_meta.csv)
  --uda-half {A,B}   DANN aligned to the target as well: unlabelled target images of that half
                     (addon_common.target_halves) get scanner label len(sources); adv variants only
  --tag NAME         output name (default: the variant). The six variant names are reserved for the
                     reference runs: a tag equal to a variant name needs that --variant with every
                     training option (above, plus --epochs/--patience/--bs/--lr) at its default
  --overwrite        replace a finished run; without it the script refuses to start when
                     results/{fold}_{tag}_seed{seed}.json exists (the JSON is the completion marker,
                     written last after the preds CSV and the checkpoint, so run_queue.py reruns a
                     job that died before finishing)

UDA note: source and target images share one forward pass in train mode (standard shared-BN DANN),
so the BatchNorm running statistics of a --uda-half model, and hence its source-val threshold and
plain predictions, mix source and adaptation-half target statistics. This is the normalisation the
network was trained with; no target labels are involved, and only the other half is evaluated.

Folds are named by the held-out (target) scanner: CS2, S360, XR.
Splits follow the official MIDOG++ split: source-scanner "train" cases are
split by case into train/val (85/15); source "test" cases are the in-domain
test; the target scanner's "test" cases are the out-of-domain test (its
other 39 cases are reported as a larger secondary out-of-domain set).

Usage:
  python scripts/train_patch.py --fold CS2 --variant baseline --seed 0
  python scripts/train_patch.py --fold CS2 --variant stain --stain-mode he_only --tag stain_he --seed 0
  python scripts/train_patch.py --fold CS2 --variant stain_adv --uda-half A --tag stain_uda_A --seed 0
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SCANNERS = ["XR", "S360", "CS2"]
VARIANTS = ["baseline", "stain", "adv", "stain_adv", "mix", "stain_mix"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Colour-deconvolution matrices (Ruifrok & Johnston 2001; same as skimage.color)
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
LOG_ADJ = float(np.log(1e-6))


# ---------------------------------------------------------------- augmentation
STAIN_MODES = ("both", "he_only", "scale_only")


def hed_jitter(x: torch.Tensor, sigma: float, mode: str = "both") -> torch.Tensor:
    """x in [0,1], (B,3,H,W). Perturb each stain channel: s' = a*s + b.
    mode: both (H, E and D jittered), he_only (D untouched: a_D=1, b_D=0), scale_only (b=0).
    a and b are drawn identically in every mode, so the RNG stream does not depend on it."""
    if mode not in STAIN_MODES:
        raise ValueError(f"unknown stain mode {mode!r}")
    B = x.shape[0]
    rgb = x.clamp_min(1e-6).permute(0, 2, 3, 1)                      # B,H,W,3
    hed = (torch.log(rgb) / LOG_ADJ) @ HED_FROM_RGB.to(x.device)      # optical density in HED
    hed = hed.clamp_min(0)
    a = torch.empty(B, 1, 1, 3, device=x.device).uniform_(1 - sigma, 1 + sigma)
    b = torch.empty(B, 1, 1, 3, device=x.device).uniform_(-sigma, sigma)
    if mode == "he_only":
        a[..., 2] = 1.0
        b[..., 2] = 0.0
    elif mode == "scale_only":
        b.zero_()
    hed = hed * a + b
    rgb = torch.exp(-(hed * -LOG_ADJ) @ RGB_FROM_HED.to(x.device))
    return rgb.clamp(0, 1).permute(0, 3, 1, 2).contiguous()


def generic_aug(x: torch.Tensor) -> torch.Tensor:
    """Flips, 90-degree rotations, brightness/contrast -- used by every variant."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(2, 3))
    B = x.shape[0]
    br = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    ct = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.9, 1.1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * ct + m + br).clamp(0, 1)


# ---------------------------------------------------------------------- model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021). With probability p, replace each sample's
    channel-wise feature mean/std by a Beta(alpha, alpha) mix of its own and those of
    a random other sample in the batch. Training only; no parameters. Uses its own RNG
    so the batch order and augmentation draws match the run without MixStyle."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, seed: int = 0):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
        self.rng = np.random.RandomState(seed)

    def forward(self, x):
        if not self.training or self.rng.rand() > self.p:
            return x
        B = x.shape[0]
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt().detach()
        lmda = torch.from_numpy(self.rng.beta(self.alpha, self.alpha, (B, 1, 1, 1))).float().to(x.device)
        perm = torch.from_numpy(self.rng.permutation(B)).to(x.device)
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]
        return (x - mu) / sig * sig_mix + mu_mix


class Net(nn.Module):
    def __init__(self, n_domains: int, mixstyle: bool = False, mix_seed: int = 0):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        r.fc = nn.Identity()
        self.encoder = r
        self.task_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
        self.scanner_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_domains))
        if mixstyle:   # after residual stages 1-3, as in the paper; hooks keep state_dict keys unchanged
            self.mixstyle = MixStyle(seed=mix_seed)
            for name in ("layer1", "layer2", "layer3"):
                getattr(r, name).register_forward_hook(lambda m, i, out: self.mixstyle(out))

    def forward(self, x, lam: float = 0.0, with_domain: bool = False):
        f = self.encoder(x)
        logit = self.task_head(f).squeeze(1)
        dom = self.scanner_head(GradReverse.apply(f, lam)) if with_domain else None
        return logit, dom, f


class EMA:
    """Exponential moving average of all parameters AND buffers (BatchNorm running statistics),
    updated after every optimizer step. Same update rule as torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True): the first update copies
    the weights, later ones do ema <- decay*ema + (1-decay)*w. Integer buffers (num_batches_tracked,
    unused with BN momentum) are copied. The copy carries no forward hooks (the MixStyle hooks
    close over the live model) and stays in eval mode; deepcopy consumes no RNG."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.model = copy.deepcopy(model)
        for m in self.model.modules():
            m._forward_hooks.clear()
        self.model.eval().requires_grad_(False)
        src = list(model.named_parameters()) + list(model.named_buffers())
        dst = list(self.model.named_parameters()) + list(self.model.named_buffers())
        assert [k for k, _ in src] == [k for k, _ in dst]
        self.pairs = [(e, w) for (_, e), (_, w) in zip(dst, src)]   # live tensors are updated in place

    @torch.no_grad()
    def update(self):
        for e, w in self.pairs:
            if self.n == 0 or not e.dtype.is_floating_point:
                e.copy_(w)
            else:
                e.lerp_(w, 1.0 - self.decay)
        self.n += 1


# ------------------------------------------------------------------- helpers
def to_input(xb_uint8: torch.Tensor, device) -> torch.Tensor:
    return xb_uint8.to(device).permute(0, 3, 1, 2).float().div_(255)


@torch.no_grad()
def predict(model, X, idx, device, bs=256):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = to_input(X[idx[i:i + bs]], device)
        xb = (xb - MEAN.to(device)) / STD.to(device)
        out.append(torch.sigmoid(model(xb)[0]).cpu())
    return torch.cat(out).numpy()


def best_threshold(y, p):
    ts = np.unique(np.round(p, 3))
    best = (0.5, -1)
    for t in ts:
        f = precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2]
        if f > best[1]:
            best = (float(t), f)
    return best


def metrics(y, p, t):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"f1": f1, "precision": pr, "recall": rc, "auroc": auc, "n": int(len(y)), "n_pos": int(y.sum())}


def check_patches(name, d, X, meta):
    """The patch file must be uint8 (N,H,W,3) and carry ann_id in exactly the row order of patches_meta.csv
    (labels, cases and splits are taken from patches_meta.csv by row index)."""
    if X.dtype != torch.uint8 or X.ndim != 4 or X.shape[3] != 3:
        raise ValueError(f"{name}: X must be uint8 (N,H,W,3), got {X.dtype} {tuple(X.shape)}")
    if X.shape[0] != len(meta):
        raise ValueError(f"{name}: {X.shape[0]} patches, patches_meta.csv has {len(meta)} rows")
    if "ann_id" not in d.files:
        raise ValueError(f"{name}: no ann_id array, so the row order cannot be checked against patches_meta.csv")
    if not np.array_equal(np.asarray(d["ann_id"]), meta.ann_id.values):
        raise ValueError(f"{name}: ann_id order differs from patches_meta.csv")


def write_atomic(path: Path, write):
    """write(tmp_path), then rename onto path: readers never see a partly written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------- main
def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    patches = Path(a.patches)
    d = np.load(patches if patches.is_absolute() else ROOT / patches)
    X = torch.from_numpy(d["X"])
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    check_patches(a.patches, d, X, meta)
    y = meta.y.values.astype(np.float32)
    y_fit = y                                                 # BCE target; evaluation always uses hard y
    if a.soft_labels:
        vote = meta.vote_mitotic.values.astype(np.float32)
        y_fit = np.where(np.isnan(vote), y, vote).astype(np.float32)
    run_tag = a.tag or a.variant

    target = a.fold
    sources = [s for s in SCANNERS if s != target]
    src_train_cases = sorted(meta[(meta.scanner.isin(sources)) & (meta.split == "train")].image_id.unique())
    rng = np.random.RandomState(a.seed)
    val_cases = set(rng.choice(src_train_cases, size=max(1, round(0.15 * len(src_train_cases))), replace=False))
    is_src = meta.scanner.isin(sources)
    sets = {
        "train": np.where(is_src & (meta.split == "train") & ~meta.image_id.isin(val_cases))[0],
        "val": np.where(is_src & meta.image_id.isin(val_cases))[0],
        "test_in": np.where(is_src & (meta.split == "test"))[0],
        "test_out": np.where((meta.scanner == target) & (meta.split == "test"))[0],
        "test_out_all": np.where(meta.scanner == target)[0],
    }
    dom_of = {s: i for i, s in enumerate(sources)}
    dom = torch.tensor(meta.scanner.map(dom_of).fillna(-1).astype(int).values)
    use_stain = a.variant in ("stain", "stain_adv", "stain_mix")
    use_adv = a.variant in ("adv", "stain_adv")
    use_mix = a.variant in ("mix", "stain_mix")

    uda_rows = None
    if a.uda_half:                     # DANN aligned to the target: unlabelled images of one target half
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from addon_common import target_halves    # lazy: addon_common imports this module
        t_rows = sets["test_out_all"]
        in_a = target_halves(meta.image_id.values[t_rows])
        uda_rows = t_rows[in_a] if a.uda_half == "A" else t_rows[~in_a]
        uda_rng = np.random.RandomState(a.seed + 2000)       # separate RNG: source batch order unchanged
        uda_queue = np.empty(0, dtype=np.int64)
    n_domains = len(sources) + (1 if uda_rows is not None else 0)

    model = Net(n_domains=n_domains, mixstyle=use_mix, mix_seed=a.seed + 1000).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    tr = sets["train"]
    pos_w = torch.tensor((1 - y[tr]).sum() / y[tr].sum(), device=device)
    steps_per_epoch = int(np.ceil(len(tr) / a.bs))
    total = steps_per_epoch * a.epochs
    mean, std = MEAN.to(device), STD.to(device)

    config = {"variant": a.variant, "tag": run_tag, "stain_mode": a.stain_mode, "sigma": a.sigma, "ema": a.ema,
              "soft_labels": a.soft_labels, "patches": a.patches, "uda_half": a.uda_half, "n_domains": n_domains,
              "mixstyle": use_mix, "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "patience": a.patience}
    ema = EMA(model, a.ema) if a.ema > 0 else None
    eval_model = ema.model if ema is not None else model     # model used for validation, threshold, output

    print(f"fold={target} variant={a.variant} seed={a.seed} device={device}")
    print({k: len(v) for k, v in sets.items()}, f"pos_weight={pos_w.item():.2f}")
    print("config", json.dumps(config))
    if uda_rows is not None:
        print(f"UDA: {len(uda_rows)} unlabelled target candidates from half {a.uda_half} "
              f"({len(np.unique(meta.image_id.values[uda_rows]))} cases), domain label {len(sources)}")
    best, best_state, bad, step, t0 = -1, None, 0, 0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = to_input(X[idx], device)
            if use_stain:
                xb = hed_jitter(xb, a.sigma, a.stain_mode)
            xb = (generic_aug(xb) - mean) / std
            yb = torch.from_numpy(y_fit[idx]).to(device)
            db = dom[idx]
            if uda_rows is not None:   # target batch: own RNG, same augmentation, one joint forward pass
                while len(uda_queue) < a.bs:
                    uda_queue = np.concatenate([uda_queue, uda_rng.permutation(uda_rows)])
                tidx, uda_queue = uda_queue[:a.bs], uda_queue[a.bs:]
                xt = to_input(X[tidx], device)
                if use_stain:
                    xt = hed_jitter(xt, a.sigma, a.stain_mode)
                xt = (generic_aug(xt) - mean) / std
                xb = torch.cat([xb, xt])               # shared BN: running stats mix source + target half
                db = torch.cat([db, torch.full((len(tidx),), len(sources), dtype=db.dtype)])
            p = step / total
            lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            logit, dlogit, _ = model(xb, lam=lam, with_domain=use_adv)
            if uda_rows is not None:
                logit = logit[:len(idx)]                       # task loss on the labelled source part only
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            if use_adv:
                loss = loss + F.cross_entropy(dlogit, db.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update()
            step += 1
        pv = predict(eval_model, X, sets["val"], device)
        t, f = best_threshold(y[sets["val"]], pv)
        print(f"  epoch {ep + 1:2d}  loss {loss.item():.3f}  val F1 {f:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if f > best:
            best, bad = f, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break

    eval_model.load_state_dict(best_state)
    pv = predict(eval_model, X, sets["val"], device)
    thr, _ = best_threshold(y[sets["val"]], pv)              # threshold from source validation only
    res = {"fold": target, "variant": a.variant, "seed": a.seed, "threshold": thr,
           "epochs_run": ep + 1, "seconds": round(time.time() - t0, 1)}
    pred_rows = []
    for name in ("val", "test_in", "test_out", "test_out_all"):
        idx = sets[name]
        p = predict(eval_model, X, idx, device)
        res[name] = metrics(y[idx], p, thr)
        pred_rows.append(pd.DataFrame({
            "set": name, "row": idx, "ann_id": meta.ann_id.values[idx], "image_id": meta.image_id.values[idx],
            "scanner": meta.scanner.values[idx], "y": y[idx].astype(int), "p": p,
            "unanimous": meta.unanimous.values[idx], "vote_mitotic": meta.vote_mitotic.values[idx]}))
        if name in ("test_out", "test_out_all"):              # recall on unanimous vs disputed mitoses
            m = meta.iloc[idx]
            for grp, sel in (("unanimous", m.unanimous.values), ("disputed", ~m.unanimous.values)):
                pos = (y[idx] == 1) & sel
                res[name][f"recall_{grp}"] = float((p[pos] >= thr).mean()) if pos.any() else float("nan")
                res[name][f"n_pos_{grp}"] = int(pos.sum())
    res["delta_f1"] = res["test_in"]["f1"] - res["test_out"]["f1"]
    res["config"] = config
    out = ROOT / "results"
    tag = f"{target}_{run_tag}_seed{a.seed}"
    # The results JSON marks a finished run (run_queue.py skips it), so it is written last; each file is
    # renamed into place. With --overwrite the old JSON goes first, so a crash below leaves no stale marker.
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (out / f"{tag}.json").unlink(missing_ok=True)
    preds = pd.concat(pred_rows)
    write_atomic(out / "preds" / f"{tag}.csv", lambda q: preds.to_csv(q, index=False))   # per-candidate predictions
    write_atomic(ROOT / "checkpoints" / f"{tag}.pt", lambda q: torch.save(best_state, q))  # best-epoch weights (EMA if --ema)
    write_atomic(out / f"{tag}.json", lambda q: q.write_text(json.dumps(res, indent=2)))
    print(json.dumps({k: (round(v['f1'], 3) if isinstance(v, dict) else v) for k, v in res.items() if k != "config"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=SCANNERS, required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--stain-mode", choices=STAIN_MODES, default="both")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights and BN buffers (0 = off)")
    ap.add_argument("--soft-labels", action="store_true", help="BCE target = rater vote fraction")
    ap.add_argument("--patches", default="data/patches_128.npz", help="relative to the project root unless absolute")
    ap.add_argument("--uda-half", choices=["A", "B"], default=None,
                    help="also align to the unlabelled target images of this half (adv / stain_adv only)")
    ap.add_argument("--tag", default=None, help="output name (default: the variant)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing results/{fold}_{tag}_seed{seed}.json and its preds CSV and checkpoint")
    a = ap.parse_args()
    if not 0.0 <= a.ema < 1.0:
        ap.error("--ema must be in [0, 1)")
    if a.uda_half and a.variant not in ("adv", "stain_adv"):
        ap.error("--uda-half needs the scanner head: --variant adv or stain_adv")
    if (a.stain_mode != ap.get_default("stain_mode") or a.sigma != ap.get_default("sigma")) \
            and a.variant not in ("stain", "stain_adv", "stain_mix"):
        ap.error("--stain-mode / --sigma only apply to stain variants (stain, stain_adv, stain_mix)")
    if a.tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.tag):
        ap.error("--tag: letters, digits, '_', '.', '-' only")
    run_tag = a.tag or a.variant
    _abs = lambda q: (Path(q) if Path(q).is_absolute() else ROOT / q).resolve()  # noqa: E731
    changed = [k for k in ("epochs", "patience", "bs", "lr", "sigma", "stain_mode", "ema", "soft_labels", "uda_half")
               if getattr(a, k) != ap.get_default(k)]
    if _abs(a.patches) != _abs(ap.get_default("patches")):
        changed.append("patches")
    if run_tag in VARIANTS and (run_tag != a.variant or changed):   # reference-run names stay reference runs
        why = f"--variant {a.variant}" if run_tag != a.variant else "non-default " + ", ".join(changed)
        ap.error(f"tag {run_tag!r} is reserved for the reference run --variant {run_tag} at default settings "
                 f"({why}); pass a new --tag")
    done = ROOT / "results" / f"{a.fold}_{run_tag}_seed{a.seed}.json"
    if done.exists() and not a.overwrite:
        ap.error(f"{done.relative_to(ROOT)} exists (finished run); pass --overwrite to replace it, "
                 f"its preds CSV and checkpoint, or use a new --tag")
    main(a)
