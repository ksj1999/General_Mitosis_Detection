#!/usr/bin/env python3
"""
Demo I model: classify a candidate cell as mitotic figure vs imposter, with
leave-one-scanner-out evaluation and two switches:

  --stain   H&E stain augmentation (HED-space jitter, Tellez et al. 2019)
  --adv     scanner-adversarial branch behind a gradient reversal layer
  --mix     MixStyle (Zhou et al. 2021): mix feature statistics between samples

Variants: baseline (none), stain, adv, stain_adv, mix, stain_mix.

Folds are named by the held-out (target) scanner: CS2, S360, XR.
Splits follow the official MIDOG++ split: source-scanner "train" cases are
split by case into train/val (85/15); source "test" cases are the in-domain
test; the target scanner's "test" cases are the out-of-domain test (its
other 39 cases are reported as a larger secondary out-of-domain set).

Usage:
  python scripts/train_patch.py --fold CS2 --variant baseline --seed 0
"""
import argparse
import json
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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Colour-deconvolution matrices (Ruifrok & Johnston 2001; same as skimage.color)
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
LOG_ADJ = float(np.log(1e-6))


# ---------------------------------------------------------------- augmentation
def hed_jitter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x in [0,1], (B,3,H,W). Perturb each stain channel: s' = a*s + b."""
    B = x.shape[0]
    rgb = x.clamp_min(1e-6).permute(0, 2, 3, 1)                      # B,H,W,3
    hed = (torch.log(rgb) / LOG_ADJ) @ HED_FROM_RGB.to(x.device)      # optical density in HED
    hed = hed.clamp_min(0)
    a = torch.empty(B, 1, 1, 3, device=x.device).uniform_(1 - sigma, 1 + sigma)
    b = torch.empty(B, 1, 1, 3, device=x.device).uniform_(-sigma, sigma)
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


# --------------------------------------------------------------------- main
def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(ROOT / "data" / "patches_128.npz")
    X = torch.from_numpy(d["X"])
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    y = meta.y.values.astype(np.float32)

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

    model = Net(n_domains=len(sources), mixstyle=use_mix, mix_seed=a.seed + 1000).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    tr = sets["train"]
    pos_w = torch.tensor((1 - y[tr]).sum() / y[tr].sum(), device=device)
    steps_per_epoch = int(np.ceil(len(tr) / a.bs))
    total = steps_per_epoch * a.epochs
    mean, std = MEAN.to(device), STD.to(device)

    print(f"fold={target} variant={a.variant} seed={a.seed} device={device}")
    print({k: len(v) for k, v in sets.items()}, f"pos_weight={pos_w.item():.2f}")
    best, best_state, bad, step, t0 = -1, None, 0, 0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            xb = to_input(X[idx], device)
            if use_stain:
                xb = hed_jitter(xb, a.sigma)
            xb = (generic_aug(xb) - mean) / std
            yb = torch.from_numpy(y[idx]).to(device)
            p = step / total
            lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            logit, dlogit, _ = model(xb, lam=lam, with_domain=use_adv)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            if use_adv:
                loss = loss + F.cross_entropy(dlogit, dom[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        pv = predict(model, X, sets["val"], device)
        t, f = best_threshold(y[sets["val"]], pv)
        print(f"  epoch {ep + 1:2d}  loss {loss.item():.3f}  val F1 {f:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if f > best:
            best, bad = f, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break

    model.load_state_dict(best_state)
    pv = predict(model, X, sets["val"], device)
    thr, _ = best_threshold(y[sets["val"]], pv)              # threshold from source validation only
    res = {"fold": target, "variant": a.variant, "seed": a.seed, "threshold": thr,
           "epochs_run": ep + 1, "seconds": round(time.time() - t0, 1)}
    pred_rows = []
    for name in ("val", "test_in", "test_out", "test_out_all"):
        idx = sets[name]
        p = predict(model, X, idx, device)
        res[name] = metrics(y[idx], p, thr)
        pred_rows.append(pd.DataFrame({
            "set": name, "row": idx, "ann_id": meta.ann_id.values[idx], "image_id": meta.image_id.values[idx],
            "scanner": meta.scanner.values[idx], "y": y[idx].astype(int), "p": p,
            "unanimous": meta.unanimous.values[idx], "vote_mitotic": meta.vote_mitotic.values[idx]}))
        if name in ("test_out", "test_out_all"):              # recall on unanimous vs disputed mitoses
            m = meta.iloc[idx]
            for tag, sel in (("unanimous", m.unanimous.values), ("disputed", ~m.unanimous.values)):
                pos = (y[idx] == 1) & sel
                res[name][f"recall_{tag}"] = float((p[pos] >= thr).mean()) if pos.any() else float("nan")
                res[name][f"n_pos_{tag}"] = int(pos.sum())
    res["delta_f1"] = res["test_in"]["f1"] - res["test_out"]["f1"]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = f"{target}_{a.variant}_seed{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    (out / "preds").mkdir(exist_ok=True)
    pd.concat(pred_rows).to_csv(out / "preds" / f"{tag}.csv", index=False)   # per-candidate predictions
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save(best_state, ROOT / "checkpoints" / f"{tag}.pt")                 # best-epoch weights
    print(json.dumps({k: (round(v['f1'], 3) if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=SCANNERS, required=True)
    ap.add_argument("--variant", choices=["baseline", "stain", "adv", "stain_adv", "mix", "stain_mix"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=0.05)
    main(ap.parse_args())
