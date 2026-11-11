"""The Virchow2 linear-probe protocol (Virchow2 sec. 5.4).

"Performance is assessed by training a linear classifier on z-score normalized frozen
embeddings generated from non-augmented samples. The classifier is trained with a batch size
of 4096 using stochastic gradient descent with a cosine learning rate schedule from 1e-2 to 0
for a total of 12.5K iterations ... The classifier checkpoint achieving the lowest validation
loss was selected for evaluation on a holdout test set."

Details the paper does not give (momentum, weight decay, how often validation loss is
measured) are fixed in ``ProbeConfig`` and recorded with every run.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ProbeConfig:
    batch_size: int = 4096
    iterations: int = 12_500
    lr: float = 1e-2
    lr_min: float = 0.0
    momentum: float = 0.0
    weight_decay: float = 0.0
    eval_every: int = 50
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeResult:
    test_probs: np.ndarray
    test_pred: np.ndarray
    val_probs: np.ndarray
    best_iteration: int
    best_val_loss: float
    history: list


def zscore(train: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    """Standardise with mean and std of the training features only."""
    mu = train.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    sd = train.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    sd = np.maximum(sd, eps)
    return [((a - mu) / sd).astype(np.float32) for a in (train, *others)]


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray, num_classes: int, cfg: ProbeConfig = ProbeConfig(),
                       device: str | None = None) -> ProbeResult:
    """Train one linear probe and return test-set probabilities of the best-validation checkpoint."""
    dev = _device(device)
    xtr, xva, xte = zscore(np.asarray(x_train, np.float32), np.asarray(x_val, np.float32), np.asarray(x_test, np.float32))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    xtr_t = torch.from_numpy(xtr).to(dev)
    ytr_t = torch.from_numpy(np.asarray(y_train, np.int64)).to(dev)
    xva_t = torch.from_numpy(xva).to(dev)
    yva_t = torch.from_numpy(np.asarray(y_val, np.int64)).to(dev)

    model = torch.nn.Linear(xtr.shape[1], num_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    def lr_at(it: int) -> float:  # cosine from lr to lr_min over the full schedule
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * it / cfg.iterations))

    n = xtr_t.shape[0]
    bs = min(cfg.batch_size, n)
    perm, pos = torch.randperm(n, generator=gen), 0
    best_loss, best_it, best_state, history = float("inf"), 0, copy.deepcopy(model.state_dict()), []

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        total = 0.0
        for s in range(0, xva_t.shape[0], 65536):
            total += F.cross_entropy(model(xva_t[s:s + 65536]), yva_t[s:s + 65536], reduction="sum").item()
        model.train()
        return total / max(1, xva_t.shape[0])

    model.train()
    for it in range(cfg.iterations):
        if pos + bs > n:  # new epoch: reshuffle, drop the incomplete tail
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + bs].to(dev)
        pos += bs
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        loss = F.cross_entropy(model(xtr_t[idx]), ytr_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % cfg.eval_every == 0 or it + 1 == cfg.iterations:
            vl = val_loss()
            history.append((it + 1, float(loss.item()), vl))
            if vl < best_loss:
                best_loss, best_it, best_state = vl, it + 1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        def probs(x: np.ndarray) -> np.ndarray:
            out = []
            for s in range(0, x.shape[0], 65536):
                out.append(torch.softmax(model(torch.from_numpy(x[s:s + 65536]).to(dev)), dim=1).cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, num_classes), np.float32)

        test_probs, val_probs = probs(xte), probs(xva)
    return ProbeResult(test_probs, test_probs.argmax(1), val_probs, best_it, best_loss, history)
