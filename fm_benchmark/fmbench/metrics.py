"""Metrics. Weighted F1 is the primary metric of the Virchow2 tables."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    pred = probs.argmax(1)
    out = {
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }
    try:
        if probs.shape[1] == 2:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            out["auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=np.arange(probs.shape[1])))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def bootstrap_ci(y_true: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the weighted F1 over test tiles."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = f1_score(y_true[idx], pred[idx], average="weighted")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def cluster_bootstrap_ci(y_true: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the weighted F1 that resamples whole groups (patients, slides).

    Tiles of one patient are correlated, so resampling single tiles gives intervals that are too narrow
    when the test split has few patients (DLBCL: 37 test patients with tiles). With fewer than two groups
    there is nothing to resample and the interval is (nan, nan).
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inverse))]
    members = [order[bounds[g]:bounds[g + 1]] for g in range(len(bounds) - 1)]
    if len(members) < 2:
        return float("nan"), float("nan")
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        stats[b] = f1_score(y_true[idx], pred[idx], average="weighted")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def majority_weighted_f1(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Weighted F1 of always predicting the most common training class: a reference a useful probe should beat.

    It is not a floor: a probe that predicts the minority class too often scores below it.
    """
    major = np.bincount(np.asarray(y_train)).argmax()
    return float(f1_score(y_test, np.full(len(y_test), major), average="weighted", zero_division=0))


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test on paired correctness (the test used for the bold marks in Virchow2).

    Uses the exact binomial test when the number of discordant pairs is below 25, else the
    chi-square test with continuity correction.
    """
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "method": "none"}
    if n < 25:
        k = min(b, c)
        p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return {"b": b, "c": c, "p": float(min(1.0, 2 * p)), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))  # survival function of chi2 with 1 dof
    return {"b": b, "c": c, "p": float(p), "method": "chi2_cc"}
