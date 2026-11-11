import numpy as np

from fmbench.metrics import bootstrap_ci, classification_metrics, mcnemar
from fmbench.probe import ProbeConfig, train_linear_probe, zscore
from fmbench.tasks.base import holdout_indices, limit_indices


def _blobs(n, d=16, k=3, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(42).normal(0, 3, (k, d))
    y = rng.integers(0, k, n)
    x = centers[y] + rng.normal(0, 1, (n, d)) + shift
    return x.astype(np.float32), y


def test_probe_learns_separable_data():
    xtr, ytr = _blobs(2000, seed=1)
    xva, yva = _blobs(300, seed=2)
    xte, yte = _blobs(500, seed=3)
    res = train_linear_probe(xtr, ytr, xva, yva, xte, 3, ProbeConfig(iterations=300, eval_every=25, batch_size=512), "cpu")
    m = classification_metrics(yte, res.test_probs)
    assert m["weighted_f1"] > 0.95
    assert 0 < res.best_iteration <= 300
    assert res.test_probs.shape == (500, 3)
    np.testing.assert_allclose(res.test_probs.sum(1), 1, atol=1e-5)


def test_probe_is_deterministic():
    xtr, ytr = _blobs(800, seed=1)
    xva, yva = _blobs(100, seed=2)
    cfg = ProbeConfig(iterations=100, eval_every=20, batch_size=256, seed=7)
    a = train_linear_probe(xtr, ytr, xva, yva, xva, 3, cfg, "cpu")
    b = train_linear_probe(xtr, ytr, xva, yva, xva, 3, cfg, "cpu")
    np.testing.assert_array_equal(a.test_pred, b.test_pred)


def test_zscore_uses_train_statistics_only():
    tr = np.array([[0.0], [2.0]], np.float32)
    te = np.array([[4.0]], np.float32)
    ztr, zte = zscore(tr, te)
    np.testing.assert_allclose(ztr.ravel(), [-1, 1], atol=1e-5)
    np.testing.assert_allclose(zte.ravel(), [3], atol=1e-5)


def test_holdout_is_ten_percent_and_stable():
    keep, hold = holdout_indices(1000)
    assert len(hold) == 100 and len(keep) == 900
    assert not set(keep) & set(hold)
    keep2, hold2 = holdout_indices(1000)
    np.testing.assert_array_equal(hold, hold2)


def test_group_holdout_keeps_groups_together():
    groups = np.repeat(np.arange(50), 10).astype(object)
    keep, hold = holdout_indices(500, groups=groups)
    assert not set(groups[keep]) & set(groups[hold])
    assert len(hold) >= 50


def test_limit_indices_balanced():
    y = np.array([0] * 900 + [1] * 100)
    idx = limit_indices(y, 50)
    assert len(idx) == 50 and (y[idx] == 1).sum() == 25


def test_mcnemar_and_ci():
    y = np.array([0, 1] * 50)
    a = y.copy()
    b = y.copy()
    b[:10] = 1 - b[:10]
    r = mcnemar(y, a, b)
    assert r["b"] == 10 and r["c"] == 0 and r["p"] < 0.01
    lo, hi = bootstrap_ci(y, b, n_boot=200)
    assert lo <= 0.9 <= hi
