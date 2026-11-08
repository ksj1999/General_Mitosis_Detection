# Confirmatory family (F1, Holm over 6; bootstrap B = 5000)

| component | ΔF1 [95 % CI] | p (raw) | p (Holm) | ΔAUROC [95 % CI] |
|---|---|---|---|---|
| Stain augmentation (all vs all minus stain) | **+0.024 [+0.011, +0.038]** | < 0.0004 | < 0.0024 | +0.014 [+0.006, +0.020] |
| GRL scanner head (all vs all minus GRL) | +0.012 [+0.001, +0.025] | 0.0432 | 0.0972 | +0.008 [+0.002, +0.014] |
| Soft labels (all vs all minus soft labels) | +0.007 [-0.006, +0.020] | 0.2844 | 0.2844 | +0.002 [-0.004, +0.007] |
| TTA (all vs all without TTA) | **+0.015 [+0.006, +0.024]** | 0.0032 | 0.0128 | +0.017 [+0.012, +0.021] |
| Cross-fitted AdaBN (all vs all without AdaBN) | +0.020 [+0.001, +0.040] | 0.0324 | 0.0972 | -0.004 [-0.010, +0.002] |
| Headline: ResNet all vs baseline as trained | **+0.093 [+0.064, +0.127]** | < 0.0004 | < 0.0024 | +0.052 [+0.041, +0.065] |

Bold: Holm-adjusted p < 0.05. p = 2 min(P, 1 - P), P = share of bootstrap draws <= 0.
