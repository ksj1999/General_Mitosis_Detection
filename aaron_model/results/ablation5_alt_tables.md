# Add-on study summary (ablation5_alt)

Folds CS2, S360, XR; seeds 0, 1, 2, 3, 4; target = all candidates of the held-out scanner (test_out_all). Means = mean over folds of the per-fold seed means; 95 % CI from the paired two-level bootstrap (B = 5000: slides within fold, seed indices shared by both arms).

## Target scanner

| tag | mode | F1 [95% CI] | AUROC [95% CI] | precision | recall | seed SD F1 | F1 CS2 | F1 S360 | F1 XR | AUROC ens. |
|---|---|---|---|---|---|---|---|---|---|---|
| all | plain | 0.733 [0.705, 0.756] | 0.884 [0.870, 0.898] | 0.686 | 0.814 | 0.012 | 0.757 | 0.698 | 0.745 | 0.904 |
| all | adabncf | 0.751 [0.725, 0.770] | 0.881 [0.868, 0.894] | 0.718 | 0.800 | 0.010 | 0.770 | 0.728 | 0.754 | 0.901 |
| all | adabncf_tta | 0.766 [0.740, 0.786] | 0.898 [0.885, 0.910] | 0.729 | 0.820 | 0.008 | 0.783 | 0.746 | 0.768 | 0.908 |
| all | tent | 0.757 [0.731, 0.777] | 0.883 [0.869, 0.896] | 0.718 | 0.811 | 0.010 | 0.772 | 0.734 | 0.765 | 0.901 |
| all | tent_tta | 0.767 [0.740, 0.788] | 0.898 [0.885, 0.911] | 0.727 | 0.824 | 0.011 | 0.785 | 0.746 | 0.769 | 0.908 |
| all | tentim | 0.756 [0.730, 0.776] | 0.883 [0.869, 0.897] | 0.707 | 0.822 | 0.008 | 0.773 | 0.730 | 0.765 | 0.901 |
| all | tentim_tta | 0.768 [0.741, 0.789] | 0.898 [0.885, 0.911] | 0.719 | 0.836 | 0.010 | 0.787 | 0.746 | 0.772 | 0.908 |
| all | adabnrand | 0.593 [0.557, 0.622] | 0.867 [0.854, 0.881] | 0.424 | 0.995 | 0.008 | 0.639 | 0.542 | 0.599 | 0.884 |
| all | adabnrand_tta | 0.587 [0.549, 0.616] | 0.886 [0.873, 0.900] | 0.417 | 0.998 | 0.006 | 0.634 | 0.537 | 0.591 | 0.898 |

## Paired differences (A minus B)

| comparison | metric | CS2 | S360 | XR | pooled [95% CI] | P(draw <= 0) |
|---|---|---|---|---|---|---|
| all@tent_tta - all@adabncf_tta | f1 | +0.002 [-0.008, +0.011] | +0.000 [-0.010, +0.009] | +0.001 [-0.007, +0.009] | +0.001 [-0.004, +0.006] | 0.335 |
| all@tent_tta - all@adabncf_tta | auroc | +0.001 [-0.003, +0.004] | -0.001 [-0.006, +0.003] | +0.001 [-0.003, +0.004] | +0.000 [-0.002, +0.003] | 0.385 |
| all@tentim_tta - all@adabncf_tta | f1 | +0.004 [-0.004, +0.012] | -0.000 [-0.010, +0.009] | +0.003 [-0.005, +0.011] | +0.003 [-0.003, +0.007] | 0.157 |
| all@tentim_tta - all@adabncf_tta | auroc | +0.002 [-0.002, +0.005] | -0.001 [-0.006, +0.003] | +0.001 [-0.003, +0.004] | +0.001 [-0.002, +0.003] | 0.316 |
| all@adabnrand_tta - all@adabncf_tta | f1 | -0.149 [-0.179, -0.121] | -0.209 [-0.262, -0.168] | -0.177 [-0.221, -0.130] | **-0.178 [-0.204, -0.156]** | 1.000 |
| all@adabnrand_tta - all@adabncf_tta | auroc | -0.004 [-0.010, +0.005] | -0.019 [-0.035, -0.006] | -0.011 [-0.024, +0.003] | **-0.011 [-0.019, -0.004]** | 0.999 |
| all@tent - all@adabncf | f1 | +0.002 [-0.006, +0.010] | +0.006 [-0.002, +0.014] | +0.011 [+0.000, +0.023] | **+0.006 [+0.001, +0.012]** | 0.010 |
| all@tent - all@adabncf | auroc | +0.003 [-0.001, +0.007] | +0.000 [-0.006, +0.005] | +0.002 [-0.002, +0.005] | +0.002 [-0.001, +0.004] | 0.108 |

Bold: pooled 95 % CI excludes 0.

## Source-val selection (ranked by mean source-val F1, then AUROC)

| rank | tag | runs | val F1 | val AUROC | test_in F1 | test_in AUROC | gap F1 (in - target, plain) | target F1 @plain (oracle) | target F1 @adabncf (oracle) | target F1 @adabncf_tta (oracle) | target F1 @tent (oracle) | target F1 @tent_tta (oracle) | target F1 @tentim (oracle) | target F1 @tentim_tta (oracle) | target F1 @adabnrand (oracle) | target F1 @adabnrand_tta (oracle) | oracle rank (plain) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | all | 15 | 0.814 | 0.905 | 0.745 | 0.875 | +0.011 | 0.733 | 0.751 | 0.766 | 0.757 | 0.767 | 0.756 | 0.768 | 0.593 | 0.587 | 1 |

Only tags with every fold x seed (15 runs) are ranked; incomplete tags are listed unranked ("-"), averaged over the seeds they have (n/a when a fold has none). Source-val and test_in use source BatchNorm, plain inference, the run's own threshold. Target columns are the oracle view (not used for selection); the oracle rank is among the ranked tags only.

