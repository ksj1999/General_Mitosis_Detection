# ResNet add-on study: protocol (fixed 2026-11-07, before any new results)

## Fixed evaluation protocol (unchanged from earlier work)
- MIDOG++ breast candidates, `data/patches_meta.csv` row order; leave-one-scanner-out folds CS2 / S360 / XR
  (fold named by the held-out target scanner).
- Source "train" cases split 85/15 by case with `RandomState(seed)` into train / val (as in `train_patch.py`).
- Main test set: `test_out_all` = all 50 target cases. Secondary: `test_in` = source "test" cases (in-domain).
- Threshold: chosen by F1 on the **source val** set only (`best_threshold`), with source BatchNorm statistics,
  in the same inference mode (plain or TTA). Target labels are never used for any choice.
- Statistics: mean over folds of the mean over seeds; paired two-level bootstrap (slides resampled within fold,
  seeds resampled and shared by both arms) for differences, 95 % percentile CI.
- Cross-fitting for anything that looks at target images: target cases sorted by `image_id`, half A = `cases[::2]`,
  half B = `cases[1::2]`; adapt on one half, predict the other, swap.

## Add-ons
Training time (each: 3 folds x 3 seeds, tag in brackets)
- Stain-augmentation variants on the +Stain model: sigma 0.02 [stain_s002], sigma 0.10 [stain_s010],
  H&E only / DAB channel fixed [stain_he], scale only sigma 0.2 [stain_scale02]. Reference: sigma 0.05 both [stain].
- Macenko per-slide stain normalisation to a fixed reference (Macenko et al. 2009 reference values):
  without [macenko] and with stain augmentation [macenko_stain].
- EMA of weights, decay 0.995 [stain_ema].
- Soft labels = rater vote fraction [stain_soft].
- DANN aligned to the *target* too (unsupervised domain adaptation), cross-fitted by half
  [stain_uda = stain_uda_A + stain_uda_B]. Uses target images -> reported in the test-time-adaptation block.

Test time (no retraining)
- TTA (8 dihedral views), AdaBN cross-fitted (existing).
- TENT (entropy minimisation of BN affine parameters; Adam lr 1e-3, batch 64, 1 epoch over the adaptation half)
  and TENT+IM (entropy minus marginal entropy), cross-fitted. Hyper-parameters fixed a priori.
- AdaBN from random tissue crops of the other half's slides instead of candidate patches [adabnrand].

## Selection rule for "ResNet all" (training-domain validation, DomainBed style)
1. Stain configuration: the one with the highest mean **source-val F1** (3 folds x 3 seeds) among
   {stain, stain_s002, stain_s010, stain_he, stain_scale02}.
2. Each binary training add-on (EMA, soft labels, Macenko) is kept if its mean source-val F1 is >= that of the
   +Stain reference trained with the same seeds.
3. GRL (scanner-adversarial head) is part of "all" by team definition; TTA and cross-fitted AdaBN are part of
   "all" by team definition. TENT, AdaBN-from-random-crops and UDA-DANN are reported as alternatives, not selected.
4. Test-domain results of every configuration are also reported ("oracle" view, flagged as such) so any
   disagreement between the two selection views is visible.

## Final ablation
"all" and "all minus one component" for every component of "all", 5 seeds (0-4) x 3 folds; the team grid
(baseline, +Stain, +GRL, all) is also extended to 5 seeds.

## Amendment (2026-11-08, after phase-1 results, before any phase-2 training run)
Phase 1 selected (results/addons_selection.json): stain config **stain_he** (source-val F1 0.7962 vs 0.7956 for
the default), soft labels kept, EMA and Macenko not kept. So "all" = H&E-only stain jitter + GRL + soft labels,
with TTA + cross-fitted AdaBN at test time. This amendment changes no choice above; it fixes how the final
ablation is read.
- **Confirmatory family** (team metric F1 on `test_out_all`; AUROC reported alongside, not tested): six paired
  differences, Holm-adjusted over the six with two-sided p = 2 min(P, 1 - P), P = bootstrap share of draws <= 0:
  1. all vs all minus stain augmentation (`all_no_stain`), both at AdaBN+TTA
  2. all vs all minus GRL (`all_no_grl`), both at AdaBN+TTA
  3. all vs all minus soft labels (`all_no_soft`), both at AdaBN+TTA
  4. all at AdaBN+TTA vs all at AdaBN (TTA removed)
  5. all at AdaBN+TTA vs all at TTA (AdaBN removed)
  6. all at AdaBN+TTA vs baseline as trained (headline)
- **Supplementary arm** `all_stain_default` (all with the default stain config), defined in `scripts/select_all.py`
  before phase-1 results existed. It is not an all-minus-one arm; it shows what the stain-config selection step
  costs or gains (the rule-4 oracle view already disagrees with the source-val choice).
- Everything else (the same contrasts at plain inference, team-grid pairs, TENT / random-crop AdaBN / UDA
  alternatives) is exploratory and reported with unadjusted CIs. Null results are stated as "no effect larger
  than the CI bound".
