# Protocol notes and deviations from Virchow2

Paige has released no benchmark code, tile lists or split files. Everything below is either stated in Virchow2 (arXiv 2408.00738v3) or Virchow (Nature Medicine 2024), or is our documented choice.

## Linear probe

The paper states:

- SGD, batch 4096, cosine learning rate from 1e-2 to 0, 12,500 iterations.
- Embeddings are z-scored and computed from tiles without augmentation.
- The checkpoint with the lowest validation loss is used.
- Weighted F1 is the metric, with McNemar's test for significance.

Our choices where the paper is silent:

- No momentum, no weight decay and no warm-up.
- Cross-entropy loss.
- Z-score statistics come from the training split only.
- Validation loss is checked every 50 iterations.
- One probe per seed.

The Virchow Methods also mention scikit-learn "logistic regression", which conflicts with the SGD description; we follow the SGD recipe that both papers describe.

## Preprocessing

Every tile is resized to 224 px with bicubic interpolation, then normalized with the model's mean and std. There is no centre crop, so all models see the same field of view. The papers do not name the interpolation method used to upsample PCam and WILDS (96 → 224) or TILS (100 → 224).

The filter matters for upsampled tiles. On real PCam test tiles, Phikon's CLS embedding of our bicubic resize has a cosine of only 0.95-0.97 with the embedding of Owkin's own processor, which resizes bilinearly; on native 224 px tiles (DLBCL) the two are identical. We keep bicubic for every model, so the comparison between models is fair, but a gap against the paper on PCam, WILDS or TILS may come partly from this. The filter is not part of the embedding cache key: an ablation needs its own model config name.

## Tasks

- **PCam**: official splits, used as published.
- **CRC**: train on NCT-CRC-HE-100K with a 10% hold-out (seed 0, not stratified); test on CRC-VAL-HE-7K. The papers say only "random 10%".
- **CRC No-Norm**: the papers say only the *test* set is unnormalized. Zenodo has no unnormalized 7K set, and Kather et al. state that CRC-VAL-HE-7K is Macenko-normalized. The only reading consistent with the files and with the 107,180-tile count is training on NCT-CRC-HE-100K-NONORM and testing on CRC-VAL-HE-7K. This is the same setup as UNI's CRC-100K-Raw. Phikon's large drop (95.9 → 88.8 CLS-only) is a useful check that the direction is right.
- **WILDS**: train on WILDS train + id_val (hospitals 0, 3 and 4; 335,996 tiles), validate on hospital 1 (34,904) and test on hospital 2 (85,054). These counts match Virchow exactly.
- **TILS**: the official Zenodo partitions (209,221 / 38,601 / 56,275). Virchow lists 247,822 training, 38,601 validation and 56,275 test tiles. That adds up to 342,698, more than the 304,097 tiles in the dataset, and 247,822 = train + val. So Virchow probably trained on train + val while selecting on val, although the paper does not say so and the count may be a reporting slip. We train on train only.
- **MHIST**: the official train/test partition. 217 of the 2,175 training tiles are held out at random (seed 0), the same count as Virchow but not the same tiles.
- **DLBCL**: Virchow2 gives four sentences. Our choices:
  - Tiles are the dataset's own 224 px `Patches/` PNGs for all six stains.
  - Label 1 means died within 5 years (OS < 5 and status 1); label 0 means OS ≥ 5.
  - The 12 patients censored before 5 years are dropped.
  - Patients are split 60/20/20, stratified by label, seed 0. The table is written to `cache/prepared/dlbcl/patients.csv`.

  The test split has 40 patients, 37 of them with tiles (31 with H&E tiles), so the score depends heavily on the split. About 79% of test tiles are from survivors, so predicting the majority class gives a tile-level weighted F1 of 69.5 on our test split (67.1 on `dlbcl_he`), above every paper number (53.9–62.9). The paper does not give its label rule or split, so the reason for the gap is unknown. The split alone moves the baseline a lot: over 2,000 random stratified patient splits with our label, its median is 69.5, its 5th percentile 63.5 and its minimum 57.1. Weighted F1 can also fall below the baseline: of the first four `dlbcl_he` probes, three score 62.4–65.9 against 67.1, and the fourth (ResNet-18 MIDOG++, 67.5) has a balanced accuracy of 50.1, so it predicts almost only the majority class. On full DLBCL our probes score about 70–72, some 16 points above the paper's rows for the same models. **DLBCL is therefore not a reproduction check.** The leaderboard shows its difference from the paper in brackets and leaves it out of the mean differences. Its class-balance table gives balanced accuracy, AUROC, the majority-class baseline and a 95% CI that resamples whole patients. The tile-level CI and McNemar's test treat tiles as independent, which they are not. `dlbcl_he` keeps H&E only; it is a fairer test for H&E-only models.
- **MIDOG**: the paper gives the recipe but not the parameters. Our rule reproduces 21,798 events against the paper's 21,806:
  - Drop events whose box centre is within 112 px of the image border.
  - Drop a negative ("non-mitotic figure") if its centred 224 px tile overlaps the centred tile of any mitotic figure in the image. Border figures count for this check.
  - Cut the tile with a random shift of up to ±86 px, so the 50 px box stays inside the tile but is not centred. The shift is seeded by the annotation id.

  For the split, the paper's counts (13,107 / 4,359 / 4,340, positive rate about 51% in every split) suggest a random split, but the unit is not stated. We split by *image* (60/20/20, stratified by tumour type, seed 0). This prevents near-duplicate tiles from the same region from landing in both train and test. Our counts are 13,505 / 4,140 / 4,153. A tile-level split would probably score higher. `midog_nobreast` drops the 200 human breast cancer images (17,928 events). It exists for models fine-tuned on that subset. The split is drawn over all images before the breast images are dropped, so every image in `midog_nobreast` is in the same split as in `midog`.
- **HEST**: we follow HEST v1.0.2 with `method=random-forest` and dataset revision `78ed54aa`, which is the only revision with 74 samples and the old Visium COAD task. That code fits cuML's `RandomForestRegressor(n_estimators=70, random_state=42)`, one model per gene, and leaves everything else at cuML's defaults. Those defaults depend on the cuML version:
  - In cuML 24.06 to 25.06 the regressor inherits `max_features='sqrt'` from the forest base class, so each split tries sqrt(d) features. The regressor's docstring says 1.0, but the code passes nothing.
  - cuML 24.04 and earlier used `'auto'`, which means 1.0 (every feature) for regression.
  - HEST v1.0.2 (July 2024) pins RAPIDS 24.6 in its README, so we take `'sqrt'`.
  - All these versions stop trees at depth 16 and search splits over 128 histogram bins per feature.

  `hest.rf_params` passes depth 16 and `'sqrt'` explicitly to cuML when it is installed and to the scikit-learn fallback otherwise, so the result does not depend on the installed cuML version. scikit-learn has no binning, so the fallback still gives slightly different numbers. Each record names the backend, the parameters and, for cuML, the version under `regressor`. `--rf-max-features 1.0` runs the pre-24.06 behaviour as an ablation. The leaderboard shows it in a separate table, and its main HEST table uses only `'sqrt'`. No HEST run was recorded before these settings were fixed (2026-11-11). With scikit-learn defaults the trees would grow until their leaves are pure and try every feature, which is further from cuML and 20-50x slower. Even at these settings the forests are the slow part (50 genes x 30 folds per embedding mode, up to 68k training spots in a CCRCC fold). So on TGI the embedding runs on a GPU node (`submit.sh hest-embed`), and the forests run on a CPU node afterwards (`DEPEND=afterany:<id> submit.sh hest-rf`, which calls `fmbench hest --stage regress --skip-done` and does not load the model). The regression stage checks the cached embeddings against the model config. Only the embedding stage can check the model fingerprint (weights, Hub revision). Virchow2 itself says its HEST numbers are "slightly higher" than the HEST paper's. `--method ridge` runs the current HEST leaderboard protocol (scaler, PCA to 256, ridge), which is not comparable with Table 3.

  **Result (TGI, 2026-11-11, scikit-learn fallback, jobs 564589 and 564726):** with `'sqrt'`, the open models reproduce Table 3. The `1.0` ablation was run on the six smaller tasks (SKCM, HCC, LUNG, COAD, READ, PAAD).

  | Pearson r | paper Avg. | ours, `'sqrt'` (10 tasks) | mean abs. diff per task, `'sqrt'` (10 tasks) | mean diff, `'sqrt'` (6 tasks) | mean diff, `1.0` (6 tasks) |
  |---|---|---|---|---|---|
  | Phikon CLS | 0.315 | 0.315 | 0.003 | -0.001 | -0.021 |
  | Phikon CLS+mean | 0.326 | 0.329 | 0.006 | +0.002 | -0.014 |
  | CTransPath mean | 0.308 | 0.308 | 0.001 | +0.001 | -0.013 |

  `1.0` scores lower on almost every task, and most where the signal is strongest: SKCM gives 0.510 against the paper's 0.581 for Phikon CLS, where `'sqrt'` gives 0.580. On weak tasks (COAD, READ, HCC) the two settings are within noise of each other. So the paper's forests tried sqrt(d) features per split, as cuML 24.06 and later does. The scikit-learn fallback, without histogram binning, is close enough that cuML is not needed to reproduce Table 3. The leaderboard's HEST section shows these differences per task. Four forests were fitted twice by overlapping jobs. Each pair agreed to within 1e-16 in Pearson r, a difference that comes only from the order in which the parallel trees are summed. So the seeded forest is reproducible.

## Baselines

- **NatImg** (Table 2, CLS-only): Virchow2 does not say which natural-image model this is, but its NatImg numbers are those of the Virchow (v1) Supplementary Table S4.2, which describes NatImg as a "1.1B parameter model trained on 142 million natural images", evaluated at 224 px. That is DINOv2 ViT-g/14 on LVD-142M, which we use (`dinov2_vitg14`, open weights). Virchow v1 does not say which embedding it probed for NatImg, although Virchow2 lists the row under CLS-only. Virchow v1 (section 4.3) defines its own embedding as CLS+mean and names a different choice only for Phikon (CLS) and CTransPath (mean), so the NatImg numbers it reports are probably CLS+mean. Our numbers support that. Over the six comparable tasks (MHIST missing, DLBCL not comparable), our CLS+mean probe is 0.5 from the paper on average (signed -0.2), and our CLS-only probe is 1.5 (signed -1.5):

  | DINOv2 ViT-g/14 | PCam | CRC | CRC No-Norm | WILDS | TILS | MIDOG |
  |---|---|---|---|---|---|---|
  | paper NatImg | 88.7 | 95.2 | 92.8 | 93.4 | 93.0 | 68.9 |
  | ours, CLS | 88.1 | 95.0 | 89.3 | 89.9 | 92.8 | 68.3 |
  | ours, CLS+mean | 88.3 | 94.9 | 93.0 | 92.0 | 93.1 | 69.4 |

  The leaderboard checks the CLS row against the paper, because that is where Virchow2 lists NatImg; read its 1.5 with this in mind.
- **NatImg at 224 px:** DINOv2 ViT-g/14 is pretrained at 518 px. timm resamples its position embedding to the 16x16 grid once, at load time, with antialiased bicubic interpolation. The official DINOv2 code and Hugging Face transformers interpolate at run time without antialiasing. The two position embeddings agree with a per-position cosine of 0.97 on average (minimum 0.92). This is a small, documented difference from "official" DINOv2 inference at 224 px.
- **CTransPath** (Table 2, `mean` row): we use the timm port `1aurent/swin_tiny_patch4_window7_224.CTransPath`. Its embedding is the mean of the final token map.

## Re-trained mitosis models

The mitosis models from this repository are re-trained on TGI (`scripts/tgi/retrain.sbatch`), because their original checkpoints are not available. The data are the same 150 annotated MIDOG++ human breast cancer images, and the commands and hyperparameters are those of `aaron_model/jobs/phase2.txt` and the `jun_model/demo2` README. Two things differ from the originals:

- **Software:** both run in the fmbench environment, which has torch 2.5.1+cu124, transformers 4.57, timm 1.0 and peft 0.13.2. aaron_model was trained on Apple MPS with newer torch. demo2 was trained with transformers 4.46.3. The demo2 LoRA targets (`query`, `value`) and the Phikon-v2 weights are the same.
- **Hardware:** both use one A100 instead of MPS or the original 24 GB GPU. Results within seed noise of the reported validation numbers count as a reproduction. Exact equality is not expected.

The benchmark uses the CS2-held-out fold, seed 0, as the encoder. The other folds and seeds are kept for the mitosis-level comparison.

Results of the re-training on TGI, 2026-11-11 (jobs 563376 and 563599):

- **ResNet-18 "all"** (3 folds x 5 seeds, without AdaBN or TTA, against the `plain` rows of `aaron_model/results/ablation5_runs.csv`): source-validation F1 0.821 vs 0.814, held-out-scanner F1 (`test_out_all`) 0.723 vs 0.733, held-out AUROC 0.883 vs 0.884, in-scanner test F1 0.743 vs 0.745 (means over the 15 runs). The paired differences are within seed noise (held-out F1 -0.011, SD 0.048). Only the S360 fold is lower (0.661 vs 0.698), and its seeds vary the most.
- **Phikon-v2 LoRA (demo2)**, seed 0, mean over the 3 folds, against the README's seed-0 table: LoRA F1 0.7915 vs 0.7901 and PR-AUC 0.8397 vs 0.8408; LoRA + stain + GRL F1 0.8015 vs 0.8017 and PR-AUC 0.8496 vs 0.8459. Each fold is within 0.006 F1 of the README. Over seeds 0-2: LoRA F1 0.795 (SD 0.007), PR-AUC 0.838; LoRA + stain + GRL F1 0.803 (SD 0.002), PR-AUC 0.849.

## Reference numbers

`reference/*.csv` are transcribed from the PDF and their averages were checked. The paper has these internal inconsistencies:

- Table A9 gives Phikon CLS-only MIDOG as 76.9, but Table 2 gives 69.9.
- The HEST text ranks Virchow2G first at 0.350, ahead of H-optimus-0 at 0.347. Table 3 gives Virchow2G 0.349 (CLS+Mean; its cells average 0.3494) and 0.339 (CLS), never 0.350. The 0.347 is H-optimus-0's CLS-only average. With CLS+Mean, H-optimus-0 averages 0.354 and ranks above Virchow2G.

## Leakage

A model config can list `trained_on` tags. If a task uses one of those data sources, its result is flagged with ‡ in the leaderboard and recorded under `leakage` in `runs.jsonl`. For example, models fine-tuned on MIDOG++ breast images get flagged on `midog`, but not on `midog_nobreast`. Phikon, Phikon-v2 and CTransPath were pretrained on TCGA, so their TILS (TCGA) results are flagged. The overlap is at the level of source slides, not necessarily shared tiles, and the paper's rows for Phikon and CTransPath have the same exposure without a mark. No other overlap was found: Phikon-v2's PANCAN-XL lists TCGA, CPTAC, GTEx, TCIA and private cohorts, but none of Camelyon16/17, NCT-CRC, MHIST, DLBCL-Morph or MIDOG++. The flag is stored when the probe runs, so re-run the probe after adding a tag.

## Data readiness and provenance

- **Readiness:** a task counts as ready only when every archive it downloads has been completely extracted. `extract()` writes `.extracted-<archive>` after the last file, and an archive (or its `.part`) without that marker means the files on disk may be partial. Data placed by hand, with no archive next to it, counts as complete. Without this rule, a GPU job that starts while a download is still running would embed part of a dataset and cache it as complete.
- **Failures:** `fmbench run` carries on with the next task when one fails and exits non-zero at the end. `run_model.sh` carries on with the next model in the same way. A running job keeps the loop it started with, because bash parses the whole `for` loop before running it, but it reads the lines after the loop from the open file when the loop ends. So replace `run_model.sh` only by rename (plain `rsync` without `--inplace`, or a temp file and `mv`), never with `cp`, `scp` or an in-place editor while a job runs it.
- **Concurrent results:** jobs on different nodes append to the same `runs.jsonl` over NFS, so each record is written under a file lock (`lockf`). A line that is still torn is skipped with a warning.
- **Provenance:** every embedding `meta.json` and every `runs.jsonl` record stores a hash of the fmbench sources (`code_sha`, taken when the process starts) and the library versions. Embedding `meta.json` files also store the weights revision (the Hugging Face snapshot), and tile records copy it from the embeddings they use (`embedding_provenance`). The git commit alone is not enough, because the TGI copy is rsynced and not tracked by git. None of these are cache keys, so recording them never forces a re-embed. How models follow the Hub differs:
  - Transformers models (Phikon, Phikon-v2, Hibou-B) have the Hub commit in `model_fingerprint`, which is a cache key. A new revision on the Hub makes their cached tasks fail with "differ ... in ['model_fingerprint']" until they are re-run with `--overwrite`.
  - demo2 LoRA checkpoints store their base revision and refuse to load on a mismatch.
  - timm models from the Hub (CTransPath, DINOv2, Virchow, Virchow2, UNI, UNI2-h, H-optimus-0, Prov-GigaPath) have no revision in their cache key and follow the Hub silently. Compare `weights_revision` in the provenance when comparing their runs months apart.
  - Embeddings cached before provenance was added (phikon, ctranspath, resnet18_midog_all and dinov2_vitg14 from the first TGI run) have none, so their records keep `embedding_provenance: null` unless they are re-embedded. Likewise, their records have no majority baseline or by-group CI; re-running `fmbench run` for the model re-uses the cached embeddings, re-trains only the probes and fills these in.
- **Deploying to TGI:** copy only the changed paths (`fmbench/`, the changed configs, `NOTES.md`, `tests/`) with plain `rsync`, never `--delete` and never the whole `fm_benchmark/` directory. The TGI checkout holds `results/`, its own Linux `.venv` and configs that exist only there. Each new `fmbench` process imports the new code. A running process keeps what it imported, but imports some modules only when it first needs them: task modules, `leaderboard` at the end of `fmbench run`, and `provenance` in `embed`. Keep those compatible with the modules already loaded, and copy a new module before the files that import it (rsync the new file on its own first).
