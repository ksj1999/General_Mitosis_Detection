# Virchow2 tile benchmark: leaderboard

Weighted F1 x 100 on the test split of each task (Virchow2 linear-probe protocol). Paper rows are copied from Table 2 of Virchow2 (arXiv 2408.00738v3). Our rows are the mean over seeds. The average is shown only when all 8 tasks are present; otherwise the cell shows the partial mean and k/8.

## CLS+Mean

| Source | Model | Mode | PCam | CRC | CRC No-Norm | WILDS | TILS | MHIST | DLBCL | MIDOG | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | Virchow2G | cls+mean | 94.7 | 97.3 | 97.0 | 98.8 | 94.8 | 86.4 | 62.9 | 83.6 | 89.4 |
| paper | Virchow2 | cls+mean | 93.5 | 97.4 | 96.9 | 98.7 | 94.8 | 85.9 | 60.6 | 80.4 | 88.5 |
| paper | Virchow2G Mini | cls+mean | 93.5 | 97.0 | 96.0 | 98.4 | 94.1 | 83.2 | 58.9 | 79.6 | 87.6 |
| paper | Virchow | cls+mean | 93.3 | 97.3 | 96.8 | 97.1 | 94.9 | 83.6 | 60.2 | 78.7 | 87.7 |
| paper | H-optimus-0 | cls+mean | 93.7 | 96.9 | 92.6 | 98.6 | 94.9 | 84.6 | 59.1 | 79.6 | 87.5 |
| paper | Prov-GigaPath | cls+mean | 94.1 | 96.9 | 96.9 | 97.9 | 94.2 | 82.3 | 60.3 | 79.4 | 87.8 |
| paper | UNI | cls+mean | 93.5 | 96.7 | 95.4 | 98.3 | 94.6 | 84.8 | 58.4 | 75.9 | 87.2 |
| paper | Phikon | cls+mean | 91.5 | 96.3 | 88.7 | 97.0 | 94.7 | 80.9 | 55.6 | 73.0 | 84.7 |
| paper | CTransPath | mean | 87.1 | 96.2 | 84.4 | 94.7 | 93.1 | 81.6 | 54.2 | 64.3 | 82.0 |
| ours | CTransPath (frozen) | pooled | 86.5 | 96.1 | 84.1 | 96.0 | 92.8 ‡ |  | 70.1 | 66.1 | (84.5, 7/8) |
| ours | DINOv2 ViT-g/14 (natural images) | cls+mean | 88.3 | 94.9 | 93.0 | 92.0 | 93.1 |  | 70.8 | 69.4 | (85.9, 7/8) |
| ours | Phikon (frozen) | cls+mean | 92.5 | 96.2 | 86.0 | 96.7 | 94.9 ‡ |  | 71.7 | 77.2 | (87.9, 7/8) |
| ours | Phikon-v2 (frozen) | cls+mean | 89.9 | 96.3 | 96.3 | 96.4 | 94.7 ‡ |  | 70.6 | 77.6 | (88.8, 7/8) |
| ours | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 90.2 | 96.0 | 95.6 | 96.2 | 94.7 ‡ |  | 70.7 | 83.3 ‡ | (89.5, 7/8) |
| ours | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 90.8 | 95.6 | 96.1 | 94.0 | 94.7 ‡ |  | 71.1 | 82.1 ‡ | (89.2, 7/8) |
| ours | ResNet-18 ImageNet | pooled | 83.4 | 89.7 | 81.1 | 88.9 | 90.1 |  | 69.8 | 61.3 | (80.6, 7/8) |
| ours | ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 80.4 | 81.7 | 53.1 | 87.3 | 89.0 |  | 70.1 | 73.6 ‡ | (76.4, 7/8) |

## CLS-Only

| Source | Model | Mode | PCam | CRC | CRC No-Norm | WILDS | TILS | MHIST | DLBCL | MIDOG | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | Virchow2G | cls | 94.4 | 97.4 | 97.0 | 98.3 | 94.7 | 85.2 | 62.4 | 80.5 | 88.7 |
| paper | Virchow2 | cls | 93.5 | 97.6 | 97.1 | 98.5 | 94.7 | 86.0 | 61.5 | 80.0 | 88.6 |
| paper | Virchow2G Mini | cls | 93.4 | 97.0 | 96.2 | 98.2 | 93.9 | 82.1 | 59.9 | 76.2 | 87.1 |
| paper | Virchow | cls | 93.4 | 97.0 | 93.2 | 96.6 | 94.8 | 83.6 | 59.1 | 76.0 | 86.7 |
| paper | H-optimus-0 | cls | 93.7 | 96.3 | 94.3 | 98.6 | 94.9 | 84.8 | 58.2 | 78.1 | 87.4 |
| paper | Prov-GigaPath | cls | 94.1 | 96.7 | 96.8 | 97.6 | 94.1 | 82.2 | 57.3 | 78.2 | 87.1 |
| paper | UNI | cls | 93.4 | 96.3 | 94.3 | 98.3 | 94.5 | 84.5 | 56.5 | 74.8 | 86.6 |
| paper | Phikon | cls | 90.5 | 95.9 | 88.8 | 97.2 | 94.4 | 79.6 | 54.4 | 69.9 | 83.8 |
| paper | NatImg | cls | 88.7 | 95.2 | 92.8 | 93.4 | 93.0 | 83.2 | 53.9 | 68.9 | 83.6 |
| ours | DINOv2 ViT-g/14 (natural images) | cls | 88.1 | 95.0 | 89.3 | 89.9 | 92.8 |  | 70.9 | 68.3 | (84.9, 7/8) |
| ours | Phikon (frozen) | cls | 91.6 | 95.3 | 90.3 | 96.3 | 94.6 ‡ |  | 71.1 | 72.2 | (87.3, 7/8) |
| ours | Phikon-v2 (frozen) | cls | 88.7 | 95.5 | 95.0 | 95.8 | 94.6 ‡ |  | 71.3 | 74.4 | (87.9, 7/8) |
| ours | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 89.2 | 95.2 | 93.5 | 93.5 | 94.6 ‡ |  | 71.2 | 82.8 ‡ | (88.6, 7/8) |
| ours | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 90.0 | 94.9 | 94.3 | 91.2 | 94.5 ‡ |  | 71.4 | 82.1 ‡ | (88.3, 7/8) |

## Reproduction check (ours minus paper)

| Model | Mode | PCam | CRC | CRC No-Norm | WILDS | TILS | MHIST | DLBCL | MIDOG | Mean abs. difference (signed) † |
|---|---|---|---|---|---|---|---|---|---|---|
| CTransPath (frozen) | pooled | -0.6 | -0.1 | -0.3 | 1.3 | -0.3 |  | (15.9) † | 1.8 | 0.7 (+0.3), 6/7 |
| DINOv2 ViT-g/14 (natural images) | cls | -0.6 | -0.2 | -3.5 | -3.5 | -0.2 |  | (17.0) † | -0.6 | 1.5 (-1.5), 6/7 |
| Phikon (frozen) | cls | 1.1 | -0.6 | 1.5 | -0.9 | 0.2 |  | (16.7) † | 2.3 | 1.1 (+0.6), 6/7 |
| Phikon (frozen) | cls+mean | 1.0 | -0.1 | -2.7 | -0.3 | 0.2 |  | (16.1) † | 4.2 | 1.4 (+0.4), 6/7 |

† Not comparable with the paper: for DLBCL the paper gives only "binary 5-year survival" with a patient-level split, not its rule for censored patients or the split itself, and with ours the majority-class baseline (69.5) is above every Table 2 DLBCL number (see the class-balance table and NOTES.md). It is shown in brackets and left out of the mean differences.

## Tasks outside Virchow2 Table 2

Weighted F1 x 100. `midog_nobreast` drops the human breast cancer images of MIDOG++; `dlbcl_he` keeps only the H&E tiles of DLBCL. No paper numbers exist for these.

| Model | Mode | dlbcl_he | midog_nobreast |
|---|---|---|---|
| CTransPath (frozen) | pooled | 65.9 | 64.5 |
| DINOv2 ViT-g/14 (natural images) | cls | 66.8 | 67.0 |
| DINOv2 ViT-g/14 (natural images) | cls+mean | 65.7 | 68.5 |
| Phikon (frozen) | cls | 62.4 | 72.4 |
| Phikon (frozen) | cls+mean | 62.7 | 76.6 |
| Phikon-v2 (frozen) | cls | 64.5 | 73.7 |
| Phikon-v2 (frozen) | cls+mean | 63.4 | 76.1 |
| Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 67.5 | 82.6 |
| Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 64.7 | 82.4 |
| Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 68.2 | 81.5 |
| Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 65.6 | 81.9 |
| ResNet-18 ImageNet | pooled | 67.1 | 60.5 |
| ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 67.5 | 71.6 |

## Class balance and group-level uncertainty

Weighted F1 rewards predicting the majority class when classes are unbalanced. `Majority` is the weighted F1 of always predicting the most common training class; a probe can score below it. The by-group CI resamples whole patients or slides, which is the more realistic interval when the test split has few of them. All values x 100. Weighted F1, Majority, balanced accuracy and AUROC are means over seeds; both CIs are from the lowest seed. Runs recorded before these columns existed leave them empty; re-running `fmbench run` for the model re-uses the cached embeddings and fills them in.

| Task | Model | Mode | Weighted F1 | Majority | Balanced acc. | AUROC | 95% CI (tiles) | 95% CI (by group) |
|---|---|---|---|---|---|---|---|---|
| dlbcl | CTransPath (frozen) | pooled | 70.1 | 69.5 | 50.5 | 58.1 | 69.5-70.7 | 52.2-86.1 (37 patients) |
| dlbcl | DINOv2 ViT-g/14 (natural images) | cls | 70.9 | 69.5 | 51.9 | 58.2 | 70.3-71.4 | 53.7-85.5 (37 patients) |
| dlbcl | DINOv2 ViT-g/14 (natural images) | cls+mean | 70.8 | 69.5 | 51.9 | 58.8 | 70.2-71.4 | 54.0-85.1 (37 patients) |
| dlbcl | Phikon (frozen) | cls | 71.1 | 69.5 | 52.9 | 55.1 | 70.5-71.7 | 55.3-84.5 (37 patients) |
| dlbcl | Phikon (frozen) | cls+mean | 71.7 | 69.5 | 54.7 | 57.2 | 71.2-72.3 | 57.7-83.7 (37 patients) |
| dlbcl | Phikon-v2 (frozen) | cls | 71.3 | 69.5 | 53.4 | 58.6 | 70.7-71.8 | 55.8-84.2 (37 patients) |
| dlbcl | Phikon-v2 (frozen) | cls+mean | 70.6 | 69.5 | 53.7 | 58.4 | 70.1-71.1 | 56.5-82.7 (37 patients) |
| dlbcl | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 71.2 | 69.5 | 52.9 | 57.9 | 70.6-71.7 | 55.2-84.3 (37 patients) |
| dlbcl | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 70.7 | 69.5 | 53.7 | 58.3 | 70.1-71.2 | 56.8-82.8 (37 patients) |
| dlbcl | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 71.4 | 69.5 | 53.2 | 58.5 | 70.8-71.9 | 55.9-84.5 (37 patients) |
| dlbcl | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 71.1 | 69.5 | 54.3 | 59.0 | 70.6-71.6 | 57.5-83.1 (37 patients) |
| dlbcl | ResNet-18 ImageNet | pooled | 69.8 | 69.5 | 50.1 | 57.6 | 69.2-70.4 | 50.7-87.1 (37 patients) |
| dlbcl | ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 70.1 | 69.5 | 50.5 | 56.3 | 69.5-70.7 | 50.8-87.7 (37 patients) |
| dlbcl_he | CTransPath (frozen) | pooled | 65.9 | 67.1 | 50.8 | 51.7 | 64.6-67.1 | 51.8-79.5 (31 patients) |
| dlbcl_he | DINOv2 ViT-g/14 (natural images) | cls | 66.8 | 67.1 | 50.5 | 51.3 | 65.4-68.0 | 51.7-80.9 (31 patients) |
| dlbcl_he | DINOv2 ViT-g/14 (natural images) | cls+mean | 65.7 | 67.1 | 49.6 | 49.4 | 64.3-67.0 | 51.4-79.5 (31 patients) |
| dlbcl_he | Phikon (frozen) | cls | 62.4 | 67.1 | 49.3 | 47.5 | 61.1-63.7 | 48.9-74.5 (31 patients) |
| dlbcl_he | Phikon (frozen) | cls+mean | 62.7 | 67.1 | 50.1 | 47.2 | 61.4-63.9 | 49.0-74.8 (31 patients) |
| dlbcl_he | Phikon-v2 (frozen) | cls | 64.5 | 67.1 | 50.4 | 50.8 | 63.2-65.7 | 50.6-77.0 (31 patients) |
| dlbcl_he | Phikon-v2 (frozen) | cls+mean | 63.4 | 67.1 | 49.5 | 50.7 | 62.1-64.5 | 50.8-75.9 (31 patients) |
| dlbcl_he | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 67.5 | 67.1 | 53.0 | 54.3 | 66.2-68.7 | 54.4-79.9 (31 patients) |
| dlbcl_he | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 64.7 | 67.1 | 51.0 | 54.0 | 63.4-65.9 | 52.9-76.5 (31 patients) |
| dlbcl_he | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 68.2 | 67.1 | 52.9 | 53.0 | 66.9-69.5 | 54.5-81.0 (31 patients) |
| dlbcl_he | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 65.6 | 67.1 | 50.4 | 52.3 | 64.3-66.9 | 52.6-78.5 (31 patients) |
| dlbcl_he | ResNet-18 ImageNet | pooled | 67.1 | 67.1 | 49.6 | 46.9 | 65.6-68.4 | 48.9-84.0 (31 patients) |
| dlbcl_he | ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 67.5 | 67.1 | 50.1 | 47.1 | 66.0-68.9 | 48.1-85.9 (31 patients) |
| wilds | CTransPath (frozen) | pooled | 96.0 | 33.3 | 96.0 | 99.2 | 95.8-96.1 | 94.0-97.2 (10 slides) |
| wilds | DINOv2 ViT-g/14 (natural images) | cls | 89.9 | 33.3 | 89.9 | 97.7 | 89.7-90.1 | 86.6-93.8 (10 slides) |
| wilds | DINOv2 ViT-g/14 (natural images) | cls+mean | 92.0 | 33.3 | 92.0 | 98.3 | 91.8-92.2 | 89.3-95.1 (10 slides) |
| wilds | Phikon (frozen) | cls | 96.3 | 33.3 | 96.3 | 98.9 | 96.1-96.4 | 93.6-97.8 (10 slides) |
| wilds | Phikon (frozen) | cls+mean | 96.7 | 33.3 | 96.7 | 99.2 | 96.6-96.9 | 94.2-98.2 (10 slides) |
| wilds | Phikon-v2 (frozen) | cls | 95.8 | 33.3 | 95.8 | 99.4 | 95.6-95.9 | 91.7-98.2 (10 slides) |
| wilds | Phikon-v2 (frozen) | cls+mean | 96.4 | 33.3 | 96.4 | 99.5 | 96.3-96.5 | 92.0-98.7 (10 slides) |
| wilds | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 93.5 | 33.3 | 93.5 | 99.1 | 93.3-93.7 | 87.5-97.4 (10 slides) |
| wilds | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 96.2 | 33.3 | 96.2 | 99.1 | 96.0-96.3 | 91.9-98.5 (10 slides) |
| wilds | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 91.2 | 33.3 | 91.3 | 99.1 | 91.0-91.4 | 82.6-96.8 (10 slides) |
| wilds | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 94.0 | 33.3 | 94.0 | 99.1 | 93.9-94.2 | 87.3-98.0 (10 slides) |
| wilds | ResNet-18 ImageNet | pooled | 88.9 | 33.3 | 88.9 | 96.5 | 88.7-89.1 | 84.3-92.2 (10 slides) |
| wilds | ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 87.3 | 33.3 | 87.4 | 93.8 | 87.1-87.5 | 83.0-89.9 (10 slides) |

## HEST-Benchmark (Pearson r, Virchow2 Table 3 protocol: random forest)

| Source | Model | Mode | IDC | PRAD | PAAD | SKCM | COAD | READ | CCRCC | HCC | LUAD | LIDC | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | Virchow2G | cls+mean | 0.559 | 0.385 | 0.458 | 0.632 | 0.139 | 0.175 | 0.222 | 0.062 | 0.588 | 0.274 | 0.349 |
| paper | Virchow2 | cls+mean | 0.539 | 0.382 | 0.425 | 0.617 | 0.127 | 0.168 | 0.226 | 0.056 | 0.586 | 0.273 | 0.340 |
| paper | Virchow | cls+mean | 0.545 | 0.372 | 0.465 | 0.624 | 0.159 | 0.133 | 0.211 | 0.065 | 0.601 | 0.269 | 0.344 |
| paper | H-optimus-0 | cls+mean | 0.555 | 0.388 | 0.469 | 0.650 | 0.158 | 0.161 | 0.222 | 0.062 | 0.596 | 0.278 | 0.354 |
| paper | Prov-GigaPath | cls+mean | 0.527 | 0.392 | 0.453 | 0.623 | 0.147 | 0.136 | 0.189 | 0.051 | 0.574 | 0.267 | 0.336 |
| paper | UNI | cls+mean | 0.531 | 0.391 | 0.444 | 0.630 | 0.149 | 0.131 | 0.183 | 0.046 | 0.571 | 0.265 | 0.334 |
| paper | Phikon | cls+mean | 0.494 | 0.388 | 0.433 | 0.595 | 0.151 | 0.123 | 0.178 | 0.062 | 0.577 | 0.259 | 0.326 |
| paper | CTransPath | mean | 0.476 | 0.363 | 0.416 | 0.557 | 0.118 | 0.093 | 0.177 | 0.063 | 0.560 | 0.255 | 0.308 |
| ours | CTransPath (frozen) | pooled | 0.475 | 0.362 | 0.419 | 0.559 | 0.119 | 0.092 | 0.176 | 0.064 | 0.559 | 0.254 | 0.308 |
| ours | DINOv2 ViT-g/14 (natural images) | cls+mean | 0.472 | 0.375 | 0.389 | 0.523 | 0.099 | 0.073 | 0.131 | 0.036 | 0.540 | 0.229 | 0.287 |
| ours | Phikon (frozen) | cls+mean | 0.504 | 0.391 | 0.438 | 0.599 | 0.147 | 0.128 | 0.185 | 0.052 | 0.588 | 0.260 | 0.329 |
| ours | Phikon-v2 (frozen) | cls+mean | 0.509 | 0.385 | 0.436 | 0.574 | 0.139 | 0.126 | 0.175 | 0.043 | 0.574 | 0.259 | 0.322 |
| ours | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 0.502 | 0.384 | 0.430 | 0.550 | 0.138 | 0.118 | 0.170 | 0.044 | 0.571 | 0.253 | 0.316 |
| ours | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls+mean | 0.501 | 0.383 | 0.433 | 0.530 | 0.123 | 0.110 | 0.173 | 0.046 | 0.567 | 0.255 | 0.312 |
| ours | ResNet-18 ImageNet | pooled | 0.397 | 0.306 | 0.366 | 0.381 | 0.095 | 0.066 | 0.134 | 0.012 | 0.517 | 0.211 | 0.249 |
| ours | ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI) | pooled | 0.349 | 0.276 | 0.360 | 0.348 | 0.074 | 0.027 | 0.137 | 0.024 | 0.459 | 0.180 | 0.223 |
| paper | Virchow2G | cls | 0.547 | 0.375 | 0.420 | 0.638 | 0.140 | 0.146 | 0.213 | 0.056 | 0.590 | 0.271 | 0.339 |
| paper | Virchow2 | cls | 0.563 | 0.379 | 0.402 | 0.631 | 0.143 | 0.169 | 0.224 | 0.060 | 0.591 | 0.275 | 0.344 |
| paper | Virchow | cls | 0.529 | 0.364 | 0.431 | 0.612 | 0.150 | 0.120 | 0.205 | 0.065 | 0.584 | 0.274 | 0.334 |
| paper | H-optimus-0 | cls | 0.536 | 0.378 | 0.446 | 0.649 | 0.159 | 0.160 | 0.210 | 0.068 | 0.591 | 0.271 | 0.347 |
| paper | Prov-GigaPath | cls | 0.514 | 0.386 | 0.437 | 0.577 | 0.144 | 0.156 | 0.188 | 0.057 | 0.568 | 0.274 | 0.330 |
| paper | UNI | cls | 0.498 | 0.371 | 0.437 | 0.637 | 0.142 | 0.160 | 0.178 | 0.052 | 0.565 | 0.261 | 0.330 |
| paper | Phikon | cls | 0.468 | 0.383 | 0.397 | 0.581 | 0.136 | 0.138 | 0.175 | 0.043 | 0.566 | 0.264 | 0.315 |
| ours | DINOv2 ViT-g/14 (natural images) | cls | 0.465 | 0.366 | 0.375 | 0.523 | 0.093 | 0.069 | 0.128 | 0.036 | 0.538 | 0.224 | 0.282 |
| ours | Phikon (frozen) | cls | 0.469 | 0.384 | 0.398 | 0.580 | 0.139 | 0.136 | 0.177 | 0.032 | 0.570 | 0.262 | 0.315 |
| ours | Phikon-v2 (frozen) | cls | 0.471 | 0.370 | 0.419 | 0.578 | 0.137 | 0.136 | 0.159 | 0.041 | 0.558 | 0.263 | 0.313 |
| ours | Phikon-v2 + LoRA (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 0.466 | 0.371 | 0.401 | 0.570 | 0.132 | 0.127 | 0.153 | 0.041 | 0.560 | 0.258 | 0.308 |
| ours | Phikon-v2 + LoRA + stain + GRL 0.01 (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI) | cls | 0.470 | 0.373 | 0.409 | 0.554 | 0.118 | 0.109 | 0.154 | 0.039 | 0.553 | 0.259 | 0.304 |

Reproduction check (ours minus paper), for models with their own Table 3 row. The mean absolute difference does not let errors of opposite sign cancel; the signed mean shows bias. The `1.0` rows cover only the tasks of the ablation:

| Model | Mode | max_features | IDC | PRAD | PAAD | SKCM | COAD | READ | CCRCC | HCC | LUAD | LIDC | Mean abs. difference (signed) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTransPath (frozen) | pooled | 'sqrt' | -0.001 | -0.001 | +0.003 | +0.002 | +0.001 | -0.001 | -0.001 | +0.001 | -0.001 | -0.001 | 0.001 (+0.000) |
| Phikon (frozen) | cls | 'sqrt' | +0.001 | +0.001 | +0.001 | -0.001 | +0.003 | -0.002 | +0.002 | -0.011 | +0.004 | -0.002 | 0.003 (-0.001) |
| Phikon (frozen) | cls+mean | 'sqrt' | +0.010 | +0.003 | +0.005 | +0.004 | -0.004 | +0.005 | +0.007 | -0.010 | +0.011 | +0.001 | 0.006 (+0.003) |
| CTransPath (frozen) | pooled | 1.0 |  |  | -0.010 | -0.025 | +0.001 | -0.012 |  | -0.001 | -0.032 |  | 0.014 (-0.013), 6/10 |
| Phikon (frozen) | cls | 1.0 |  |  | -0.021 | -0.071 | +0.002 | +0.002 |  | -0.007 | -0.031 |  | 0.022 (-0.021), 6/10 |
| Phikon (frozen) | cls+mean | 1.0 |  |  | -0.015 | -0.037 | -0.005 | -0.001 |  | -0.010 | -0.017 |  | 0.014 (-0.014), 6/10 |

Ablation: random forest with max_features=1.0 (every feature tried per split, as in cuML 24.04 and earlier); the rows above use 'sqrt', as in cuML 24.06 and later (see NOTES.md):

| Model | Mode | IDC | PRAD | PAAD | SKCM | COAD | READ | CCRCC | HCC | LUAD | LIDC | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTransPath (frozen) | pooled |  |  | 0.406 | 0.532 | 0.119 | 0.081 |  | 0.062 | 0.528 |  |  |
| Phikon (frozen) | cls |  |  | 0.376 | 0.510 | 0.138 | 0.140 |  | 0.036 | 0.535 |  |  |
| Phikon (frozen) | cls+mean |  |  | 0.418 | 0.558 | 0.146 | 0.122 |  | 0.052 | 0.560 |  |  |

‡ The model was trained on data from the same source as this task (see `leakage` in runs.jsonl): models fine-tuned on the MIDOG++ breast images, for MIDOG, or models pretrained on TCGA slides (Phikon, Phikon-v2, CTransPath), for TILS. Where Table 2 has a row for the same model (Phikon, CTransPath), that row has the same exposure but is not marked. Use `midog_nobreast` for a cleaner MIDOG comparison; TILS has no clean alternative.
