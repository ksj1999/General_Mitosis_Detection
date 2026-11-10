# Jun's Mitosis Generalization Work

This directory contains Jun's contributions to the collaborative MIDOG++ project. Run commands from the repository root so paths resolve consistently.

## Contents

- `breast_cancer_subset/`: prepared breast-cancer metadata and annotation documentation; TIFF images are intentionally excluded from Git.
- `demo1/`: classical patch baselines and Faster R-CNN scanner-generalization experiments.
- `demo2/`: pathology foundation-model, LoRA, stain-augmentation, and scanner-probe experiments.
- `mitosis_scanner_workflow/`: reusable orchestration package for scanner-robust detection experiments.

Start with [`demo2/README.md`](demo2/README.md) for the current candidate-classification work and [`demo2/reports/demo2_detailed_summary.md`](demo2/reports/demo2_detailed_summary.md) for methods, results, and limitations.

Generated patches, model weights, embeddings, logs, caches, and copied pathology images remain local and are excluded by the repository `.gitignore`.
