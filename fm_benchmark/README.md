# fmbench: the Virchow2 tile benchmark, for any encoder

`fmbench` reproduces the tile-level evaluation of Virchow2 (Zimmermann et al., arXiv 2408.00738v3, `papers/virchow2.pdf`). It does this so that any frozen encoder can be scored on the same tasks, including the mitosis models in this repository. Every run is appended to `results/runs.jsonl`. The leaderboard puts our numbers next to the paper's Table 2 and Table 3. Its reproduction checks subtract the paper's numbers from our runs of the same models. So far these are the open ones (Phikon, CTransPath, and DINOv2 as NatImg). Frozen Virchow2 joins them once its gated weights are available (see "Adding a model").

| Task | Paper column | What it is | Data | Access |
|---|---|---|---|---|
| `pcam` | PCam | lymph node metastasis, 96 px at 10x, official splits | Zenodo 2546921, 7.5 GB | automatic |
| `crc` | CRC | 9 colorectal tissue classes, NCT-CRC-HE-100K → CRC-VAL-HE-7K | Zenodo 1214456, 12.5 GB | automatic |
| `crc_nonorm` | CRC No-Norm | same, unnormalized training pool (stain shift) | Zenodo 1214456, +11.7 GB | automatic |
| `wilds` | WILDS | Camelyon17-WILDS, train on 3 hospitals, test on a 5th | HF mirror, 10.7 GB | automatic |
| `tils` | TILS | TCGA tumour-infiltrating lymphocytes, 100 px at 20x | Zenodo 6604094, 6.9 GB | automatic |
| `mhist` | MHIST | colorectal polyps HP vs SSA, 224 px at 5x | BMIRDS, 0.4 GB | **form** (see below) |
| `dlbcl` | DLBCL | 5-year survival from DLBCL-Morph TMA tiles, all 6 stains, 40x | figshare, 31.8 GB zip | automatic |
| `midog` | MIDOG | mitotic figure vs imposter, MIDOG++ events as 224 px tiles, 40x | figshare (MIDOG++) | automatic |
| `midog_nobreast` | (extra) | `midog` without the human breast cancer images | same | automatic |
| `dlbcl_he` | (extra) | `dlbcl` with H&E tiles only | same | automatic |
| HEST (10 tasks) | Table 3 | gene expression from spot patches, random forest, Pearson r | HF `MahmoodLab/hest-bench`, 41 GB | automatic |

PanMSK (Table 1) is internal to MSKCC and cannot be reproduced.

## Protocol

This is the Virchow2 linear probe, sec. 5.4:

- Embeddings come from frozen, non-augmented tiles. Every tile is resized to 224 px with bicubic interpolation and normalized with the model's own mean and std.
- The embeddings are z-scored with training-set statistics.
- One linear layer is trained with SGD, batch 4096, cosine learning rate from 1e-2 to 0, for 12,500 iterations.
- The checkpoint with the lowest validation loss is evaluated on the test split.
- The headline metric is weighted F1 × 100. The run also records macro F1, accuracy, balanced accuracy, AUROC and a 1,000-sample bootstrap 95% CI.
- Test predictions are saved, so any two runs can be compared with McNemar's test (`fmbench compare`), which is the paper's significance test.
- Datasets without a validation split hold out 10% of training with seed 0.
- Two embedding modes match the paper's two blocks:
  - `cls+mean`: the CLS token concatenated with the mean of the patch tokens, registers excluded.
  - `cls`: the CLS token only.
  - CNNs and CTransPath use `pooled`, the mean of the final feature map.

`NOTES.md` lists every place where the paper is silent and what we chose. This covers MIDOG construction, DLBCL labels and splits, the CRC No-Norm pairing, probe momentum, and others.

## Setup on the GPU machine

```bash
git clone <this repo> && cd General_Mitosis_Detection/fm_benchmark
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
huggingface-cli login          # for gated models (Virchow, Virchow2, UNI, H-optimus-0, Prov-GigaPath, Hibou)
cp fmbench.local.yaml.example fmbench.local.yaml   # then point data_root / cache_root at big disks
```

Storage. The archives total about 150 GB, plus HEST (41 GB) and the MIDOG++ images. Extracted data takes roughly as much again; delete the archives in `<data>/<task>/_downloads/` after `prepare` succeeds. Embedding caches use about 1 GB per ViT-H model per 100k tiles (float16), so about 5 GB per model for all tasks.

Gated models need you to accept their terms on the Hugging Face model page first. Virchow and Virchow2 require an institutional email; UNI requires the same.

## Running on TGI (NCSA RAILS)

Everything is wrapped in `scripts/tgi/`. `env.sh` puts data, caches, logs and Hugging Face weights under `~/fmbench/`, because the home quota is large (4.6 TB, 12M files) and `/projects/bdbk` is close to its file quota. `submit.sh` refuses to submit if the account would go above **3 jobs, running or pending**, which is the limit for this account.

```bash
cd ~/code/General_Mitosis_Detection/fm_benchmark
~/.local/bin/uv sync --locked                    # once: Python 3.11, torch 2.5.1+cu124 (pyproject.toml / uv.lock)
scripts/tgi/submit.sh download                   # CPU job: all datasets + HEST, then downloads and test-loads every model
scripts/tgi/submit.sh smoke                      # 1 h GPU job: tests on CUDA and a 256-tile PCam run
scripts/tgi/submit.sh models phikon ctranspath dinov2_vitg14 resnet18_imagenet   # one GPU job, models in sequence
HEST=1 SEEDS="0 1 2" scripts/tgi/submit.sh models virchow2   # HEST in the same GPU job (forests on its CPUs)
scripts/tgi/submit.sh hest-embed phikon ctranspath            # or: HEST embeddings only, on a GPU node ...
DEPEND=afterany:<job id> scripts/tgi/submit.sh hest-rf phikon ctranspath  # ... then the forests on a CPU node
scripts/tgi/submit.sh retrain                    # re-train the mitosis models (below), then benchmark them; one GPU job
source scripts/tgi/env.sh && $PY -m fmbench tasks --check   # interactive use on a login node (no heavy work there)
```

- **Billing and placement:** jobs bill to `bdbk-tgirails`, which has more fair share than `bdbl` so A100 jobs start sooner (`ACCOUNT=` to change). GPU jobs use one A100 on `gpu_a100` (`PARTITION=` to change).
- **Logs:** they go to `~/fmbench/logs/`. Results go to `fm_benchmark/results/` in the checkout.
- **Gated models:** `fmbench models --check` reports `NOACCESS` for any gated model the Hugging Face account has not been approved for. Request access on each model page, then re-run the check.
- **Re-trained mitosis models:** the original aaron_model and jun_model/demo2 checkpoints are not available, so `retrain` trains them again on TGI from the 150 annotated MIDOG++ breast cancer images. `link_breast_subset.sh` links those TIFFs from the fmbench download into `jun_model/breast_cancer_subset/images/`. The job runs `aaron_resnet18.sbatch` (ResNet-18 "all": 3 held-out-scanner folds x 5 seeds), then `demo2_lora.sbatch` (Phikon-v2 LoRA, and LoRA + stain + GRL 0.01, each 3 folds x 3 seeds, with the README settings). It writes `configs/models/resnet18_midog_all.yaml`, `phikon_v2_lora_demo2.yaml` and `phikon_v2_lora_stain_grl001_demo2.yaml` (CS2 fold, seed 0) and benchmarks them. `submit.sh aaron` and `submit.sh demo2` run one part alone. The Hibou-B and Virchow2 LoRA variants need gated access.

## Getting the data

```bash
scripts/download_all.sh                    # everything except MHIST (resumable; re-run after a failure)
python -m fmbench tasks --check            # which tasks are ready, with split sizes
```

Or one task at a time, for example `python -m fmbench download pcam wilds`. Downloads resume after a failure and are checked against the published md5 or sha256. A file that is already in place, such as a copy from another machine, is verified and reused. Tasks whose data is already ready, or whose archive was already extracted, are not downloaded again (`--force` overrides this), so deleting archives after `prepare` is safe.

- **MHIST** needs a person to fill in the form at https://bmirds.github.io/MHIST/. Each user registers individually, and the links expire after 4 hours. Put `annotations.csv` and `images.zip` in `<data>/mhist/`, then run `python -m fmbench prepare mhist`. The data use agreement forbids redistribution, so do not copy it elsewhere.
- **MIDOG** reads `MIDOG++.json` from `midogpp/databases/`. The images go to `<data>/midogpp/images/`; set `FMBENCH_MIDOGPP` to reuse an existing copy. `prepare midog` cuts the 224 px tiles into an HDF5 file.
- **HEST** defaults to the dataset revision that Virchow2 used (`--revision` to change it).

## Running a model

```bash
python -m fmbench models                                   # configs in configs/models/
scripts/run_model.sh virchow2                              # all paper tasks + extra tasks, seed 0
SEEDS="0 1 2" scripts/run_model.sh uni h_optimus_0 resnet18_imagenet
HEST=1 scripts/run_model.sh virchow2                       # also HEST (install cuML for speed)
python -m fmbench hest --model virchow2 --stage embed      # HEST embeddings only (GPU)
python -m fmbench hest --model virchow2 --stage regress    # HEST forests from cached embeddings (CPU)
python -m fmbench run --model virchow2 --task pcam --limit 256 --iterations 200   # quick smoke test
python -m fmbench leaderboard                              # rebuild results/leaderboard.md
python -m fmbench compare --task midog --a virchow2:cls+mean --b virchow2_lora_demo2:cls+mean
```

`run` embeds each task once (the cache is reused and resumable) and then trains a probe per mode and seed. Tasks whose data is missing are skipped with a message. Smoke runs (`--limit`) are kept out of `results/leaderboard.md`; `run --limit` writes them to `results/smoke/leaderboard.md`, labelled `[limit N]`.

Run `virchow2` first. Its "Reproduction check" rows in the leaderboard should be within about one point of the paper on the tasks with public splits (PCam, CRC, WILDS, TILS). Larger gaps point to a data or preprocessing problem, so check those before trusting other numbers.

## Adding a model

Add `configs/models/<name>.yaml`:

```yaml
display_name: My encoder
adapter: timm            # timm | transformers | torchvision | demo2_lora
params:
  model: hf-hub:org/repo # or a timm architecture name plus checkpoint: path/to/weights.pt
  num_prefix_tokens: 5   # CLS + registers, so the patch mean skips them
modes: [cls, cls+mean]
batch_size: 128
precision: fp16
trained_on: []           # e.g. [midogpp_breast]: flags MIDOG results as in-distribution (‡)
paper_name: null         # the model's row name in Virchow2 Table 2, if it has one
```

Then run `scripts/run_model.sh <name>`. For an architecture none of the adapters cover, subclass `Encoder` in `fmbench/models/adapters.py`. Implement `features(x) -> {"cls": ..., "patch_mean": ...}` or `{"pooled": ...}` and register the class in `ADAPTERS`.

The models trained in this repository have example configs (copy them and fill in the checkpoint path):

- `resnet18_midog_all.yaml.example`: the aaron_model ResNet-18 encoder.
- `virchow2_lora_demo2.yaml.example`, `phikon_v2_lora_demo2.yaml.example` and `hibou_b_lora_demo2.yaml.example`: the jun_model/demo2 LoRA encoders. Only the encoder is used; the mitosis head is ignored.

All of these were trained on MIDOG++ breast images. Their `midog` scores are therefore marked ‡, because the model has seen some of those images. `midog_nobreast` is the clean comparison.

## Layout

```
fmbench/tasks/      one module per dataset (download specs, prepare, splits)
fmbench/models/     ModelSpec + encoder adapters
fmbench/embed.py    cached, resumable embedding extraction (float16 .npy per split)
fmbench/probe.py    the Virchow2 linear probe
fmbench/runner.py   probe run -> results/runs.jsonl + predictions
fmbench/hest.py     HEST-Benchmark (random forest / ridge)
fmbench/leaderboard.py  results/leaderboard.{md,csv}
reference/          numbers transcribed from Virchow2 Tables 1, 2, 3 and A9
tests/              unit tests on tiny fake copies of every dataset (pytest -q tests/)
```
