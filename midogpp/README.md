# Original MIDOG++ Infrastructure

This directory contains the upstream MIDOG++ annotations, download notebook, data loaders,
and original training, inference, evaluation, and aggregation code. The collaborative models
are kept separately in `../aaron_model/` and `../jun_model/`.

## Contents

- `databases/`: MIDOG++ annotations in COCO JSON and SlideRunner SQLite formats.
- `images/`: downloaded TIFF images; only `.gitkeep` is committed.
- `configs/`: configuration for the original model pipeline.
- `slide/` and `utils/`: upstream loading and detection utilities.
- `Setup.ipynb`: downloads the original image collection from Figshare.
- `training.py`, `inference.py`, `evaluation.py`, and `aggregate.py`: original pipeline code.

## Setup

Create the legacy environment from the collaboration repository root:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r midogpp\requirements.txt
```

Open `midogpp/Setup.ipynb` in VS Code, select the `.venv` kernel, and run all cells. The
notebook accepts either the repository root or this directory as its working directory and
writes downloaded files to `midogpp/images/` or `images/`, respectively.

## Original pipeline

The original code uses paths relative to this directory. Enter it before using those modules:

```powershell
Set-Location midogpp
..\.venv\Scripts\python.exe main.py
```

`main.py` is retained as upstream reference code, not a turnkey command. Training is
commented out by default, and inference expects configured Weights & Biases artifacts.
Review `main.py` and `configs/all.yaml` before running it.

See the repository-level [`README.md`](../README.md) for the collaborative research question,
methods, results, and teammate workflows.
