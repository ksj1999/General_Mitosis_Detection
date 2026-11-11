"""PCam (PatchCamelyon): lymph node metastasis vs normal, 96 px tiles at 10x.

Virchow2 sec. 5.3: 327,680 tiles of 96 x 96 px at 10x, upsampled to 224 x 224, official splits
(train 262,144 / valid 32,768 / test 32,768). The label is positive when the centre 32 x 32 px holds
tumour. Data: Zenodo record 2546921 (gzip-compressed HDF5). The md5 values in the PCam GitHub
README swap the valid and test rows; the ones below come from the Zenodo API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import DownloadSpec, H5ArraySplit, TileTask

ZENODO = "https://zenodo.org/records/2546921/files/{}?download=1"
FILES = {
    # (split tag in the file name, x/y) -> md5 of the .h5.gz file
    ("train", "x"): "1571f514728f59376b705fc836ff4b63",
    ("train", "y"): "35c2d7259d906cfc8143347bb8e05be7",
    ("valid", "x"): "d5b63470df7cfa627aeec8b9dc0c066e",
    ("valid", "y"): "2b85f58b927af9964a4c15b8f7e8f179",
    ("test", "x"): "d8c2d60d490dbd479f8199bdfa0cf6ec",
    ("test", "y"): "60a7035772fbdb7f34eb86d4420cf66a",
}
SPLIT_TAG = {"train": "train", "val": "valid", "test": "test"}
EXPECTED = {"train": 262144, "val": 32768, "test": 32768}


def _name(tag: str, xy: str) -> str:
    return f"camelyonpatch_level_2_split_{tag}_{xy}.h5"


class PCamTask(TileTask):
    name = "pcam"
    display_name = "PCam"
    paper_column = "PCam"
    classes = ("normal", "tumor")
    access = "open"
    magnification = "10x (96 px, upsampled to 224)"
    native_size = 96
    data_sources = ("camelyon16",)
    official_val = True
    description = "PatchCamelyon, official h5 splits from Zenodo 2546921 (about 7.5 GB compressed)."

    def download_specs(self):
        return [DownloadSpec(url=ZENODO.format(_name(tag, xy) + ".gz"), filename=_name(tag, xy) + ".gz", md5=md5,
                             extract=True, size="6.2 GB" if (tag, xy) == ("train", "x") else "")
                for (tag, xy), md5 in FILES.items()]

    def instructions(self) -> str:
        return ("  PCam downloads automatically (python -m fmbench download pcam). To use files you already\n"
                f"  have, put the six camelyonpatch_level_2_split_*.h5 (or .h5.gz) files in {self.raw_dir}.")

    def _find(self, tag: str, xy: str) -> Path:
        for cand in (self.raw_dir / _name(tag, xy), self.raw_dir / "pcam" / _name(tag, xy)):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"{self.raw_dir / _name(tag, xy)} (run: python -m fmbench download pcam)")

    def prepare(self, force: bool = False) -> None:
        import gzip
        import shutil

        for (tag, xy) in FILES:  # accept .h5.gz files dropped in raw_dir by hand
            h5 = self.raw_dir / _name(tag, xy)
            gz = self.raw_dir / (h5.name + ".gz")
            if not h5.exists() and gz.exists():
                print(f"  gunzip {gz.name}")
                with gzip.open(gz, "rb") as src, h5.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 23)

    def official_splits(self):
        import h5py

        out = {}
        for split, tag in SPLIT_TAG.items():
            xpath, ypath = self._find(tag, "x"), self._find(tag, "y")
            with h5py.File(ypath, "r") as f:
                y = np.asarray(f["y"][:]).reshape(-1).astype(np.int64)
            out[split] = H5ArraySplit(xpath, "x", y, ids=[f"{tag}:{i}" for i in range(len(y))])
        return out
