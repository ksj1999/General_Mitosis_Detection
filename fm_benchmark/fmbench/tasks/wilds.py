"""WILDS (Camelyon17-WILDS v1.0): tumour vs normal 96 px tiles, hospital shift.

Virchow (Nat. Med. 2024) and Virchow2 sec. 6.5: 455,954 tiles at 10x, upsampled to 224 x 224;
train 335,996 tiles from three hospitals, validation 34,904 from a fourth and test 85,054 from a
fifth. 335,996 is the WILDS train split (302,436) plus id_val (33,560), so here:

    train = hospitals 0, 3, 4 (WILDS train + id_val);  val = hospital 1;  test = hospital 2.

The raw ``split`` column of metadata.csv only holds 0/1 for every hospital; the WILDS loader assigns
hospitals 1 and 2 to the OOD val/test splits, which we do by ``center``. PNG tiles are RGBA.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .base import DownloadSpec, FileListSplit, TileTask

URL = ("https://huggingface.co/datasets/pangwei/wilds-camelyon17/resolve/"
       "2ed40cf44d4e63de0e89f756e5ef95108127c1df/camelyon17_v1.0.tar.gz")
SHA256 = "9895fa1f425cf4bc3454ca48dc18bd0a861c893d22df2b80cbd0d0e6d9cfe0b0"
SPLIT_OF_CENTER = {0: "train", 3: "train", 4: "train", 1: "val", 2: "test"}
EXPECTED = {"train": 335996, "val": 34904, "test": 85054}


class WildsTask(TileTask):
    name = "wilds"
    display_name = "WILDS"
    paper_column = "WILDS"
    classes = ("normal", "tumor")
    access = "open"
    magnification = "10x (96 px, upsampled to 224)"
    native_size = 96
    data_sources = ("camelyon17",)
    official_val = True
    cluster_unit = "slide"  # the test hospital contributes only 10 slides
    description = "Camelyon17-WILDS v1.0; train on hospitals 0/3/4, select on hospital 1, test on hospital 2."

    def download_specs(self):
        return [DownloadSpec(url=URL, filename="camelyon17_v1.0.tar.gz", size="10.7 GB", sha256=SHA256,
                             extract=True, extract_to="camelyon17_v1.0")]

    def instructions(self) -> str:
        return ("  WILDS downloads automatically (python -m fmbench download wilds). An existing copy of\n"
                f"  camelyon17_v1.0/ (metadata.csv + patches/) can be placed or symlinked at {self.raw_dir}/camelyon17_v1.0.")

    def _root(self) -> Path:
        for cand in (self.raw_dir / "camelyon17_v1.0", self.raw_dir):
            if (cand / "metadata.csv").exists():
                return cand
        raise FileNotFoundError(f"{self.raw_dir}/camelyon17_v1.0/metadata.csv (run: python -m fmbench download wilds)")

    def official_splits(self):
        root = self._root()
        rows = {"train": [], "val": [], "test": []}
        with (root / "metadata.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                split = SPLIT_OF_CENTER[int(r["center"])]
                p, n = r["patient"].zfill(3), r["node"]
                rel = f"patches/patient_{p}_node_{n}/patch_patient_{p}_node_{n}_x_{r['x_coord']}_y_{r['y_coord']}.png"
                rows[split].append((rel, int(r["tumor"]), f"slide{r['slide']}"))
        return {split: FileListSplit([root / rel for rel, _, _ in rs], [y for _, y, _ in rs],
                                     ids=[rel for rel, _, _ in rs], groups=[g for _, _, g in rs])
                for split, rs in rows.items()}
