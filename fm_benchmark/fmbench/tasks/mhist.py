"""MHIST: hyperplastic polyp (HP) vs sessile serrated adenoma (SSA), 224 px tiles at 5x.

Virchow2 sec. 5.3: 3,152 colorectal polyp tiles; official train 2,175 / test 977. There is no
validation split, so 10% of train is held out at random (Virchow: 217 tiles), leaving 1,958.

Access is by form only (https://bmirds.github.io/MHIST/): the links arrive by email and expire, and
the data use agreement forbids redistribution. Put the files in <data>/mhist/:
    annotations.csv        (Image Name, Majority Vote Label, ..., Partition)
    images.zip or images/  (MHIST_*.png)
"""
from __future__ import annotations

import csv
from pathlib import Path

from .base import FileListSplit, TileTask

LABELS = {"HP": 0, "SSA": 1}
EXPECTED = {"train": 2175, "test": 977}


class MhistTask(TileTask):
    name = "mhist"
    display_name = "MHIST"
    paper_column = "MHIST"
    classes = ("HP", "SSA")
    access = "form_request"
    magnification = "5x (224 px)"
    native_size = 224
    data_sources = ("mhist",)
    official_val = False  # random 10% of train, seed 0
    description = "MHIST colorectal polyps (form access); official train/test, 10% of train held out."

    def instructions(self) -> str:
        return (f"  MHIST needs a request form: fill it in at https://bmirds.github.io/MHIST/ (each user must\n"
                f"  register; links expire after 4 h). Then put annotations.csv and images.zip (or the\n"
                f"  unzipped MHIST_*.png files) in {self.raw_dir} and run: python -m fmbench prepare mhist")

    def prepare(self, force: bool = False) -> None:
        from ..download import extract

        z = self.raw_dir / "images.zip"
        if z.exists() and not any(self.raw_dir.rglob("MHIST_*.png")):
            extract(z, self.raw_dir)

    def official_splits(self):
        ann = self.raw_dir / "annotations.csv"
        if not ann.exists():
            raise FileNotFoundError(f"{ann} (form access; see: python -m fmbench download mhist)")
        images = {p.name: p for p in self.raw_dir.rglob("MHIST_*.png")}
        rows = {"train": [], "test": []}
        with ann.open(newline="") as f:
            for r in csv.DictReader(f):
                name = r["Image Name"].strip()
                if name not in images:
                    raise FileNotFoundError(f"{name} listed in annotations.csv but not found under {self.raw_dir}")
                rows[r["Partition"].strip().lower()].append((images[name], LABELS[r["Majority Vote Label"].strip()], name))
        return {split: FileListSplit([p for p, _, _ in rs], [y for _, y, _ in rs], ids=[n for _, _, n in rs])
                for split, rs in rows.items()}
