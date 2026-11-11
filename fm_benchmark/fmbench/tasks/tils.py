"""TCGA-TILs: does a 100 px TCGA H&E patch hold at least two tumour-infiltrating lymphocytes.

Virchow2 sec. 6.5 / Virchow (Nat. Med. 2024): 304,097 images of 100 x 100 px at 20x, upsampled
to 224 x 224, official patient-level partitions of Zenodo record 6604094
(train 209,221 / val 38,601 / test 56,275). Virchow's Methods list 247,822 training images,
which equals train + val; we use the official partitions (see NOTES.md).

Layout after extraction: TCGA-TILs/images-tcga-tils/<study>/{train,val,test}/{til-negative,
til-positive}/*.png. ``pancancer/`` only holds symlinks into the study folders and is skipped.
"""
from __future__ import annotations

from pathlib import Path

from .base import DownloadSpec, FileListSplit, TileTask

URL = "https://zenodo.org/records/6604094/files/TCGA-TILs.tar.gz?download=1"
MD5 = "42d387c2f8d300dea8bb8a544b43ad3c"
CLASSES = ("til-negative", "til-positive")
EXPECTED = {"train": 209221, "val": 38601, "test": 56275}


class TilsTask(TileTask):
    name = "tils"
    display_name = "TILS"
    paper_column = "TILS"
    classes = CLASSES
    access = "open"
    magnification = "20x (100 px, upsampled to 224)"
    native_size = 100
    data_sources = ("tcga",)
    official_val = True
    description = "TCGA tumour-infiltrating lymphocytes, Zenodo 6604094, official patient-level splits."

    def download_specs(self):
        return [DownloadSpec(url=URL, filename="TCGA-TILs.tar.gz", size="6.9 GB", md5=MD5, extract=True)]

    def instructions(self) -> str:
        return ("  TILS downloads automatically (python -m fmbench download tils); an existing extraction can be\n"
                f"  placed at {self.raw_dir}/TCGA-TILs/images-tcga-tils.")

    def _base(self) -> Path:
        for cand in (self.raw_dir / "TCGA-TILs" / "images-tcga-tils", self.raw_dir / "images-tcga-tils"):
            if cand.is_dir():
                return cand
        raise FileNotFoundError(f"{self.raw_dir}/TCGA-TILs/images-tcga-tils (run: python -m fmbench download tils)")

    def official_splits(self):
        base = self._base()
        out = {}
        for split in ("train", "val", "test"):
            files, labels, groups = [], [], []
            for study in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "pancancer"):
                for y, c in enumerate(CLASSES):
                    d = study / split / c
                    if not d.is_dir():
                        continue
                    for f in sorted(d.glob("*.png")):
                        if f.is_symlink():
                            continue
                        files.append(f)
                        labels.append(y)
                        groups.append(study.name)
            if not files:
                raise FileNotFoundError(f"no {split} tiles under {base}")
            out[split] = FileListSplit(files, labels, groups=groups, root=base)
        return out
