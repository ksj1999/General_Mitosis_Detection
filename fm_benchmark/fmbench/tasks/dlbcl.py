"""DLBCL: 5-year overall survival from DLBCL-Morph TMA tiles (H&E and 5 IHC stains, 40x).

Virchow2 sec. 6.5 gives only: binary 5-year survival, 42 TMAs, 209 patients, H&E + CD10, BCL6,
MUM1, BCL2 and MYC at 40x, patient-level train/val/test. Everything else is our choice
(documented in NOTES.md):
  * tiles: the dataset's own 224 x 224 Patches/<stain>/<patient>/*.png at 40x (152,194 tiles);
  * label: died within 5 years (OS < 5 and Follow-up Status == 1) = 1; OS >= 5 = 0; patients censored
    before 5 years (OS < 5, status 0; 12 patients) are dropped. OS is in years from end of treatment;
  * split: by patient, stratified by label, 60/20/20, seed 0; all stains of a patient in one split.
``dlbcl_he`` keeps only the H&E tiles (170 patients), a fairer test for H&E-only models.

Data: figshare record 12964772 (DLBCL-Morph.zip, 31.8 GB, CC0). Only Patches/ and the CSVs are
extracted; the TMA slides and ~1M cell masks are skipped.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .base import DownloadSpec, FileListSplit, TileTask

URL = "https://ndownloader.figshare.com/files/24697142"
MD5 = "96cf88ad80f26cca82f7e346bad49936"
STAINS = ("HE", "CD10", "BCL6", "MUM1", "BCL2", "MYC")
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 0


def patient_labels(clinical_csv: Path) -> dict[str, int]:
    """patient_id -> 1 (died within 5 years) / 0 (alive at 5 years); censored-before-5y patients omitted."""
    out = {}
    with clinical_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            os_years = float(r["OS"])
            status = int(float(r["Follow-up Status"]))
            if os_years >= 5:
                out[r["patient_id"].strip()] = 0
            elif status == 1:
                out[r["patient_id"].strip()] = 1
    return out


def patient_split(labels: dict[str, int], seed: int = SEED) -> dict[str, str]:
    """Stratified patient-level split; deterministic for a given label table and seed."""
    rng = np.random.default_rng(seed)
    split_of = {}
    for y in sorted(set(labels.values())):
        ids = rng.permutation(sorted(p for p, v in labels.items() if v == y)).tolist()
        n = len(ids)
        n_test = int(round(n * SPLIT_FRACTIONS["test"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        for i, p in enumerate(ids):
            split_of[p] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return split_of


class DlbclTask(TileTask):
    name = "dlbcl"
    display_name = "DLBCL"
    paper_column = "DLBCL"
    classes = ("alive_5y", "died_within_5y")
    access = "open"
    magnification = "40x (224 px)"
    native_size = 224
    data_sources = ("dlbcl_morph",)
    official_val = True  # our own patient-level split includes val
    cluster_unit = "patient"  # 40 test patients, 37 with tiles (26-1,453 each; 31 with H&E tiles): tile-level CIs are far too narrow
    stains: tuple[str, ...] = STAINS
    description = "DLBCL-Morph 5-year survival, all six stains, patient-level 60/20/20 split (ours)."

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "dlbcl"  # shared by dlbcl and dlbcl_he

    def download_specs(self):
        return [DownloadSpec(url=URL, filename="DLBCL-Morph.zip", size="31.8 GB", md5=MD5, extract=True,
                             extract_members=("DLBCL-Morph/Patches/", "DLBCL-Morph/clinical_data_cleaned.csv",
                                              "DLBCL-Morph/TMA/core.csv", "DLBCL-Morph/TMA/annotations.csv"))]

    def instructions(self) -> str:
        return ("  DLBCL downloads automatically (python -m fmbench download dlbcl; 31.8 GB, only Patches/ and\n"
                f"  the CSVs are extracted, about 15 GB). The zip in {self.download_dir} can be deleted afterwards.")

    def _root(self) -> Path:
        for cand in (self.raw_dir / "DLBCL-Morph", self.raw_dir):
            if (cand / "clinical_data_cleaned.csv").exists() and (cand / "Patches").is_dir():
                return cand
        raise FileNotFoundError(f"{self.raw_dir}/DLBCL-Morph/{{clinical_data_cleaned.csv,Patches/}} "
                                "(run: python -m fmbench download dlbcl)")

    def official_splits(self):
        root = self._root()
        labels = patient_labels(root / "clinical_data_cleaned.csv")
        split_of = patient_split(labels)
        rows = {"train": [], "val": [], "test": []}
        for stain in self.stains:
            sdir = root / "Patches" / stain
            if not sdir.is_dir():
                continue
            for pdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
                pid = pdir.name
                if pid not in labels:
                    continue
                for f in sorted(pdir.glob("*.png")):
                    rows[split_of[pid]].append((f, labels[pid], pid))
        self._write_split_table(labels, split_of)
        return {split: FileListSplit([f for f, _, _ in rs], [y for _, y, _ in rs], groups=[g for _, _, g in rs], root=root)
                for split, rs in rows.items()}

    def _write_split_table(self, labels, split_of) -> None:
        try:
            self.prepared_dir.mkdir(parents=True, exist_ok=True)
            with (self.prepared_dir / "patients.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patient_id", "label", "split"])
                for p in sorted(labels):
                    w.writerow([p, labels[p], split_of[p]])
        except OSError:
            pass


class DlbclHETask(DlbclTask):
    name = "dlbcl_he"
    display_name = "DLBCL (H&E)"
    paper_column = None
    stains = ("HE",)
    description = "DLBCL-Morph 5-year survival, H&E tiles only, same patient split as dlbcl."
