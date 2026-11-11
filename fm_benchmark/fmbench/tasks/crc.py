"""CRC and CRC No-Norm: 9-class colorectal tissue tiles (Kather et al., Zenodo 1214456), 224 px at 20x.

Virchow2 sec. 5.3 / 6.5 and Virchow (Nat. Med. 2024): train on NCT-CRC-HE-100K with a random 10% held
out for validation, test on CRC-VAL-HE-7K (7,180 tiles; both papers list 107,180 tiles per task).

CRC No-Norm is described as "the test set is not Macenko-normalized", but Zenodo has no unnormalized
7K set and Kather et al. say CRC-VAL-HE-7K is normalized. The only reading consistent with the file
list and the tile count is: train/val on NCT-CRC-HE-100K-NONORM, test on CRC-VAL-HE-7K (the same setup
as UNI's "CRC-100K-Raw"). That is what ``crc_nonorm`` does; see NOTES.md.
"""
from __future__ import annotations

from pathlib import Path

from .base import DownloadSpec, FileListSplit, TileTask

CLASSES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
ZENODO = "https://zenodo.org/records/1214456/files/{}?download=1"
ARCHIVES = {
    "NCT-CRC-HE-100K": ("6fd702d11df6292bc054397ae038a464", "11.7 GB"),
    "NCT-CRC-HE-100K-NONORM": ("035777cf327776a71a05c95da6d6325f", "11.7 GB"),
    "CRC-VAL-HE-7K": ("2fd1651b4f94ebd818ebf90ad2b6ce06", "0.8 GB"),
}
TEST_SET = "CRC-VAL-HE-7K"
EXPECTED = {"pool": 100000, "test": 7180}


def _spec(folder: str) -> DownloadSpec:
    md5, size = ARCHIVES[folder]
    return DownloadSpec(url=ZENODO.format(folder + ".zip"), filename=folder + ".zip", md5=md5, size=size, extract=True)


class CRCTask(TileTask):
    name = "crc"
    display_name = "CRC"
    paper_column = "CRC"
    classes = CLASSES
    access = "open"
    magnification = "20x (224 px, 0.5 um/px)"
    native_size = 224
    data_sources = ("nct_crc",)
    official_val = False  # random 10% of the training pool, seed 0
    train_folder = "NCT-CRC-HE-100K"
    description = "NCT-CRC-HE-100K (Macenko) -> CRC-VAL-HE-7K, 9 tissue classes."

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "crc"  # shared by crc and crc_nonorm (the 7K test set is downloaded once)

    def download_specs(self):
        return [_spec(self.train_folder), _spec(TEST_SET)]

    def instructions(self) -> str:
        return (f"  CRC downloads automatically from Zenodo 1214456. Existing extractions can be placed at\n"
                f"  {self.raw_dir}/{self.train_folder}/<CLASS>/*.tif and {self.raw_dir}/{TEST_SET}/<CLASS>/*.tif.")

    def _folder_split(self, folder: str) -> FileListSplit:
        root = self.raw_dir / folder
        if not root.is_dir():
            raise FileNotFoundError(f"{root} (run: python -m fmbench download {self.name})")
        files, labels = [], []
        for y, c in enumerate(CLASSES):
            for f in sorted((root / c).glob("*.tif")):
                files.append(f)
                labels.append(y)
        if not files:
            raise FileNotFoundError(f"no .tif tiles under {root}/<CLASS>/")
        return FileListSplit(files, labels, root=self.raw_dir)

    def official_splits(self):
        return {"train": self._folder_split(self.train_folder), "test": self._folder_split(TEST_SET)}


class CRCNoNormTask(CRCTask):
    name = "crc_nonorm"
    display_name = "CRC No-Norm"
    paper_column = "CRC No-Norm"
    train_folder = "NCT-CRC-HE-100K-NONORM"
    description = "NCT-CRC-HE-100K-NONORM (unnormalized) -> CRC-VAL-HE-7K (Macenko), a stain shift."
