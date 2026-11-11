"""Common interface for the tile-classification tasks of the benchmark.

A task gives three splits (train, val, test). Each split is a ``SplitSource``: integer labels,
stable sample ids and random access to RGB tiles. Tasks without an official validation split
hold out a random 10% of the training split with a fixed seed, as in Virchow2 (sec. 5.4).
"""
from __future__ import annotations

import abc
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, Sequence

import numpy as np
from PIL import Image

from .. import paths

SPLITS = ("train", "val", "test")
HOLDOUT_SEED = 0
HOLDOUT_FRACTION = 0.10


# --------------------------------------------------------------------------- splits
class SplitSource(abc.ABC):
    """Labels, ids and lazy image access for one split."""

    labels: np.ndarray
    ids: np.ndarray
    groups: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.labels))

    @abc.abstractmethod
    def image(self, i: int) -> Image.Image:
        """Return tile ``i`` as an RGB PIL image at its native size."""

    def subset(self, indices: Sequence[int]) -> "SplitSource":
        return SubsetSplit(self, np.asarray(indices, dtype=np.int64))

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for sid, lab in zip(self.ids.tolist(), self.labels.tolist()):
            h.update(f"{sid}\t{lab}\n".encode())
        return h.hexdigest()[:16]


class SubsetSplit(SplitSource):
    def __init__(self, parent: SplitSource, indices: np.ndarray):
        self.parent = parent
        self.indices = indices
        self.labels = parent.labels[indices]
        self.ids = parent.ids[indices]
        self.groups = None if parent.groups is None else parent.groups[indices]

    def image(self, i: int) -> Image.Image:
        return self.parent.image(int(self.indices[i]))


class FileListSplit(SplitSource):
    """Tiles stored as one image file each."""

    def __init__(self, files: Sequence[Path], labels: Sequence[int], ids: Optional[Sequence[str]] = None,
                 groups: Optional[Sequence[str]] = None, root: Optional[Path] = None):
        self.files = [Path(f) for f in files]
        self.labels = np.asarray(labels, dtype=np.int64)
        root = Path(root) if root else None
        if ids is None:
            ids = [str(f.relative_to(root)) if root else f.name for f in self.files]
        self.ids = np.asarray(list(ids), dtype=object)
        self.groups = None if groups is None else np.asarray(list(groups), dtype=object)
        if not (len(self.files) == len(self.labels) == len(self.ids)):
            raise ValueError("files, labels and ids must have the same length")

    def image(self, i: int) -> Image.Image:
        with Image.open(self.files[i]) as im:
            return im.convert("RGB")


class H5ArraySplit(SplitSource):
    """Tiles stored as an (N, H, W, 3) uint8 array inside an HDF5 file.

    The file handle is opened lazily per process, so the split works inside DataLoader workers.
    """

    def __init__(self, path: Path, key: str, labels: np.ndarray, ids: Optional[Sequence[str]] = None):
        self.path, self.key = Path(path), key
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.ids = np.asarray(ids if ids is not None else [f"{self.path.stem}:{i}" for i in range(len(self.labels))], dtype=object)
        self._h5 = None
        self._pid = None

    def _array(self):
        if self._h5 is None or self._pid != os.getpid():
            import h5py

            self._h5 = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._h5[self.key]

    def image(self, i: int) -> Image.Image:
        return Image.fromarray(np.asarray(self._array()[i], dtype=np.uint8), mode="RGB")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_pid"] = None
        return state


# --------------------------------------------------------------------------- helpers
def holdout_indices(n: int, fraction: float = HOLDOUT_FRACTION, seed: int = HOLDOUT_SEED,
                    groups: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Random hold-out of ``fraction`` of the samples (or of the groups, if given).

    Returns sorted (keep, holdout) index arrays. With groups (for example patients), whole
    groups move to the hold-out until it holds at least ``fraction`` of the samples.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        k = int(n * fraction)  # floor: MHIST 2,175 -> 217 held out, as in Virchow
        return np.sort(perm[k:]), np.sort(perm[:k])
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    order = rng.permutation(len(uniq))
    target, chosen, count = n * fraction, set(), 0
    sizes = {g: int((groups == g).sum()) for g in uniq.tolist()}
    for j in order:
        if count >= target:
            break
        g = uniq[j]
        chosen.add(g)
        count += sizes[g]
    mask = np.array([g in chosen for g in groups.tolist()])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def limit_indices(labels: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    """Deterministic class-balanced-ish subsample of at most ``limit`` indices (smoke tests)."""
    n = len(labels)
    if limit <= 0 or limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = max(1, limit // len(classes))
    picked = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        picked.extend(rng.permutation(idx)[:per].tolist())
    if len(picked) < limit:
        rest = np.setdiff1d(np.arange(n), picked)
        picked.extend(rng.permutation(rest)[: limit - len(picked)].tolist())
    return np.sort(np.asarray(picked[:limit], dtype=np.int64))


@dataclass
class DownloadSpec:
    url: str
    filename: str
    size: str = ""
    md5: str = ""
    sha256: str = ""
    extract: bool = False
    extract_to: str = ""                     # sub-directory of raw_dir to extract into ("" = raw_dir)
    extract_members: tuple[str, ...] = ()    # only extract archive members starting with one of these
    note: str = ""


@dataclass
class TaskStatus:
    available: bool
    message: str
    counts: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- task
class TileTask(abc.ABC):
    """One tile-level benchmark task."""

    name: ClassVar[str]                      # registry key, e.g. "pcam"
    display_name: ClassVar[str]              # e.g. "PCam"
    paper_column: ClassVar[Optional[str]] = None  # column name in Virchow2 Table 2, if any
    classes: ClassVar[tuple[str, ...]]
    access: ClassVar[str] = "open"           # open | gated_hf | form_request | registration | local
    magnification: ClassVar[str] = ""
    native_size: ClassVar[int] = 224
    data_sources: ClassVar[tuple[str, ...]] = ()  # dataset families, used for leakage flags
    official_val: ClassVar[bool] = True      # False -> random 10% hold-out of train
    group_holdout: ClassVar[bool] = False    # hold out whole groups (patients) instead of tiles
    cluster_unit: ClassVar[Optional[str]] = None  # e.g. "patient": also report a CI that resamples these groups
    description: ClassVar[str] = ""

    def __init__(self, data_root: Optional[Path] = None, cache_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else paths.data_root()
        self.cache_root = Path(cache_root) if cache_root else paths.cache_root()

    # -- to implement ------------------------------------------------------
    @abc.abstractmethod
    def official_splits(self) -> dict[str, SplitSource]:
        """Splits as published. Must include "train" and "test"; "val" when the dataset has one."""

    def download_specs(self) -> list[DownloadSpec]:
        return []

    def instructions(self) -> str:
        """How to get the data when it cannot be downloaded automatically."""
        return ""

    def prepare(self, force: bool = False) -> None:
        """Turn raw downloads into the files that ``official_splits`` reads (unzip, tile, index)."""

    def unfinished_extractions(self) -> list[str]:
        """Archives that are still downloading or extracting, or whose extraction failed.

        ``download.extract`` writes ``.extracted-<archive>`` only after the last member is written, so an
        archive (or its ``.part``) in ``download_dir`` without that marker means the extracted tree may be
        partial: a loader that globs files would list only what is there so far. Data placed by hand,
        with no archive next to it, counts as complete.
        """
        out = []
        for spec in self.download_specs():
            if not spec.extract:
                continue
            dest = self.raw_dir / spec.extract_to if spec.extract_to else self.raw_dir
            if (dest / f".extracted-{spec.filename}").exists():
                continue
            archive = self.download_dir / spec.filename
            if archive.exists() or archive.with_name(archive.name + ".part").exists():
                out.append(spec.filename)
        return out

    def status(self) -> TaskStatus:
        try:
            unfinished = self.unfinished_extractions()
            if unfinished:
                return TaskStatus(False, f"download or extraction of {', '.join(unfinished)} not finished "
                                         f"(run: python -m fmbench download {self.name})")
            sp = self.splits()
        except FileNotFoundError as e:
            return TaskStatus(False, f"missing: {e}")
        except Exception as e:  # noqa: BLE001 - report any loader failure as "not ready"
            return TaskStatus(False, f"error: {type(e).__name__}: {e}")
        return TaskStatus(True, "ready", {k: len(v) for k, v in sp.items()})

    # -- shared ------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_root / self.name

    @property
    def download_dir(self) -> Path:
        """Where archives are saved; they can be deleted after ``prepare``."""
        return self.raw_dir / "_downloads"

    @property
    def prepared_dir(self) -> Path:
        return self.cache_root / "prepared" / self.name

    def splits(self) -> dict[str, SplitSource]:
        sp = dict(self.official_splits())
        missing = {"train", "test"} - set(sp)
        if missing:
            raise ValueError(f"{self.name}: official_splits() lacks {sorted(missing)}")
        if "val" not in sp:
            train = sp["train"]
            groups = train.groups if self.group_holdout else None
            keep, hold = holdout_indices(len(train), groups=groups)
            sp["train"], sp["val"] = train.subset(keep), train.subset(hold)
        return {k: sp[k] for k in SPLITS}

    @property
    def num_classes(self) -> int:
        return len(self.classes)
