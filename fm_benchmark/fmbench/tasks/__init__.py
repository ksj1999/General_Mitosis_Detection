"""Task registry. Modules are imported lazily so a missing optional dependency (h5py,
tifffile, ...) only affects the task that needs it."""
from __future__ import annotations

import importlib

from .base import TileTask

# name -> "module:Class"
_REGISTRY = {
    "pcam": "pcam:PCamTask",
    "crc": "crc:CRCTask",
    "crc_nonorm": "crc:CRCNoNormTask",
    "wilds": "wilds:WildsTask",
    "tils": "tils:TilsTask",
    "mhist": "mhist:MhistTask",
    "dlbcl": "dlbcl:DlbclTask",
    "dlbcl_he": "dlbcl:DlbclHETask",
    "midog": "midog:MidogTask",
    "midog_nobreast": "midog:MidogNoBreastTask",
}

# The 8 public out-of-distribution tasks of Virchow2 Table 2, in the paper's column order.
PAPER8 = ["pcam", "crc", "crc_nonorm", "wilds", "tils", "mhist", "dlbcl", "midog"]
TASK_SETS = {
    "paper8": PAPER8,
    # everything that downloads without a form (MHIST needs one)
    "open": ["pcam", "crc", "crc_nonorm", "wilds", "tils", "dlbcl", "midog"],
    # tasks outside Virchow2 Table 2, for models trained on MIDOG++ breast or on H&E only
    "extra": ["midog_nobreast", "dlbcl_he"],
    "all": list(_REGISTRY),
}


def _load(name: str) -> type[TileTask]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {', '.join(_REGISTRY)}")
    mod, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), cls)


def get_task(name: str, **kwargs) -> TileTask:
    return _load(name)(**kwargs)


def list_tasks() -> dict[str, type[TileTask]]:
    out = {}
    for name in _REGISTRY:
        try:
            out[name] = _load(name)
        except ImportError as e:  # report, do not crash the listing
            print(f"[tasks] {name}: cannot import ({e})")
    return out
