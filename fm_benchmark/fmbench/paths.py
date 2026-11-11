"""Locations of raw data, caches and results.

Resolution order for each root: environment variable, then ``fmbench.local.yaml`` next to
the package (git-ignored, for per-machine settings), then the default below.

    FMBENCH_DATA     raw downloads, one folder per dataset      (default: fm_benchmark/data)
    FMBENCH_CACHE    prepared tiles and cached embeddings       (default: fm_benchmark/cache)
    FMBENCH_RESULTS  run records, predictions and leaderboards  (default: fm_benchmark/results)
    FMBENCH_MIDOGPP  MIDOG++ images and JSON (reuses midogpp/)  (default: <data>/midogpp)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BENCH_DIR.parent
CONFIG_DIR = BENCH_DIR / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
REFERENCE_DIR = BENCH_DIR / "reference"
LOCAL_CONFIG = BENCH_DIR / "fmbench.local.yaml"


@lru_cache(maxsize=1)
def _local() -> dict:
    if LOCAL_CONFIG.exists():
        return yaml.safe_load(LOCAL_CONFIG.read_text()) or {}
    return {}


def _root(env: str, key: str, default: Path) -> Path:
    value = os.environ.get(env) or _local().get(key)
    return Path(value).expanduser().resolve() if value else default


def data_root() -> Path:
    return _root("FMBENCH_DATA", "data_root", BENCH_DIR / "data")


def cache_root() -> Path:
    return _root("FMBENCH_CACHE", "cache_root", BENCH_DIR / "cache")


def results_root() -> Path:
    return _root("FMBENCH_RESULTS", "results_root", BENCH_DIR / "results")


def midogpp_root() -> Path:
    return _root("FMBENCH_MIDOGPP", "midogpp_root", data_root() / "midogpp")


def embeddings_dir(model: str, task: str) -> Path:
    return cache_root() / "embeddings" / model / task


def prepared_dir(task: str) -> Path:
    return cache_root() / "prepared" / task
