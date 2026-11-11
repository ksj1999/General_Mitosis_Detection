"""Each task loader against a tiny copy of its dataset in the official on-disk format (no downloads)."""
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import zipfile

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results"),
                     ("FMBENCH_MIDOGPP", "data/midogpp")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


def _img(path, size, seed=0, mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, len(mode)), dtype=np.uint8)
    if mode == "RGBA":
        arr[..., 3] = 255
    Image.fromarray(arr, mode).save(path)
    return path


def _counts(task):
    return {k: len(v) for k, v in task.splits().items()}


# --------------------------------------------------------------------------- PCam
def test_pcam(env):
    import h5py

    from fmbench.tasks import get_task

    t = get_task("pcam")
    t.raw_dir.mkdir(parents=True)
    for tag, n in (("train", 20), ("valid", 6), ("test", 8)):
        with h5py.File(t.raw_dir / f"camelyonpatch_level_2_split_{tag}_x.h5", "w") as f:
            f["x"] = np.random.default_rng(n).integers(0, 255, (n, 96, 96, 3), dtype=np.uint8)
        y = (np.arange(n) % 2).reshape(n, 1, 1, 1).astype(np.uint8)
        if tag == "test":  # also exercise the gunzip path of prepare()
            tmp = t.raw_dir / "y.h5"
            with h5py.File(tmp, "w") as f:
                f["y"] = y
            with tmp.open("rb") as src, gzip.open(t.raw_dir / f"camelyonpatch_level_2_split_{tag}_y.h5.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.unlink()
        else:
            with h5py.File(t.raw_dir / f"camelyonpatch_level_2_split_{tag}_y.h5", "w") as f:
                f["y"] = y
    t.prepare()
    assert _counts(t) == {"train": 20, "val": 6, "test": 8}
    sp = t.splits()
    im = sp["val"].image(1)
    assert im.size == (96, 96) and im.mode == "RGB"
    assert sp["test"].labels.tolist() == [0, 1] * 4
    assert sp["train"].ids[0] == "train:0"


# --------------------------------------------------------------------------- WILDS
def test_wilds_splits_by_hospital(env):
    from fmbench.tasks import get_task

    t = get_task("wilds")
    root = t.raw_dir / "camelyon17_v1.0"
    rows = []
    for center in range(5):
        for k in range(2 + center):
            patient, node, x, y = f"{center * 10 + 4:03d}", center % 5, 100 * k, 200
            rows.append(["", patient, node, x, y, k % 2, center * 2, center, 0])
            _img(root / f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png",
                 96, seed=k, mode="RGBA")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "patient", "node", "x_coord", "y_coord", "tumor", "slide", "center", "split"])
        for i, r in enumerate(rows):
            w.writerow([i, *r[1:]])
    # hospitals 0, 3, 4 train (2 + 5 + 6), hospital 1 val (3), hospital 2 test (4)
    assert _counts(t) == {"train": 13, "val": 3, "test": 4}
    assert t.splits()["val"].image(0).mode == "RGB"  # RGBA -> RGB


# --------------------------------------------------------------------------- TILS
def test_tils_skips_pancancer_symlinks(env):
    from fmbench.tasks import get_task

    t = get_task("tils")
    base = t.raw_dir / "TCGA-TILs" / "images-tcga-tils"
    n = {"train": 4, "val": 2, "test": 3}
    for study in ("acc", "brca"):
        for split, k in n.items():
            for i in range(k):
                cls = "til-positive" if i % 2 else "til-negative"
                f = _img(base / study / split / cls / f"{study}_{split}_{i}.png", 100, seed=i)
                link = base / "pancancer" / split / cls / f.name
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.path.relpath(f, link.parent), link)
    assert _counts(t) == {k: 2 * v for k, v in n.items()}
    sp = t.splits()
    assert set(sp["train"].groups.tolist()) == {"acc", "brca"}
    assert sorted(set(sp["test"].labels.tolist())) == [0, 1]


# --------------------------------------------------------------------------- MHIST
def test_mhist_holds_out_ten_percent(env):
    from fmbench.tasks import get_task

    t = get_task("mhist")
    with pytest.raises(FileNotFoundError):
        t.official_splits()
    names = [f"MHIST_{chr(97 + i // 26)}{chr(97 + i % 26)}a.png" for i in range(25)]
    t.raw_dir.mkdir(parents=True)
    with zipfile.ZipFile(t.raw_dir / "images.zip", "w") as z:
        for i, nm in enumerate(names):
            p = _img(env / "tmpimg" / nm, 224, seed=i)
            z.write(p, f"images/{nm}")
    with (t.raw_dir / "annotations.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Name", "Majority Vote Label", "Number of Annotators who Selected SSA (Out of 7)", "Partition"])
        for i, nm in enumerate(names):
            w.writerow([nm, "SSA" if i % 3 == 0 else "HP", 5, "train" if i < 20 else "test"])
    t.prepare()
    assert _counts(t) == {"train": 18, "val": 2, "test": 5}  # floor(10% of 20)
    assert t.splits()["test"].image(0).size == (224, 224)


# --------------------------------------------------------------------------- DLBCL
def test_dlbcl_patient_split(env):
    from fmbench.tasks import get_task
    from fmbench.tasks.dlbcl import patient_labels

    t, t_he = get_task("dlbcl"), get_task("dlbcl_he")
    assert t.raw_dir == t_he.raw_dir
    root = t.raw_dir / "DLBCL-Morph"
    root.mkdir(parents=True)
    patients = []
    for i in range(20):
        if i < 12:
            os_y, status = 6.0 + i, i % 2           # alive at 5 years -> 0
        elif i < 18:
            os_y, status = 1.0 + 0.3 * (i - 12), 1  # died within 5 years -> 1
        else:
            os_y, status = 2.0, 0                   # censored before 5 years -> dropped
        patients.append((str(13900 + i), os_y, status))
    with (root / "clinical_data_cleaned.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "Age", "OS", "PFS", "Follow-up Status"])
        for pid, os_y, st in patients:
            w.writerow([pid, 60, os_y, os_y, st])
    for pid, _, _ in patients:
        for stain in ("HE", "CD10"):
            for k in range(3):
                _img(root / "Patches" / stain / pid / f"{k}.png", 224, seed=k)
    labels = patient_labels(root / "clinical_data_cleaned.csv")
    assert len(labels) == 18 and sum(labels.values()) == 6
    sp = t.splits()
    assert sum(len(v) for v in sp.values()) == 18 * 2 * 3
    groups = {k: set(v.groups.tolist()) for k, v in sp.items()}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) and not (groups["val"] & groups["test"])
    assert all(1 in set(v.labels.tolist()) for v in sp.values())  # stratified: every split has deaths
    sp_he = t_he.splits()
    assert sum(len(v) for v in sp_he.values()) == 18 * 3
    assert {k: set(v.groups.tolist()) for k, v in sp_he.items()} == groups  # same patient split
    assert (t.prepared_dir / "patients.csv").exists()


# --------------------------------------------------------------------------- CRC
def test_crc_and_nonorm(env):
    from fmbench.tasks import get_task
    from fmbench.tasks.crc import CLASSES

    crc, nonorm = get_task("crc"), get_task("crc_nonorm")
    assert crc.raw_dir == nonorm.raw_dir
    for folder, per in (("NCT-CRC-HE-100K", 4), ("NCT-CRC-HE-100K-NONORM", 3), ("CRC-VAL-HE-7K", 2)):
        for c in CLASSES:
            for k in range(per):
                _img(crc.raw_dir / folder / c / f"{c}-{folder[-6:]}{k}.tif", 224, seed=k)
    assert _counts(crc) == {"train": 33, "val": 3, "test": 18}
    assert _counts(nonorm) == {"train": 25, "val": 2, "test": 18}
    sp = crc.splits()
    assert sp["test"].fingerprint() == nonorm.splits()["test"].fingerprint()
    assert sp["train"].image(0).mode == "RGB"
    assert crc.download_specs()[1].filename == nonorm.download_specs()[1].filename == "CRC-VAL-HE-7K.zip"


# --------------------------------------------------------------------------- MIDOG
def _coco(n_images=5):
    images, anns, aid = [], [], 1
    for i in range(1, n_images + 1):
        images.append({"id": i, "file_name": f"{i:03d}.tiff", "width": 1200, "height": 900,
                       "tumor_type": "human breast cancer" if i <= 2 else "canine lymphoma"})
        # positive near the centre, a negative overlapping it, a far negative, a positive at the border
        for (cx, cy, cat) in ((400, 400, 1), (500, 450, 2), (1000, 700, 2), (60, 450, 1)):
            anns.append({"id": aid, "image_id": i, "category_id": cat, "bbox": [cx - 25, cy - 25, cx + 25, cy + 25]})
            aid += 1
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "mitotic figure"},
                                                                   {"id": 2, "name": "non-mitotic figure"}]}


def test_midog_rules():
    from fmbench.tasks.midog import TILE, build_index

    coco = _coco()
    recs = build_index(coco)
    # per image: the centre positive and the far negative survive; the border positive and the
    # negative overlapping the positive are dropped
    assert len(recs) == 2 * 5
    for r in recs:
        a = next(a for a in coco["annotations"] if a["id"] == r["ann_id"])
        x0, y0, x1, y1 = a["bbox"]
        assert r["x"] <= x0 and x1 <= r["x"] + TILE and r["y"] <= y0 and y1 <= r["y"] + TILE
        assert 0 <= r["x"] <= 1200 - TILE and 0 <= r["y"] <= 900 - TILE
    split_imgs = {s: {r["image_id"] for r in recs if r["split"] == s} for s in ("train", "val", "test")}
    assert not (split_imgs["train"] & split_imgs["test"])
    nobreast = build_index(coco, ("human breast cancer",))
    assert {r["image_id"] for r in nobreast} == {3, 4, 5}


def test_midog_prepare(env):
    import tifffile

    from fmbench.tasks import get_task

    t = get_task("midog")
    coco = _coco(n_images=10)
    t.raw_dir.mkdir(parents=True)
    (t.raw_dir / "MIDOG++.json").write_text(json.dumps(coco))
    for im in coco["images"]:
        arr = np.random.default_rng(im["id"]).integers(0, 255, (im["height"], im["width"], 3), dtype=np.uint8)
        (t.raw_dir / "images").mkdir(exist_ok=True)
        tifffile.imwrite(t.raw_dir / "images" / im["file_name"], arr)
    t.prepare()
    c = _counts(t)
    assert sum(c.values()) == 20 and all(c.values())
    sp = t.splits()
    assert sp["train"].image(0).size == (224, 224)
    assert json.loads((t.prepared_dir / "counts.json").read_text())


# --------------------------------------------------------------------------- downloads
def test_fetch_accepts_local_file_and_checks_hash(env):
    from fmbench.download import fetch
    from fmbench.tasks.base import DownloadSpec

    d = env / "dl"
    d.mkdir()
    (d / "a.bin").write_bytes(b"hello")
    md5 = hashlib.md5(b"hello").hexdigest()
    sha = hashlib.sha256(b"hello").hexdigest()
    assert fetch(DownloadSpec("http://invalid.example/a.bin", "a.bin", md5=md5, sha256=sha), d, log=lambda *a: None).exists()
    (d / "b.bin").write_bytes(b"other")
    with pytest.raises(RuntimeError):
        fetch(DownloadSpec("http://invalid.example/b.bin", "b.bin", md5=md5), d, log=lambda *a: None)


def test_extract_members(env):
    from fmbench.download import extract

    src = env / "src"
    for rel in ("top/keep/a.txt", "top/skip/b.txt", "top/c.csv"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    with zipfile.ZipFile(env / "x.zip", "w") as z:
        for rel in ("top/keep/a.txt", "top/skip/b.txt", "top/c.csv"):
            z.write(src / rel, rel)
    with tarfile.open(env / "x.tar.gz", "w:gz") as tf:
        tf.add(src / "top", arcname="./top")
    for arch in ("x.zip", "x.tar.gz"):
        out = env / f"out_{arch}"
        extract(env / arch, out, members=("top/keep/", "top/c.csv"), log=lambda *a: None)
        assert (out / "top/keep/a.txt").exists() and (out / "top/c.csv").exists()
        assert not (out / "top/skip/b.txt").exists()


def test_registry_lists_all_tasks():
    from fmbench.tasks import PAPER8, TASK_SETS, list_tasks

    tasks = list_tasks()
    assert set(PAPER8) <= set(tasks) and set(TASK_SETS["extra"]) <= set(tasks)
    assert [tasks[n].paper_column for n in PAPER8] == ["PCam", "CRC", "CRC No-Norm", "WILDS", "TILS", "MHIST", "DLBCL", "MIDOG"]
