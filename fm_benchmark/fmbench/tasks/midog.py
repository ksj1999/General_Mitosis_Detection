"""MIDOG: binary mitotic-figure vs imposter tile classification built from MIDOG++.

Virchow2 sec. 5.3: "MIDOG is a binary mitotic cell detection task that we created from the
public MItosis DOmain Generalization (MIDOG++) dataset, containing 21,806 mitotic and
non-mitotic events labeled on 503 7K x 5K pixel WSI regions from several tumor, species, and
scanner types at 40x magnification. This data was converted into a binary classification task
by expanding each 50 x 50 pixel annotation to 224 x 224 regions and then randomly shifting in the
horizontal and vertical regions such that the event is not centered in the tile. All negative
instances that overlapped with positive instances were removed from the dataset."

Our construction (the paper gives neither the shift range, the overlap rule nor the split). It reproduces
the paper's event count to within 8 (21,798 vs 21,806):
  * every annotation of MIDOG++.json (11,937 mitotic figures, 14,349 imposters on 503 images) whose box
    centre is at least 112 px from the image border, so a centred 224 px tile fits (11,193 mitotic);
  * a negative is dropped if its centred 224 px tile overlaps the centred tile of any mitotic figure in the
    same image, border figures included (|dx| < 224 and |dy| < 224 between centres; 10,605 imposters left);
  * the crop is then shifted by a uniform integer offset in [-86, 86] px on each axis (seeded by the
    annotation id), so the 50 px box stays inside the tile but is not centred, and clamped to the image;
  * split by image (no image in two splits), stratified by tumour type: 60% train, 20% val, 20% test of
    the images, fixed seed. The paper's split is about 60/20/20 but its unit is not stated (see NOTES.md).

``midog_nobreast`` drops the human breast cancer images (MIDOG++ ids 1-200), so models fine-tuned
on the MIDOG++ breast subset can be compared without training-image leakage.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from .. import paths
from .base import DownloadSpec, H5ArraySplit, TaskStatus, TileTask

TILE = 224
HALF = TILE // 2
BOX = 50
MAX_SHIFT = HALF - BOX // 2 - 1  # 86: keeps the 50 px box (half-pixel coordinates) inside the tile
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 0
BREAST = "human breast cancer"


def _find_json(root: Path) -> Path:
    for cand in (root / "MIDOG++.json", root / "databases" / "MIDOG++.json",
                 paths.REPO_DIR / "midogpp" / "databases" / "MIDOG++.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"MIDOG++.json not found under {root} or midogpp/databases/")


def _find_image(root: Path, file_name: str) -> Path:
    for cand in (root / "images" / file_name, root / file_name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{file_name} not found under {root}/images (set FMBENCH_MIDOGPP)")


def figshare_files() -> dict[str, str]:
    """file name -> figshare file id, read from the ``filename`` dict in midogpp/Setup.ipynb."""
    import ast

    nb = json.loads((paths.REPO_DIR / "midogpp" / "Setup.ipynb").read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.lstrip().startswith("filename = {"):
            mapping = ast.literal_eval(src[src.index("{"):src.index("}") + 1])
            return {name: fid for fid, name in mapping.items()}
    raise RuntimeError("no figshare file map in midogpp/Setup.ipynb")


def build_index(coco: dict, exclude_tumours: tuple[str, ...] = ()) -> list[dict]:
    """Return one record per kept event: image id, file, label, tile origin, tumour, split."""
    images = {im["id"]: im for im in coco["images"]}
    by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    rng_shift = {}
    for a in coco["annotations"]:
        r = np.random.default_rng([SEED, int(a["id"])])
        rng_shift[a["id"]] = r.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=2)

    records = []
    for img_id in sorted(by_img):
        im = images[img_id]  # excluded tumours are dropped only after the split, so the other images keep midog's split
        w, h = int(im["width"]), int(im["height"])

        def centre(a):
            x0, y0, x1, y1 = a["bbox"]
            return (x0 + x1) / 2, (y0 + y1) / 2

        def origin(a):
            cx, cy = centre(a)
            dx, dy = rng_shift[a["id"]]
            ox = int(np.floor(cx - HALF + dx))
            oy = int(np.floor(cy - HALF + dy))
            return min(max(ox, 0), w - TILE), min(max(oy, 0), h - TILE)

        anns = by_img[img_id]
        pos_centres = np.array([centre(a) for a in anns if a["category_id"] == 1]).reshape(-1, 2)
        for a in sorted(anns, key=lambda a: a["id"]):
            cx, cy = centre(a)
            if cx < HALF or cy < HALF or cx > w - HALF or cy > h - HALF:
                continue  # a centred tile would leave the image
            label = 1 if a["category_id"] == 1 else 0
            if label == 0 and len(pos_centres):
                if np.all(np.abs(pos_centres - np.array([cx, cy])) < TILE, axis=1).any():
                    continue
            ox, oy = origin(a)
            records.append({"ann_id": int(a["id"]), "image_id": int(img_id), "file_name": im["file_name"],
                            "tumor_type": im.get("tumor_type", ""), "label": label, "x": ox, "y": oy})

    # image-level split, stratified by tumour type
    by_tumour = collections.defaultdict(list)
    for img_id in sorted({r["image_id"] for r in records}):
        by_tumour[images[img_id].get("tumor_type", "")].append(img_id)
    split_of = {}
    rng = np.random.default_rng(SEED)
    for tumour in sorted(by_tumour):
        ids = rng.permutation(by_tumour[tumour]).tolist()
        n = len(ids)
        n_test = int(round(n * SPLIT_FRACTIONS["test"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        for i, img_id in enumerate(ids):
            split_of[img_id] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    for r in records:
        r["split"] = split_of[r["image_id"]]
    return [r for r in records if r["tumor_type"] not in exclude_tumours]


class MidogTask(TileTask):
    name = "midog"
    display_name = "MIDOG"
    paper_column = "MIDOG"
    classes = ("imposter", "mitotic_figure")
    access = "open"
    magnification = "40x (~0.25 um/px)"
    native_size = TILE
    data_sources = ("midogpp",)
    official_val = True
    exclude_tumours: tuple[str, ...] = ()
    description = "MIDOG++ events as 224 px tiles, shifted off-centre; image-level 70/10/20 split."

    @property
    def raw_dir(self) -> Path:
        return paths.midogpp_root()

    @property
    def download_dir(self) -> Path:
        return self.raw_dir / "images"  # the TIFFs are used in place

    def download_specs(self):
        """One figshare download per annotated image (file ids from midogpp/Setup.ipynb)."""
        coco = json.loads(_find_json(self.raw_dir).read_text())
        used = {a["image_id"] for a in coco["annotations"]}
        wanted = sorted(im["file_name"] for im in coco["images"] if im["id"] in used
                        and im.get("tumor_type") not in self.exclude_tumours)
        ids = figshare_files()
        return [DownloadSpec(url=f"https://ndownloader.figshare.com/files/{ids[f]}", filename=f) for f in wanted]

    def instructions(self) -> str:
        return (f"  MIDOG++ images download from figshare into {self.raw_dir}/images (503 TIFFs, tens of GB;\n"
                "  set FMBENCH_MIDOGPP to reuse an existing copy). MIDOG++.json is read from there or from\n"
                "  midogpp/databases/ in this repository.")

    def index(self) -> list[dict]:
        coco = json.loads(_find_json(self.raw_dir).read_text())
        return build_index(coco, self.exclude_tumours)

    def _h5(self) -> Path:
        return self.prepared_dir / "tiles.h5"

    def prepare(self, force: bool = False) -> None:
        import h5py
        import tifffile

        out = self._h5()
        if out.exists() and not force:
            print(f"  {out} exists")
            return
        recs = self.index()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as f:
            for split in ("train", "val", "test"):
                rs = [r for r in recs if r["split"] == split]
                ds = f.create_dataset(f"{split}/x", (len(rs), TILE, TILE, 3), dtype="uint8", chunks=(1, TILE, TILE, 3))
                f.create_dataset(f"{split}/y", data=np.array([r["label"] for r in rs], np.int64))
                f.create_dataset(f"{split}/ann_id", data=np.array([r["ann_id"] for r in rs], np.int64))
                f.create_dataset(f"{split}/image_id", data=np.array([r["image_id"] for r in rs], np.int64))
                order = sorted(range(len(rs)), key=lambda i: rs[i]["file_name"])
                cur_name, cur_img = None, None
                for i in order:
                    r = rs[i]
                    if r["file_name"] != cur_name:
                        cur_img = tifffile.imread(_find_image(self.raw_dir, r["file_name"]))
                        if cur_img.ndim == 3 and cur_img.shape[-1] == 4:
                            cur_img = cur_img[..., :3]
                        cur_name = r["file_name"]
                    ds[i] = cur_img[r["y"]:r["y"] + TILE, r["x"]:r["x"] + TILE, :3]
                print(f"  {split}: {len(rs)} tiles")
        tmp.replace(out)
        counts = collections.Counter((r["split"], r["label"]) for r in recs)
        (self.prepared_dir / "counts.json").write_text(json.dumps({f"{s}_{l}": n for (s, l), n in sorted(counts.items())}, indent=2))

    def official_splits(self):
        import h5py

        p = self._h5()
        if not p.exists():
            raise FileNotFoundError(f"{p} (run: python -m fmbench prepare {self.name})")
        out = {}
        with h5py.File(p, "r") as f:
            for split in ("train", "val", "test"):
                y = f[f"{split}/y"][:]
                ids = [f"ann{a}" for a in f[f"{split}/ann_id"][:]]
                groups = f[f"{split}/image_id"][:].astype(object)
                s = H5ArraySplit(p, f"{split}/x", y, ids)
                s.groups = groups
                out[split] = s
        return out

    def status(self) -> TaskStatus:
        if not self._h5().exists():
            try:
                _find_json(self.raw_dir)
                return TaskStatus(False, f"not prepared (run: python -m fmbench prepare {self.name})")
            except FileNotFoundError as e:
                return TaskStatus(False, f"missing: {e}")
        return super().status()


class MidogNoBreastTask(MidogTask):
    name = "midog_nobreast"
    display_name = "MIDOG (no breast)"
    paper_column = None
    exclude_tumours = (BREAST,)
    data_sources = ("midogpp_nonbreast",)
    description = "MIDOG without the human breast cancer images, for models fine-tuned on MIDOG++ breast."
