#!/usr/bin/env python3
"""
Slide figure of the real MIDOG 2021 breast data: one case per scanner.

Row 1: the whole region of interest, with every annotation marked
        (green = mitotic figure, orange = imposter).
Row 2: close-up of one mitotic figure.
Row 3: close-up of one imposter (a look-alike that is NOT mitosis).
GT450 has no annotations, so its close-ups are random tissue.

Output: figures/midog_scanner_samples.png
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "midog2021"
OUT = ROOT / "figures" / "midog_scanner_samples.png"

CASES = [  # (file id, image_id, scanner name)
    ("024", 24, "Hamamatsu XR"),
    ("078", 78, "Hamamatsu S360"),
    ("101", 101, "Aperio CS2"),
    ("151", 151, "Leica GT450"),
]
THUMB_W, CROP, ZOOM, PAD, LABEL_W, HEAD_H = 300, 128, 220, 14, 120, 40
GREEN, ORANGE = (0, 170, 60), (240, 130, 0)


def font(sz, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", sz)
    except Exception:
        return ImageFont.load_default()


def centre(a):
    x1, y1, x2, y2 = a["bbox"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def closeup(img, cx, cy, colour):
    h = CROP // 2
    c = img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((ZOOM, ZOOM), Image.LANCZOS)
    if colour:
        d = ImageDraw.Draw(c)
        s = ZOOM / CROP
        d.rectangle([ZOOM / 2 - 25 * s, ZOOM / 2 - 25 * s, ZOOM / 2 + 25 * s, ZOOM / 2 + 25 * s],
                    outline=colour, width=3)
    return c


def pick(anns, cat, img):
    h = CROP // 2
    ok = [a for a in anns if a["category_id"] == cat
          and h <= centre(a)[0] <= img.width - h and h <= centre(a)[1] <= img.height - h]
    return ok[len(ok) // 2] if ok else None


def main():
    meta = json.load(open(DATA / "MIDOG.json"))
    by_img = {}
    for a in meta["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    col_w = max(THUMB_W, ZOOM)
    thumb_h = 230
    W = LABEL_W + len(CASES) * (col_w + PAD) + PAD
    H = HEAD_H + thumb_h + 2 * (ZOOM + PAD) + 3 * PAD + 30
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    f_head, f_lab, f_sm = font(17, True), font(15, True), font(12)

    rows_y = [HEAD_H, HEAD_H + thumb_h + PAD, HEAD_H + thumb_h + ZOOM + 2 * PAD]
    for y, txt in zip(rows_y, ["Whole region", "Mitotic\nfigure", "Imposter\n(look-alike)"]):
        dr.multiline_text((PAD, y + 20), txt, fill="black", font=f_lab, spacing=4)

    rnd = random.Random(0)
    for k, (fid, iid, name) in enumerate(CASES):
        x0 = LABEL_W + k * (col_w + PAD)
        img = Image.open(DATA / f"{fid}.tiff").convert("RGB")
        anns = by_img.get(iid, [])
        tw = dr.textlength(name, font=f_head)
        dr.text((x0 + (col_w - tw) / 2, 10), name, fill="black", font=f_head)

        # whole region with markers
        s = THUMB_W / img.width
        th = img.resize((THUMB_W, int(img.height * s)), Image.LANCZOS)
        td = ImageDraw.Draw(th)
        for a in anns:
            cx, cy = centre(a)
            col = GREEN if a["category_id"] == 1 else ORANGE
            td.ellipse([cx * s - 3, cy * s - 3, cx * s + 3, cy * s + 3], outline=col, width=2)
        canvas.paste(th, (x0 + (col_w - THUMB_W) // 2, rows_y[0]))
        n_mf = sum(a["category_id"] == 1 for a in anns)
        n_im = sum(a["category_id"] == 2 for a in anns)
        cap = f"{img.width}x{img.height} px  |  {n_mf} mitoses, {n_im} imposters" if anns else \
              f"{img.width}x{img.height} px  |  no annotations"
        cw = dr.textlength(cap, font=f_sm)
        dr.text((x0 + (col_w - cw) / 2, rows_y[0] + th.height + 2), cap, fill="#555555", font=f_sm)

        # close-ups
        for row, cat, colour in ((1, 1, GREEN), (2, 2, ORANGE)):
            a = pick(anns, cat, img)
            if a:
                tile = closeup(img, *centre(a), colour)
            else:
                cx = rnd.randint(CROP, img.width - CROP)
                cy = rnd.randint(CROP, img.height - CROP)
                tile = closeup(img, cx, cy, None)
                td2 = ImageDraw.Draw(tile)
                td2.rectangle([0, ZOOM - 22, ZOOM, ZOOM], fill=(40, 40, 40))
                td2.text((6, ZOOM - 18), "no label: random tissue", fill="white", font=f_sm)
            canvas.paste(tile, (x0 + (col_w - ZOOM) // 2, rows_y[row]))

    dr.text((PAD, H - 22),
            "MIDOG 2021 training set, human breast cancer (Aubreville et al., CC BY 4.0). "
            "Close-ups are 128x128 px (~30 um); boxes are the 50x50 px annotations.",
            fill="#555555", font=f_sm)
    OUT.parent.mkdir(exist_ok=True)
    canvas.save(OUT)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    main()
