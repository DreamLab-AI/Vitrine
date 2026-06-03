#!/usr/bin/env python3
"""Crop a poster into 4 quadrants, or stitch 4 refined quadrants back into one A0 image.

Usage:
  quadrants.py crop  MASTER.png  OUTDIR              # writes q_tl/q_tr/q_bl/q_br.png
  quadrants.py stitch OUTDIR FINAL.png TILE_W TILE_H # montage 2x2 to (2*TILE_W)x(2*TILE_H)
"""
import sys
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def crop(master: str, outdir: str) -> None:
    im = Image.open(master).convert("RGB")
    w, h = im.size
    mw, mh = w // 2, h // 2
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = {
        "q_tl": (0, 0, mw, mh),
        "q_tr": (mw, 0, w, mh),
        "q_bl": (0, mh, mw, h),
        "q_br": (mw, mh, w, h),
    }
    for name, box in boxes.items():
        im.crop(box).save(out / f"{name}.png")
        print(f"OK {name}.png {box[2]-box[0]}x{box[3]-box[1]}")


def stitch(outdir: str, final: str, tile_w: int, tile_h: int) -> None:
    out = Path(outdir)
    tiles = {}
    for name in ("q_tl", "q_tr", "q_bl", "q_br"):
        p = out / f"{name}_4k.png"
        if not p.exists():
            p = out / f"{name}.png"
        tiles[name] = Image.open(p).convert("RGB").resize((tile_w, tile_h), Image.LANCZOS)
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), "white")
    canvas.paste(tiles["q_tl"], (0, 0))
    canvas.paste(tiles["q_tr"], (tile_w, 0))
    canvas.paste(tiles["q_bl"], (0, tile_h))
    canvas.paste(tiles["q_br"], (tile_w, tile_h))
    canvas.save(final)
    print(f"OK {final} {tile_w*2}x{tile_h*2}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "crop":
        crop(sys.argv[2], sys.argv[3])
    elif mode == "stitch":
        stitch(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        sys.exit("mode must be crop|stitch")
