#!/usr/bin/env python3
"""Finalise the A0 poster: patch the one text typo on the clean master, then
upscale 2x to A0 print resolution with a mild unsharp mask."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

Image.MAX_IMAGE_PIXELS = None
FB = "/nix/store/s160g9cfrqr70cpyxrmhnbkcsgk9mm9f-liberation-fonts-2.1.5/share/fonts/truetype/LiberationSans-Bold.ttf"
FR = "/nix/store/s160g9cfrqr70cpyxrmhnbkcsgk9mm9f-liberation-fonts-2.1.5/share/fonts/truetype/LiberationSans-Regular.ttf"

im = Image.open("work/poster_clean4k.png").convert("RGB")
d = ImageDraw.Draw(im)

# --- patch the System Architecture navy pill: "USD I/D" -> "USD I/O" ---
x0, y0, x1, y1 = 2450, 4089, 3278, 4169
navy = (20, 56, 75)
d.rounded_rectangle([x0, y0, x1, y1], radius=38, fill=navy)
bold_txt = "LichtFeld Studio v0.5.2"
reg_txt = "   ·   USD I/O   ·   MCP control (70+ tools)"
pad = 34
maxw = (x1 - x0) - 2 * pad
size = 50
while size > 20:
    fb = ImageFont.truetype(FB, size)
    fr = ImageFont.truetype(FR, size)
    wb = d.textlength(bold_txt, font=fb)
    wr = d.textlength(reg_txt, font=fr)
    if wb + wr <= maxw:
        break
    size -= 1
total = wb + wr
sx = (x0 + x1) / 2 - total / 2
cy = (y0 + y1) / 2
asc, desc = fb.getmetrics()
ty = cy - (asc + desc) / 2
white = (245, 246, 248)
d.text((sx, ty), bold_txt, font=fb, fill=white)
d.text((sx + wb, ty), reg_txt, font=fr, fill=white)

# --- upscale 2x to A0 print resolution, mild sharpen ---
big = im.resize((im.size[0] * 2, im.size[1] * 2), Image.LANCZOS)
big = big.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
big.save("work/poster_a0_final.png")
big.resize((big.size[0] // 5, big.size[1] // 5), Image.LANCZOS).save("work/poster_a0_final_preview.png")
print("OK", big.size)
