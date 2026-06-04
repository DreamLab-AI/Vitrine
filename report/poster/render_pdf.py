#!/usr/bin/env python3
"""Rasterise a PDF page to PNG using PyMuPDF. Run via: uv run --with pymupdf."""
import sys
import fitz  # PyMuPDF

pdf, out = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 100
page_no = int(sys.argv[4]) if len(sys.argv) > 4 else 0

doc = fitz.open(pdf)
page = doc[page_no]
pix = page.get_pixmap(dpi=dpi, alpha=False)
pix.save(out)
print(f"OK {out} {pix.width}x{pix.height} @ {dpi}dpi")
