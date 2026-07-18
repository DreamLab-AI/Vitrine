# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""HEIC/HEIF frame ingestion (Apple capture support).

iPhone/iPad captures are HEIC by default, but COLMAP and the pipeline's frame
globbing only read JPEG/PNG — so a HEIC-only capture silently reconstructs
NOTHING (0 frames). This module converts HEIC/HEIF to full-resolution JPEG
before COLMAP, baking EXIF orientation so the camera poses are correct.

Modern iPhone HEIC files carry HDR gain-map auxiliary images that break older
ImageMagick/libheif; ``pillow-heif`` decodes them reliably, so that is the
backend. When ``pillow-heif`` is unavailable the frames are left untouched and
the caller proceeds with whatever JPEG/PNG are present (a logged no-op, never a
crash).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_HEIC_SUFFIXES = {".heic", ".heif", ".HEIC", ".HEIF"}
_DECODABLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def has_heic(frames_dir: str | Path) -> bool:
    """True if the directory contains any HEIC/HEIF files."""
    p = Path(frames_dir)
    if not p.is_dir():
        return False
    return any(f.suffix in _HEIC_SUFFIXES for f in p.iterdir() if f.is_file())


def ensure_jpeg_frames(frames_dir: str | Path,
                       quality: int = 95) -> Path:
    """Return a directory of COLMAP-readable frames, converting HEIC if needed.

    If ``frames_dir`` has no HEIC, it is returned unchanged. Otherwise HEIC/HEIF
    files are converted to JPEG in a sibling ``<name>_jpeg`` directory (existing
    JPEG/PNG are copied across so the returned dir is the complete frame set),
    and that directory is returned. Idempotent: already-converted files are
    skipped, so it is safe to call repeatedly.
    """
    src = Path(frames_dir)
    if not has_heic(src):
        return src

    out = src.parent / f"{src.name}_jpeg"
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pillow_heif  # noqa: F401
        from PIL import Image, ImageOps
        pillow_heif.register_heif_opener()
    except ImportError as exc:
        logger.warning("HEIC frames present but pillow-heif unavailable (%s) — "
                       "leaving frames as-is; COLMAP will only see JPEG/PNG", exc)
        return src

    converted = skipped = failed = 0
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if f.suffix in _DECODABLE_SUFFIXES:
            # Carry existing JPEG/PNG into the unified output dir.
            dst = out / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())
            continue
        if f.suffix not in _HEIC_SUFFIXES:
            continue
        dst = out / (f.stem + ".jpg")
        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        try:
            im = Image.open(f)
            im = ImageOps.exif_transpose(im)      # bake orientation for COLMAP
            im = im.convert("RGB")
            im.save(dst, "JPEG", quality=quality)
            converted += 1
        except Exception as exc:  # noqa: BLE001 — one bad frame must not stop ingest
            logger.warning("HEIC convert failed for %s: %s", f.name, exc)
            failed += 1

    logger.info("HEIC ingestion: %d converted, %d already done, %d failed -> %s",
                converted, skipped, failed, out)
    return out
