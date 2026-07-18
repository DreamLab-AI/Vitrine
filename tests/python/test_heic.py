# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.heic (Apple HEIC frame ingestion)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import heic  # noqa: E402


def test_has_heic_detection(tmp_path):
    assert not heic.has_heic(tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert not heic.has_heic(tmp_path)
    (tmp_path / "b.HEIC").write_bytes(b"x")
    assert heic.has_heic(tmp_path)
    assert not heic.has_heic(tmp_path / "does_not_exist")


def test_ensure_jpeg_frames_noop_without_heic(tmp_path):
    # No HEIC -> the same directory is returned untouched.
    (tmp_path / "frame_001.jpg").write_bytes(b"x")
    out = heic.ensure_jpeg_frames(tmp_path)
    assert out == tmp_path
    assert not (tmp_path.parent / f"{tmp_path.name}_jpeg").exists()


def test_ensure_jpeg_frames_converts_heic_roundtrip(tmp_path):
    pytest.importorskip("pillow_heif")
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()

    src = tmp_path / "frames"
    src.mkdir()
    # A real HEIC + a plain JPEG that should be carried across.
    Image.new("RGB", (64, 48), (200, 120, 40)).save(src / "IMG_1.heic", "HEIF")
    Image.new("RGB", (64, 48), (10, 20, 30)).save(src / "IMG_2.jpg", "JPEG")

    out = heic.ensure_jpeg_frames(src)
    assert out == src.parent / "frames_jpeg"
    # HEIC converted to JPEG; existing JPEG carried over.
    assert (out / "IMG_1.jpg").is_file()
    assert (out / "IMG_2.jpg").is_file()
    # The converted JPEG is a valid, correctly-sized image.
    im = Image.open(out / "IMG_1.jpg")
    assert im.size == (64, 48)

    # Idempotent: a second call re-uses the converted files (no error).
    out2 = heic.ensure_jpeg_frames(src)
    assert out2 == out
