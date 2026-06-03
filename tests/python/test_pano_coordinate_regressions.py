# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for viewer-side equirectangular coordinate conventions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_viewer_equirectangular_rasterizer_uses_y_up_screen_mapping():
    source = _read("src/rendering/rasterizer/gsplat_fwd/Cameras.cuh")

    assert "auto py = (elevation / PI + 0.5f) * parameters.resolution[1];" in source
    assert "auto elevation = PI * (image_point.y / static_cast<float>(parameters.resolution[1]) - 0.5);" in source


def test_viewer_equirectangular_shaders_use_y_down_ndc_mapping():
    point_cloud = _read("src/rendering/resources/shaders/point_cloud.vert")
    frustum = _read("src/rendering/resources/shaders/camera_frustum.vert")
    axes = _read("src/rendering/resources/shaders/coordinate_axes.vert")

    assert "gl_Position = vec4(v_ndc_x, -phi / (PI * 0.5), -1.0 / depth, 1.0);" in point_cloud
    assert "gl_Position = vec4(ndcX, -phi / (PI * 0.5), -1.0 / depth, 1.0);" in frustum
    assert "gl_Position = vec4(v_ndc_x, -phi / (PI * 0.5), -1.0 / depth, 1.0);" in axes
