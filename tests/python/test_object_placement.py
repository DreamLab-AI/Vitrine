# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.object_placement (ADR-025 D3 / PRD v4 R10).

Pure math — no deps, no I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.object_placement import (  # noqa: E402
    Placement,
    build_placements,
    solve_placement,
)


def test_scale_ratio_is_real_over_normalized():
    # A 0.8 m real object generated as a ~unit-diameter mesh scales down.
    p = solve_placement("vase", [1.0, 2.0, 3.0], [0.8, 0.6, 0.8], [1.0, 0.86, 1.0])
    assert p.scale_ratio == pytest.approx(0.8 / 1.0)
    assert p.world_centroid == (1.0, 2.0, 3.0)
    assert p.orientation == "unsolved"


def test_scale_uses_the_longest_axis_on_both_sides():
    p = solve_placement("slab", [0, 0, 0], [2.0, 0.1, 0.5], [0.5, 0.02, 0.1])
    # real max 2.0 / glb max 0.5 = 4.0
    assert p.scale_ratio == pytest.approx(4.0)


def test_degenerate_glb_extent_falls_back_to_unit_scale():
    # No usable normalized size -> position-only, never a guessed scale.
    p = solve_placement("x", [1, 1, 1], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0])
    assert p.scale_ratio == 1.0
    assert p.world_centroid == (1.0, 1.0, 1.0)


def test_missing_glb_extent_is_safe():
    p = solve_placement("x", [0, 0, 0], [1, 1, 1], None)
    assert p.scale_ratio == 1.0


def test_degenerate_world_extent_falls_back_to_unit_scale():
    p = solve_placement("x", [3, 0, 0], [0, 0, 0], [1, 1, 1])
    assert p.scale_ratio == 1.0
    assert p.world_centroid == (3.0, 0.0, 0.0)


def test_to_dict_schema():
    d = solve_placement("vase", [1, 2, 3], [0.8, 0.8, 0.8], [1, 1, 1]).to_dict()
    assert set(d) == {"label", "world_centroid", "scale_ratio", "orientation",
                      "world_extent", "glb_extent"}
    assert d["world_centroid"] == [1.0, 2.0, 3.0]


def test_build_placements_only_covers_generator_meshes():
    meshes = [
        {"label": "full_scene", "mesh": "/x/scene.glb"},          # no placement
        {"label": "vase", "mesh": "/x/vase.glb",
         "placement": {"centroid": [1, 0, 0], "extent": [0.5, 0.5, 0.5]},
         "glb_extent": [1.0, 1.0, 1.0]},
        {"label": "block", "mesh": "/x/block.glb",
         "placement": {"centroid": [0, 2, 0], "extent": [0.2, 0.1, 0.3]},
         "glb_extent": [1.0, 0.5, 1.5]},
    ]
    out = build_placements(meshes)
    assert set(out) == {"vase", "block"}          # full_scene excluded
    assert out["vase"]["scale_ratio"] == pytest.approx(0.5)
    assert out["block"]["world_centroid"] == [0.0, 2.0, 0.0]


def test_build_placements_tolerates_missing_glb_extent():
    meshes = [{"label": "vase", "placement": {"centroid": [1, 0, 0],
                                              "extent": [0.5, 0.5, 0.5]}}]
    out = build_placements(meshes)
    assert out["vase"]["scale_ratio"] == 1.0    # unknown normalized size


def test_build_placements_solves_orientation_from_camera_pose():
    # R10 orientation integration: a mesh carrying a crop camera pose gets a
    # yaw-solved orientation quaternion in its placement (camera on +X_usd
    # looking at origin -> object front faces +X, yaw 90deg).
    meshes = [{
        "label": "vase",
        "placement": {"centroid": [0, 0, 0], "extent": [0.5, 0.5, 0.5]},
        "glb_extent": [1.0, 1.0, 1.0],
        "camera_pose": {"quaternion_wxyz": [0.7071, 0.0, 0.7071, 0.0],
                        "translation": [0.0, 0.0, 2.0]},
    }]
    out = build_placements(meshes)["vase"]
    assert out["orientation"] == "yaw-solved"
    assert len(out["orientation_quat"]) == 4
    assert out["orientation_yaw_deg"] == pytest.approx(90.0, abs=1.0)


def test_build_placements_without_pose_stays_unsolved():
    meshes = [{"label": "vase",
               "placement": {"centroid": [0, 0, 0], "extent": [0.5, 0.5, 0.5]},
               "glb_extent": [1.0, 1.0, 1.0]}]
    out = build_placements(meshes)["vase"]
    assert out["orientation"] == "unsolved"
    assert "orientation_quat" not in out
