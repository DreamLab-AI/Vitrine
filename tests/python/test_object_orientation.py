# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.object_orientation (ADR-025 D3 / PRD v4 R10).

Pure math — no deps, no I/O. Every fixture below is derived by hand from the
COLMAP world→camera convention (p_cam = R p + t; camera RDF: +X right, +Y
down, +Z forward) and the assembler's colmap→USD map (x, y, z) → (x, -y, -z).

Fixture derivation (level cameras, object at the COLMAP/USD origin):
  * "camera on +X_usd looking at origin": desired USD viewing dir (-1,0,0)
    → COLMAP forward (-1,0,0) = row3(R). Level ⇒ camera-down = world-down
    = (0,+1,0) (COLMAP up is -Y) = row2. row1 = down × forward = (0,0,1).
    R = Ry(90°) ⇒ q = (cos45°, 0, sin45°, 0). Center C = (2,0,0) ⇒
    t = -R C = (0,0,2). Expected: object front faces +X_usd ⇒ yaw = 90°,
    quat = (cos45°, 0, sin45°, 0).
  * "+Z_usd": R = I, C = (0,0,-2), t = (0,0,2) ⇒ yaw 0 (identity).
  * "-X_usd": R = Ry(-90°) ⇒ q = (cos45°, 0, -sin45°, 0), t = (0,0,2)
    ⇒ yaw -90°.
  * "-Z_usd": R = Ry(180°) ⇒ q = (0,0,1,0), t = (0,0,2) ⇒ |yaw| = 180°.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.object_orientation import (  # noqa: E402
    FRONT_AXIS_USD,
    IDENTITY_QUAT_WXYZ,
    camera_center_colmap,
    camera_forward_colmap,
    colmap_dir_to_usd,
    normalize_quat_wxyz,
    quat_multiply,
    quat_to_rotation_matrix,
    quat_wxyz_from_yaw,
    rotate_vec_by_quat,
    solve_yaw,
    yaw_from_front_usd,
)

S = math.sqrt(0.5)  # cos45° = sin45°

# Level cameras looking at the origin, hand-derived above.
CAM_PLUS_X = ((S, 0.0, S, 0.0), (0.0, 0.0, 2.0))       # on +X_usd
CAM_PLUS_Z = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 2.0))   # on +Z_usd
CAM_MINUS_X = ((S, 0.0, -S, 0.0), (0.0, 0.0, 2.0))     # on -X_usd
CAM_MINUS_Z = ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 2.0))  # on -Z_usd


def _assert_unit_pure_yaw(quat):
    w, x, y, z = quat
    assert math.sqrt(w * w + x * x + y * y + z * z) == pytest.approx(1.0)
    assert x == pytest.approx(0.0, abs=1e-12)   # never tips the object
    assert z == pytest.approx(0.0, abs=1e-12)   # never rolls the object


def _rotated_front(quat):
    return rotate_vec_by_quat(quat, FRONT_AXIS_USD)


# ---------------------------------------------------------------------------
#  Geometry helpers pin the conventions
# ---------------------------------------------------------------------------

def test_camera_center_is_minus_R_transpose_t():
    # R = Ry(90°) as world→cam, t = (0,0,2): C = -R^T t = (2, 0, 0).
    c = camera_center_colmap(*CAM_PLUS_X)
    assert c == pytest.approx((2.0, 0.0, 0.0))


def test_camera_forward_is_third_row_of_R():
    # Identity pose looks along +Z_colmap.
    assert camera_forward_colmap((1, 0, 0, 0)) == pytest.approx((0.0, 0.0, 1.0))
    # The +X_usd camera looks along -X_colmap (toward the origin).
    assert camera_forward_colmap(CAM_PLUS_X[0]) == pytest.approx((-1.0, 0.0, 0.0))


def test_colmap_dir_to_usd_flips_y_and_z_without_scale():
    assert colmap_dir_to_usd((1.0, 2.0, 3.0)) == pytest.approx((1.0, -2.0, -3.0))


def test_quat_multiply_composes_rotations():
    qa = quat_wxyz_from_yaw(math.radians(30))
    qb = quat_wxyz_from_yaw(math.radians(60))
    Rab = quat_to_rotation_matrix(quat_multiply(qa, qb))
    R90 = quat_to_rotation_matrix(quat_wxyz_from_yaw(math.radians(90)))
    for i in range(3):
        assert Rab[i] == pytest.approx(R90[i])


def test_yaw_round_trip_through_front_vector():
    for deg in (-179.0, -90.0, -45.0, 0.0, 30.0, 90.0, 135.0, 179.0):
        yaw = math.radians(deg)
        front = (math.sin(yaw), 0.0, math.cos(yaw))
        assert math.degrees(yaw_from_front_usd(front)) == pytest.approx(deg)


# ---------------------------------------------------------------------------
#  The headline solve: known camera pose → expected yaw
# ---------------------------------------------------------------------------

def test_camera_on_plus_x_makes_object_face_plus_x():
    out = solve_yaw(*CAM_PLUS_X)
    assert out["method"] == "optical-axis"
    assert out["yaw_deg"] == pytest.approx(90.0)
    assert out["quat_wxyz"] == pytest.approx([S, 0.0, S, 0.0])
    _assert_unit_pure_yaw(out["quat_wxyz"])
    assert _rotated_front(out["quat_wxyz"]) == pytest.approx((1.0, 0.0, 0.0))
    assert out["elevation_deg"] == pytest.approx(0.0)


def test_camera_on_plus_z_is_identity():
    out = solve_yaw(*CAM_PLUS_Z)
    assert out["yaw_deg"] == pytest.approx(0.0)
    assert out["quat_wxyz"] == pytest.approx(list(IDENTITY_QUAT_WXYZ))
    assert _rotated_front(out["quat_wxyz"]) == pytest.approx((0.0, 0.0, 1.0))


def test_camera_on_minus_x_makes_object_face_minus_x():
    out = solve_yaw(*CAM_MINUS_X)
    assert out["yaw_deg"] == pytest.approx(-90.0)
    assert _rotated_front(out["quat_wxyz"]) == pytest.approx((-1.0, 0.0, 0.0))


def test_camera_behind_turns_object_around():
    out = solve_yaw(*CAM_MINUS_Z)
    assert abs(out["yaw_deg"]) == pytest.approx(180.0)
    assert _rotated_front(out["quat_wxyz"]) == pytest.approx((0.0, 0.0, -1.0))
    _assert_unit_pure_yaw(out["quat_wxyz"])


# ---------------------------------------------------------------------------
#  The upright constraint: elevation is discarded, never tips the object
# ---------------------------------------------------------------------------

def test_elevated_camera_keeps_object_upright_optical_axis():
    # The +X_usd camera pitched 45° down toward the origin: world→cam gains
    # a premultiplied Rx(45°) (verified: R = Rx(45°) @ Ry(90°) has
    # row3 = forward = normalize((-1, 1, 0)) in COLMAP = toward the origin
    # from the raised center C = (2, -2, 0); t = -R C = (0, 0, 2√2)).
    q_pitch = (math.cos(math.radians(22.5)), math.sin(math.radians(22.5)), 0.0, 0.0)
    q = quat_multiply(q_pitch, CAM_PLUS_X[0])
    t = (0.0, 0.0, 2.0 * math.sqrt(2.0))
    # Self-check the composed fixture before trusting the assertion.
    assert camera_forward_colmap(q) == pytest.approx((-S, S, 0.0))
    assert camera_center_colmap(q, t) == pytest.approx((2.0, -2.0, 0.0))

    out = solve_yaw(q, t)
    assert out["method"] == "optical-axis"
    assert out["yaw_deg"] == pytest.approx(90.0)          # pitch discarded
    assert out["elevation_deg"] == pytest.approx(45.0)    # ...but reported
    _assert_unit_pure_yaw(out["quat_wxyz"])
    # Rotated front stays horizontal: the object is never tipped.
    assert _rotated_front(out["quat_wxyz"]) == pytest.approx((1.0, 0.0, 0.0))


def test_elevated_camera_keeps_object_upright_camera_ray():
    # Same level +X-camera ROTATION but center raised to C = (2, -2, 0)
    # (t = -R C = (0, 2, 2)); the object centroid pins the exact ray.
    q = CAM_PLUS_X[0]
    t = (0.0, 2.0, 2.0)
    assert camera_center_colmap(q, t) == pytest.approx((2.0, -2.0, 0.0))

    out = solve_yaw(q, t, object_centroid=(0.0, 0.0, 0.0))
    assert out["method"] == "camera-ray"
    assert out["yaw_deg"] == pytest.approx(90.0)
    assert out["elevation_deg"] == pytest.approx(45.0)
    _assert_unit_pure_yaw(out["quat_wxyz"])


def test_camera_ray_beats_optical_axis_for_off_centre_objects():
    # Identity camera at C = (0,0,-2) looking along +Z_colmap; the object
    # sits off-axis at (2, 0, 0). Optical axis says yaw 0; the true
    # object→camera ray normalize((-2,0,-2)) → USD (-S, 0, S) says -45°.
    q, t = CAM_PLUS_Z
    assert solve_yaw(q, t)["yaw_deg"] == pytest.approx(0.0)
    out = solve_yaw(q, t, object_centroid=(2.0, 0.0, 0.0))
    assert out["method"] == "camera-ray"
    assert out["yaw_deg"] == pytest.approx(-45.0)


# ---------------------------------------------------------------------------
#  Degenerate + malformed inputs
# ---------------------------------------------------------------------------

def test_top_down_camera_is_degenerate_identity():
    # Straight-down camera: forward = world-down = (0, +1, 0) in COLMAP
    # (COLMAP up is -Y). R = Rx(90°) ⇒ q = (cos45°, sin45°, 0, 0); center
    # C = (0, -3, 0) above the object ⇒ t = -R C = (0, 0, 3).
    q = (S, S, 0.0, 0.0)
    t = (0.0, 0.0, 3.0)
    assert camera_forward_colmap(q) == pytest.approx((0.0, 1.0, 0.0))

    out = solve_yaw(q, t)
    assert out["method"] == "degenerate"
    assert out["quat_wxyz"] == pytest.approx(list(IDENTITY_QUAT_WXYZ))
    assert out["yaw_deg"] == 0.0
    assert out["elevation_deg"] == pytest.approx(90.0)

    # Same verdict when the ray comes from a centroid directly underneath.
    out = solve_yaw(q, t, object_centroid=(0.0, 0.0, 0.0))
    assert out["method"] == "degenerate"


def test_centroid_coincident_with_camera_falls_back_to_optical_axis():
    q, t = CAM_PLUS_Z
    out = solve_yaw(q, t, object_centroid=(0.0, 0.0, -2.0))  # == camera center
    assert out["method"] == "optical-axis"
    assert out["yaw_deg"] == pytest.approx(0.0)


def test_malformed_inputs_raise_value_error():
    with pytest.raises(ValueError):
        solve_yaw((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0))     # zero quat
    with pytest.raises(ValueError):
        solve_yaw((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))          # 3-quat
    with pytest.raises(ValueError):
        solve_yaw((1.0, 0.0, 0.0, 0.0), (0.0, 1.0))          # 2-translation
    with pytest.raises(ValueError):
        solve_yaw((1.0, 0.0, 0.0, 0.0), None)                # missing t
    with pytest.raises(ValueError):
        normalize_quat_wxyz(None)


def test_unnormalized_quaternion_is_accepted():
    # COLMAP text quats can drift off unit norm; scaling must not change yaw.
    q, t = CAM_PLUS_X
    scaled = tuple(3.7 * c for c in q)
    assert solve_yaw(scaled, t)["yaw_deg"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
#  Output schema (the placements.json fragment)
# ---------------------------------------------------------------------------

def test_result_schema_is_json_ready():
    out = solve_yaw(*CAM_PLUS_X, object_centroid=(0.0, 0.0, 0.0))
    assert set(out) == {"quat_wxyz", "yaw_deg", "method", "elevation_deg",
                        "front_usd"}
    assert isinstance(out["quat_wxyz"], list) and len(out["quat_wxyz"]) == 4
    assert isinstance(out["front_usd"], list) and len(out["front_usd"]) == 3
    assert all(isinstance(v, float) for v in out["quat_wxyz"] + out["front_usd"])
    assert isinstance(out["yaw_deg"], float)
    assert isinstance(out["elevation_deg"], float)
    assert isinstance(out["method"], str)
