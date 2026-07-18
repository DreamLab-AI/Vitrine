# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Upright-object yaw solve from the crop camera pose (ADR-025 D3 / PRD v4 R10).

``object_placement`` solves WHERE a generated object goes (position) and HOW
BIG it is (uniform scale). This module solves WHICH WAY IT FACES — the piece
``object_placement`` honestly flags ``orientation: "unsolved"``.

The key observation
-------------------
TRELLIS.2 (and Hunyuan3D) canonicalize their output relative to the
conditioning image: the surface visible in the crop becomes the mesh's
canonical FRONT, which in the exporter's glTF convention is **+Z with +Y up**
(``FRONT_AXIS_USD`` / ``UP_AXIS_USD`` below). The crop was taken from ONE
known COLMAP camera (``object_crops`` records ``quaternion_wxyz`` +
``translation`` in its provenance). Therefore the mesh's +Z axis should
point *from the object toward that camera* in the assembled scene — that
single correspondence pins the yaw.

A single view cannot pin all three rotational degrees of freedom (the crop
says nothing about the object's roll, and its pitch is confounded with the
camera's elevation). Real captured objects overwhelmingly rest upright, so we
adopt the upright prior: keep the mesh's +Y aligned with scene up and solve
**yaw only** — rotate about +Y until canonical front points at the horizontal
projection of the object→camera ray.

Frame conventions (all verified against the pipeline sources)
--------------------------------------------------------------
COLMAP (``colmap_parser.ColmapImage``) stores WORLD→CAMERA:

    p_cam = R(q) @ p_world + t          # camera frame: +X right, +Y down,
                                        # +Z forward (RDF)

hence

    camera center (world):     C   = -R^T @ t
    optical axis (world):      fwd = R^T @ (0,0,1) = third ROW of R

``scripts/assemble_usd_scene.py`` maps COLMAP points into the Y-up USD stage
with ``colmap_to_usd_position``: ``(x, y, z) -> (x*s, -y*s, -z*s)`` where
``s = SCENE_SCALE``. For DIRECTIONS only the linear part applies (a 180°
rotation about X — the same map, sans scale):

    d_usd = (d_x, -d_y, -d_z)

Scene "up" is therefore +Y in USD, equivalently -Y in COLMAP world. The
generated GLB is authored Y-up (glTF), inlined into the Y-up stage verbatim,
so mesh-local up already matches stage up: the upright prior costs nothing.

The solve
---------
1. front ray (COLMAP world): f = normalize(C - object_centroid) when the
   Gaussian-subset centroid is available (exact), else f = -fwd (optical
   axis — exact only when the object sat dead-centre in the frame, which
   the crop selector's centrality score biases toward).
2. map to USD: f_usd = (f_x, -f_y, -f_z).
3. upright projection: f_h = (f_x, 0, f_z) in USD. Degenerate when the
   camera looked straight down/up at the object (‖f_h‖ ≈ 0) — then yaw is
   unobservable and we return identity, flagged.
4. yaw about +Y taking FRONT_AXIS_USD = +Z onto f_h:
   R_y(yaw) @ (0,0,1) = (sin yaw, 0, cos yaw)  =>  yaw = atan2(f_x, f_z).
5. quaternion (wxyz): (cos(yaw/2), 0, sin(yaw/2), 0).

All outputs are in the USD stage frame, ready for an ``AddOrientOp`` between
the assembler's translate and scale ops. Everything here is pure math — no
numpy, no I/O — mirroring ``object_placement``'s testability contract.

See ``research/2026-07-orientation-solve.md`` for the full derivation,
limitations (roll ambiguity, upright assumption, canonical-front assumption)
and the integration plan.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

# The generated mesh's canonical axes in its own (glTF Y-up) frame, which is
# also the USD stage frame once inlined. If empirical validation ever shows
# the generator fronts -Z instead of +Z, flip this ONE constant (equivalent
# to a 180° yaw offset) — nothing else changes.
FRONT_AXIS_USD: tuple[float, float, float] = (0.0, 0.0, 1.0)
UP_AXIS_USD: tuple[float, float, float] = (0.0, 1.0, 0.0)

# Below this horizontal-component norm the viewing ray is straight up/down
# and yaw is unobservable (elevation ~> 89.9997°).
DEFAULT_DEGENERATE_EPS: float = 1e-8

_EPS = 1e-12

IDENTITY_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
#  Small pure helpers (quaternion / vector)
# ---------------------------------------------------------------------------

def _as_vec3(v: Sequence[float] | None, name: str) -> tuple[float, float, float]:
    if v is None or len(v) != 3:
        raise ValueError(f"{name} must be a 3-sequence, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _normalize3(v: tuple[float, float, float]) -> Optional[tuple[float, float, float]]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < _EPS:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def normalize_quat_wxyz(q: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate + unit-normalize a (w, x, y, z) quaternion.

    Raises ValueError on wrong arity or (near-)zero norm — a zero quaternion
    is corrupt input, not a solvable pose.
    """
    if q is None or len(q) != 4:
        raise ValueError(f"quaternion must be a 4-sequence (w,x,y,z), got {q!r}")
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < _EPS:
        raise ValueError("zero-norm quaternion")
    return (w / n, x / n, y / n, z / n)


def quat_to_rotation_matrix(q: Sequence[float]) -> list[list[float]]:
    """Rotation matrix for a (w, x, y, z) quaternion (COLMAP/USD convention).

    Same formula as ``assemble_usd_scene._quat_to_rotation_matrix`` so the
    two stay numerically consistent.
    """
    w, x, y, z = normalize_quat_wxyz(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """Hamilton product a ⊗ b, both (w, x, y, z). R(a⊗b) = R(a) @ R(b)."""
    aw, ax, ay, az = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bw, bx, by, bz = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate_vec_by_quat(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a 3-vector by a (w, x, y, z) quaternion: R(q) @ v."""
    R = quat_to_rotation_matrix(q)
    x, y, z = _as_vec3(v, "v")
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z,
        R[1][0] * x + R[1][1] * y + R[1][2] * z,
        R[2][0] * x + R[2][1] * y + R[2][2] * z,
    )


# ---------------------------------------------------------------------------
#  COLMAP camera geometry
# ---------------------------------------------------------------------------

def camera_center_colmap(
    quat_wxyz: Sequence[float], translation: Sequence[float]
) -> tuple[float, float, float]:
    """Camera center in COLMAP world coordinates: C = -R^T @ t.

    COLMAP stores world→camera (p_cam = R p_world + t); the center is the
    world point mapping to the camera origin.
    """
    R = quat_to_rotation_matrix(quat_wxyz)
    tx, ty, tz = _as_vec3(translation, "translation")
    return (
        -(R[0][0] * tx + R[1][0] * ty + R[2][0] * tz),
        -(R[0][1] * tx + R[1][1] * ty + R[2][1] * tz),
        -(R[0][2] * tx + R[1][2] * ty + R[2][2] * tz),
    )


def camera_forward_colmap(quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Camera optical axis (+Z_cam, the viewing direction) in COLMAP world.

    A world direction d maps to camera coords as R @ d, so the world vector
    mapping onto +Z_cam is R^T @ (0,0,1) — the third ROW of R.
    """
    R = quat_to_rotation_matrix(quat_wxyz)
    return (R[2][0], R[2][1], R[2][2])


def colmap_dir_to_usd(v: Sequence[float]) -> tuple[float, float, float]:
    """Map a DIRECTION from COLMAP world into the USD stage frame.

    The linear part of ``assemble_usd_scene.colmap_to_usd_position`` —
    (x, y, z) → (x, -y, -z), a 180° rotation about X. SCENE_SCALE is a
    point-only concern and never applies to directions.
    """
    x, y, z = _as_vec3(v, "direction")
    return (x, -y, -z)


# ---------------------------------------------------------------------------
#  The upright yaw solve
# ---------------------------------------------------------------------------

def yaw_from_front_usd(
    front_usd: Sequence[float],
    degenerate_eps: float = DEFAULT_DEGENERATE_EPS,
) -> Optional[float]:
    """Yaw (radians, about +Y_usd) taking FRONT_AXIS_USD (+Z) onto ``front``.

    Only the horizontal (XZ) component of ``front`` is used — the upright
    constraint. Returns None when that component is degenerate (near-vertical
    viewing ray: yaw unobservable).

    Derivation: R_y(θ) @ (0,0,1) = (sin θ, 0, cos θ), so matching the
    horizontal front direction (f_x, f_z) gives θ = atan2(f_x, f_z).
    """
    fx, _fy, fz = _as_vec3(front_usd, "front_usd")
    if math.hypot(fx, fz) <= degenerate_eps:
        return None
    return math.atan2(fx, fz)


def quat_wxyz_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a rotation of ``yaw_rad`` about +Y."""
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def solve_yaw(
    camera_quaternion_wxyz: Sequence[float],
    camera_translation: Sequence[float],
    object_centroid: Sequence[float] | None = None,
    degenerate_eps: float = DEFAULT_DEGENERATE_EPS,
) -> dict[str, Any]:
    """Solve the upright-object yaw from the crop camera's COLMAP pose.

    Parameters
    ----------
    camera_quaternion_wxyz, camera_translation :
        The crop frame's WORLD→CAMERA pose exactly as recorded by
        ``object_crops`` provenance (``camera_pose.quaternion_wxyz`` /
        ``camera_pose.translation``, i.e. COLMAP images.txt fields).
    object_centroid :
        Optional Gaussian-subset centroid in COLMAP world coordinates (the
        ``placement.centroid`` the position solve already uses). When given,
        the front direction is the exact object→camera ray (method
        ``"camera-ray"``); otherwise the camera's optical axis is used as a
        proxy (method ``"optical-axis"`` — exact only for a centred object).
    degenerate_eps :
        Horizontal-norm threshold below which yaw is unobservable.

    Returns
    -------
    dict with plain-JSON values (the placements.json fragment):
        quat_wxyz     [w, x, y, z] rotation in the USD stage frame — a pure
                      rotation about +Y (identity when degenerate).
        yaw_deg       solved yaw in degrees (0 when degenerate).
        method        "camera-ray" | "optical-axis" | "degenerate".
        elevation_deg camera elevation above the object's horizontal plane —
                      a confidence signal (high |elevation| → the visible
                      surface was mostly top/bottom; treat yaw as weak).
        front_usd     the (unit) desired front direction, USD frame,
                      pre-projection — diagnostic.

    Deterministic, pure, raises ValueError only on malformed input.
    """
    q = normalize_quat_wxyz(camera_quaternion_wxyz)
    t = _as_vec3(camera_translation, "camera_translation")

    front_colmap: Optional[tuple[float, float, float]] = None
    method = "optical-axis"
    if object_centroid is not None:
        cx, cy, cz = _as_vec3(object_centroid, "object_centroid")
        cam = camera_center_colmap(q, t)
        front_colmap = _normalize3((cam[0] - cx, cam[1] - cy, cam[2] - cz))
        if front_colmap is not None:
            method = "camera-ray"
        # else: centroid coincides with the camera center (corrupt hint) —
        # fall through to the optical-axis proxy rather than failing.

    if front_colmap is None:
        fwd = camera_forward_colmap(q)
        # The object is IN FRONT of the camera; its front points BACK at it.
        front_colmap = (-fwd[0], -fwd[1], -fwd[2])  # unit: row of a rotation
        method = "optical-axis"

    front_usd = colmap_dir_to_usd(front_colmap)
    horiz = math.hypot(front_usd[0], front_usd[2])
    elevation_deg = math.degrees(math.atan2(front_usd[1], horiz))

    yaw = yaw_from_front_usd(front_usd, degenerate_eps)
    if yaw is None:
        return {
            "quat_wxyz": list(IDENTITY_QUAT_WXYZ),
            "yaw_deg": 0.0,
            "method": "degenerate",
            "elevation_deg": elevation_deg,
            "front_usd": list(front_usd),
        }

    return {
        "quat_wxyz": list(quat_wxyz_from_yaw(yaw)),
        "yaw_deg": math.degrees(yaw),
        "method": method,
        "elevation_deg": elevation_deg,
        "front_usd": list(front_usd),
    }
