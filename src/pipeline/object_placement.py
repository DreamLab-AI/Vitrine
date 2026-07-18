# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Object pose-solve for scene assembly (ADR-025 D3 / PRD v4 R10).

The 3D generator (TRELLIS.2) emits each object in its OWN normalized frame —
centred near the origin, roughly unit-diameter — with no knowledge of where or
how big the object is in the captured scene. This module solves the rigid
placement that puts that normalized mesh back into the scene, using the
per-object Gaussian subset (which IS in COLMAP-world coordinates) as the
ground truth for position and size:

    generated GLB (normalized)  ──scale──►  real-world size
                                ──translate──►  real-world position

Position and uniform scale are fully determined by the recorded data and are
computed here. **Orientation is left unsolved** (identity) and flagged as such
in the output: TRELLIS's canonical output orientation does not correspond to
the object's real orientation without a further solve from the crop camera
pose + silhouette, which is future work (the honest ADR-025 D3 posture — a
generated backside is already `surface: inferred`).

All values here stay in RAW COLMAP-WORLD units and a dimensionless scale
ratio. The USD/coordinate-system convention (Y-flip, SCENE_SCALE) is applied
by the consumer (``scripts/assemble_usd_scene.py``) so there is exactly one
place that owns it. This module is pure (no I/O, no pxr, no numpy) and unit-
tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Placement:
    """Solved rigid placement of a generated object in the scene.

    Attributes
    ----------
    label : str
        Object label (matches the mesh prim name).
    world_centroid : tuple[float, float, float]
        Object centre in COLMAP-world coordinates (raw, pre-USD-conversion).
    scale_ratio : float
        Uniform scale mapping the normalized GLB to real-world size
        (real_max_extent / glb_max_extent).
    orientation : str
        ``"unsolved"`` — placement applies position + uniform scale only.
    world_extent : tuple[float, float, float]
        Real-world bbox size (diagnostic / UE-side sanity check).
    glb_extent : tuple[float, float, float]
        The generated mesh's own normalized bbox size (diagnostic).
    """
    label: str
    world_centroid: tuple[float, float, float]
    scale_ratio: float
    orientation: str = "unsolved"
    world_extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    glb_extent: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "world_centroid": list(self.world_centroid),
            "scale_ratio": self.scale_ratio,
            "orientation": self.orientation,
            "world_extent": list(self.world_extent),
            "glb_extent": list(self.glb_extent),
        }


def _as3(v: Sequence[float] | None) -> tuple[float, float, float]:
    if not v or len(v) < 3:
        return (0.0, 0.0, 0.0)
    return (float(v[0]), float(v[1]), float(v[2]))


def solve_placement(
    label: str,
    world_centroid: Sequence[float],
    world_extent: Sequence[float],
    glb_extent: Sequence[float] | None,
) -> Placement:
    """Solve position + uniform scale for one generated object.

    Parameters
    ----------
    label : str
        Object label.
    world_centroid : sequence of 3 floats
        Centroid of the object's Gaussian subset, COLMAP-world coords.
    world_extent : sequence of 3 floats
        Bounding-box size of that subset in COLMAP-world coords (real size).
    glb_extent : sequence of 3 floats, optional
        Bounding-box size of the generated mesh in its normalized frame. When
        unknown/degenerate, the scale defaults to 1.0 (place at position only)
        rather than guessing — a wrong scale is worse than an unscaled asset.

    Returns
    -------
    Placement
    """
    wc = _as3(world_centroid)
    we = _as3(world_extent)
    ge = _as3(glb_extent)

    world_max = max(we)
    glb_max = max(ge)
    if world_max <= 0.0 or glb_max <= 0.0:
        # No usable size on one side — position only, unit scale, flagged.
        return Placement(label=label, world_centroid=wc, scale_ratio=1.0,
                         orientation="unsolved", world_extent=we, glb_extent=ge)

    return Placement(
        label=label,
        world_centroid=wc,
        scale_ratio=world_max / glb_max,
        orientation="unsolved",
        world_extent=we,
        glb_extent=ge,
    )


def build_placements(meshes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Solve placements for every generator mesh that carries placement hints.

    ``meshes`` is the mesh_objects artifact list. Only entries with a
    ``placement`` block (generator assets — environment meshes are already in
    world coordinates) are solved. Returns ``{label: placement_dict}``, the
    schema the standalone assembler reads from ``usd/placements.json``.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in meshes:
        placement = m.get("placement")
        if not placement:
            continue
        label = m.get("label", "object")
        p = solve_placement(
            label,
            placement.get("centroid", [0, 0, 0]),
            placement.get("extent", [0, 0, 0]),
            m.get("glb_extent"),
        )
        d = p.to_dict()

        # R10 orientation solve (PRD v4): when the crop's COLMAP camera pose is
        # recorded, solve the upright yaw so the object faces the direction it
        # was observed from. Additive + best-effort — a bad/absent pose leaves
        # orientation "unsolved" (identity), which the assembler already handles.
        pose = m.get("camera_pose")
        if pose and pose.get("quaternion_wxyz") and pose.get("translation"):
            try:
                from pipeline.object_orientation import solve_yaw
                o = solve_yaw(pose["quaternion_wxyz"], pose["translation"],
                              object_centroid=placement.get("centroid"))
                if o.get("method") != "degenerate":
                    d["orientation"] = "yaw-solved"
                    d["orientation_quat"] = o["quat_wxyz"]
                    d["orientation_yaw_deg"] = o["yaw_deg"]
                    d["orientation_method"] = o["method"]
                    d["orientation_elevation_deg"] = o.get("elevation_deg")
            except (ValueError, KeyError, ImportError):
                pass  # stay "unsolved"

        out[label] = d
    return out
