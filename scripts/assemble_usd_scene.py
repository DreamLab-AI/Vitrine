#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Standalone USD scene assembler. Run with Python 3.12 + usd-core.

Reads a job directory produced by the LichtFeld pipeline and assembles a
proper Blender-loadable USD scene with textured meshes, cameras, and a full
scene graph hierarchy.

Usage:
    python3 scripts/assemble_usd_scene.py \
        --job-dir /data/output/JOB_ID \
        --output /data/output/JOB_ID/usd/scene.usda

The script looks for:
  - colmap/undistorted/sparse/0/  (binary or text COLMAP reconstruction)
  - colmap/exported/              (fallback text-format COLMAP)
  - objects/meshes/*/*.obj        (per-object meshes)
  - full_scene_textured.obj       (whole-scene mesh)
  - *.ply                         (trained Gaussian splat)
  - full_scene_diffuse.png        (diffuse texture)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  COLMAP parsing (self-contained — no project imports needed)
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"^\s*#")

CAMERA_MODEL_PARAMS: Dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

# Map COLMAP camera model IDs to names (for binary parsing)
CAMERA_MODEL_IDS: Dict[int, str] = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "SIMPLE_RADIAL_FISHEYE",
    8: "RADIAL_FISHEYE",
    9: "THIN_PRISM_FISHEYE",
}


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]

    @property
    def focal_x(self) -> float:
        if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            return self.params[0]
        return self.params[0]

    @property
    def focal_y(self) -> float:
        if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            return self.params[1]
        return self.params[0]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str

    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)

    @property
    def translation(self) -> Tuple[float, float, float]:
        return (self.tx, self.ty, self.tz)


def _iter_data_lines(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or _COMMENT_RE.match(stripped):
                continue
            yield stripped


def parse_cameras_txt(filepath: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    for line in _iter_data_lines(filepath):
        tokens = line.split()
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = tuple(float(t) for t in tokens[4:])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id, model=model,
            width=width, height=height, params=params,
        )
    return cameras


def parse_images_txt(filepath: Path) -> List[ColmapImage]:
    images: List[ColmapImage] = []
    data_lines = list(_iter_data_lines(filepath))
    idx = 0
    while idx < len(data_lines):
        tokens = data_lines[idx].split()
        if len(tokens) < 10:
            idx += 1
            continue
        images.append(ColmapImage(
            image_id=int(tokens[0]),
            qw=float(tokens[1]), qx=float(tokens[2]),
            qy=float(tokens[3]), qz=float(tokens[4]),
            tx=float(tokens[5]), ty=float(tokens[6]),
            tz=float(tokens[7]),
            camera_id=int(tokens[8]), name=tokens[9],
        ))
        idx += 2
    images.sort(key=lambda img: img.image_id)
    return images


# ---------------------------------------------------------------------------
#  COLMAP binary parsers
# ---------------------------------------------------------------------------

def parse_cameras_bin(filepath: Path) -> Dict[int, ColmapCamera]:
    """Parse COLMAP cameras.bin (binary format)."""
    cameras: Dict[int, ColmapCamera] = {}
    with open(filepath, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiii", f.read(16))
            # Convert unsigned width/height (stored as int but always positive)
            width = width & 0xFFFFFFFF
            height = height & 0xFFFFFFFF
            model_name = CAMERA_MODEL_IDS.get(model_id, "PINHOLE")
            num_params = CAMERA_MODEL_PARAMS.get(model_name, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id, model=model_name,
                width=width, height=height, params=params,
            )
    return cameras


def parse_images_bin(filepath: Path) -> List[ColmapImage]:
    """Parse COLMAP images.bin (binary format)."""
    images: List[ColmapImage] = []
    with open(filepath, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            # Read null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="replace")
            # Skip 2D points: num_points2d, then each point is (x, y, point3d_id)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # each: 2 doubles + 1 int64 = 24 bytes
            images.append(ColmapImage(
                image_id=image_id, qw=qw, qx=qx, qy=qy, qz=qz,
                tx=tx, ty=ty, tz=tz, camera_id=camera_id, name=name,
            ))
    images.sort(key=lambda img: img.image_id)
    return images


def convert_colmap_binary_to_text(sparse_dir: Path, out_dir: Path) -> bool:
    """Try to convert binary COLMAP to text using colmap CLI."""
    colmap_bin = None
    for candidate in ["/usr/local/bin/colmap", "/usr/bin/colmap", "colmap"]:
        try:
            subprocess.run([candidate, "--help"], capture_output=True, timeout=5)
            colmap_bin = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if colmap_bin is None:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [colmap_bin, "model_converter",
             "--input_path", str(sparse_dir),
             "--output_path", str(out_dir),
             "--output_type", "TXT"],
            capture_output=True, timeout=60, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def load_colmap_data(
    sparse_dir: Path,
) -> Tuple[Dict[int, ColmapCamera], List[ColmapImage]]:
    """Load COLMAP cameras and images from text or binary format.

    Tries text first, then binary, then attempts conversion.
    """
    # Try text format
    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    if cameras_txt.exists() and images_txt.exists():
        print(f"  Loading COLMAP text format from {sparse_dir}")
        return parse_cameras_txt(cameras_txt), parse_images_txt(images_txt)

    # Try binary format directly
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    if cameras_bin.exists() and images_bin.exists():
        print(f"  Loading COLMAP binary format from {sparse_dir}")
        return parse_cameras_bin(cameras_bin), parse_images_bin(images_bin)

    # Try converting binary to text
    if cameras_bin.exists():
        txt_dir = sparse_dir / "_txt_export"
        if convert_colmap_binary_to_text(sparse_dir, txt_dir):
            return parse_cameras_txt(txt_dir / "cameras.txt"), parse_images_txt(txt_dir / "images.txt")
        # Fall back to direct binary parse
        if images_bin.exists():
            return parse_cameras_bin(cameras_bin), parse_images_bin(images_bin)

    raise FileNotFoundError(
        f"No COLMAP reconstruction found in {sparse_dir}. "
        f"Expected cameras.txt/images.txt or cameras.bin/images.bin"
    )


# ---------------------------------------------------------------------------
#  Coordinate transforms (COLMAP RDF -> USD Y-up)
# ---------------------------------------------------------------------------

SCENE_SCALE: float = 0.5


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_rotation_matrix(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


_RDF_TO_YUP_QUAT = (0.0, 1.0, 0.0, 0.0)


def colmap_to_usd_position(tx, ty, tz, apply_scene_scale=True):
    scale = SCENE_SCALE if apply_scene_scale else 1.0
    return (tx * scale, -ty * scale, -tz * scale)


def colmap_to_usd_rotation(qw, qx, qy, qz):
    return _quat_multiply(_RDF_TO_YUP_QUAT, (qw, qx, qy, qz))


def colmap_camera_world_position(image: ColmapImage) -> Tuple[float, float, float]:
    R = _quat_to_rotation_matrix(image.quaternion)
    tx, ty, tz = image.translation
    cx = -(R[0][0] * tx + R[1][0] * ty + R[2][0] * tz)
    cy = -(R[0][1] * tx + R[1][1] * ty + R[2][1] * tz)
    cz = -(R[0][2] * tx + R[1][2] * ty + R[2][2] * tz)
    return (cx, cy, cz)


def build_usd_transform_from_colmap(image: ColmapImage, apply_scene_scale=True):
    cx, cy, cz = colmap_camera_world_position(image)
    ux, uy, uz = colmap_to_usd_position(cx, cy, cz, apply_scene_scale=apply_scene_scale)
    usd_q = colmap_to_usd_rotation(*image.quaternion)
    R = _quat_to_rotation_matrix(usd_q)
    return [
        [R[0][0], R[0][1], R[0][2], 0.0],
        [R[1][0], R[1][1], R[1][2], 0.0],
        [R[2][0], R[2][1], R[2][2], 0.0],
        [ux, uy, uz, 1.0],
    ]


# ---------------------------------------------------------------------------
#  USD scene builder
# ---------------------------------------------------------------------------

def _sanitize_prim_name(name: str) -> str:
    sanitized = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            sanitized += ch
        else:
            sanitized += "_"
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"_{sanitized}"
    return sanitized


def _find_colmap_dir(job_dir: Path) -> Optional[Path]:
    """Search for COLMAP reconstruction data in the job directory."""
    candidates = [
        job_dir / "colmap" / "undistorted" / "sparse" / "0",
        job_dir / "colmap" / "undistorted" / "sparse",
        job_dir / "colmap" / "sparse" / "0",
        job_dir / "colmap" / "sparse",
        job_dir / "colmap" / "exported",
        job_dir / "colmap",
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        if (d / "cameras.txt").exists() or (d / "cameras.bin").exists():
            return d
    return None


def _find_meshes(job_dir: Path) -> List[Dict[str, Any]]:
    """Find all mesh files in the job directory."""
    meshes: List[Dict[str, Any]] = []

    # Per-object meshes: objects/meshes/*/*.{obj,glb}
    obj_meshes_dir = job_dir / "objects" / "meshes"
    if obj_meshes_dir.is_dir():
        for sub in sorted(obj_meshes_dir.iterdir()):
            if sub.is_dir():
                # Prefer GLB over OBJ (GLB embeds textures)
                mesh_files = sorted(sub.glob("*.glb")) or sorted(sub.glob("*.obj"))
                for mesh_file in mesh_files:
                    label = sub.name
                    # Look for accompanying texture
                    diffuse = None
                    for ext in (".png", ".jpg", ".jpeg"):
                        for pattern in [f"{mesh_file.stem}_diffuse{ext}", f"diffuse{ext}",
                                        f"{mesh_file.stem}{ext}", f"texture{ext}"]:
                            tex_candidate = sub / pattern
                            if tex_candidate.exists():
                                diffuse = str(tex_candidate)
                                break
                        if diffuse:
                            break
                    meshes.append({
                        "label": label,
                        "mesh": str(mesh_file),
                        "diffuse": diffuse,
                    })

    # Full scene mesh
    for name in ["full_scene_textured.obj", "full_scene.obj"]:
        full_scene = job_dir / name
        if full_scene.exists():
            diffuse = None
            for ext in (".png", ".jpg", ".jpeg"):
                for tex_name in ["full_scene_diffuse", "diffuse", "texture"]:
                    tex = job_dir / f"{tex_name}{ext}"
                    if tex.exists():
                        diffuse = str(tex)
                        break
                if diffuse:
                    break
            meshes.append({
                "label": "full_scene",
                "mesh": str(full_scene),
                "diffuse": diffuse,
            })
            break

    return meshes


def _find_ply(job_dir: Path) -> Optional[Path]:
    """Find a Gaussian splat PLY file in the job directory."""
    candidates = [
        job_dir / "scene.ply",
        job_dir / "point_cloud.ply",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Search more broadly
    plys = sorted(job_dir.glob("**/*.ply"))
    if plys:
        return plys[0]
    return None


def _compute_obj_centroid(obj_path: Path) -> Tuple[float, float, float]:
    """Compute the centroid of an OBJ file from its vertices."""
    sx, sy, sz, count = 0.0, 0.0, 0.0, 0
    try:
        with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        sx += float(parts[1])
                        sy += float(parts[2])
                        sz += float(parts[3])
                        count += 1
    except (OSError, ValueError):
        pass
    if count == 0:
        return (0.0, 0.0, 0.0)
    return (sx / count, sy / count, sz / count)


def _inline_mesh_geometry(stage, mesh_prim_path: str, mesh_file: Path) -> bool:
    """Load an OBJ/GLB mesh via trimesh and write it as UsdGeom.Mesh.

    Returns True if the mesh was successfully inlined, False on failure.
    """
    try:
        import numpy as np
        import trimesh
    except ImportError:
        return False

    if not mesh_file.exists():
        return False

    try:
        scene_or_mesh = trimesh.load(str(mesh_file), force=None)
    except Exception:
        return False

    # Flatten to a single mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return False
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        return False

    from pxr import Gf, Sdf, UsdGeom, Vt

    usd_mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

    # Vertices
    points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray(points))

    # Face topology
    faces = mesh.faces
    face_vertex_counts = [3] * len(faces)
    face_vertex_indices = faces.flatten().tolist()
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_vertex_counts))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_vertex_indices))

    # Normals (if available)
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        normals = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh.vertex_normals]
        usd_mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # UVs (if available via trimesh visual)
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        uvs = mesh.visual.uv
        uv_primvar = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv_values = [Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs]
        uv_primvar.Set(uv_values)

    # Extent
    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)
    usd_mesh.CreateExtentAttr([
        Gf.Vec3f(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])),
        Gf.Vec3f(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])),
    ])

    usd_mesh.GetPrim().SetCustomDataByKey("mesh:source_file", str(mesh_file))
    usd_mesh.GetPrim().SetCustomDataByKey("mesh:vertex_count", len(mesh.vertices))
    usd_mesh.GetPrim().SetCustomDataByKey("mesh:face_count", len(mesh.faces))

    return True


def assemble_scene(
    job_dir: Path,
    output_path: Path,
) -> None:
    """Build a complete USD scene from a pipeline job directory."""
    from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    print(f"Assembling USD scene from job dir: {job_dir}")
    print(f"Output: {output_path}")

    # ---- Discover data ----

    # COLMAP cameras
    colmap_dir = _find_colmap_dir(job_dir)
    cameras: Dict[int, ColmapCamera] = {}
    images: List[ColmapImage] = []
    if colmap_dir:
        try:
            cameras, images = load_colmap_data(colmap_dir)
            print(f"  Found {len(cameras)} camera(s), {len(images)} image(s)")
        except Exception as e:
            print(f"  Warning: Failed to load COLMAP data: {e}")
    else:
        print("  No COLMAP reconstruction found")

    # Meshes
    meshes = _find_meshes(job_dir)
    print(f"  Found {len(meshes)} mesh(es)")

    # R10 pose-solve (PRD v4): per-object world placement + uniform scale for
    # generated objects, written by the pipeline's assemble_usd stage. Absent
    # for legacy runs -> fall back to the mesh-self-centroid placement below.
    placements: Dict[str, Dict[str, Any]] = {}
    placements_path = job_dir / "usd" / "placements.json"
    if placements_path.exists():
        try:
            placements = json.loads(placements_path.read_text())
            print(f"  Loaded R10 placements for {len(placements)} object(s)")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read placements.json: {e}")

    # Gaussian PLY
    ply_path = _find_ply(job_dir)
    if ply_path:
        print(f"  Found PLY: {ply_path}")

    # ---- Create USD stage ----

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove existing file if present (Usd.Stage.CreateNew fails if file exists)
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))

    # Stage metrics
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World"))

    # ---- Build hierarchy ----

    world = stage.DefinePrim("/World", "Xform")
    Usd.ModelAPI(world).SetKind(Kind.Tokens.assembly)

    stage.DefinePrim("/World/Environment", "Xform")
    stage.DefinePrim("/World/Environment/Background", "Xform")
    stage.DefinePrim("/World/Objects", "Xform")
    stage.DefinePrim("/World/Cameras", "Xform")
    materials_scope = stage.DefinePrim("/World/Materials", "Scope")

    # Dome light
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Background/DomeLight")
    dome.CreateIntensityAttr(1.0)

    # ---- Reference background PLY ----

    if ply_path:
        bg_xform = UsdGeom.Xform.Define(stage, "/World/Environment/Background/GaussianSplat")
        bg_prim = bg_xform.GetPrim()
        # Store PLY path as custom data (PLY can't be directly referenced in USD,
        # but Blender/viewers can use this metadata)
        bg_prim.SetCustomDataByKey("gaussian:ply_path", str(ply_path))
        bg_prim.SetCustomDataByKey("gaussian:format", "3dgs")
        print(f"  Referenced PLY: {ply_path}")

    # ---- Place objects ----

    for idx, mesh_info in enumerate(meshes):
        label = mesh_info.get("label", f"object_{idx}")
        safe_name = _sanitize_prim_name(label)
        mesh_path_str = mesh_info.get("mesh", "")
        diffuse_path = mesh_info.get("diffuse")
        obj_path = f"/World/Objects/{safe_name}"

        xform = UsdGeom.Xform.Define(stage, obj_path)
        prim = xform.GetPrim()
        Usd.ModelAPI(prim).SetKind(Kind.Tokens.component)

        # Positioning: prefer the R10 solved placement (real world position +
        # uniform scale of a normalized generator mesh); else fall back to the
        # mesh-self centroid (correct for world-frame OBJ meshes like the
        # environment, meaningless for a normalized generator GLB).
        placement = placements.get(label)
        if placement is not None:
            wc = placement.get("world_centroid", [0.0, 0.0, 0.0])
            ux, uy, uz = colmap_to_usd_position(wc[0], wc[1], wc[2])
            xform.AddTranslateOp().Set(Gf.Vec3d(ux, uy, uz))
            # R10 orientation: apply the solved upright yaw (TRS order; a pure-Y
            # rotation commutes with uniform scale anyway). Absent/unsolved ->
            # identity (no op), matching legacy placements.json.
            oq = placement.get("orientation_quat")
            if placement.get("orientation") == "yaw-solved" and oq and len(oq) == 4:
                xform.AddOrientOp().Set(Gf.Quatf(oq[0], oq[1], oq[2], oq[3]))
                prim.SetCustomDataByKey("v2g:orientation_yaw_deg",
                                        float(placement.get("orientation_yaw_deg", 0.0)))
                prim.SetCustomDataByKey("v2g:orientation_method",
                                        placement.get("orientation_method", ""))
            # scale_ratio is in raw world units; USD positions carry SCENE_SCALE,
            # so the object's size must be scaled by the same factor to match.
            s = float(placement.get("scale_ratio", 1.0)) * SCENE_SCALE
            xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            prim.SetCustomDataByKey("v2g:placement_orientation",
                                    placement.get("orientation", "unsolved"))
            prim.SetCustomDataByKey("v2g:placement_scale_ratio",
                                    float(placement.get("scale_ratio", 1.0)))
        elif mesh_path_str and Path(mesh_path_str).exists():
            cx, cy, cz = _compute_obj_centroid(Path(mesh_path_str))
            ux, uy, uz = colmap_to_usd_position(cx, cy, cz)
            xform.AddTranslateOp().Set(Gf.Vec3d(ux, uy, uz))

        # Variant set: gaussian vs mesh
        vset = prim.GetVariantSets().AddVariantSet("representation")

        if ply_path:
            vset.AddVariant("gaussian")
            vset.SetVariantSelection("gaussian")
            with vset.GetVariantEditContext():
                gs_prim = stage.DefinePrim(f"{obj_path}/GaussianData", "Xform")
                gs_prim.SetCustomDataByKey("gaussian:ply_path", str(ply_path))

        if mesh_path_str:
            vset.AddVariant("mesh")
            vset.SetVariantSelection("mesh")
            with vset.GetVariantEditContext():
                mesh_data_path = f"{obj_path}/MeshData"
                # Inline mesh geometry into the USD stage using trimesh
                inlined = _inline_mesh_geometry(
                    stage, mesh_data_path, Path(mesh_path_str),
                )
                if not inlined:
                    # Fallback: create Xform with path metadata
                    ref_prim = stage.DefinePrim(mesh_data_path, "Xform")
                    ref_prim.SetCustomDataByKey("mesh:source_path", mesh_path_str)

        # Default to mesh variant (Blender can load mesh, not raw PLY gaussians)
        vset.SetVariantSelection("mesh" if mesh_path_str else "gaussian")

        # Material with UsdPreviewSurface
        mat_prim_path = f"/World/Materials/{safe_name}_mat"
        material = UsdShade.Material.Define(stage, mat_prim_path)

        shader_path = f"{mat_prim_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        if diffuse_path and Path(diffuse_path).exists():
            # Texture-mapped material
            try:
                rel_tex = os.path.relpath(diffuse_path, output_path.parent)
            except ValueError:
                rel_tex = diffuse_path

            tex_reader_path = f"{mat_prim_path}/DiffuseTexture"
            tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
            tex_reader.CreateIdAttr("UsdUVTexture")
            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel_tex)
            tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(0.8, 0.8, 0.8, 1.0)
            )
            tex_output = tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f,
            ).ConnectToSource(tex_output)

            # UV reader
            uv_path = f"{mat_prim_path}/UVReader"
            uv_reader = UsdShade.Shader.Define(stage, uv_path)
            uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
            uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv_output = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv_output
            )
            print(f"  Object '{label}': textured material ({rel_tex})")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.8, 0.8, 0.8)
            )
            print(f"  Object '{label}': solid material")

        # Bind material to mesh data prim
        mesh_data_prim = stage.GetPrimAtPath(f"{obj_path}/MeshData")
        if mesh_data_prim and mesh_data_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(mesh_data_prim).Bind(material)

        # Store mesh path as custom data
        prim.SetCustomDataByKey("lichtfeld:mesh_path", mesh_path_str)
        if diffuse_path:
            prim.SetCustomDataByKey("lichtfeld:diffuse_path", diffuse_path)

    # ---- Place cameras ----

    for img in images:
        cam_intrinsic = cameras.get(img.camera_id)
        if cam_intrinsic is None:
            continue

        safe_name = _sanitize_prim_name(
            Path(img.name).stem if img.name else f"cam_{img.image_id:04d}"
        )
        cam_path = f"/World/Cameras/{safe_name}"

        cam = UsdGeom.Camera.Define(stage, cam_path)

        # Transform (COLMAP -> USD)
        xf_matrix = build_usd_transform_from_colmap(img)
        xform_op = cam.AddTransformOp()
        xform_op.Set(Gf.Matrix4d(
            xf_matrix[0][0], xf_matrix[0][1], xf_matrix[0][2], xf_matrix[0][3],
            xf_matrix[1][0], xf_matrix[1][1], xf_matrix[1][2], xf_matrix[1][3],
            xf_matrix[2][0], xf_matrix[2][1], xf_matrix[2][2], xf_matrix[2][3],
            xf_matrix[3][0], xf_matrix[3][1], xf_matrix[3][2], xf_matrix[3][3],
        ))

        # Intrinsics: USD focal length in mm, 36mm horizontal aperture
        horizontal_aperture_mm = 36.0
        focal_mm = (cam_intrinsic.focal_x / cam_intrinsic.width) * horizontal_aperture_mm
        vertical_aperture_mm = horizontal_aperture_mm * (cam_intrinsic.height / cam_intrinsic.width)

        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        cam.CreateVerticalApertureAttr(vertical_aperture_mm)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

        # COLMAP metadata
        cam_prim = cam.GetPrim()
        cam_prim.SetCustomDataByKey("colmap:camera_id", img.camera_id)
        cam_prim.SetCustomDataByKey("colmap:image_id", img.image_id)
        cam_prim.SetCustomDataByKey("colmap:image_name", img.name)
        cam_prim.SetCustomDataByKey("colmap:model", cam_intrinsic.model)
        cam_prim.SetCustomDataByKey("colmap:width", cam_intrinsic.width)
        cam_prim.SetCustomDataByKey("colmap:height", cam_intrinsic.height)

    print(f"  Placed {len(images)} camera(s)")

    # ---- Scene metadata ----

    world.SetCustomDataByKey("lichtfeld:pipeline_version", "1.0")
    world.SetCustomDataByKey("lichtfeld:up_axis", "Y")
    world.SetCustomDataByKey("lichtfeld:meters_per_unit", 1.0)
    world.SetCustomDataByKey("lichtfeld:object_count", len(meshes))
    world.SetCustomDataByKey("lichtfeld:camera_count", len(images))
    world.SetCustomDataByKey("lichtfeld:job_dir", str(job_dir))
    if ply_path:
        world.SetCustomDataByKey("lichtfeld:gaussian_ply", str(ply_path))

    # ---- Save ----

    stage.GetRootLayer().Save()

    # Print summary
    prim_count = sum(1 for _ in stage.Traverse())
    cam_count = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Camera")
    file_size = output_path.stat().st_size
    print(f"\nScene assembled:")
    print(f"  Prims: {prim_count}")
    print(f"  Cameras: {cam_count}")
    print(f"  Objects: {len(meshes)}")
    print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Output: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a USD scene from a LichtFeld pipeline job directory"
    )
    parser.add_argument(
        "--job-dir", type=Path, required=True,
        help="Path to the pipeline job output directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output .usda file path",
    )
    args = parser.parse_args()

    if not args.job_dir.is_dir():
        print(f"Error: job directory does not exist: {args.job_dir}", file=sys.stderr)
        return 1

    try:
        assemble_scene(args.job_dir, args.output)
        return 0
    except Exception as e:
        print(f"Error assembling scene: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
