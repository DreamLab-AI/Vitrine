"""TSDF-based mesh extraction from Gaussian splat objects.

Supports two rendering backends:
- gsplat: GPU-accelerated gaussian splatting for high-quality depth+color (preferred)
- LichtFeld MCP: subprocess-based rendering via the studio's MCP server (legacy)

Pipeline: render depth from multiple views -> TSDF fusion -> marching cubes -> mesh
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from skimage.measure import marching_cubes
import trimesh

from .mesh_cleaner import MeshCleaner

logger = logging.getLogger(__name__)

LFS_MCP = "/usr/local/bin/lfs-mcp"


# ---------------------------------------------------------------------------
# gsplat PLY loader
# ---------------------------------------------------------------------------

def load_3dgs_ply(path: str) -> dict:
    """Load a 3DGS PLY file into gsplat-ready tensors on CUDA.

    Expects the standard 3DGS format with properties:
        x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3

    Returns a dict with keys: means, scales, quats, opacities, sh_coeffs,
    colors_dc (numpy RGB 0-1), count.
    """
    import torch
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply['vertex']
    n = len(v['x'])

    means = torch.tensor(
        np.column_stack([v['x'], v['y'], v['z']]),
        dtype=torch.float32, device='cuda',
    )

    # Scales stored as log-space; apply exp
    scales = torch.exp(torch.tensor(
        np.column_stack([v['scale_0'], v['scale_1'], v['scale_2']]),
        dtype=torch.float32, device='cuda',
    ))

    # Quaternions wxyz; normalize
    quats = torch.tensor(
        np.column_stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]),
        dtype=torch.float32, device='cuda',
    )
    quats = quats / quats.norm(dim=1, keepdim=True)

    # Opacities stored as logit; apply sigmoid
    opacities = torch.sigmoid(torch.tensor(
        np.array(v['opacity'], dtype=np.float32),
        dtype=torch.float32, device='cuda',
    ))

    # SH coefficients: DC (3) + rest. The PLY may carry fewer SH bands than
    # degree 3 — e.g. LichtFeld's indoor preset trains at SH degree 1, giving 9
    # f_rest fields, not 45 — so the old hardcoded range(45) raised
    # "no field of name f_rest_9". Count what's present and zero-pad the higher
    # bands up to degree 3 (15 rest bands) so the gsplat sh_degree=3 path holds.
    sh_dc = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).astype(np.float32)
    n_rest = sum(1 for p in v.properties if p.name.startswith('f_rest_'))
    rest_bands = n_rest // 3
    if rest_bands:
        sh_rest = np.column_stack(
            [v[f'f_rest_{i}'] for i in range(n_rest)]
        ).astype(np.float32)
        sh_rest_r = sh_rest.reshape(n, rest_bands, 3)
    else:
        sh_rest_r = np.zeros((n, 0, 3), dtype=np.float32)
    if rest_bands < 15:
        pad = np.zeros((n, 15 - rest_bands, 3), dtype=np.float32)
        sh_rest_r = np.concatenate([sh_rest_r, pad], axis=1)
    sh_dc_r = sh_dc.reshape(n, 1, 3)
    sh_all = np.concatenate([sh_dc_r, sh_rest_r], axis=1)  # [N, 16, 3]
    sh_coeffs = torch.tensor(sh_all, dtype=torch.float32, device='cuda')

    # Vertex colors from SH DC band (C0 = 0.2820948)
    colors_dc = np.clip(0.5 + 0.2820948 * sh_dc, 0.0, 1.0)

    logger.info("Loaded 3DGS PLY: %d gaussians from %s", n, path)
    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'sh_coeffs': sh_coeffs,
        'colors_dc': colors_dc,
        'count': n,
    }


# ---------------------------------------------------------------------------
# gsplat rendering
# ---------------------------------------------------------------------------

def render_gsplat(
    gaussians: dict,
    viewmat: 'torch.Tensor',
    K: 'torch.Tensor',
    width: int,
    height: int,
    sh_degree: int = 3,
    near_plane: float = 0.01,
    far_plane: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB + expected-depth + alpha using gsplat rasterization.

    Args:
        gaussians: Dict from load_3dgs_ply.
        viewmat: [4,4] world-to-camera transform (float32, CUDA).
        K: [3,3] camera intrinsics matrix (float32, CUDA).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        (depth_map, color_map, alpha_map) as numpy float32 arrays.
        depth_map: [H,W], color_map: [H,W,3], alpha_map: [H,W].
        Invalid depth (alpha < 0.5) is set to 0.
    """
    import torch
    from gsplat import rasterization

    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=gaussians['means'],
            quats=gaussians['quats'],
            scales=gaussians['scales'],
            opacities=gaussians['opacities'],
            colors=gaussians['sh_coeffs'],
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            render_mode='RGB+ED',
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=True,
            rasterize_mode='antialiased',
        )

    rgb = renders[0, :, :, :3].cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)  # Clamp SH color to valid range
    depth = renders[0, :, :, 3].cpu().numpy()
    alpha = alphas[0, :, :, 0].cpu().numpy()

    # Zero out depth where alpha is too low (empty regions)
    # Keep threshold low to preserve thin structures; use alpha as
    # confidence weight during TSDF integration instead of binary gate
    depth[alpha < 0.1] = 0.0

    return depth, rgb, alpha


# ---------------------------------------------------------------------------
# Camera generation from gaussian distribution
# ---------------------------------------------------------------------------

def generate_orbit_cameras_gsplat(
    means: 'torch.Tensor',
    num_views: int,
    image_size: int,
    focal_factor: float = 0.8,
    distance_factor: float = 2.5,
) -> list[tuple['torch.Tensor', 'torch.Tensor']]:
    """Generate orbit cameras covering the scene based on gaussian positions.

    Uses Fibonacci sphere sampling for near-uniform coverage.

    Args:
        means: [N,3] gaussian positions tensor (CUDA).
        num_views: Number of camera viewpoints.
        image_size: Render resolution (square).
        focal_factor: Focal length as fraction of image size.
        distance_factor: Camera distance as multiple of scene extent.

    Returns:
        List of (viewmat_4x4, K_3x3) tensor pairs on CUDA.
    """
    import torch

    pts = means.cpu().numpy()
    centroid = pts.mean(axis=0)
    p5, p95 = np.percentile(pts, [5, 95], axis=0)
    extent = np.max(p95 - p5)
    distance = extent * distance_factor

    focal = image_size * focal_factor
    K = torch.tensor([
        [focal, 0.0, image_size / 2.0],
        [0.0, focal, image_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    cameras = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    for i in range(num_views):
        # Fibonacci sphere
        theta = np.arccos(1.0 - 2.0 * (i + 0.5) / num_views)
        phi = 2.0 * np.pi * i / golden_ratio

        pos = centroid + distance * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], dtype=np.float64)

        # Look-at: camera looks toward centroid
        forward = centroid - pos
        forward = forward / (np.linalg.norm(forward) + 1e-12)

        # Choose up vector that is not parallel to forward
        up_world = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, up_world)) > 0.99:
            up_world = np.array([0.0, 1.0, 0.0])

        right = np.cross(forward, up_world)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-12)

        # Build world-to-camera matrix (COLMAP convention: +Z forward)
        # R columns = [right, -up, forward] in world space
        # viewmat = [R^T | -R^T @ pos]
        R = np.stack([right, -up, forward], axis=0)  # 3x3, rows
        t = -R @ pos

        viewmat = torch.eye(4, dtype=torch.float32, device='cuda')
        viewmat[:3, :3] = torch.tensor(R, dtype=torch.float32, device='cuda')
        viewmat[:3, 3] = torch.tensor(t, dtype=torch.float32, device='cuda')

        cameras.append((viewmat, K))

    return cameras


# ---------------------------------------------------------------------------
# COLMAP camera loading — for TSDF from training viewpoints
# ---------------------------------------------------------------------------

def load_colmap_cameras(
    colmap_dir: str,
    render_size: int = 1024,
) -> list[tuple['torch.Tensor', 'torch.Tensor']]:
    """Load COLMAP camera poses as gsplat-compatible (viewmat, K) pairs.

    Uses the undistorted COLMAP model to get camera intrinsics and extrinsics.
    Returns cameras suitable for render_gsplat() and TSDF integration.

    Args:
        colmap_dir: Path to COLMAP output (containing sparse/0/ or undistorted/sparse/0/).
        render_size: Render resolution (cameras are rescaled to this).

    Returns:
        List of (viewmat_4x4, K_3x3) tensor pairs on CUDA.
    """
    import torch
    from pathlib import Path

    colmap_path = Path(colmap_dir)

    # Find the sparse model — try undistorted first, then sparse/0
    for candidate in [
        colmap_path / "undistorted" / "sparse" / "0",
        colmap_path / "sparse" / "0",
        colmap_path / "sparse",
        colmap_path,
    ]:
        if (candidate / "images.bin").exists() or (candidate / "images.txt").exists():
            model_dir = candidate
            break
    else:
        logger.warning("No COLMAP model found in %s, falling back to orbit cameras", colmap_dir)
        return []

    # Read cameras (intrinsics)
    cameras_file = model_dir / "cameras.bin"
    cameras_txt = model_dir / "cameras.txt"

    fx, fy, cx, cy = None, None, None, None
    img_w, img_h = render_size, render_size

    if cameras_txt.exists():
        with open(cameras_txt) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    model = parts[1]
                    img_w, img_h = int(parts[2]), int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
                        fx = params[0]
                        fy = params[1] if len(params) > 1 and model == "PINHOLE" else params[0]
                        cx = params[-2] if model == "PINHOLE" else img_w / 2
                        cy = params[-1] if model == "PINHOLE" else img_h / 2
                    elif model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
                        fx = params[0]
                        fy = params[1] if model == "OPENCV" else params[0]
                        cx = params[1] if model != "OPENCV" else params[2]
                        cy = params[2] if model != "OPENCV" else params[3]
                    break
    elif cameras_file.exists():
        import struct
        with open(cameras_file, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            if num_cameras > 0:
                cam_id = struct.unpack("<i", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                img_w = struct.unpack("<Q", f.read(8))[0]
                img_h = struct.unpack("<Q", f.read(8))[0]
                # Model 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), 2=SIMPLE_RADIAL(4), etc.
                n_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
                params = struct.unpack(f"<{n_params}d", f.read(n_params * 8))
                fx = params[0]
                fy = params[1] if n_params >= 4 and model_id in (1, 4) else params[0]
                cx = params[-2] if n_params >= 3 else img_w / 2
                cy = params[-1] if n_params >= 3 else img_h / 2

    if fx is None:
        logger.warning("Could not parse COLMAP cameras, falling back to orbit cameras")
        return []

    # Scale to render_size
    scale_x = render_size / img_w
    scale_y = render_size / img_h
    scale = min(scale_x, scale_y)
    fx_s, fy_s = fx * scale, fy * scale
    cx_s, cy_s = cx * scale, cy * scale

    K = torch.tensor([
        [fx_s, 0.0, cx_s],
        [0.0, fy_s, cy_s],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device='cuda')

    # Read images (extrinsics)
    images_txt = model_dir / "images.txt"
    images_bin = model_dir / "images.bin"
    cameras = []

    if images_txt.exists():
        with open(images_txt) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        # images.txt has 2 lines per image: extrinsics then 2D points
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 8:
                continue
            qw, qx, qy, qz = [float(parts[j]) for j in range(1, 5)]
            tx, ty, tz = [float(parts[j]) for j in range(5, 8)]

            # COLMAP quaternion to rotation matrix (qw, qx, qy, qz)
            R = _quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])

            # COLMAP stores world-to-camera: [R | t]
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t
            cameras.append((
                torch.tensor(viewmat, dtype=torch.float32, device='cuda'),
                K,
            ))
    elif images_bin.exists():
        import struct
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                img_id = struct.unpack("<i", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                cam_id = struct.unpack("<i", f.read(4))[0]
                # Read image name (null-terminated)
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name += c
                # Read 2D points
                n_pts = struct.unpack("<Q", f.read(8))[0]
                f.read(n_pts * 24)  # skip point2D data

                R = _quat_to_rotmat(qw, qx, qy, qz)
                t = np.array([tx, ty, tz], dtype=np.float32)
                viewmat = np.eye(4, dtype=np.float32)
                viewmat[:3, :3] = R
                viewmat[:3, 3] = t
                cameras.append((
                    torch.tensor(viewmat, dtype=torch.float32, device='cuda'),
                    K,
                ))

    logger.info("Loaded %d COLMAP cameras from %s (scaled to %d)", len(cameras), model_dir, render_size)
    return cameras


def _quat_to_rotmat(qw, qx, qy, qz):
    """Convert COLMAP quaternion (w,x,y,z) to 3x3 rotation matrix."""
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def default(cls, width: int = 512, height: int = 512, fov_deg: float = 60.0) -> CameraIntrinsics:
        fov_rad = np.radians(fov_deg)
        f = (width / 2.0) / np.tan(fov_rad / 2.0)
        return cls(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


@dataclass
class TSDFConfig:
    """Configuration for TSDF volume and mesh extraction."""
    voxel_size: float = 0.005
    sdf_trunc: float = 0.04
    volume_bounds_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, -1.0]))
    volume_bounds_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    num_viewpoints: int = 36
    render_width: int = 512
    render_height: int = 512
    camera_distance: float = 2.5
    camera_fov_deg: float = 60.0
    target_faces: int = 50000
    smooth_iterations: int = 3
    min_component_ratio: float = 0.005
    min_component_faces: int = 100
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp"


# ---------------------------------------------------------------------------
# TSDF Volume
# ---------------------------------------------------------------------------

class TSDFVolume:
    """Truncated Signed Distance Function volume for depth fusion.

    Implements volumetric TSDF integration and marching-cubes extraction
    without requiring Open3D, using raw numpy arrays.
    """

    def __init__(self, config: TSDFConfig):
        self.config = config
        self.voxel_size = config.voxel_size
        self.sdf_trunc = config.sdf_trunc

        bounds_min = config.volume_bounds_min
        bounds_max = config.volume_bounds_max

        self.origin = bounds_min.copy()
        dims = ((bounds_max - bounds_min) / self.voxel_size).astype(int)
        self.nx, self.ny, self.nz = int(dims[0]), int(dims[1]), int(dims[2])

        logger.info("TSDF volume: %d x %d x %d voxels (%.1fM)",
                     self.nx, self.ny, self.nz,
                     self.nx * self.ny * self.nz / 1e6)

        self.tsdf = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)
        self.weight = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.color = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float32)

    def _voxel_centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute world coordinates of all voxel centers."""
        x = np.arange(self.nx) * self.voxel_size + self.origin[0] + self.voxel_size / 2
        y = np.arange(self.ny) * self.voxel_size + self.origin[1] + self.voxel_size / 2
        z = np.arange(self.nz) * self.voxel_size + self.origin[2] + self.voxel_size / 2
        return x, y, z

    def integrate(
        self,
        depth_map: np.ndarray,
        color_image: np.ndarray,
        intrinsics: Union[CameraIntrinsics, np.ndarray],
        extrinsics: np.ndarray,
        weight: Union[float, np.ndarray] = 1.0,
    ) -> None:
        """Integrate a single depth frame into the TSDF volume.

        Args:
            depth_map: HxW float array of depth values in meters.
            weight: Scalar or HxW per-pixel weight array (e.g. alpha map).
            color_image: HxWx3 uint8 or float array of color values.
            intrinsics: CameraIntrinsics object or 3x3 numpy K matrix.
            extrinsics: 4x4 matrix. If intrinsics is CameraIntrinsics,
                this is camera-to-world (pose). If intrinsics is a 3x3 K
                matrix, this is world-to-camera (viewmat).
            weight: Integration weight for this frame.
        """
        h, w = depth_map.shape[:2]

        # Unify intrinsics to fx, fy, cx, cy
        if isinstance(intrinsics, np.ndarray):
            # Raw K matrix [3x3], extrinsics is world-to-camera viewmat
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
            world_to_cam = extrinsics.astype(np.float64)
        else:
            # CameraIntrinsics object, extrinsics is camera-to-world
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.cx
            cy = intrinsics.cy
            world_to_cam = np.linalg.inv(extrinsics)

        x_coords, y_coords, z_coords = self._voxel_centers()

        color_f = color_image.astype(np.float32)
        if color_f.max() > 1.5:  # Likely uint8 [0,255]
            color_f /= 255.0
        color_f = np.clip(color_f, 0.0, 1.0)  # Ensure valid range

        # Process in slabs along x to manage memory
        slab_size = max(1, min(self.nx, 64))
        for x_start in range(0, self.nx, slab_size):
            x_end = min(x_start + slab_size, self.nx)
            nx_slab = x_end - x_start

            # Build world-space coordinates for this slab
            xx, yy, zz = np.meshgrid(
                x_coords[x_start:x_end], y_coords, z_coords, indexing="ij"
            )
            pts_world = np.stack(
                [xx.ravel(), yy.ravel(), zz.ravel(), np.ones(nx_slab * self.ny * self.nz)],
                axis=0,
            )

            # Transform to camera coordinates
            pts_cam = world_to_cam @ pts_world
            cam_x = pts_cam[0]
            cam_y = pts_cam[1]
            cam_z = pts_cam[2]

            # Skip voxels behind camera
            valid = cam_z > 0
            if not np.any(valid):
                continue

            # Project to image plane
            u = (fx * cam_x / cam_z + cx).astype(np.float64)
            v = (fy * cam_y / cam_z + cy).astype(np.float64)

            # Bounds check
            valid &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            if not np.any(valid):
                continue

            u_int = np.clip(u.astype(int), 0, w - 1)
            v_int = np.clip(v.astype(int), 0, h - 1)

            # Look up depth at projected pixel
            depth_vals = depth_map[v_int, u_int]
            valid &= depth_vals > 0

            # Compute SDF
            sdf = depth_vals - cam_z
            valid &= sdf >= -self.sdf_trunc
            tsdf_val = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

            # Reshape back to slab dimensions
            tsdf_slab = tsdf_val.reshape(nx_slab, self.ny, self.nz)
            valid_slab = valid.reshape(nx_slab, self.ny, self.nz)

            # Look up colors
            col_r = color_f[v_int, u_int, 0].reshape(nx_slab, self.ny, self.nz)
            col_g = color_f[v_int, u_int, 1].reshape(nx_slab, self.ny, self.nz)
            col_b = color_f[v_int, u_int, 2].reshape(nx_slab, self.ny, self.nz)

            # Per-pixel weight: use alpha confidence if weight is an array
            if isinstance(weight, np.ndarray):
                pixel_w = weight[v_int, u_int].reshape(nx_slab, self.ny, self.nz)
            else:
                pixel_w = weight

            # Weighted running average update
            old_w = self.weight[x_start:x_end]
            new_w = old_w + pixel_w * valid_slab.astype(np.float32)
            mask = valid_slab & (new_w > 0)

            old_tsdf = self.tsdf[x_start:x_end]
            self.tsdf[x_start:x_end] = np.where(
                mask,
                (old_w * old_tsdf + pixel_w * tsdf_slab) / np.maximum(new_w, 1e-8),
                old_tsdf,
            )
            for c_idx, col_ch in enumerate([col_r, col_g, col_b]):
                old_c = self.color[x_start:x_end, :, :, c_idx]
                self.color[x_start:x_end, :, :, c_idx] = np.where(
                    mask,
                    (old_w * old_c + pixel_w * col_ch) / np.maximum(new_w, 1e-8),
                    old_c,
                )
            self.weight[x_start:x_end] = new_w

    def extract_mesh(self) -> trimesh.Trimesh:
        """Extract a triangle mesh from the TSDF using marching cubes.

        Returns:
            A trimesh.Trimesh with vertex colors.
        """
        # Require multi-view consensus: voxels seen from 2+ views with confidence
        # Single-view voxels produce noisy surfaces
        observed = self.weight > 2.0
        if not np.any(observed):
            # Fall back to any observed voxel
            observed = self.weight > 0
        if not np.any(observed):
            raise ValueError("TSDF volume is empty -- no depth frames were integrated")

        tsdf_vol = self.tsdf.copy()
        tsdf_vol[~observed] = 1.0  # mark unobserved as outside

        try:
            verts, faces, normals, _ = marching_cubes(
                tsdf_vol,
                level=0.0,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size),
            )
        except ValueError as e:
            raise ValueError(f"Marching cubes failed (surface may not cross zero): {e}") from e

        # Shift vertices to world coordinates
        verts += self.origin

        # Interpolate vertex colors from the volume
        voxel_indices = ((verts - self.origin) / self.voxel_size).astype(int)
        voxel_indices = np.clip(
            voxel_indices,
            [0, 0, 0],
            [self.nx - 1, self.ny - 1, self.nz - 1],
        )
        vertex_colors = self.color[
            voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
        ]
        vertex_colors = np.clip(vertex_colors * 255, 0, 255).astype(np.uint8)
        alpha_col = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([vertex_colors, alpha_col])

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            vertex_colors=vertex_colors,
            process=False,
        )
        logger.info("Extracted mesh: %d vertices, %d faces", len(verts), len(faces))
        return mesh


# ---------------------------------------------------------------------------
# Legacy MCP orbit camera generation
# ---------------------------------------------------------------------------

def _generate_orbit_cameras(
    n_views: int,
    distance: float,
    fov_deg: float,
    width: int,
    height: int,
    elevation_angles: Optional[list[float]] = None,
) -> list[tuple[np.ndarray, CameraIntrinsics]]:
    """Generate camera poses orbiting around the origin.

    Returns a list of (extrinsics_4x4, intrinsics) pairs.
    Cameras look at the origin from a sphere of given radius.
    """
    if elevation_angles is None:
        elevation_angles = [-20.0, 0.0, 30.0]

    intrinsics = CameraIntrinsics.default(width, height, fov_deg)
    cameras = []

    views_per_ring = max(1, n_views // len(elevation_angles))
    for elev_deg in elevation_angles:
        elev = np.radians(elev_deg)
        for i in range(views_per_ring):
            azimuth = 2.0 * np.pi * i / views_per_ring

            cx = distance * np.cos(elev) * np.cos(azimuth)
            cy = distance * np.cos(elev) * np.sin(azimuth)
            cz = distance * np.sin(elev)
            pos = np.array([cx, cy, cz])

            forward = -pos / np.linalg.norm(pos)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            up /= np.linalg.norm(up)

            extrinsics = np.eye(4, dtype=np.float64)
            extrinsics[:3, 0] = right
            extrinsics[:3, 1] = up
            extrinsics[:3, 2] = forward
            extrinsics[:3, 3] = pos

            cameras.append((extrinsics, intrinsics))

    return cameras


# ---------------------------------------------------------------------------
# Legacy MCP rendering
# ---------------------------------------------------------------------------

def _call_mcp_render(
    width: int,
    height: int,
    camera_pose: np.ndarray,
    render_type: str = "color",
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp",
) -> Optional[np.ndarray]:
    """Call LichtFeld MCP render.capture to get an image."""
    pose_list = camera_pose.flatten().tolist()
    args = {
        "width": width,
        "height": height,
        "camera_pose": pose_list,
        "render_type": render_type,
        "output_format": "numpy",
    }
    args_json = json.dumps(args)

    try:
        result = subprocess.run(
            [LFS_MCP, "call", "render.capture", args_json],
            capture_output=True, text=True, timeout=30,
            env={"LICHTFELD_MCP_ENDPOINT": mcp_endpoint, "PATH": "/usr/bin:/usr/local/bin"},
        )
        if result.returncode != 0:
            logger.warning("MCP render.capture failed: %s", result.stderr)
            return None

        response = json.loads(result.stdout)
        content = response.get("result", {}).get("content", [])
        for item in content:
            if item.get("type") == "resource":
                uri = item.get("resource", {}).get("uri", "")
                if uri.startswith("file://"):
                    path = uri[7:]
                    return np.load(path) if path.endswith(".npy") else None
            if item.get("type") == "text":
                data = json.loads(item["text"])
                if "data" in data:
                    arr = np.array(data["data"], dtype=np.float32)
                    if render_type == "depth":
                        return arr.reshape(height, width)
                    return arr.reshape(height, width, -1)
        logger.warning("No parseable image data in MCP response")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.warning("MCP render call failed: %s", e)
        return None


def _render_depth_from_gaussians(
    camera_pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    mcp_endpoint: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Render depth and color from LichtFeld MCP."""
    depth = _call_mcp_render(
        intrinsics.width, intrinsics.height, camera_pose,
        render_type="depth", mcp_endpoint=mcp_endpoint,
    )
    color = _call_mcp_render(
        intrinsics.width, intrinsics.height, camera_pose,
        render_type="color", mcp_endpoint=mcp_endpoint,
    )
    return depth, color


# ---------------------------------------------------------------------------
# MeshExtractor
# ---------------------------------------------------------------------------

class MeshExtractor:
    """Extract polygonal meshes from Gaussian splat objects.

    Supports three extraction modes:
    - extract_from_gsplat: GPU rendering via gsplat -> TSDF (preferred)
    - extract_from_mcp: Legacy MCP rendering -> TSDF
    - extract_from_arrays: Pre-rendered depth/color -> TSDF
    - extract_from_pointcloud: Point cloud density -> marching cubes
    """

    def __init__(self, config: Optional[TSDFConfig] = None):
        self.config = config or TSDFConfig()

    def extract_from_gsplat(
        self,
        ply_path: str,
        num_views: int = 64,
        render_size: int = 1024,
        target_faces: Optional[int] = None,
        preview_dir: Optional[Path] = None,
        colmap_dir: Optional[str] = None,
    ) -> tuple[trimesh.Trimesh, list[np.ndarray], list]:
        """Full gsplat pipeline: load PLY -> render depth+color -> TSDF -> mesh.

        If colmap_dir is provided, uses COLMAP training cameras for TSDF
        (much better for interior scenes). Falls back to orbit cameras.

        Args:
            ply_path: Path to 3DGS PLY file.
            num_views: Number of orbit viewpoints for depth rendering.
            render_size: Square render resolution in pixels.
            target_faces: Override face count target (uses config if None).
            colmap_dir: Path to COLMAP output for camera loading.

        Returns:
            (mesh, color_images, cameras) where:
            - mesh: cleaned trimesh.Trimesh with vertex colors
            - color_images: list of [H,W,3] float32 color renders
            - cameras: list of (viewmat, K) tuples for texture baking
        """
        t_start = time.time()
        target = target_faces or self.config.target_faces

        # Load gaussians
        gaussians = load_3dgs_ply(ply_path)
        logger.info("Loaded %d gaussians in %.1fs",
                     gaussians['count'], time.time() - t_start)

        # Compute scene bounds for TSDF volume (1st-99th percentile captures more)
        pts = gaussians['means'].cpu().numpy()
        p1, p99 = np.percentile(pts, [1, 99], axis=0)
        center = (p1 + p99) / 2.0
        extent = p99 - p1
        padding = extent * 0.15  # Was 0.3 — less padding = more voxels on actual scene
        vol_min = p1 - padding
        vol_max = p99 + padding
        vol_extent = vol_max - vol_min

        # Adaptive voxel size: aim for ~300 voxels per axis
        # 200^3 was too coarse (7.5cm voxels for 15m scene = lumpy output)
        # 300^3 gives 5cm voxels = adequate for most architectural features
        max_dim = vol_extent.max()
        voxel_size = max(max_dim / 300.0, 0.005)
        sdf_trunc = voxel_size * 3.0  # Was 5.0 — too thick for thin structures

        tsdf_config = TSDFConfig(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            volume_bounds_min=vol_min,
            volume_bounds_max=vol_max,
            target_faces=target,
        )

        logger.info("TSDF config: voxel=%.4f, trunc=%.4f, bounds=[%.2f..%.2f]",
                     voxel_size, sdf_trunc, vol_min.min(), vol_max.max())

        # Generate cameras — prefer COLMAP cameras (training viewpoints)
        max_tsdf_views = 100  # More views = better coverage. 300^3 grid is the bottleneck, not rendering.
        cameras = []
        if colmap_dir:
            cameras = load_colmap_cameras(colmap_dir, render_size)
            if cameras:
                # Subsample evenly if too many
                if len(cameras) > max_tsdf_views:
                    step = len(cameras) / max_tsdf_views
                    cameras = [cameras[int(i * step)] for i in range(max_tsdf_views)]
                logger.info("Using %d COLMAP cameras for TSDF (better for interiors)", len(cameras))

        if not cameras:
            cameras = generate_orbit_cameras_gsplat(
                gaussians['means'], min(num_views, max_tsdf_views), render_size,
            )
            logger.info("Using %d orbit cameras for TSDF", len(cameras))

        # Render and integrate
        tsdf = TSDFVolume(tsdf_config)
        color_images = []
        integrated = 0
        t_render = time.time()

        for i, (viewmat, K) in enumerate(cameras):
            depth, rgb, alpha = render_gsplat(
                gaussians, viewmat, K, render_size, render_size,
            )

            valid_pixels = (depth > 0).sum()
            if valid_pixels < 100:
                logger.debug("View %d: only %d valid pixels, skipping", i, valid_pixels)
                continue

            # Save depth colormap preview (every 8th view)
            if i % 8 == 0 and preview_dir:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
                    ax.imshow(depth, cmap='turbo')
                    ax.axis('off')
                    fig.savefig(
                        str(preview_dir / f'depth_view_{i:03d}.jpg'),
                        bbox_inches='tight', dpi=100,
                    )
                    plt.close(fig)
                except Exception as preview_exc:
                    logger.debug("Failed to save depth preview for view %d: %s", i, preview_exc)

            # Store color for texture baking
            color_images.append(rgb.copy())

            # Integrate into TSDF: pass K as numpy 3x3, viewmat as numpy 4x4
            K_np = K.cpu().numpy().astype(np.float64)
            viewmat_np = viewmat.cpu().numpy().astype(np.float64)

            # Make color HxWx3 for TSDF — use alpha as per-pixel confidence weight
            # Expected depth is unreliable at low alpha (multi-surface blending)
            rgb_for_tsdf = rgb.copy()
            tsdf.integrate(depth, rgb_for_tsdf, K_np, viewmat_np,
                           weight=alpha)
            integrated += 1

            if (i + 1) % 16 == 0:
                elapsed = time.time() - t_render
                logger.info("Rendered+integrated %d/%d views (%.1fs, %.0fms/view)",
                            i + 1, num_views, elapsed,
                            elapsed / (i + 1) * 1000)

        elapsed_render = time.time() - t_render
        logger.info("Rendering complete: %d/%d views integrated in %.1fs",
                     integrated, num_views, elapsed_render)

        if integrated == 0:
            raise RuntimeError("No valid depth frames from gsplat rendering")

        # Extract mesh
        mesh = tsdf.extract_mesh()

        # Clean
        cleaner = MeshCleaner()
        mesh = cleaner.clean(
            mesh,
            target_faces=target,
            smooth_iterations=self.config.smooth_iterations,
            min_component_ratio=self.config.min_component_ratio,
            min_component_faces=self.config.min_component_faces,
        )

        total_time = time.time() - t_start
        logger.info("gsplat mesh extraction complete: %d verts, %d faces in %.1fs",
                     len(mesh.vertices), len(mesh.faces), total_time)

        return mesh, color_images, cameras

    def extract_from_mcp(self) -> trimesh.Trimesh:
        """Full pipeline: render from MCP, fuse, extract, clean.

        Requires LichtFeld Studio MCP to be running with a loaded scene.
        """
        cfg = self.config
        cameras = _generate_orbit_cameras(
            n_views=cfg.num_viewpoints,
            distance=cfg.camera_distance,
            fov_deg=cfg.camera_fov_deg,
            width=cfg.render_width,
            height=cfg.render_height,
        )

        tsdf = TSDFVolume(cfg)
        integrated_count = 0

        for i, (extrinsics, intrinsics) in enumerate(cameras):
            logger.info("Rendering viewpoint %d/%d", i + 1, len(cameras))
            depth, color = _render_depth_from_gaussians(
                extrinsics, intrinsics, cfg.mcp_endpoint,
            )
            if depth is None or color is None:
                logger.warning("Skipping viewpoint %d (render failed)", i + 1)
                continue

            tsdf.integrate(depth, color, intrinsics, extrinsics)
            integrated_count += 1

        if integrated_count == 0:
            raise RuntimeError("No depth frames could be rendered from MCP")

        logger.info("Integrated %d/%d viewpoints", integrated_count, len(cameras))
        mesh = tsdf.extract_mesh()

        cleaner = MeshCleaner()
        mesh = cleaner.clean(
            mesh,
            target_faces=cfg.target_faces,
            smooth_iterations=cfg.smooth_iterations,
            min_component_ratio=cfg.min_component_ratio,
            min_component_faces=cfg.min_component_faces,
        )
        return mesh

    def extract_from_arrays(
        self,
        depth_maps: list[np.ndarray],
        color_images: list[np.ndarray],
        extrinsics_list: list[np.ndarray],
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> trimesh.Trimesh:
        """Extract mesh from pre-rendered depth/color arrays."""
        if not (len(depth_maps) == len(color_images) == len(extrinsics_list)):
            raise ValueError("depth_maps, color_images, and extrinsics_list must have equal length")

        cfg = self.config
        if intrinsics is None:
            h, w = depth_maps[0].shape[:2]
            intrinsics = CameraIntrinsics.default(w, h, cfg.camera_fov_deg)

        tsdf = TSDFVolume(cfg)

        for i, (depth, color, ext) in enumerate(zip(depth_maps, color_images, extrinsics_list)):
            logger.info("Integrating frame %d/%d", i + 1, len(depth_maps))
            tsdf.integrate(depth, color, intrinsics, ext)

        mesh = tsdf.extract_mesh()

        cleaner = MeshCleaner()
        mesh = cleaner.clean(
            mesh,
            target_faces=cfg.target_faces,
            smooth_iterations=cfg.smooth_iterations,
            min_component_ratio=cfg.min_component_ratio,
            min_component_faces=cfg.min_component_faces,
        )
        return mesh

    def extract_from_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        target_faces: Optional[int] = None,
    ) -> trimesh.Trimesh:
        """Extract mesh from a point cloud using density-based marching cubes.

        Falls back to convex hull if advanced reconstruction fails.
        """
        target = target_faces or self.config.target_faces

        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        k = min(30, len(points))
        _, indices = tree.query(points, k=k)

        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]

        centroid = points.mean(axis=0)
        for i in range(len(normals)):
            if np.dot(normals[i], points[i] - centroid) < 0:
                normals[i] = -normals[i]

        padding = 0.1
        bounds_min = points.min(axis=0) - padding
        bounds_max = points.max(axis=0) + padding

        cfg = TSDFConfig(
            voxel_size=self.config.voxel_size,
            volume_bounds_min=bounds_min,
            volume_bounds_max=bounds_max,
        )
        dims = ((bounds_max - bounds_min) / cfg.voxel_size).astype(int)
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        voxel_idx = ((points - bounds_min) / cfg.voxel_size).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, [nx - 1, ny - 1, nz - 1])
        for idx in voxel_idx:
            grid[idx[0], idx[1], idx[2]] += 1.0

        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=2.0)
        threshold = grid.max() * 0.1

        try:
            verts, faces, mesh_normals, _ = marching_cubes(
                grid, level=threshold,
                spacing=(cfg.voxel_size, cfg.voxel_size, cfg.voxel_size),
            )
            verts += bounds_min
        except ValueError:
            logger.warning("Marching cubes on point cloud failed, falling back to convex hull")
            cloud = trimesh.PointCloud(points)
            mesh = cloud.convex_hull
            cleaner = MeshCleaner()
            return cleaner.clean(mesh, target_faces=target)

        vertex_colors = None
        if colors is not None:
            colors_f = colors.astype(np.float32)
            if colors_f.max() > 1.0:
                colors_f /= 255.0
            _, nn_idx = tree.query(verts, k=1)
            vc = colors_f[nn_idx]
            vc = np.clip(vc * 255, 0, 255).astype(np.uint8)
            alpha_col = np.full((len(vc), 1), 255, dtype=np.uint8)
            vertex_colors = np.hstack([vc, alpha_col])

        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces, vertex_normals=mesh_normals,
            vertex_colors=vertex_colors, process=True,
        )

        cleaner = MeshCleaner()
        return cleaner.clean(mesh, target_faces=target)
