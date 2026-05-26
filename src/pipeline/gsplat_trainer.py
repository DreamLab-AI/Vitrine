"""Minimal 3DGS trainer using gsplat for COLMAP datasets.

Produces a standard 3DGS PLY compatible with load_3dgs_ply().
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.colmap_parser import (
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def load_colmap_dataset(colmap_dir: Path) -> Tuple[
    List[torch.Tensor],  # images [H,W,3] float32 on CPU
    torch.Tensor,        # viewmats [N,4,4]
    torch.Tensor,        # Ks [N,3,3]
    int, int,            # width, height
    np.ndarray,          # points [M,3]
    np.ndarray,          # colors [M,3] in [0,1]
]:
    sparse_dir = colmap_dir / "sparse" / "0"
    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    images_meta = parse_images_txt(sparse_dir / "images.txt")
    points3d = parse_points3d_txt(sparse_dir / "points3D.txt")

    cam = list(cameras.values())[0]
    W, H = cam.width, cam.height
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = cam.center_x, cam.center_y

    images_meta.sort(key=lambda im: im.name)

    image_tensors = []
    viewmats_list = []
    Ks_list = []
    images_dir = colmap_dir / "images"

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    for img in images_meta:
        img_path = images_dir / img.name
        if not img_path.exists():
            continue
        pil_img = Image.open(img_path).convert("RGB")
        if pil_img.size != (W, H):
            pil_img = pil_img.resize((W, H), Image.LANCZOS)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        image_tensors.append(torch.from_numpy(arr))

        q = np.array([img.qw, img.qx, img.qy, img.qz])
        R = quat_to_rotmat(q)
        t = np.array([img.tx, img.ty, img.tz])
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t
        viewmats_list.append(torch.from_numpy(vm))
        Ks_list.append(torch.from_numpy(K.copy()))

    pts = np.array([[p.x, p.y, p.z] for p in points3d], dtype=np.float32)
    cols = np.array([[p.r, p.g, p.b] for p in points3d], dtype=np.float32) / 255.0

    return (
        image_tensors,
        torch.stack(viewmats_list),
        torch.stack(Ks_list),
        W, H,
        pts, cols,
    )


def init_gaussians(points: np.ndarray, colors: np.ndarray, device: str = "cuda") -> Dict[str, torch.nn.Parameter]:
    N = len(points)
    means = torch.from_numpy(points).float().to(device)

    # Estimate initial scale from nearest-neighbor distances
    from scipy.spatial import KDTree
    tree = KDTree(points)
    dists, _ = tree.query(points, k=4)  # k=4: self + 3 neighbors
    avg_dist = np.mean(dists[:, 1:], axis=1).astype(np.float32)
    avg_dist = np.clip(avg_dist, 1e-5, None)
    log_scales = torch.from_numpy(np.log(avg_dist)).float().to(device)
    log_scales = log_scales.unsqueeze(-1).expand(-1, 3).clone()

    quats = torch.zeros((N, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity quaternion wxyz

    logit_opacities = torch.full((N,), 2.0, dtype=torch.float32, device=device)  # sigmoid(2) ~ 0.88

    # SH coefficients: DC band from colors
    sh0 = torch.zeros((N, 1, 3), dtype=torch.float32, device=device)
    sh0[:, 0, :] = (torch.from_numpy(colors).float().to(device) - 0.5) / SH_C0

    shN = torch.zeros((N, 15, 3), dtype=torch.float32, device=device)

    return {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(log_scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(logit_opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }


def save_3dgs_ply(path: Path, params: Dict[str, torch.nn.Parameter], sh_degree: int = 3):
    """Save Gaussians in standard 3DGS PLY format."""
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()  # already log-space
    quats = params["quats"].detach().cpu().numpy()
    opacities = params["opacities"].detach().cpu().numpy()  # already logit-space
    sh0 = params["sh0"].detach().cpu().numpy()  # [N,1,3]
    shN = params["shN"].detach().cpu().numpy()  # [N,15,3]

    N = len(means)

    # Normalize quaternions
    qnorm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(qnorm, 1e-8, None)

    # Build structured array
    props = []
    props.append(('x', 'f4'))
    props.append(('y', 'f4'))
    props.append(('z', 'f4'))
    props.append(('nx', 'f4'))
    props.append(('ny', 'f4'))
    props.append(('nz', 'f4'))
    for i in range(3):
        props.append((f'f_dc_{i}', 'f4'))
    n_sh_rest = (sh_degree + 1) ** 2 - 1
    for i in range(n_sh_rest * 3):
        props.append((f'f_rest_{i}', 'f4'))
    props.append(('opacity', 'f4'))
    props.append(('scale_0', 'f4'))
    props.append(('scale_1', 'f4'))
    props.append(('scale_2', 'f4'))
    props.append(('rot_0', 'f4'))
    props.append(('rot_1', 'f4'))
    props.append(('rot_2', 'f4'))
    props.append(('rot_3', 'f4'))

    arr = np.zeros(N, dtype=props)
    arr['x'] = means[:, 0]
    arr['y'] = means[:, 1]
    arr['z'] = means[:, 2]
    arr['nx'] = 0
    arr['ny'] = 0
    arr['nz'] = 0

    # DC SH coefficients [N,1,3] -> interleaved as f_dc_0, f_dc_1, f_dc_2
    for i in range(3):
        arr[f'f_dc_{i}'] = sh0[:, 0, i]

    # Rest SH coefficients [N,15,3] -> interleaved
    # Standard 3DGS PLY stores SH rest as: for each band, rgb interleaved
    # shN shape: [N, 15, 3] but PLY wants [N, 45] where layout is
    # f_rest_0..f_rest_44 corresponding to shN transposed to [N, 3, 15] then flattened
    sh_rest_flat = shN[:, :n_sh_rest, :].reshape(N, -1)  # [N, n_sh_rest*3]
    # Reorder: PLY stores per-channel groups: all R bands, all G bands, all B bands
    # Actually standard 3DGS stores as: sh[band][channel], interleaved differently
    # Let's check: f_rest_0..14 are band1..15 for R, f_rest_15..29 for G, f_rest_30..44 for B
    for c in range(3):
        for b in range(n_sh_rest):
            arr[f'f_rest_{c * n_sh_rest + b}'] = shN[:, b, c]

    arr['opacity'] = opacities
    arr['scale_0'] = scales[:, 0]
    arr['scale_1'] = scales[:, 1]
    arr['scale_2'] = scales[:, 2]
    arr['rot_0'] = quats[:, 0]
    arr['rot_1'] = quats[:, 1]
    arr['rot_2'] = quats[:, 2]
    arr['rot_3'] = quats[:, 3]

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el]).write(str(path))
    logger.info("Saved PLY: %s (%d gaussians, %.1f MB)", path, N, path.stat().st_size / 1e6)


def train(
    colmap_dir: str,
    output_dir: str,
    iterations: int = 30000,
    sh_degree: int = 3,
    device: str = "cuda",
) -> Path:
    """Train 3DGS from COLMAP dataset, return path to output PLY."""
    from gsplat import rasterization, DefaultStrategy

    colmap_path = Path(colmap_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading COLMAP dataset from %s", colmap_path)
    image_tensors, viewmats, Ks, W, H, points, colors = load_colmap_dataset(colmap_path)
    n_images = len(image_tensors)
    logger.info("Loaded %d images (%dx%d), %d points", n_images, W, H, len(points))

    # Downscale for memory if images are large
    scale_factor = 1
    if W > 1600:
        scale_factor = 2
        W //= 2
        H //= 2
        Ks[:, :2, :] /= 2
        image_tensors = [
            F.interpolate(
                img.permute(2, 0, 1).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            for img in image_tensors
        ]
        logger.info("Downscaled to %dx%d (factor %d)", W, H, scale_factor)

    # Move images to GPU
    gt_images = torch.stack(image_tensors).to(device)  # [N, H, W, 3]
    viewmats = viewmats.to(device)
    Ks = Ks.to(device)

    # Initialize Gaussians
    params = init_gaussians(points, colors, device)
    logger.info("Initialized %d Gaussians", len(points))

    # Scene scale for strategy
    cam_centers = []
    for i in range(n_images):
        vm = viewmats[i]
        R = vm[:3, :3]
        t = vm[:3, 3]
        center = -R.T @ t
        cam_centers.append(center)
    cam_centers = torch.stack(cam_centers)
    scene_scale = (cam_centers.max(dim=0).values - cam_centers.min(dim=0).values).norm().item() / 2

    # Optimizers
    lr_dict = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {}
    for name, param in params.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_dict[name], eps=1e-15)

    # Strategy
    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=min(15000, iterations),
        refine_every=100,
        grow_grad2d=0.0002,
        prune_opa=0.005,
        grow_scale3d=0.01 * scene_scale,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # Background color - shape (3,) for packed mode
    bg = torch.zeros(3, device=device)

    logger.info("Starting training for %d iterations (scene_scale=%.3f)", iterations, scene_scale)

    for step in range(iterations):
        # Random camera
        idx = torch.randint(0, n_images, (1,)).item()
        gt = gt_images[idx]  # [H, W, 3]

        # Build SH coefficients [N, K, 3]
        sh_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)

        renders, alphas, info = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=sh_coeffs,
            viewmats=viewmats[idx:idx+1],
            Ks=Ks[idx:idx+1],
            width=W,
            height=H,
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=1000.0,
            packed=True,
            render_mode="RGB",
            rasterize_mode="antialiased",
            backgrounds=bg,
            absgrad=True,
        )

        rendered = renders[0]  # [H, W, 3]

        # Pre-backward
        strategy.step_pre_backward(params, optimizers, state, step, info)

        # L1 loss
        loss = F.l1_loss(rendered, gt)

        # SSIM-like loss (simplified: just use L1 for speed)
        if step > 500:
            # Add a simple MSE component for better convergence
            loss = 0.8 * loss + 0.2 * F.mse_loss(rendered, gt)

        loss.backward()

        with torch.no_grad():
            # Post-backward (densification)
            strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

            # Step optimizers
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

        if step % 1000 == 0 or step == iterations - 1:
            n_gs = len(params["means"])
            logger.info("Step %d/%d: loss=%.5f, #GS=%d", step, iterations, loss.item(), n_gs)

        # Save intermediate checkpoint
        if step > 0 and step % 10000 == 0:
            ckpt_path = out_path / f"splat_{step}.ply"
            save_3dgs_ply(ckpt_path, params, sh_degree)

    # Save final
    final_path = out_path / f"splat_{iterations}.ply"
    save_3dgs_ply(final_path, params, sh_degree)

    return final_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    args = parser.parse_args()
    result = train(args.colmap_dir, args.output_dir, args.iterations)
    print(f"Training complete: {result}")
