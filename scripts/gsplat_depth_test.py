"""
gsplat depth rendering test: load 3DGS PLY, render proper splatted depth maps,
then run TSDF fusion to extract a mesh.

Uses gsplat's render_mode="ED" (expected depth) for dense, smooth depth maps
where each gaussian's full 2D splat contributes depth weighted by opacity.
"""
import time
import math
import struct
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths (override base via TEST_DATA_DIR env var) ──
from pathlib import Path as _Path
_GALLERY = _Path(os.environ.get(
    "TEST_DATA_DIR", str(_Path(__file__).resolve().parent.parent.parent / "test-data")
)) / "gallery_output"
PLY_PATH = str(_GALLERY / "model_quality" / "splat_30000.ply")
COLMAP_CAMERAS = str(_GALLERY / "colmap" / "exported" / "cameras.txt")
COLMAP_IMAGES  = str(_GALLERY / "colmap" / "exported" / "images.txt")
OUT_DIR = str(_GALLERY / "gsplat_depth_test")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load PLY ──
print("=" * 60)
print("STEP 1: Loading PLY")
t0 = time.time()

from plyfile import PlyData
ply = PlyData.read(PLY_PATH)
v = ply.elements[0]
N = v.count
print(f"  Gaussians: {N:,}")

# Extract arrays
xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
opacities_raw = v['opacity'].astype(np.float32)
scales_raw = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32)
rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)

# SH coefficients (degree 3 = 16 coeffs per channel, 48 total)
sh_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).astype(np.float32)
sh_rest = np.zeros((N, 45), dtype=np.float32)
for i in range(45):
    sh_rest[:, i] = v[f'f_rest_{i}'].astype(np.float32)

# Reshape SH: dc is [N,1,3], rest is [N,15,3] -> total [N,16,3]
sh_dc = sh_dc.reshape(N, 1, 3)
sh_rest = sh_rest.reshape(N, 15, 3)
shs = np.concatenate([sh_dc, sh_rest], axis=1)  # [N, 16, 3]

dt = time.time() - t0
print(f"  Loaded in {dt:.1f}s")
print(f"  Position range: {xyz.min(0)} to {xyz.max(0)}")
print(f"  Opacity (raw) range: [{opacities_raw.min():.3f}, {opacities_raw.max():.3f}]")
print(f"  Scale (raw log) range: [{scales_raw.min():.3f}, {scales_raw.max():.3f}]")

# ── 2. Convert to gsplat format ──
print("\n" + "=" * 60)
print("STEP 2: Converting to gsplat tensors")

device = torch.device("cuda:0")

means = torch.from_numpy(xyz).to(device)                        # [N, 3]
quats = torch.from_numpy(rots).to(device)                       # [N, 4] wxyz
scales = torch.exp(torch.from_numpy(scales_raw).to(device))     # [N, 3] - stored as log, need exp
opacities = torch.sigmoid(torch.from_numpy(opacities_raw).to(device))  # [N] - stored as logit
colors = torch.from_numpy(shs).to(device)                       # [N, 16, 3]

# Normalize quaternions
quats = quats / quats.norm(dim=-1, keepdim=True)

print(f"  means: {means.shape}, device={means.device}")
print(f"  scales (after exp): [{scales.min().item():.6f}, {scales.max().item():.6f}]")
print(f"  opacities (after sigmoid): [{opacities.min().item():.4f}, {opacities.max().item():.4f}]")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ── 3. Parse COLMAP cameras ──
print("\n" + "=" * 60)
print("STEP 3: Setting up camera")

# Camera: SIMPLE_RADIAL 1920 1080 1514.07 960 540 0.0296
W, H = 1920, 1080
fx = fy = 1514.0663
cx, cy = 960.0, 540.0

K = torch.tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1],
], dtype=torch.float32, device=device)

# Parse first few images from COLMAP for viewmats
def parse_colmap_images(path, max_images=5):
    """Parse COLMAP images.txt to get world-to-camera transforms."""
    viewmats = []
    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    count = 0
    while i < len(lines) and count < max_images:
        line = lines[i].strip()
        if line.startswith('#') or len(line) == 0:
            i += 1
            continue
        # Image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

            # Build rotation matrix from quaternion
            q = np.array([qw, qx, qy, qz])
            q = q / np.linalg.norm(q)
            w, x, y, z = q
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ])
            t = np.array([tx, ty, tz])

            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = t
            viewmats.append(viewmat)
            count += 1
            i += 2  # skip the points2D line
        else:
            i += 1

    return viewmats

colmap_viewmats = parse_colmap_images(COLMAP_IMAGES, max_images=3)
print(f"  Parsed {len(colmap_viewmats)} COLMAP views")
print(f"  Image size: {W}x{H}")
print(f"  Focal: {fx:.2f}")

# ── 4. Render depth with gsplat ──
print("\n" + "=" * 60)
print("STEP 4: Rendering depth maps with gsplat")

from gsplat import rasterization

def render_view(viewmat_np, render_mode="RGB+ED", width=W, height=H):
    """Render a single view returning (rgb, depth) images."""
    viewmat = torch.from_numpy(viewmat_np).unsqueeze(0).to(device)  # [1, 4, 4]
    Ks = K.unsqueeze(0)  # [1, 3, 3]

    with torch.no_grad():
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=100.0,
            sh_degree=3,
            render_mode=render_mode,
            packed=True,
            rasterize_mode="antialiased",
        )

    # render_colors shape: [1, H, W, C] where C = 3(rgb) + 1(depth) for RGB+ED
    out = render_colors[0].cpu().numpy()   # [H, W, C]
    alpha = render_alphas[0].cpu().numpy()  # [H, W, 1]

    if render_mode == "RGB+ED":
        rgb = out[..., :3]
        depth = out[..., 3:4]
    elif render_mode == "ED":
        depth = out
        rgb = None
    elif render_mode == "D":
        depth = out
        rgb = None
    else:
        rgb = out
        depth = None

    return rgb, depth, alpha.squeeze(-1)


# Render the first COLMAP view
print("  Rendering COLMAP view 0...")
t0 = time.time()
rgb0, depth0, alpha0 = render_view(colmap_viewmats[0])
dt = time.time() - t0
print(f"  Render time: {dt:.2f}s")
print(f"  RGB shape: {rgb0.shape}, range: [{rgb0.min():.3f}, {rgb0.max():.3f}]")
print(f"  Depth shape: {depth0.shape}, range: [{depth0.min():.3f}, {depth0.max():.3f}]")
print(f"  Alpha shape: {alpha0.shape}, range: [{alpha0.min():.3f}, {alpha0.max():.3f}]")

# Filter depth stats for valid regions
valid_mask = alpha0 > 0.5
if valid_mask.any():
    valid_depths = depth0.squeeze()[valid_mask]
    print(f"  Valid depth pixels: {valid_mask.sum()} / {valid_mask.size}")
    print(f"  Valid depth range: [{valid_depths.min():.3f}, {valid_depths.max():.3f}]")
    print(f"  Valid depth mean: {valid_depths.mean():.3f}, std: {valid_depths.std():.3f}")

# Save visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(np.clip(rgb0, 0, 1))
axes[0].set_title("RGB (gsplat)")
axes[0].axis('off')

depth_vis = depth0.squeeze().copy()
depth_vis[~valid_mask] = np.nan
im = axes[1].imshow(depth_vis, cmap='turbo')
axes[1].set_title("Expected Depth (gsplat ED)")
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(alpha0, cmap='gray')
axes[2].set_title("Alpha (opacity accumulation)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gsplat_render_colmap0.png"), dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR}/gsplat_render_colmap0.png")

# Save raw depth
np.save(os.path.join(OUT_DIR, "depth_colmap0.npy"), depth0)
print(f"  Saved: {OUT_DIR}/depth_colmap0.npy")


# ── 5. Generate orbit cameras and render depth maps for TSDF ──
print("\n" + "=" * 60)
print("STEP 5: Rendering orbit depth maps for TSDF fusion")

# Compute scene center and radius from gaussian positions
scene_center = means.mean(dim=0).cpu().numpy()
scene_std = means.std(dim=0).cpu().numpy()
scene_radius = np.linalg.norm(scene_std) * 2.0
print(f"  Scene center: {scene_center}")
print(f"  Scene std: {scene_std}")
print(f"  Estimated radius: {scene_radius:.3f}")

# Use actual scene bounds for better orbit
pos_np = means.cpu().numpy()
scene_min = np.percentile(pos_np, 5, axis=0)
scene_max = np.percentile(pos_np, 95, axis=0)
center = (scene_min + scene_max) / 2.0
extent = scene_max - scene_min
orbit_radius = np.linalg.norm(extent) * 0.8
print(f"  Orbit center: {center}")
print(f"  Scene extent: {extent}")
print(f"  Orbit radius: {orbit_radius:.3f}")

def look_at(eye, target, up=np.array([0, -1, 0], dtype=np.float32)):
    """Create a world-to-camera matrix (OpenGL convention -> COLMAP convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_new = np.cross(right, forward)

    # COLMAP convention: camera looks along +Z, Y is down
    R = np.stack([right, -up_new, forward], axis=0)  # 3x3
    t = -R @ eye

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


N_VIEWS = 48  # Number of orbit views
orbit_viewmats = []
# Two elevation rings
for elev_deg in [-15, 0, 15]:
    elev = math.radians(elev_deg)
    n_ring = N_VIEWS // 3
    for i in range(n_ring):
        azimuth = 2.0 * math.pi * i / n_ring
        eye = center + orbit_radius * np.array([
            math.cos(azimuth) * math.cos(elev),
            math.sin(elev),
            math.sin(azimuth) * math.cos(elev),
        ], dtype=np.float32)
        viewmat = look_at(eye, center)
        orbit_viewmats.append(viewmat)

print(f"  Generated {len(orbit_viewmats)} orbit views")

# Render at lower resolution for speed
RENDER_W = 960
RENDER_H = 540
# Scale intrinsics
K_half = K.clone()
K_half[0, 0] *= RENDER_W / W
K_half[1, 1] *= RENDER_H / H
K_half[0, 2] *= RENDER_W / W
K_half[1, 2] *= RENDER_H / H

# Override the render function for half-res
def render_view_halfres(viewmat_np):
    viewmat = torch.from_numpy(viewmat_np).unsqueeze(0).to(device)
    Ks_h = K_half.unsqueeze(0)
    with torch.no_grad():
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks_h,
            width=RENDER_W,
            height=RENDER_H,
            near_plane=0.01,
            far_plane=100.0,
            sh_degree=3,
            render_mode="ED",
            packed=True,
            rasterize_mode="antialiased",
        )
    depth = render_colors[0].cpu().numpy().squeeze(-1)  # [H, W]
    alpha = render_alphas[0].cpu().numpy().squeeze(-1)   # [H, W]
    return depth, alpha

t0 = time.time()
all_depths = []
all_alphas = []
all_viewmats_np = []

for i, vm in enumerate(orbit_viewmats):
    depth, alpha = render_view_halfres(vm)
    all_depths.append(depth)
    all_alphas.append(alpha)
    all_viewmats_np.append(vm)
    if (i + 1) % 12 == 0:
        elapsed = time.time() - t0
        print(f"  Rendered {i+1}/{len(orbit_viewmats)} views ({elapsed:.1f}s)")

elapsed = time.time() - t0
print(f"  Total render: {elapsed:.1f}s for {len(orbit_viewmats)} views")
print(f"  Average: {elapsed/len(orbit_viewmats):.2f}s per view")

# Save a montage of depth maps
n_show = min(16, len(all_depths))
fig, axes = plt.subplots(4, 4, figsize=(16, 9))
for idx in range(n_show):
    ax = axes[idx // 4][idx % 4]
    d = all_depths[idx].copy()
    a = all_alphas[idx]
    d[a < 0.3] = np.nan
    ax.imshow(d, cmap='turbo')
    ax.set_title(f"View {idx}", fontsize=8)
    ax.axis('off')
plt.suptitle("gsplat Expected Depth Maps (orbit views)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "depth_montage.png"), dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR}/depth_montage.png")


# ── 6. TSDF Fusion ──
print("\n" + "=" * 60)
print("STEP 6: TSDF Fusion")

try:
    import open3d as o3d
    USE_O3D = True
    print("  Using Open3D TSDF")
except ImportError:
    USE_O3D = False
    print("  Open3D not available, using custom TSDF")


if USE_O3D:
    # Open3D TSDF pipeline
    voxel_size = orbit_radius / 256.0  # Adaptive voxel size
    sdf_trunc = voxel_size * 5.0

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        RENDER_W, RENDER_H,
        K_half[0, 0].item(), K_half[1, 1].item(),
        K_half[0, 2].item(), K_half[1, 2].item()
    )

    integrated = 0
    for i in range(len(all_depths)):
        depth = all_depths[i]
        alpha = all_alphas[i]

        # Mask invalid regions
        depth_clean = depth.copy()
        depth_clean[alpha < 0.3] = 0.0
        depth_clean[depth_clean < 0.01] = 0.0
        depth_clean[depth_clean > 50.0] = 0.0

        if (depth_clean > 0).sum() < 1000:
            continue

        depth_o3d = o3d.geometry.Image(depth_clean.astype(np.float32))

        # Convert viewmat to Open3D extrinsic (same convention)
        extrinsic = all_viewmats_np[i].astype(np.float64)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)),
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=50.0,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))
        integrated += 1

    print(f"  Integrated {integrated}/{len(all_depths)} views")

    print("  Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    print(f"  Raw mesh: {len(vertices):,} vertices, {len(triangles):,} faces")

    # Save
    mesh_path = os.path.join(OUT_DIR, "tsdf_mesh_gsplat.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"  Saved: {mesh_path}")

    # Check watertight
    mesh_trimesh = None
    try:
        import trimesh
        mesh_trimesh = trimesh.load(mesh_path)
        print(f"  Watertight: {mesh_trimesh.is_watertight}")
        print(f"  Trimesh vertices: {len(mesh_trimesh.vertices):,}")
        print(f"  Trimesh faces: {len(mesh_trimesh.faces):,}")
    except Exception as e:
        print(f"  Trimesh check failed: {e}")

else:
    # Minimal custom TSDF (CPU, numpy-based)
    print("  Running custom numpy TSDF (slower)...")

    # Determine volume bounds from depth back-projection
    voxel_size = orbit_radius / 128.0
    vol_min = center - orbit_radius * 1.2
    vol_max = center + orbit_radius * 1.2
    vol_shape = np.ceil((vol_max - vol_min) / voxel_size).astype(int)
    print(f"  Volume shape: {vol_shape}, voxel_size: {voxel_size:.4f}")

    # Limit volume size
    max_dim = 256
    if vol_shape.max() > max_dim:
        voxel_size = ((vol_max - vol_min) / max_dim).max()
        vol_shape = np.ceil((vol_max - vol_min) / voxel_size).astype(int)
        print(f"  Clamped volume shape: {vol_shape}, voxel_size: {voxel_size:.4f}")

    tsdf = np.ones(vol_shape, dtype=np.float32)
    weight = np.zeros(vol_shape, dtype=np.float32)
    trunc = voxel_size * 5.0

    fx_h = K_half[0, 0].item()
    fy_h = K_half[1, 1].item()
    cx_h = K_half[0, 2].item()
    cy_h = K_half[1, 2].item()

    # Create voxel centers
    x = np.arange(vol_shape[0]) * voxel_size + vol_min[0]
    y = np.arange(vol_shape[1]) * voxel_size + vol_min[1]
    z = np.arange(vol_shape[2]) * voxel_size + vol_min[2]

    integrated = 0
    for vi in range(len(all_depths)):
        depth = all_depths[vi]
        alpha = all_alphas[vi]
        vm = all_viewmats_np[vi]

        depth_clean = depth.copy()
        depth_clean[alpha < 0.3] = 0.0
        if (depth_clean > 0).sum() < 1000:
            continue

        R = vm[:3, :3]
        t = vm[:3, 3]

        # Project each voxel to this camera (in chunks for memory)
        chunk_size = vol_shape[0]
        for xi in range(0, vol_shape[0], chunk_size):
            xe = min(xi + chunk_size, vol_shape[0])
            xx, yy, zz = np.meshgrid(x[xi:xe], y, z, indexing='ij')
            pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

            # Transform to camera space
            cam_pts = (R @ pts.T).T + t  # [M, 3]
            cam_z = cam_pts[:, 2]

            # Project to pixel
            u = (fx_h * cam_pts[:, 0] / cam_z + cx_h).astype(int)
            v_px = (fy_h * cam_pts[:, 1] / cam_z + cy_h).astype(int)

            # Valid mask
            valid = (cam_z > 0.01) & (u >= 0) & (u < RENDER_W) & (v_px >= 0) & (v_px < RENDER_H)

            # Look up depth
            measured_depth = np.zeros(len(cam_pts))
            measured_depth[valid] = depth_clean[v_px[valid], u[valid]]

            # SDF
            sdf = measured_depth - cam_z

            # Truncation
            valid_sdf = valid & (measured_depth > 0) & (np.abs(sdf) < trunc)
            sdf_trunc = np.clip(sdf / trunc, -1, 1)

            # Update volume
            sub_shape = (xe - xi, vol_shape[1], vol_shape[2])
            sdf_vol = sdf_trunc.reshape(sub_shape)
            valid_vol = valid_sdf.reshape(sub_shape)

            w_old = weight[xi:xe]
            tsdf_old = tsdf[xi:xe]

            w_new = w_old + valid_vol.astype(np.float32)
            tsdf[xi:xe] = np.where(w_new > 0,
                (w_old * tsdf_old + valid_vol * sdf_vol) / np.maximum(w_new, 1),
                tsdf_old)
            weight[xi:xe] = w_new

        integrated += 1
        if (integrated) % 12 == 0:
            print(f"  Integrated {integrated}/{len(all_depths)} views")

    print(f"  Integrated {integrated} views total")
    print(f"  TSDF filled voxels: {(weight > 0).sum():,} / {weight.size:,}")

    # Marching cubes
    try:
        from skimage.measure import marching_cubes
        print("  Running marching cubes...")
        verts, faces, normals, values = marching_cubes(tsdf, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
        verts += vol_min
        print(f"  Mesh: {len(verts):,} vertices, {len(faces):,} faces")

        # Save OBJ
        mesh_path = os.path.join(OUT_DIR, "tsdf_mesh_gsplat.obj")
        with open(mesh_path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"  Saved: {mesh_path}")

        # Check with trimesh
        try:
            import trimesh
            mesh_trimesh = trimesh.load(mesh_path)
            print(f"  Watertight: {mesh_trimesh.is_watertight}")
        except:
            pass
    except ImportError:
        print("  ERROR: scikit-image not available for marching cubes")


# ── 7. Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  gsplat version: 1.5.3")
print(f"  Render modes available: RGB, D, ED, RGB+D, RGB+ED")
print(f"  D = accumulated depth (sum w_i * z_i)")
print(f"  ED = expected depth (sum w_i * z_i / sum w_i) -- RECOMMENDED")
print(f"  PLY loaded: {N:,} gaussians from {PLY_PATH}")
print(f"  Orbit views rendered: {len(orbit_viewmats)}")
print(f"  Output directory: {OUT_DIR}")
print(f"  VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
