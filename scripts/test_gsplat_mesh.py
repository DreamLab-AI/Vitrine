"""
Test gsplat-based mesh extraction pipeline.

Loads the gallery quality model, extracts a mesh via gsplat depth rendering + TSDF,
decimates, bakes UV texture, exports OBJ + PNG, and reports quality metrics.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_gsplat_mesh")

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_GALLERY = Path(os.environ.get("TEST_DATA_DIR", str(PROJECT_ROOT.parent / "test-data"))) / "gallery_output"
PLY_PATH = str(_GALLERY / "model_quality" / "splat_30000.ply")
OUT_DIR = _GALLERY / "gsplat_mesh_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_total = time.time()

    # ── 1. Verify PLY exists ──
    if not Path(PLY_PATH).exists():
        logger.error("PLY not found: %s", PLY_PATH)
        sys.exit(1)
    logger.info("PLY: %s", PLY_PATH)

    # ── 2. Extract mesh via gsplat ──
    logger.info("=" * 60)
    logger.info("STEP 1: gsplat mesh extraction")
    from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

    extractor = MeshExtractor(config=TSDFConfig(
        target_faces=100000,
        smooth_iterations=3,
        min_component_ratio=0.01,
    ))

    t0 = time.time()
    mesh, color_images, cameras = extractor.extract_from_gsplat(
        PLY_PATH,
        num_views=64,
        render_size=1024,
        target_faces=100000,
    )
    dt_extract = time.time() - t0
    logger.info("Extraction: %.1fs", dt_extract)
    logger.info("Raw mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

    # ── 3. Decimate to 100K faces ──
    logger.info("=" * 60)
    logger.info("STEP 2: Decimation")
    from pipeline.mesh_cleaner import MeshCleaner
    cleaner = MeshCleaner()

    if len(mesh.faces) > 100000:
        mesh = cleaner.decimate(mesh, target_faces=100000)
        logger.info("Decimated: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
    else:
        logger.info("Already under 100K faces, skipping decimation")

    # ── 4. Bake UV texture from vertex colors ──
    # Note: Multi-view projection baking is available via TextureBaker.bake()
    # but is slow for large meshes (pure Python loops). For production use,
    # consider decimating further or using GPU-accelerated UV projection.
    logger.info("=" * 60)
    logger.info("STEP 3: UV texture baking (vertex colors)")
    texture_path = OUT_DIR / "diffuse_texture.png"

    t0 = time.time()
    try:
        from pipeline.texture_baker import TextureBaker, BakeConfig

        baker = TextureBaker(config=BakeConfig(texture_size=2048))
        textured_mesh, tex_path = baker.bake_from_vertex_colors(
            mesh, output_texture_path=str(texture_path),
        )
        dt_bake = time.time() - t0
        logger.info("Vertex color texture baked: %.1fs -> %s", dt_bake, tex_path)
        mesh = textured_mesh
    except Exception as e:
        dt_bake = time.time() - t0
        logger.warning("Texture baking failed (%.1fs): %s", dt_bake, e)

    # ── 5. Export OBJ + PNG ──
    logger.info("=" * 60)
    logger.info("STEP 4: Export")
    obj_path = OUT_DIR / "gallery_mesh.obj"
    glb_path = OUT_DIR / "gallery_mesh.glb"

    mesh.export(str(obj_path))
    mesh.export(str(glb_path))
    logger.info("Exported: %s", obj_path)
    logger.info("Exported: %s", glb_path)

    # ── 6. Quality report ──
    logger.info("=" * 60)
    logger.info("STEP 5: Quality report")

    is_watertight = mesh.is_watertight
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]

    logger.info("  Vertices:    %d", len(mesh.vertices))
    logger.info("  Faces:       %d", len(mesh.faces))
    logger.info("  Watertight:  %s", is_watertight)
    logger.info("  Bounds min:  [%.3f, %.3f, %.3f]", *bounds[0])
    logger.info("  Bounds max:  [%.3f, %.3f, %.3f]", *bounds[1])
    logger.info("  Extent:      [%.3f, %.3f, %.3f]", *extent)

    if mesh.visual and hasattr(mesh.visual, 'material'):
        logger.info("  Has texture:  Yes")
    elif mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
        logger.info("  Has vertex colors: Yes")
    else:
        logger.info("  Visual:      None")

    if texture_path.exists():
        from PIL import Image
        tex_img = Image.open(str(texture_path))
        logger.info("  Texture:     %dx%d (%s)", tex_img.width, tex_img.height, texture_path.name)
        # Check texture coverage (non-background pixels)
        tex_arr = np.array(tex_img)
        bg = np.array([128, 128, 128])
        non_bg = np.any(np.abs(tex_arr.astype(float) - bg) > 10, axis=2)
        coverage = non_bg.sum() / non_bg.size
        logger.info("  Tex coverage: %.1f%%", coverage * 100)

    # ── 7. Render preview with depth montage ──
    logger.info("=" * 60)
    logger.info("STEP 6: Saving depth montage preview")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_show = min(16, len(color_images))
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for idx in range(n_show):
            ax = axes[idx // 4][idx % 4]
            img = color_images[idx]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"View {idx}", fontsize=8)
            ax.axis('off')
        for idx in range(n_show, 16):
            axes[idx // 4][idx % 4].axis('off')
        plt.suptitle("gsplat Color Renders (used for texture baking)")
        plt.tight_layout()
        montage_path = OUT_DIR / "color_montage.png"
        plt.savefig(str(montage_path), dpi=150)
        plt.close()
        logger.info("Saved montage: %s", montage_path)
    except Exception as e:
        logger.warning("Montage failed: %s", e)

    # ── Summary ──
    dt_total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("COMPLETE in %.1fs", dt_total)
    logger.info("  Output dir:  %s", OUT_DIR)
    logger.info("  OBJ:         %s (%.1f MB)", obj_path, obj_path.stat().st_size / 1e6)
    logger.info("  GLB:         %s (%.1f MB)", glb_path, glb_path.stat().st_size / 1e6)
    if texture_path.exists():
        logger.info("  Texture:     %s (%.1f MB)", texture_path, texture_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
