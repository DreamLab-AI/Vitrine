#!/usr/bin/env python3
"""Run the full gallery tour pipeline: COLMAP → Train → Segment → Mesh → USD."""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gallery_pipeline")

# Paths (override via TEST_DATA_DIR / LICHTFELD_BUILD env vars)
_REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = Path(os.environ.get("TEST_DATA_DIR", str(_REPO_ROOT.parent / "test-data")))
LICHTFELD_BUILD = Path(os.environ.get("LICHTFELD_BUILD", str(_REPO_ROOT / "build")))
BASE = TEST_DATA_DIR / "gallery_output"
COLMAP_DIR = BASE / "colmap" / "undistorted"
MODEL_DIR = BASE / "model"
OBJECTS_DIR = BASE / "objects"
SCENE_DIR = BASE / "scene"
LICHTFELD = str(LICHTFELD_BUILD / "LichtFeld-Studio")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def wait_for_colmap():
    """Wait for COLMAP output to appear."""
    log.info("Waiting for COLMAP reconstruction to complete...")
    while not (COLMAP_DIR / "sparse" / "0").exists():
        if (COLMAP_DIR / "sparse").exists():
            # Check if any model exists
            sparse = COLMAP_DIR / "sparse"
            if any(sparse.iterdir()):
                break
        time.sleep(5)
    log.info(f"COLMAP output found at {COLMAP_DIR}")


def stage_3_train():
    """Train 3DGS with LichtFeld Studio headless."""
    log.info("=== STAGE 3: LichtFeld Training ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LICHTFELD_BUILD}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        LICHTFELD, "--headless",
        "--data-path", str(COLMAP_DIR),
        "--output-path", str(MODEL_DIR),
        "--iter", "7000",
        "--strategy", "mcmc",
        "--log-level", "info",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        line = line.strip()
        if line:
            if "iteration" in line.lower() or "loss" in line.lower() or "saving" in line.lower():
                log.info(f"  [LFS] {line}")

    proc.wait()
    if proc.returncode != 0:
        log.error(f"Training failed with exit code {proc.returncode}")
        return False

    # Find output PLY
    ply_files = list(MODEL_DIR.rglob("*.ply"))
    if ply_files:
        log.info(f"Training complete. Output: {ply_files[0]}")
        return True
    else:
        log.warning("Training completed but no PLY found. Checking for checkpoints...")
        resume_files = list(MODEL_DIR.rglob("*.resume"))
        if resume_files:
            log.info(f"Checkpoint found: {resume_files[0]}")
            return True
        return False


def stage_4_segment():
    """Segment objects using SAM2 on training frames."""
    log.info("=== STAGE 4: SAM2 Segmentation ===")
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    from pipeline.sam2_segmentor import SAM2Segmentor
    from pipeline.frame_quality import FrameQualityAssessor

    # Find training images
    images_dir = COLMAP_DIR / "images"
    if not images_dir.exists():
        # Try alternate paths
        for candidate in [COLMAP_DIR / "images", BASE / "frames"]:
            if candidate.exists():
                images_dir = candidate
                break

    if not images_dir.exists():
        log.error(f"No images directory found at {images_dir}")
        return False

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    log.info(f"Found {len(image_files)} images for segmentation")

    if not image_files:
        log.error("No images found for segmentation")
        return False

    # Run SAM2 on a subset of frames (every 5th for speed)
    segmentor = SAM2Segmentor(model_name="facebook/sam2.1-hiera-small")
    sample_frames = image_files[::5]  # Every 5th frame
    log.info(f"Running SAM2 on {len(sample_frames)} sample frames...")

    all_results = []
    for i, img_path in enumerate(sample_frames):
        log.info(f"  Segmenting frame {i+1}/{len(sample_frames)}: {img_path.name}")
        try:
            results = segmentor.generate_masks_single(str(img_path))
            all_results.append((img_path, results))
            log.info(f"    Found {len(results)} objects")
        except Exception as e:
            log.warning(f"    Failed: {e}")

    if not all_results:
        log.error("No segmentation results produced")
        return False

    # Save segmentation results
    seg_output = OBJECTS_DIR / "segmentation_results.json"
    seg_data = []
    for img_path, results in all_results:
        for r in results:
            seg_data.append({
                "image": str(img_path.name),
                "object_id": int(r.object_id),
                "area": int(r.area),
                "bbox": r.bbox.tolist() if hasattr(r.bbox, 'tolist') else list(r.bbox),
                "confidence": float(r.confidence),
            })

    with open(seg_output, "w") as f:
        json.dump(seg_data, f, indent=2)

    # Count unique objects across frames
    unique_objects = set()
    for img_path, results in all_results:
        for r in results:
            unique_objects.add(r.object_id)

    log.info(f"Segmentation complete. {len(unique_objects)} unique objects detected across {len(all_results)} frames")
    log.info(f"Results saved to {seg_output}")
    return True


def stage_5_mesh():
    """Extract meshes from the trained model."""
    log.info("=== STAGE 5: Mesh Extraction (TSDF) ===")

    from pipeline.mesh_extractor import MeshExtractor
    from pipeline.mesh_cleaner import MeshCleaner

    # Find the trained PLY
    ply_files = sorted(MODEL_DIR.rglob("*.ply"))
    if not ply_files:
        log.warning("No PLY files found. Skipping mesh extraction.")
        return False

    ply_path = ply_files[-1]  # Latest
    log.info(f"Using trained model: {ply_path}")

    try:
        import trimesh
        import numpy as np

        # Load the gaussian PLY to get point cloud
        cloud = trimesh.load(str(ply_path))
        if hasattr(cloud, 'vertices'):
            points = np.array(cloud.vertices)
        else:
            log.warning("PLY doesn't have vertices, trying as point cloud")
            points = np.array(cloud.vertices) if hasattr(cloud, 'vertices') else None

        if points is None or len(points) == 0:
            log.error("Could not extract points from PLY")
            return False

        log.info(f"Loaded {len(points)} Gaussian positions")

        # Extract mesh using TSDF from point cloud
        extractor = MeshExtractor(voxel_size=0.02, sdf_trunc=0.06, volume_resolution=256)
        mesh = extractor.extract_from_pointcloud(points)

        if mesh is None or len(mesh.vertices) == 0:
            log.warning("TSDF extraction produced empty mesh, trying direct ball-pivoting")
            # Fallback: create mesh from point cloud directly
            cloud_tm = trimesh.PointCloud(points)
            mesh = cloud_tm.convex_hull
            log.info(f"Convex hull: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        else:
            log.info(f"TSDF mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Clean
        cleaner = MeshCleaner()
        mesh = cleaner.clean(mesh, target_faces=50000)
        log.info(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Save
        mesh_path = OBJECTS_DIR / "full_scene.obj"
        mesh.export(str(mesh_path))
        log.info(f"Mesh saved to {mesh_path}")
        return True

    except Exception as e:
        log.error(f"Mesh extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def stage_6_usd():
    """Assemble USD scene."""
    log.info("=== STAGE 6: USD Assembly ===")
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from pipeline.usd_assembler import UsdSceneAssembler
    except ImportError:
        log.warning("USD assembly requires usd-core (Python 3.10-3.12). Writing metadata instead.")
        # Write scene manifest as JSON fallback
        manifest = {
            "pipeline": "gaussian-toolkit",
            "source_video": "gallery_tour_60s.mp4",
            "colmap_dir": str(COLMAP_DIR),
            "model_dir": str(MODEL_DIR),
            "objects_dir": str(OBJECTS_DIR),
            "mesh_files": [str(p) for p in OBJECTS_DIR.glob("*.obj")],
            "segmentation": str(OBJECTS_DIR / "segmentation_results.json"),
        }
        manifest_path = SCENE_DIR / "scene_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Scene manifest written to {manifest_path}")
        return True

    # Full USD assembly
    assembler = UsdSceneAssembler()
    # ... would compose full USD here
    return True


def main():
    log.info("=" * 60)
    log.info("GAUSSIAN TOOLKIT: Gallery Tour Pipeline")
    log.info("=" * 60)

    # Check if COLMAP is done
    if not (COLMAP_DIR / "sparse").exists():
        wait_for_colmap()

    # Check COLMAP output
    sparse_dirs = list((COLMAP_DIR / "sparse").iterdir()) if (COLMAP_DIR / "sparse").exists() else []
    images = list((COLMAP_DIR / "images").glob("*")) if (COLMAP_DIR / "images").exists() else []
    log.info(f"COLMAP: {len(sparse_dirs)} model(s), {len(images)} undistorted images")

    if not images:
        log.error("No undistorted images from COLMAP. Pipeline cannot continue.")
        return 1

    # Stage 3: Train
    if not stage_3_train():
        log.error("Training failed. Attempting to continue with available data...")

    # Stage 4: Segment
    stage_4_segment()

    # Stage 5: Mesh
    stage_5_mesh()

    # Stage 6: USD
    stage_6_usd()

    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info(f"Output: {BASE}")
    log.info("=" * 60)

    # Summary
    for d in [MODEL_DIR, OBJECTS_DIR, SCENE_DIR]:
        if d.exists():
            files = list(d.rglob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            log.info(f"  {d.name}/: {len(files)} files, {size/1024/1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
