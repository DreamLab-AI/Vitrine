#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mask projection and per-object Gaussian extraction pipeline.

Loads trained 3DGS PLY, runs SAM2 auto-segmentation on all 121 frames with
cross-frame IoU tracking, projects masks onto Gaussians via majority voting,
and writes per-object PLY files.

Usage:
    python scripts/run_object_separation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("object_separation")

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
import os
TEST_DATA = Path(os.environ.get("TEST_DATA_DIR", str(PROJECT_ROOT.parent / "test-data"))) / "gallery_output"
PLY_PATH = TEST_DATA / "model" / "splat_7000.ply"
COLMAP_TXT_DIR = TEST_DATA / "colmap" / "undistorted" / "sparse" / "0_txt"
IMAGE_DIR = TEST_DATA / "colmap" / "undistorted" / "images"
OUTPUT_DIR = TEST_DATA / "objects"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Step 1: Load Gaussian positions from PLY
# ---------------------------------------------------------------------------
def load_gaussian_positions(ply_path: Path) -> np.ndarray:
    """Load (N, 3) float64 positions from a 3DGS PLY file."""
    from plyfile import PlyData

    logger.info("Loading PLY from %s", ply_path)
    t0 = time.time()
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    logger.info("Loaded %d Gaussians in %.1fs", len(xyz), time.time() - t0)
    return xyz, ply


def filter_outliers(xyz: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Return boolean mask of inliers (within percentile distance from median)."""
    median = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - median, axis=1)
    threshold = np.percentile(dists, percentile)
    mask = dists <= threshold
    logger.info(
        "Outlier filter: keeping %d / %d (threshold=%.4f)",
        mask.sum(), len(mask), threshold,
    )
    return mask


# ---------------------------------------------------------------------------
#  Step 2: Load COLMAP cameras
# ---------------------------------------------------------------------------
def load_colmap(colmap_dir: Path):
    """Parse COLMAP text-format cameras and images."""
    from src.pipeline.colmap_parser import parse_cameras_txt, parse_images_txt

    cameras = parse_cameras_txt(colmap_dir / "cameras.txt")
    images = parse_images_txt(colmap_dir / "images.txt")
    logger.info("COLMAP: %d cameras, %d images", len(cameras), len(images))
    return cameras, images


# ---------------------------------------------------------------------------
#  Step 3: SAM2 segmentation on all 121 frames
# ---------------------------------------------------------------------------
def run_sam2_segmentation(image_dir: Path):
    """Run SAM2 automatic segmentation with cross-frame IoU tracking."""
    from src.pipeline.sam2_segmentor import SAM2Segmentor

    seg = SAM2Segmentor(
        model_id="facebook/sam2-hiera-large",
        device="cuda",
        points_per_side=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )

    logger.info("Running SAM2 segment_video_auto on %s", image_dir)
    t0 = time.time()
    results = seg.segment_video_auto(
        image_dir,
        iou_threshold=0.30,
        extensions=(".jpg", ".jpeg", ".png"),
    )
    elapsed = time.time() - t0
    logger.info("SAM2 completed: %d frames in %.1fs", len(results), elapsed)

    # Collect stats
    all_ids = set()
    for r in results:
        all_ids.update(r.object_ids.tolist())
    logger.info("Total unique object IDs across all frames: %d", len(all_ids))

    # Free GPU memory
    seg.unload()

    return results


# ---------------------------------------------------------------------------
#  Step 4: Build label maps and project onto Gaussians
# ---------------------------------------------------------------------------
def project_masks_to_gaussians(
    gaussian_xyz: np.ndarray,
    cameras: dict,
    images: list,
    seg_results: list,
    image_dir: Path,
) -> np.ndarray:
    """Project SAM2 masks onto Gaussians via multi-view majority voting."""
    from src.pipeline.mask_projector import MaskProjector

    # Build image_name -> index mapping from sorted filenames
    frame_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    frame_names = [p.name for p in frame_paths]

    # Convert SegmentationResults to label maps
    label_maps = MaskProjector.segmentation_results_to_label_maps(
        seg_results, frame_names,
    )
    logger.info("Built label maps for %d frames", len(label_maps))

    # Build projector from COLMAP data
    # cameras is already Dict[int, ColmapCamera] from parse_cameras_txt
    # images is List[ColmapImage] from parse_images_txt - convert to dict
    cam_dict = cameras
    img_dict = {i.image_id: i for i in images}

    projector = MaskProjector(
        cameras=cam_dict,
        images=img_dict,
        image_hw=(899, 1600),  # H, W from the actual images
    )

    # Use batched assignment for memory efficiency
    logger.info("Projecting masks onto %d Gaussians across %d views",
                gaussian_xyz.shape[0], len(projector.views))
    t0 = time.time()
    labels = projector.assign_labels_batched(
        gaussian_xyz,
        label_maps,
        background_label=0,
        min_votes=2,
        batch_size=200_000,
    )
    elapsed = time.time() - t0
    logger.info("Projection complete in %.1fs", elapsed)

    return labels


# ---------------------------------------------------------------------------
#  Step 5: Save per-object PLY files
# ---------------------------------------------------------------------------
def save_per_object_ply(
    ply_data,
    labels: np.ndarray,
    inlier_mask: np.ndarray,
    output_dir: Path,
) -> dict:
    """Write a separate PLY for each object label. Returns summary stats."""
    from plyfile import PlyData, PlyElement

    vertex = ply_data["vertex"]
    all_data = vertex.data

    # Map labels back to full array (outliers get label 0)
    full_labels = np.zeros(len(all_data), dtype=np.int32)
    full_labels[inlier_mask] = labels

    unique_labels = np.unique(full_labels)
    unique_labels = unique_labels[unique_labels != 0]  # skip background

    logger.info("Saving %d object PLY files", len(unique_labels))
    summary = {}

    for obj_id in unique_labels:
        obj_mask = full_labels == int(obj_id)
        obj_count = int(obj_mask.sum())

        if obj_count < 10:
            continue

        obj_vertices = all_data[obj_mask]
        obj_xyz = np.stack([
            obj_vertices["x"], obj_vertices["y"], obj_vertices["z"]
        ], axis=1).astype(np.float64)

        centroid = obj_xyz.mean(axis=0).tolist()
        bbox_min = obj_xyz.min(axis=0).tolist()
        bbox_max = obj_xyz.max(axis=0).tolist()

        # Write PLY
        out_path = output_dir / f"object_{int(obj_id):03d}.ply"
        el = PlyElement.describe(obj_vertices, "vertex")
        PlyData([el], text=False).write(str(out_path))

        summary[int(obj_id)] = {
            "gaussian_count": obj_count,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "ply_file": out_path.name,
        }
        logger.info(
            "  Object %d: %d Gaussians, centroid=(%.3f, %.3f, %.3f)",
            obj_id, obj_count, *centroid,
        )

    # Also save background
    bg_mask = full_labels == 0
    bg_count = int(bg_mask.sum())
    if bg_count > 0:
        bg_vertices = all_data[bg_mask]
        bg_path = output_dir / "background.ply"
        el = PlyElement.describe(bg_vertices, "vertex")
        PlyData([el], text=False).write(str(bg_path))
        summary[0] = {
            "gaussian_count": bg_count,
            "centroid": [0, 0, 0],
            "bbox_min": [0, 0, 0],
            "bbox_max": [0, 0, 0],
            "ply_file": "background.ply",
            "note": "background / unlabeled Gaussians",
        }
        logger.info("  Background: %d Gaussians", bg_count)

    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # Step 1: Load PLY
    gaussian_xyz, ply_data = load_gaussian_positions(PLY_PATH)
    inlier_mask = filter_outliers(gaussian_xyz, percentile=95.0)
    filtered_xyz = gaussian_xyz[inlier_mask]

    # Step 2: Load COLMAP
    cameras, images = load_colmap(COLMAP_TXT_DIR)

    # Step 3: SAM2 on all 121 frames
    seg_results = run_sam2_segmentation(IMAGE_DIR)

    # Step 4: Project masks onto Gaussians
    labels = project_masks_to_gaussians(
        filtered_xyz, cameras, images, seg_results, IMAGE_DIR,
    )

    # Stats
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Label distribution:")
    for lbl, cnt in zip(unique, counts):
        logger.info("  Label %d: %d Gaussians (%.1f%%)",
                     lbl, cnt, 100.0 * cnt / len(labels))

    # Step 5: Save per-object PLY files
    summary = save_per_object_ply(ply_data, labels, inlier_mask, OUTPUT_DIR)

    # Write summary JSON
    summary_path = OUTPUT_DIR / "object_separation_summary.json"
    summary_out = {
        "total_gaussians": int(len(gaussian_xyz)),
        "inlier_gaussians": int(inlier_mask.sum()),
        "labeled_gaussians": int((labels != 0).sum()),
        "background_gaussians": int((labels == 0).sum()),
        "num_objects": len([k for k in summary if k != 0]),
        "objects": summary,
        "pipeline": {
            "sam2_model": "facebook/sam2-hiera-large",
            "points_per_side": 32,
            "iou_threshold": 0.30,
            "min_votes": 2,
            "outlier_percentile": 95.0,
            "num_views": 121,
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Objects: %d, Labeled: %d/%d (%.1f%%)",
                summary_out["num_objects"],
                summary_out["labeled_gaussians"],
                summary_out["inlier_gaussians"],
                100.0 * summary_out["labeled_gaussians"] / max(summary_out["inlier_gaussians"], 1))


if __name__ == "__main__":
    main()
