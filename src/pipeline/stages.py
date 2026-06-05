# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent pipeline stages callable by Claude Code or scripts.

Each stage is a self-contained method that takes explicit inputs and returns
explicit outputs as a dict. There is no hidden state, no state machine, and
no automatic transitions. Claude Code (or any caller) decides what to run
next based on the results.

Usage from the terminal::

    from pipeline.stages import PipelineStages
    p = PipelineStages('/data/output/JOB_ID')
    result = p.ingest('/data/output/JOB_ID/input.mp4', fps=2.0)
    print(result)

All heavy imports (torch, cv2, trimesh, etc.) are deferred to the methods
that need them so that the module loads instantly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage names (ordered) -- used by the web UI for display
# ---------------------------------------------------------------------------

STAGE_NAMES: list[str] = [
    "ingest",
    "remove_people",
    "select_frames",
    "reconstruct",
    "train",
    "render_previews",
    "segment",
    "extract_objects",
    "mesh_objects",
    "texture_bake",
    "assemble_usd",
    "validate",
]


# ---------------------------------------------------------------------------
# Result dataclass returned by every stage
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Outcome of a single pipeline stage execution."""
    success: bool
    stage: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Static helpers (carried over from old orchestrator)
# ---------------------------------------------------------------------------

def _load_ply_points(ply_path: str) -> tuple:
    """Load points and colors from a PLY file (handles both 3DGS and standard PLYs).

    Returns:
        Tuple of (points: np.ndarray Nx3, colors: np.ndarray Nx3 or None)
    """
    import numpy as np

    points = None
    colors = None

    # Try plyfile first (handles 3DGS PLYs with custom properties)
    try:
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        vertex = ply["vertex"]
        x = np.asarray(vertex["x"])
        y = np.asarray(vertex["y"])
        z = np.asarray(vertex["z"])
        points = np.column_stack([x, y, z])

        for r_name, g_name, b_name in [
            ("red", "green", "blue"),
            ("f_dc_0", "f_dc_1", "f_dc_2"),
        ]:
            if r_name in vertex.data.dtype.names:
                r = np.asarray(vertex[r_name])
                g = np.asarray(vertex[g_name])
                b = np.asarray(vertex[b_name])
                c = np.column_stack([r, g, b])
                if c.max() <= 1.0:
                    c = (c * 255).clip(0, 255).astype(np.uint8)
                elif c.max() > 255:
                    c = ((c * 0.2822 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                colors = c
                break

        valid = np.isfinite(points).all(axis=1)
        if not valid.all():
            points = points[valid]
            if colors is not None:
                colors = colors[valid]

        return points, colors
    except Exception:
        pass

    # Fallback: trimesh
    try:
        import trimesh
        loaded = trimesh.load(ply_path)
        if hasattr(loaded, "vertices") and loaded.vertices is not None:
            points = np.asarray(loaded.vertices)
        elif hasattr(loaded, "points"):
            points = np.asarray(loaded.points) if hasattr(loaded.points, '__len__') else None

        if points is not None:
            try:
                if hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors"):
                    vc = np.asarray(loaded.visual.vertex_colors)
                    if vc.ndim == 2 and vc.shape[1] >= 3:
                        colors = vc[:, :3]
            except Exception:
                pass

            valid = np.isfinite(points).all(axis=1)
            if not valid.all():
                points = points[valid]
                if colors is not None:
                    colors = colors[valid]

        return points, colors
    except Exception:
        pass

    return None, None


def _mesh_with_open3d(ply_path: str, output_path: str) -> None:
    """Fallback Poisson surface reconstruction using Open3D."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_normals():
        pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.io.write_triangle_mesh(output_path, mesh)


def _write_minimal_usda(path: Path, meshes: list[dict[str, Any]]) -> None:
    """Write a minimal USDA that references extracted meshes."""
    lines = [
        '#usda 1.0',
        '(',
        '    defaultPrim = "Scene"',
        '    metersPerUnit = 1',
        '    upAxis = "Y"',
        ')',
        '',
        'def Xform "Scene"',
        '{',
    ]
    for i, mesh in enumerate(meshes):
        label = mesh.get("label", f"object_{i}").replace(" ", "_")
        mesh_path = mesh.get("mesh", "")
        lines.append(f'    def Xform "{label}"')
        lines.append('    {')
        lines.append(f'        # Reference: {mesh_path}')
        lines.append(f'        custom string mesh:path = "{mesh_path}"')
        lines.append('    }')
    lines.append('}')
    lines.append('')
    path.write_text("\n".join(lines), encoding="utf-8")



# _find_usd_python removed: python3 (3.12) has usd-core installed directly


def _get_file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# PipelineStages -- the main class
# ---------------------------------------------------------------------------

class PipelineStages:
    """Individual pipeline stages callable by Claude Code or scripts.

    Each method is self-contained, takes explicit inputs, returns explicit
    outputs as a StageResult. No hidden state, no automatic transitions.
    Claude Code calls them in sequence, inspecting results between each call.
    """

    def __init__(self, job_dir: str, config: PipelineConfig | None = None) -> None:
        self.job_dir = Path(job_dir)
        self.config = config or PipelineConfig()
        self.config.output_dir = str(self.job_dir)

        # Pre-flight dependency check -- fail hard if critical deps are missing
        from pipeline.preflight import check_all as preflight_check
        self._preflight = preflight_check()

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------

    def ingest(self, video_path: str, fps: float = 2.0) -> StageResult:
        """Extract frames from video.

        Returns: {frames_dir, frame_count}
        """
        video = Path(video_path)
        if not video.exists():
            return StageResult(
                success=False, stage="ingest",
                error=f"Video not found: {video_path}",
            )

        frame_dir = self.job_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(frame_dir / "frame_%05d.jpg"),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            return StageResult(success=False, stage="ingest", error="ffmpeg not found in PATH")
        except subprocess.TimeoutExpired:
            return StageResult(success=False, stage="ingest", error="Frame extraction timed out (300s)")

        if proc.returncode != 0:
            return StageResult(
                success=False, stage="ingest",
                error=f"ffmpeg failed (rc={proc.returncode}): {proc.stderr[:500]}",
            )

        frames = sorted(frame_dir.glob("*.jpg"))
        return StageResult(
            success=True, stage="ingest",
            metrics={"frame_count": len(frames), "fps": fps},
            artifacts={"frames_dir": str(frame_dir), "frame_count": str(len(frames))},
        )

    # ------------------------------------------------------------------
    # Stage 2: Remove people
    # ------------------------------------------------------------------

    def remove_people(self, frames_dir: str) -> StageResult:
        """Detect and remove people from frames.

        Returns: {cleaned_dir, manifest}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="remove_people",
                error=f"Frames directory not found: {frames_dir}",
            )

        cfg = self.config.person_removal
        if not cfg.enabled:
            logger.info("Person removal disabled, skipping")
            return StageResult(
                success=True, stage="remove_people",
                metrics={"skipped": True},
                artifacts={"cleaned_frames_dir": frames_dir},
            )

        from pipeline.person_remover import PersonRemover

        cleaned_dir = self.job_dir / "frames_cleaned"
        remover = PersonRemover(
            method=cfg.method,
            comfyui_url=cfg.comfyui_url or None,
            flux_endpoint=cfg.flux_endpoint or None,
            confidence=cfg.confidence,
            dilation_px=cfg.dilation_px,
            drop_threshold=cfg.drop_threshold,
            flag_threshold=cfg.flag_threshold,
            comfyui_timeout=cfg.comfyui_timeout,
        )

        try:
            manifest = remover.process_directory(frames_dir, str(cleaned_dir))
        except Exception as exc:
            return StageResult(
                success=False, stage="remove_people",
                error=f"Person removal failed: {exc}",
            )

        summary = manifest.get("summary", {})
        remaining = summary.get("clean", 0) + summary.get("inpainted", 0) + summary.get("flagged_inpainted", 0)

        if remaining < self.config.ingest.min_frames:
            return StageResult(
                success=False, stage="remove_people",
                error=f"Too few frames after person removal: {remaining} (need {self.config.ingest.min_frames})",
                metrics=summary,
            )

        manifest_path = cleaned_dir / "person_removal_manifest.json"
        return StageResult(
            success=True, stage="remove_people",
            metrics={"total_frames": manifest.get("total_frames", 0), "remaining_frames": remaining, **summary},
            artifacts={"cleaned_frames_dir": str(cleaned_dir), "manifest": str(manifest_path)},
        )

    # ------------------------------------------------------------------
    # Stage 3: Select frames
    # ------------------------------------------------------------------

    def select_frames(
        self,
        frames_dir: str,
        target: int = 150,
        camera_positions: "Optional[np.ndarray]" = None,
    ) -> StageResult:
        """Select best frames for COLMAP.

        Args:
            frames_dir: Directory of extracted frames to score and curate.
            target: Desired number of selected frames.
            camera_positions: Optional ``(N, 3)`` array of camera centres
                aligned to the scored frames, for Fibonacci-coverage selection
                (ADR-007). Normally ``None`` on the first (pre-COLMAP) pass —
                poses do not exist yet — so callers may run a second selection
                pass after ``reconstruct`` to activate coverage. If ``None`` and
                ``ingest.use_fibonacci_coverage`` is set, this stage tries to
                load centres from an existing COLMAP text model in the job dir.

        Returns: {selected_dir, count}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="select_frames",
                error=f"Frames directory not found: {frames_dir}",
            )

        from pipeline.frame_selector import FrameSelector, SelectionConfig

        sel_cfg = SelectionConfig(
            target_frames=min(self.config.ingest.max_frames, max(self.config.ingest.min_frames, target)),
            min_frames=self.config.ingest.min_frames,
            max_frames=self.config.ingest.max_frames,
            blur_threshold=self.config.ingest.blur_threshold,
            # ADR-007 Fibonacci-sphere coverage selection (off by default).
            use_fibonacci_coverage=self.config.ingest.use_fibonacci_coverage,
            coverage_weight=self.config.ingest.coverage_weight,
        )
        selector = FrameSelector(config=sel_cfg)

        try:
            scores = selector.score_frames(frames_dir)
        except Exception as exc:
            logger.warning("Frame scoring failed (%s), keeping all frames", exc)
            return StageResult(
                success=True, stage="select_frames",
                metrics={"skipped": True, "reason": str(exc)},
                artifacts={"selected_frames_dir": frames_dir},
            )

        if not scores:
            return StageResult(
                success=False, stage="select_frames",
                error=f"No frames found in {frames_dir}",
            )

        # Fibonacci coverage (ADR-007) needs camera poses. They do not exist on
        # the first pass (before COLMAP); if enabled and not supplied, try to
        # load centres from an existing COLMAP model (two-pass re-selection),
        # else fall back to quality-only selection below.
        coverage_used = False
        if self.config.ingest.use_fibonacci_coverage and camera_positions is None:
            camera_positions = self._load_colmap_camera_centers(scores)
            if camera_positions is None:
                logger.info(
                    "Fibonacci coverage enabled but no camera poses available yet "
                    "(no COLMAP text model in job dir); using quality-only selection. "
                    "Run a second select_frames pass after reconstruct to apply coverage."
                )
        coverage_used = (
            self.config.ingest.use_fibonacci_coverage and camera_positions is not None
        )

        selected = selector.select(scores, camera_positions=camera_positions)

        if len(selected) < self.config.ingest.min_frames:
            return StageResult(
                success=False, stage="select_frames",
                error=f"Only {len(selected)} frames after selection (need {self.config.ingest.min_frames})",
                metrics={"scored": len(scores), "selected": len(selected)},
            )

        selected_dir = self.job_dir / "frames_selected"
        selector.copy_selected(selected, str(selected_dir))

        # FR-3 / ADR-009: write a per-image metadata sidecar for each selected
        # frame so the quality scores (otherwise discarded after selection)
        # become a durable lineage root (video→frame→object→USD). Best-effort.
        sidecars = self._write_frame_sidecars(selected, selected_dir)

        return StageResult(
            success=True, stage="select_frames",
            metrics={
                "scored": len(scores),
                "selected": len(selected),
                "target": sel_cfg.target_frames,
                "fibonacci_coverage": coverage_used,
                "sidecars_written": sidecars,
            },
            artifacts={"selected_frames_dir": str(selected_dir), "count": str(len(selected))},
        )

    def _write_frame_sidecars(self, selected: list, selected_dir: Path) -> int:
        """Write a ``<frame>.json`` metadata sidecar (schema ``v2g.frame.1``) for
        each selected frame, under ``frames_selected/frame_metadata/`` so the
        image directory stays clean for COLMAP. Captures the quality scores that
        are otherwise discarded after selection (FR-3 / ADR-009). Fields not yet
        known at this stage (source_video, capture_session, source_timestamp_pts,
        pose_hint) are written as ``null`` and backfilled by later stages.
        Never raises.
        """
        meta_dir = selected_dir / "frame_metadata"
        try:
            meta_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("sidecar dir create failed: %s", exc)
            return 0

        written = 0
        for s in selected:
            try:
                payload = {
                    "schema_version": "v2g.frame.1",
                    "frame_name": Path(s.path).name,
                    "source_video": None,
                    "capture_session": None,
                    "frame_index": getattr(s, "index", None),
                    "source_timestamp_pts": None,
                    "blur_score": getattr(s, "blur_score", None),
                    "exposure_score": getattr(s, "exposure_score", None),
                    "sharpness": getattr(s, "sharpness", None),
                    "phash": getattr(s, "phash", None),
                    "composite_score": getattr(s, "composite_score", None),
                    "kept": True,
                    "selection_reason": "quality+coverage",
                    "pose_hint": None,
                }
                (meta_dir / f"{Path(s.path).stem}.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                written += 1
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("sidecar write failed for %s: %s", getattr(s, "path", "?"), exc)
        return written

    def _load_colmap_camera_centers(self, scores: list) -> "Optional[np.ndarray]":
        """Best-effort load of per-frame camera centres from an existing COLMAP
        text model in the job dir, aligned to ``scores`` order (row i ↔ scores[i]).

        Returns an ``(N, 3)`` float array, or ``None`` when no usable text model
        is found or too few frames are registered. COLMAP *binary* models are not
        parsed (``colmap_parser`` handles ``.txt`` only); callers then fall back
        to quality-only selection. Never raises.
        """
        try:
            import numpy as np
            from pipeline.colmap_parser import parse_images_txt
        except Exception as exc:  # pragma: no cover - import guard
            logger.debug("camera-centre load skipped (import failed): %s", exc)
            return None

        candidates = [
            self.job_dir / "colmap" / "sparse" / "0" / "images.txt",
            self.job_dir / "colmap" / "sparse" / "images.txt",
            self.job_dir / "colmap" / "undistorted" / "sparse" / "0" / "images.txt",
            self.job_dir / "colmap" / "undistorted" / "sparse" / "images.txt",
        ]
        images_txt = next((c for c in candidates if c.exists()), None)
        if images_txt is None:
            return None

        try:
            images = parse_images_txt(images_txt)
            name_to_center: dict[str, Any] = {}
            for img in images:
                name = getattr(img, "name", None)
                if not name:
                    continue
                q = np.array(img.quaternion(), dtype=np.float64)  # wxyz
                norm = float(np.linalg.norm(q))
                if norm == 0.0:
                    continue
                w, x, y, z = q / norm
                R = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ], dtype=np.float64)
                t = np.array(img.translation(), dtype=np.float64)
                name_to_center[name] = -R.T @ t

            if not name_to_center:
                return None

            positions = np.full((len(scores), 3), np.nan, dtype=np.float64)
            matched = 0
            for i, s in enumerate(scores):
                center = name_to_center.get(Path(s.path).name)
                if center is not None:
                    positions[i] = center
                    matched += 1

            if matched < max(2, int(0.5 * len(scores))):
                logger.info(
                    "Only %d/%d frames registered in COLMAP model; skipping coverage",
                    matched, len(scores),
                )
                return None

            # Replace any unmatched rows with the centroid so they read as
            # low-coverage rather than producing NaNs downstream.
            nan_rows = np.isnan(positions).any(axis=1)
            if nan_rows.any():
                centroid = positions[~nan_rows].mean(axis=0)
                positions[nan_rows] = centroid

            logger.info("Loaded %d/%d camera centres for Fibonacci coverage", matched, len(scores))
            return positions
        except Exception as exc:
            logger.debug("camera-centre load failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Stage 4: Reconstruct (COLMAP SfM)
    # ------------------------------------------------------------------

    def reconstruct(self, frames_dir: str) -> StageResult:
        """Run COLMAP SfM reconstruction.

        Returns: {colmap_dir, cameras, points}
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="reconstruct",
                error=f"Frames directory not found: {frames_dir}",
            )

        colmap_dir = self.job_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)

        # Try SplatReady first
        splatready_config = {
            "video_path": "",
            "base_output_folder": str(self.job_dir),
            "frame_rate": self.config.ingest.fps,
            "skip_extraction": True,
            "reconstruction_method": self.config.reconstruct.method,
            "colmap_exe_path": self.config.reconstruct.colmap_exe,
            "use_fisheye": self.config.reconstruct.use_fisheye,
            "max_image_size": self.config.ingest.max_image_size,
            "min_scale": self.config.reconstruct.min_scale,
            "skip_reconstruction": False,
        }

        config_path = self.job_dir / "splatready_config.json"
        config_path.write_text(json.dumps(splatready_config, indent=2), encoding="utf-8")

        plugin_dir = Path.home() / ".lichtfeld" / "plugins" / "splat_ready"
        runner = plugin_dir / "core" / "runner.py"

        if runner.exists():
            try:
                proc = subprocess.run(
                    ["python3", str(runner), str(config_path)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    return StageResult(
                        success=False, stage="reconstruct",
                        error=f"SplatReady failed: {proc.stderr[:500]}",
                    )
            except subprocess.TimeoutExpired:
                return StageResult(
                    success=False, stage="reconstruct",
                    error="COLMAP reconstruction timed out",
                )
        else:
            # Fallback: run COLMAP directly
            try:
                self._run_colmap_direct(colmap_dir, frames_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return StageResult(
                    success=False, stage="reconstruct",
                    error=f"COLMAP failed: {exc}",
                )

        dataset_dir = colmap_dir / "undistorted"
        if not dataset_dir.exists():
            for alt in [colmap_dir, self.job_dir / "colmap"]:
                if (alt / "images").exists() and (alt / "sparse").exists():
                    dataset_dir = alt
                    break
            else:
                return StageResult(
                    success=False, stage="reconstruct",
                    error="COLMAP output not found at expected paths",
                )

        # Check COLMAP registration rate
        sparse_images_bin = dataset_dir / "sparse" / "0" / "images.bin"
        if sparse_images_bin.exists():
            import struct as _struct
            try:
                with open(sparse_images_bin, 'rb') as f:
                    n_registered = _struct.unpack('<Q', f.read(8))[0]
                n_total = len(list(frames_path.glob('*.jpg'))) + len(list(frames_path.glob('*.png')))
                rate = n_registered / max(n_total, 1)
                logger.info("COLMAP registration: %d/%d (%.0f%%)", n_registered, n_total, rate * 100)
                if rate < 0.3:
                    return StageResult(
                        success=False, stage="reconstruct",
                        error=f"COLMAP only registered {rate*100:.0f}% of frames (need >30%)",
                        metrics={"registered": n_registered, "total": n_total, "rate": round(rate, 3)},
                    )
            except Exception as reg_exc:
                logger.warning("Could not check registration rate: %s", reg_exc)

        # Count cameras / points if available
        sparse_dir = dataset_dir / "sparse" / "0"
        cameras = len(list(sparse_dir.glob("cameras.*"))) if sparse_dir.exists() else 0
        points_file = sparse_dir / "points3D.bin" if sparse_dir.exists() else None
        points_size = _get_file_size_mb(points_file) if points_file else 0.0

        return StageResult(
            success=True, stage="reconstruct",
            metrics={"colmap_dir": str(dataset_dir), "cameras_found": cameras, "points_size_mb": points_size},
            artifacts={"colmap_dir": str(dataset_dir)},
        )

    def _run_colmap_direct(self, output_dir: Path, frame_dir: Path) -> None:
        """Fallback: run COLMAP feature extraction + matching + mapper."""
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        colmap = self.config.reconstruct.colmap_exe

        subprocess.run([
            colmap, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--ImageReader.single_camera", "1",
        ], check=True, capture_output=True, timeout=300)

        matcher_type = self.config.reconstruct.matcher + "_matcher"
        matcher_cmd = [
            colmap, matcher_type,
            "--database_path", str(db_path),
        ]
        # Add sequential-matcher-specific parameters for better overlap coverage
        if self.config.reconstruct.matcher == "sequential":
            matcher_cmd += [
                "--SequentialMatching.overlap", "30",
                "--SequentialMatching.loop_detection", "1",
            ]
        subprocess.run(matcher_cmd, check=True, capture_output=True, timeout=600)

        subprocess.run([
            colmap, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.multiple_models", "0",
        ], check=True, capture_output=True, timeout=1800)

        undist_dir = output_dir / "undistorted"
        subprocess.run([
            colmap, "image_undistorter",
            "--image_path", str(frame_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(undist_dir),
            "--output_type", "COLMAP",
        ], check=True, capture_output=True, timeout=300)

        undist_sparse = undist_dir / "sparse"
        if undist_sparse.exists() and not (undist_sparse / "0").exists():
            (undist_sparse / "0").mkdir(exist_ok=True)
            for f in undist_sparse.glob("*.bin"):
                shutil.move(str(f), str(undist_sparse / "0" / f.name))

    # ------------------------------------------------------------------
    # Stage 5: Train (Gaussian splatting)
    # ------------------------------------------------------------------

    def train(self, colmap_dir: str, iterations: int = 30000) -> StageResult:
        """Run gaussian splatting training.

        Supports multiple backends via ``config.training.mesh_method``:

        - ``tsdf`` (default): LichtFeld headless training, mesh extracted later
          in ``mesh_objects``.
        - ``milo``: MILo (SIGGRAPH Asia 2025) differentiable mesh-in-the-loop
          training. Produces BOTH the gaussian PLY and a high-quality mesh in a
          single pass, so the ``mesh_objects`` stage can be skipped for the
          scene-level mesh.

        Returns: {ply_path, loss} -- and for MILo also {milo_mesh_path}
        """
        dataset_dir = Path(colmap_dir)
        if not dataset_dir.exists():
            return StageResult(
                success=False, stage="train",
                error=f"COLMAP directory not found: {colmap_dir}",
            )

        # -- Resolve effective backend (auto-selection per ADR-003) -----------
        effective_method = self.config.training.mesh_method
        if (
            effective_method == "auto"
            or (effective_method == "tsdf" and self.config.training.mesh_backend_auto)
        ):
            effective_method = self._select_mesh_backend()

        # -- MILo backend ----------------------------------------------------
        if effective_method == "milo":
            return self._train_milo(colmap_dir, dataset_dir)

        # -- CoMe backend ----------------------------------------------------
        if effective_method == "come":
            return self._train_come(colmap_dir, dataset_dir)

        # -- GaussianWrapping backend ----------------------------------------
        if effective_method == "gaussianwrapping":
            return self._train_gaussianwrapping(colmap_dir, dataset_dir)

        # -- Default LichtFeld backend ---------------------------------------
        model_dir = self.job_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        lfs_binary = self.config.training.lichtfeld_binary
        if not Path(lfs_binary).exists():
            for candidate in [
                "/opt/gaussian-toolkit/build/LichtFeld-Studio",
                "/usr/local/bin/lichtfeld-studio",
                str(Path.home() / "workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio"),
            ]:
                if Path(candidate).exists():
                    lfs_binary = candidate
                    break

        if not Path(lfs_binary).exists():
            return StageResult(
                success=False, stage="train",
                error=f"LichtFeld binary not found at {lfs_binary}",
            )

        strategy = self.config.training.resolved_strategy()
        sh_degree = self.config.training.resolved_sh_degree()
        effective_iters = self.config.training.resolved_iterations()
        # Caller-supplied iterations override only for default preset
        if self.config.training.scene_preset == "default" and iterations != 30000:
            effective_iters = iterations

        train_cmd = [
            lfs_binary, "--headless",
            "--data-path", str(dataset_dir),
            "--output-path", str(model_dir),
            "--iter", str(effective_iters),
            "--strategy", strategy,
            "--sh-degree", str(sh_degree),
            "--log-level", "info",
        ]

        # Indoor-reflective preset: write a config JSON with bg_color and
        # regularization overrides that cannot be set via CLI flags alone.
        indoor_config_path = None
        if self.config.training.scene_preset == "indoor_reflective":
            indoor_config_path = model_dir / "_indoor_overrides.json"
            bg = self.config.training.indoor_bg_color
            indoor_overrides = {
                "sh_degree": sh_degree,
                "strategy": strategy,
                "iterations": effective_iters,
                "opacity_reg": self.config.training.indoor_opacity_reg,
                "scale_reg": self.config.training.indoor_scale_reg,
                "bg_color": list(bg),
                "bg_modulation": False,
                "mip_filter": True,
            }
            indoor_config_path.write_text(
                json.dumps(indoor_overrides, indent=2), encoding="utf-8",
            )
            train_cmd.extend(["--config", str(indoor_config_path)])
            train_cmd.append("--enable-mip")

        logger.info("Training: %s", " ".join(train_cmd))

        try:
            # The prebuilt LichtFeld binary links the host's CUDA runtime (e.g.
            # libcudart.so.13, absent from the container toolkit) and the vcpkg
            # OpenUSD libs (libusd_*.so). The compose mounts host CUDA libs at
            # /opt/host-cuda-libs; the USD libs sit under the build's
            # vcpkg_installed tree. Add all of them to LD_LIBRARY_PATH.
            _blib = Path(lfs_binary).parent
            _ld = ":".join(p for p in [
                str(_blib),
                str(_blib / "vcpkg_installed" / "x64-linux" / "debug" / "lib"),
                str(_blib / "vcpkg_installed" / "x64-linux" / "lib"),
                "/opt/host-cuda-libs",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ] if p)
            proc = subprocess.run(
                train_cmd, capture_output=True, text=True,
                timeout=3600,
                env={**os.environ, "LD_LIBRARY_PATH": _ld},
            )
            if proc.returncode != 0:
                logger.warning("Training stderr: %s", proc.stderr[-500:] if proc.stderr else "none")
        except subprocess.TimeoutExpired:
            return StageResult(success=False, stage="train", error="Training timed out (3600s)")

        # Find trained PLY
        ply_files = sorted(model_dir.rglob("*.ply"))
        if not ply_files:
            return StageResult(
                success=False, stage="train",
                error="No PLY output from training",
                metrics={"stderr_tail": (proc.stderr or "")[-300:]},
            )

        ply_path = ply_files[-1]
        ply_size_mb = _get_file_size_mb(ply_path)

        result = StageResult(
            success=True, stage="train",
            metrics={
                "ply_path": str(ply_path),
                "ply_size_mb": round(ply_size_mb, 1),
                "iterations": effective_iters,
                "strategy": strategy,
                "sh_degree": sh_degree,
                "scene_preset": self.config.training.scene_preset,
            },
            artifacts={"ply_path": str(ply_path), "model_dir": str(model_dir)},
        )
        self._optimize_splat(ply_path, result)
        return result

    def _train_milo(self, colmap_dir: str, dataset_dir: Path) -> StageResult:
        """Run MILo training + mesh extraction in the isolated conda env.

        MILo produces both gaussian splats AND a mesh in a single pipeline,
        so when this succeeds the caller can skip the ``mesh_objects`` stage
        for the scene-level mesh.
        """
        from pipeline.milo_extractor import run_milo, is_milo_available, MiloConfig

        if not is_milo_available():
            logger.warning("MILo requested but not available, falling back to LichtFeld")
            # Recurse with tsdf to use the default backend
            original_method = self.config.training.mesh_method
            self.config.training.mesh_method = "tsdf"
            result = self.train(colmap_dir, self.config.training.iterations)
            self.config.training.mesh_method = original_method
            if result.success:
                result.metrics["fallback"] = "lichtfeld (milo unavailable)"
            return result

        milo_output = self.job_dir / "model_milo"

        milo_cfg = MiloConfig(
            imp_metric="indoor",
            iterations=self.config.training.iterations
            if self.config.training.iterations <= 30000
            else 18000,
        )

        logger.info("Running MILo training on %s -> %s", colmap_dir, milo_output)
        milo_result = run_milo(str(dataset_dir), str(milo_output), config=milo_cfg)

        if not milo_result["success"]:
            return StageResult(
                success=False, stage="train",
                error=f"MILo failed: {milo_result['error']}",
                metrics={"backend": "milo", "duration": milo_result["duration"]},
            )

        artifacts: dict[str, str] = {"model_dir": str(milo_output)}
        metrics: dict[str, Any] = {
            "backend": "milo",
            "duration": round(milo_result["duration"], 1),
            "milo_mesh_path": milo_result["mesh_path"],
        }

        if milo_result["ply_path"]:
            ply_path = Path(milo_result["ply_path"])
            artifacts["ply_path"] = str(ply_path)
            metrics["ply_path"] = str(ply_path)
            metrics["ply_size_mb"] = round(_get_file_size_mb(ply_path), 1)

        if milo_result["mesh_path"]:
            artifacts["milo_mesh_path"] = milo_result["mesh_path"]

        # Copy MILo mesh to standard viewer location
        milo_mesh = Path(milo_result.get("glb_path") or milo_result.get("mesh_path", ""))
        if milo_mesh.exists():
            std_mesh_dir = self.job_dir / "objects" / "meshes" / "full_scene"
            std_mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = std_mesh_dir / "full_scene.glb"
            shutil.copy2(str(milo_mesh), str(dest))
            artifacts["scene_mesh_glb"] = str(dest)
            logger.info("MILo mesh copied to %s", dest)

        stage_result = StageResult(
            success=True, stage="train",
            metrics=metrics,
            artifacts=artifacts,
        )
        ply_for_opt = Path(artifacts["ply_path"]) if "ply_path" in artifacts else None
        if ply_for_opt:
            self._optimize_splat(ply_for_opt, stage_result)
        return stage_result

    def _select_mesh_backend(self) -> str:
        """Apply the ADR-003 auto-selection policy and return a concrete backend name.

        Precedence (highest → lowest), per the ADR-003 decision table:
        1. thin-structure hint → gaussianwrapping (when available)
        2. speed priority AND come available → come (~25 min vs MILo's ~69 min)
        3. milo available → milo  (the DEFAULT high-quality path)
        4. come available → come  (fallback when MILo's sidecar is down)
        5. fallback → tsdf

        Note CoMe does NOT override MILo by default — MILo is the default
        high-quality path and CoMe is chosen only when speed is prioritised
        (``training.mesh_speed_priority``). The thin-structure hint is carried
        as ``LICHTFELD_THIN_STRUCTURE=1`` for now (full SAM-label integration
        is a Phase-2c item per ADR-003).
        """
        import os as _os
        from pipeline.come_extractor import is_come_available
        from pipeline.milo_extractor import is_milo_available
        from pipeline.gaussianwrapping_extractor import is_gaussianwrapping_available

        thin_hint = _os.environ.get("LICHTFELD_THIN_STRUCTURE", "0").strip().lower() in ("1", "true", "yes")
        speed_priority = bool(getattr(self.config.training, "mesh_speed_priority", False))

        # 1. Thin structures → GaussianWrapping
        if thin_hint and is_gaussianwrapping_available():
            logger.info("Auto backend: gaussianwrapping (thin-structure hint)")
            return "gaussianwrapping"

        # 2. Speed priority → CoMe (only when explicitly prioritised)
        if speed_priority and is_come_available():
            logger.info("Auto backend: come (speed priority)")
            return "come"

        # 3. Default high-quality path → MILo
        if is_milo_available():
            logger.info("Auto backend: milo (default high-quality)")
            return "milo"

        # 4. MILo unavailable → CoMe as the next-best quality/speed option
        if is_come_available():
            logger.info("Auto backend: come (milo unavailable)")
            return "come"

        # 5. Nothing available → TSDF fallback
        logger.info("Auto backend: tsdf (fallback)")
        return "tsdf"

    def _optimize_splat(self, ply_path: Path, stage_result: StageResult) -> None:
        """Optionally run splat-transform on *ply_path* (ADR-006).

        Non-fatal: on any error the stage_result is annotated but not failed.
        Only runs when delivery.enable_splat_optimize is True and the CLI is
        available.  The original PLY is never modified.
        """
        if not self.config.delivery.enable_splat_optimize:
            return

        try:
            from pipeline.splat_optimizer import (
                is_splat_transform_available,
                optimize,
                SplatOptConfig,
            )
        except ImportError:
            logger.warning("splat_optimizer not importable; skipping splat optimisation")
            return

        if not is_splat_transform_available():
            logger.warning("splat-transform CLI not available; skipping splat optimisation")
            return

        d = self.config.delivery
        opt_cfg = SplatOptConfig(
            opacity_min_threshold=d.opacity_min_threshold,
            max_scale=d.max_scale,
            sort=d.sort,
            output_format=d.output_format,
            generate_html_viewer=d.generate_html_viewer,
        )

        delivery_dir = self.job_dir / "delivery"
        logger.info("Running splat optimisation: %s -> %s", ply_path, delivery_dir)
        opt_result = optimize(str(ply_path), str(delivery_dir), opt_cfg)

        if opt_result["success"]:
            stage_result.artifacts["delivery_splat"] = opt_result["output_path"]
            stage_result.metrics["splat_compression_ratio"] = round(
                opt_result["compression_ratio"], 2
            )
            stage_result.metrics["splat_output_size_mb"] = round(
                opt_result["output_size_mb"], 1
            )
            logger.info(
                "Splat optimisation complete: %s (%.1fx compression)",
                opt_result["output_path"],
                opt_result["compression_ratio"],
            )
        else:
            logger.warning(
                "Splat optimisation failed (non-fatal): %s", opt_result["error"]
            )
            stage_result.metrics["splat_optimize_error"] = opt_result["error"]

    def _train_come(self, colmap_dir: str, dataset_dir: Path) -> StageResult:
        """Run CoMe training + mesh extraction in the dedicated sidecar.

        CoMe produces both gaussian splats AND a high-quality mesh in a
        single pipeline pass (confidence-based marching tetrahedra), so when
        this succeeds the ``mesh_objects`` stage can be skipped for the
        scene-level mesh.  Falls back to tsdf on availability failure.

        See ADR-004 for the licensing gate and sidecar requirements.
        """
        from pipeline.come_extractor import run_come, is_come_available, CoMeConfig

        if not is_come_available():
            logger.warning("CoMe requested but not available, falling back to LichtFeld")
            original_method = self.config.training.mesh_method
            self.config.training.mesh_method = "tsdf"
            result = self.train(colmap_dir, self.config.training.iterations)
            self.config.training.mesh_method = original_method
            if result.success:
                result.metrics["fallback"] = "lichtfeld (come unavailable)"
            return result

        come_output = self.job_dir / "model_come"

        come_cfg = CoMeConfig(
            iterations=self.config.training.iterations
            if self.config.training.iterations <= 30000
            else 30000,
        )

        logger.info("Running CoMe training on %s -> %s", colmap_dir, come_output)
        come_result = run_come(str(dataset_dir), str(come_output), config=come_cfg)

        if not come_result["success"]:
            return StageResult(
                success=False, stage="train",
                error=f"CoMe failed: {come_result['error']}",
                metrics={"backend": "come", "duration": come_result["duration"]},
            )

        artifacts: dict[str, str] = {"model_dir": str(come_output)}
        metrics: dict[str, Any] = {
            "backend": "come",
            "duration": round(come_result["duration"], 1),
            "come_mesh_path": come_result["mesh_path"],
        }

        if come_result["ply_path"]:
            ply_path = Path(come_result["ply_path"])
            artifacts["ply_path"] = str(ply_path)
            metrics["ply_path"] = str(ply_path)
            metrics["ply_size_mb"] = round(_get_file_size_mb(ply_path), 1)

        if come_result["mesh_path"]:
            artifacts["come_mesh_path"] = come_result["mesh_path"]

        # Copy CoMe mesh to standard viewer location
        come_mesh = Path(come_result.get("glb_path") or come_result.get("mesh_path", ""))
        if come_mesh.exists():
            std_mesh_dir = self.job_dir / "objects" / "meshes" / "full_scene"
            std_mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = std_mesh_dir / "full_scene.glb"
            shutil.copy2(str(come_mesh), str(dest))
            artifacts["scene_mesh_glb"] = str(dest)
            logger.info("CoMe mesh copied to %s", dest)

        stage_result = StageResult(
            success=True, stage="train",
            metrics=metrics,
            artifacts=artifacts,
        )
        ply_for_opt = Path(artifacts["ply_path"]) if "ply_path" in artifacts else None
        if ply_for_opt:
            self._optimize_splat(ply_for_opt, stage_result)
        return stage_result

    def _train_gaussianwrapping(self, colmap_dir: str, dataset_dir: Path) -> StageResult:
        """Run GaussianWrapping training + mesh extraction in the MILo sidecar.

        GaussianWrapping specialises in thin-structure scenes (bicycle spokes,
        wires, fences, railings) and shares the MILo sidecar container.
        Falls back to tsdf on availability failure.

        See ADR-005 for the licensing gate and sidecar requirements.
        """
        from pipeline.gaussianwrapping_extractor import (
            run_gaussianwrapping,
            is_gaussianwrapping_available,
            GWConfig,
        )

        if not is_gaussianwrapping_available():
            logger.warning(
                "GaussianWrapping requested but not available, falling back to LichtFeld"
            )
            original_method = self.config.training.mesh_method
            self.config.training.mesh_method = "tsdf"
            result = self.train(colmap_dir, self.config.training.iterations)
            self.config.training.mesh_method = original_method
            if result.success:
                result.metrics["fallback"] = "lichtfeld (gaussianwrapping unavailable)"
            return result

        gw_output = self.job_dir / "model_gaussianwrapping"

        gw_cfg = GWConfig(
            iterations=self.config.training.iterations
            if self.config.training.iterations <= 30000
            else 30000,
        )

        logger.info(
            "Running GaussianWrapping training on %s -> %s", colmap_dir, gw_output
        )
        gw_result = run_gaussianwrapping(str(dataset_dir), str(gw_output), config=gw_cfg)

        if not gw_result["success"]:
            return StageResult(
                success=False, stage="train",
                error=f"GaussianWrapping failed: {gw_result['error']}",
                metrics={"backend": "gaussianwrapping", "duration": gw_result["duration"]},
            )

        artifacts: dict[str, str] = {"model_dir": str(gw_output)}
        metrics: dict[str, Any] = {
            "backend": "gaussianwrapping",
            "duration": round(gw_result["duration"], 1),
            "gaussianwrapping_mesh_path": gw_result["mesh_path"],
        }

        if gw_result["ply_path"]:
            ply_path = Path(gw_result["ply_path"])
            artifacts["ply_path"] = str(ply_path)
            metrics["ply_path"] = str(ply_path)
            metrics["ply_size_mb"] = round(_get_file_size_mb(ply_path), 1)

        if gw_result["mesh_path"]:
            artifacts["gaussianwrapping_mesh_path"] = gw_result["mesh_path"]

        # Copy GaussianWrapping mesh to standard viewer location
        gw_mesh = Path(gw_result.get("glb_path") or gw_result.get("mesh_path", ""))
        if gw_mesh.exists():
            std_mesh_dir = self.job_dir / "objects" / "meshes" / "full_scene"
            std_mesh_dir.mkdir(parents=True, exist_ok=True)
            dest = std_mesh_dir / "full_scene.glb"
            shutil.copy2(str(gw_mesh), str(dest))
            artifacts["scene_mesh_glb"] = str(dest)
            logger.info("GaussianWrapping mesh copied to %s", dest)

        stage_result = StageResult(
            success=True, stage="train",
            metrics=metrics,
            artifacts=artifacts,
        )
        ply_for_opt = Path(artifacts["ply_path"]) if "ply_path" in artifacts else None
        if ply_for_opt:
            self._optimize_splat(ply_for_opt, stage_result)
        return stage_result

    # ------------------------------------------------------------------
    # Stage 5b: Render previews
    # ------------------------------------------------------------------

    def render_previews(self, ply_path: str, colmap_dir: str, num_views: int = 8) -> StageResult:
        """Render RGB + depth previews from COLMAP camera positions using gsplat.

        Saves to job_dir/previews/ for the web carousel.

        Args:
            ply_path: Path to trained 3DGS PLY.
            colmap_dir: Path to COLMAP reconstruction (with sparse/0/).
            num_views: Number of preview views to render.

        Returns: {preview_dir, rendered_count}
        """
        import struct
        import numpy as np

        trained_ply = Path(ply_path)
        dataset_dir = Path(colmap_dir)
        if not trained_ply.exists():
            return StageResult(success=False, stage="render_previews",
                               error=f"PLY not found: {ply_path}")

        preview_dir = self.job_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        saved_files: list[str] = []

        try:
            import torch
            from pipeline.mesh_extractor import load_3dgs_ply, render_gsplat

            gaussians = load_3dgs_ply(str(trained_ply))

            # Parse COLMAP cameras (binary format)
            sparse_dir = dataset_dir / "sparse" / "0"
            images_bin = sparse_dir / "images.bin"
            cameras_bin = sparse_dir / "cameras.bin"

            viewmats = []
            K_matrix = None

            if cameras_bin.exists() and images_bin.exists():
                # Read camera intrinsics
                with open(cameras_bin, 'rb') as f:
                    n_cams = struct.unpack('<Q', f.read(8))[0]
                    for _ in range(n_cams):
                        cam_id, model_id, w, h = struct.unpack('<iiii', f.read(16))
                        w = w & 0xFFFFFFFF
                        h = h & 0xFFFFFFFF
                        # Read params (assume PINHOLE: fx, fy, cx, cy)
                        n_params = 4
                        params = struct.unpack(f'<{n_params}d', f.read(8 * n_params))
                        if K_matrix is None:
                            K_matrix = torch.tensor([
                                [params[0], 0.0, params[2]],
                                [0.0, params[1], params[3]],
                                [0.0, 0.0, 1.0],
                            ], dtype=torch.float32, device='cuda')
                            render_w, render_h = int(w), int(h)

                # Read image poses
                with open(images_bin, 'rb') as f:
                    n_images = struct.unpack('<Q', f.read(8))[0]
                    for _ in range(n_images):
                        img_id = struct.unpack('<i', f.read(4))[0]
                        qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
                        tx, ty, tz = struct.unpack('<3d', f.read(24))
                        cam_id = struct.unpack('<i', f.read(4))[0]
                        # Read null-terminated name
                        name_bytes = b""
                        while True:
                            c = f.read(1)
                            if c == b"\x00" or c == b"":
                                break
                            name_bytes += c
                        n_pts2d = struct.unpack('<Q', f.read(8))[0]
                        f.read(n_pts2d * 24)

                        # Build viewmat from quaternion + translation
                        q = np.array([qw, qx, qy, qz], dtype=np.float64)
                        q /= np.linalg.norm(q)
                        w_ = q[0]; x_ = q[1]; y_ = q[2]; z_ = q[3]
                        R = np.array([
                            [1-2*(y_*y_+z_*z_), 2*(x_*y_-z_*w_), 2*(x_*z_+y_*w_)],
                            [2*(x_*y_+z_*w_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-x_*w_)],
                            [2*(x_*z_-y_*w_), 2*(y_*z_+x_*w_), 1-2*(x_*x_+y_*y_)],
                        ], dtype=np.float64)
                        t = np.array([tx, ty, tz], dtype=np.float64)
                        vm = np.eye(4, dtype=np.float64)
                        vm[:3, :3] = R
                        vm[:3, 3] = t
                        viewmats.append(vm)
            else:
                # No COLMAP cameras; generate orbit cameras
                from pipeline.mesh_extractor import generate_orbit_cameras_gsplat
                orbit_cams = generate_orbit_cameras_gsplat(
                    gaussians['means'], num_views, 1024,
                )
                for vm_t, K_t in orbit_cams:
                    viewmats.append(vm_t.cpu().numpy())
                    if K_matrix is None:
                        K_matrix = K_t
                        render_w, render_h = 1024, 1024

            # Render first num_views cameras
            step = max(1, len(viewmats) // num_views)
            selected_indices = list(range(0, len(viewmats), step))[:num_views]

            for idx in selected_indices:
                vm = viewmats[idx]
                vm_tensor = torch.tensor(vm, dtype=torch.float32, device='cuda')
                depth, rgb, alpha = render_gsplat(
                    gaussians, vm_tensor, K_matrix, render_w, render_h,
                )

                # Save RGB preview
                from PIL import Image
                rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                rgb_path = preview_dir / f"preview_render_view{idx:02d}.jpg"
                Image.fromarray(rgb_uint8).save(str(rgb_path), quality=90)
                saved_files.append(str(rgb_path))

                # Save depth colormap
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
                ax.imshow(depth, cmap='turbo')
                ax.axis('off')
                depth_path = preview_dir / f"preview_depth_view{idx:02d}.jpg"
                fig.savefig(str(depth_path), bbox_inches='tight', dpi=100)
                plt.close(fig)
                saved_files.append(str(depth_path))

            logger.info("Rendered %d preview views to %s", len(selected_indices), preview_dir)

        except Exception as exc:
            logger.warning("Preview rendering failed: %s", exc)
            return StageResult(
                success=True, stage="render_previews",
                metrics={"skipped": True, "reason": str(exc)},
                artifacts={"preview_dir": str(preview_dir)},
            )

        return StageResult(
            success=True, stage="render_previews",
            metrics={"rendered_count": len(selected_indices), "files": len(saved_files)},
            artifacts={"preview_dir": str(preview_dir), "previews": json.dumps(saved_files)},
        )

    # ------------------------------------------------------------------
    # Stage 6: Segment (SAM2/SAM3)
    # ------------------------------------------------------------------

    def segment(self, ply_path: str, frames_dir: str) -> StageResult:
        """SAM2/SAM3 segmentation + mask projection.

        Returns: {objects: list[{label, object_id, mask_pixels}], masks_dir}
        """
        import cv2

        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="segment",
                error=f"Frames directory not found: {frames_dir}",
            )

        frame_paths = sorted(
            p for p in frames_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_paths:
            return StageResult(
                success=False, stage="segment",
                error=f"No frames found in {frames_dir}",
            )

        decompose_cfg = self.config.decompose
        concepts = decompose_cfg.sam3_concepts or decompose_cfg.descriptions or [
            "paintings", "frames", "sculptures", "furniture",
            "walls", "floor", "ceiling", "fixtures", "doorways",
        ]

        # Try SAM3 first
        if decompose_cfg.use_sam3:
            try:
                from pipeline.sam3_segmentor import SAM3Segmentor
                import numpy as np

                seg = SAM3Segmentor(
                    device="cuda",
                    confidence_threshold=decompose_cfg.sam3_confidence_threshold,
                )

                # Segment every 10th frame and merge results for robustness
                sample_step = max(1, min(10, len(frame_paths) // 3))
                sampled_frames = frame_paths[::sample_step]
                logger.info("SAM3: segmenting %d/%d frames (step=%d)",
                            len(sampled_frames), len(frame_paths), sample_step)

                all_masks: dict[str, list[np.ndarray]] = {}  # concept -> list of masks
                all_id_to_concept: dict[int, str] = {}

                for frame_path in sampled_frames:
                    image = cv2.imread(str(frame_path))
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    frame_result, frame_id_map = seg.segment_by_concepts(
                        image, concepts,
                        confidence_threshold=decompose_cfg.sam3_confidence_threshold,
                    )
                    all_id_to_concept.update(frame_id_map)

                    for obj_id in frame_result.object_ids:
                        concept = frame_id_map.get(int(obj_id), "unknown")
                        mask = frame_result.masks[frame_result.object_ids == obj_id]
                        if concept not in all_masks:
                            all_masks[concept] = []
                        all_masks[concept].append(mask[0] if mask.ndim == 3 else mask)

                # Merge masks per concept: union across frames (resize to first frame's shape)
                first_image = cv2.imread(str(frame_paths[0]))
                if first_image is None:
                    return StageResult(
                        success=False, stage="segment",
                        error=f"Failed to read frame {frame_paths[0]}",
                    )
                ref_h, ref_w = first_image.shape[:2]

                objects: list[dict[str, Any]] = []
                merged_masks: list[np.ndarray] = []
                merged_ids: list[int] = []
                next_id = 1

                for concept, mask_list in all_masks.items():
                    # Union all masks for this concept (resize if needed)
                    merged = np.zeros((ref_h, ref_w), dtype=bool)
                    for m in mask_list:
                        if m.shape != (ref_h, ref_w):
                            m_resized = cv2.resize(
                                m.astype(np.uint8), (ref_w, ref_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        else:
                            m_resized = m
                        merged |= m_resized

                    mask_pixels = int(merged.sum())
                    if mask_pixels > 0:
                        objects.append({
                            "label": concept,
                            "object_id": next_id,
                            "mask_pixels": mask_pixels,
                        })
                        merged_masks.append(merged)
                        merged_ids.append(next_id)
                        next_id += 1

                id_to_concept = {obj["object_id"]: obj["label"] for obj in objects}

                # Save merged masks
                masks_dir = self.job_dir / "sam3_masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                for obj_id, mask in zip(merged_ids, merged_masks):
                    mask_path = masks_dir / f"mask_{obj_id:04d}.npy"
                    np.save(str(mask_path), mask)

                seg.unload()

                if not objects:
                    objects = [{"label": "full_scene", "count": -1}]

                return StageResult(
                    success=True, stage="segment",
                    metrics={
                        "object_count": len(objects),
                        "method": "sam3",
                        "concepts": concepts,
                    },
                    artifacts={
                        "objects": json.dumps(objects),
                        "masks_dir": str(masks_dir),
                    },
                )
            except Exception as exc:
                if not decompose_cfg.sam3_fallback_to_sam2:
                    return StageResult(
                        success=False, stage="segment",
                        error=f"SAM3 segmentation failed: {exc}",
                    )
                logger.warning("SAM3 failed (%s), falling back to full-scene", exc)

        # Fallback: treat entire scene as one object
        return StageResult(
            success=True, stage="segment",
            metrics={"object_count": 1, "method": "full_scene"},
            artifacts={"objects": json.dumps([{"label": "full_scene", "count": -1}])},
        )

    # ------------------------------------------------------------------
    # Stage 7: Extract objects
    # ------------------------------------------------------------------

    def extract_objects(self, ply_path: str, labels: dict | list | str | None = None) -> StageResult:
        """Extract per-object PLY files from the trained model.

        Args:
            ply_path: Path to the trained gaussian PLY.
            labels: Object definitions from segment(). Can be a JSON string,
                    list of dicts, or None (full_scene fallback).

        Returns: {object_plys: list[str]}
        """
        trained_ply = Path(ply_path)
        if not trained_ply.exists():
            return StageResult(
                success=False, stage="extract_objects",
                error=f"PLY not found: {ply_path}",
            )

        # Parse labels
        if labels is None:
            objects = [{"label": "full_scene", "count": -1}]
        elif isinstance(labels, str):
            objects = json.loads(labels)
        elif isinstance(labels, dict):
            objects = [labels]
        else:
            objects = list(labels)

        objects_dir = self.job_dir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = self.job_dir / "sam3_masks"
        has_masks = masks_dir.exists() and any(masks_dir.glob("*.npy"))

        # ADR-010 / FR-9: concept-priority weights for key-item ranking. Key
        # gallery items (artworks, furniture, fixtures) outrank structural
        # surfaces; unknown concepts default to 1.0.
        concept_priority = {
            "paintings": 3.0, "frames": 2.5, "sculptures": 3.0, "furniture": 2.0,
            "fixtures": 1.5, "doorways": 1.0,
            "walls": 0.5, "floor": 0.3, "ceiling": 0.3,
        }
        min_gaussians = self.config.decompose.min_object_gaussians

        extracted: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []

        for obj in objects:
            label = obj.get("label", "unknown")
            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            out_ply = objects_dir / f"{safe_name}.ply"

            # full_scene (and the no-mask fallback) is the whole environment —
            # never keyness-ranked or dropped.
            if label == "full_scene" or not has_masks:
                shutil.copy2(str(trained_ply), str(out_ply))
                extracted.append({"label": label, "ply": str(out_ply),
                                  "gaussian_count": -1, "keyness": float("inf")})
                logger.info("Copied trained PLY as '%s': %s", label, out_ply)
                continue

            # Mask-based extraction returns the number of gaussians assigned.
            count = 0
            try:
                count = self._extract_with_mask(obj, trained_ply, out_ply, masks_dir)
            except Exception as exc:
                logger.warning("Mask extraction failed for '%s': %s", label, exc)
                count = 0

            if count <= 0:
                # Un-isolable object: preserve prior behaviour (copy full PLY).
                shutil.copy2(str(trained_ply), str(out_ply))
                extracted.append({"label": label, "ply": str(out_ply),
                                  "gaussian_count": -1, "keyness": 0.0})
                continue

            # FR-9: enforce the (previously dead) min_object_gaussians threshold.
            if count < min_gaussians:
                logger.info("Dropping '%s': %d gaussians < min_object_gaussians=%d",
                            label, count, min_gaussians)
                dropped.append({"label": label, "gaussian_count": count})
                try:
                    out_ply.unlink(missing_ok=True)
                except OSError:
                    pass
                continue

            keyness = count * concept_priority.get(label.lower(), 1.0)
            extracted.append({"label": label, "ply": str(out_ply),
                              "gaussian_count": count, "keyness": round(keyness, 2)})

        if not extracted:
            return StageResult(
                success=False, stage="extract_objects",
                error=(f"No objects met min_object_gaussians={min_gaussians} "
                       f"({len(dropped)} dropped below threshold)"),
                metrics={"dropped_below_threshold": len(dropped)},
            )

        # FR-9: rank by keyness (descending) so downstream per-object hull
        # reconstruction processes the most significant items first.
        extracted.sort(key=lambda e: e.get("keyness", 0.0), reverse=True)

        return StageResult(
            success=True, stage="extract_objects",
            metrics={
                "extracted_count": len(extracted),
                "dropped_below_threshold": len(dropped),
                "min_object_gaussians": min_gaussians,
                "ranking": [e["label"] for e in extracted],
            },
            artifacts={
                "object_plys": json.dumps([e["ply"] for e in extracted]),
                "object_ranking": json.dumps(extracted),
                **{f"ply:{e['label']}": e["ply"] for e in extracted},
            },
        )

    def _extract_with_mask(
        self,
        obj: dict[str, Any],
        trained_ply: Path,
        output_path: Path,
        masks_dir: Path,
    ) -> int:
        """Extract a subset of gaussians using a SAM3 mask.

        Returns the number of gaussians assigned to the object (0 on any
        failure). Callers use the count to enforce min_object_gaussians (FR-9).
        """
        import numpy as np

        object_id = obj.get("object_id")
        if object_id is None:
            return 0

        mask_path = masks_dir / f"mask_{int(object_id):04d}.npy"
        if not mask_path.exists():
            return 0

        mask = np.load(str(mask_path))

        import trimesh
        pcd = trimesh.load(str(trained_ply))
        if hasattr(pcd, "vertices"):
            points = np.asarray(pcd.vertices)
        else:
            points = np.asarray(pcd.points) if hasattr(pcd, "points") else None
        if points is None or len(points) == 0:
            return 0

        h, w = mask.shape[-2], mask.shape[-1]
        xy = points[:, :2]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_range = xy_max - xy_min
        xy_range[xy_range == 0] = 1.0

        px = ((xy[:, 0] - xy_min[0]) / xy_range[0] * (w - 1)).astype(int).clip(0, w - 1)
        py = ((xy[:, 1] - xy_min[1]) / xy_range[1] * (h - 1)).astype(int).clip(0, h - 1)

        mask_2d = mask[0] if mask.ndim == 3 else mask
        inside = mask_2d[py, px] > 0
        if inside.sum() == 0:
            return 0

        filtered_points = points[inside]
        colors = None
        if hasattr(pcd, "visual") and hasattr(pcd.visual, "vertex_colors"):
            all_colors = np.asarray(pcd.visual.vertex_colors)
            colors = all_colors[inside]

        filtered_pcd = trimesh.PointCloud(filtered_points)
        if colors is not None:
            filtered_pcd.colors = colors

        filtered_pcd.export(str(output_path))
        logger.info("Extracted %d/%d points for '%s'", inside.sum(), len(points), obj.get("label", "unknown"))
        return int(inside.sum())

    # ------------------------------------------------------------------
    # Stage 8: Mesh objects
    # ------------------------------------------------------------------

    def mesh_objects(self, object_plys: list[str] | str) -> StageResult:
        """Generate meshes per object PLY.

        Args:
            object_plys: List of PLY paths, or a JSON string of a list.

        Returns: {meshes: list[{label, mesh, ply, vertex_count, method}]}
        """
        if isinstance(object_plys, str):
            object_plys = json.loads(object_plys)

        import trimesh

        meshes_dir = self.job_dir / "objects" / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        for ply_path in object_plys:
            ply = Path(ply_path)
            if not ply.exists():
                logger.warning("PLY not found: %s", ply_path)
                continue

            label = ply.stem
            obj_dir = meshes_dir / label
            obj_dir.mkdir(parents=True, exist_ok=True)
            mesh_glb = obj_dir / f"{label}.glb"
            mesh_obj = obj_dir / f"{label}.obj"

            # Run each object mesh with a 600s (10 min) timeout to prevent
            # the stage from hanging in subprocess environments where
            # SIGALRM doesn't propagate.
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._mesh_single, label, str(ply), obj_dir, mesh_obj, mesh_glb,
                    )
                    mesh_result = future.result(timeout=3600)
            except FuturesTimeout:
                logger.warning("Meshing timed out (3600s) for '%s', skipping", label)
                mesh_result = None
            except Exception as exc:
                logger.warning("Meshing failed for '%s': %s", label, exc)
                mesh_result = None

            if mesh_result is not None:
                results.append(mesh_result)

        if not results:
            return StageResult(
                success=False, stage="mesh_objects",
                error="No meshes could be generated",
            )

        return StageResult(
            success=True, stage="mesh_objects",
            metrics={"mesh_count": len(results)},
            artifacts={
                "meshes": json.dumps(results),
                **{f"mesh:{r['label']}": r["mesh"] for r in results},
            },
        )

    def _mesh_single(
        self,
        label: str,
        ply_path: str,
        obj_dir: Path,
        mesh_obj_path: Path,
        mesh_glb_path: Path,
    ) -> dict[str, Any] | None:
        """Try all meshing strategies for a single object PLY."""
        import numpy as np
        import trimesh

        points, colors = _load_ply_points(ply_path)
        if points is None or len(points) == 0:
            logger.warning("No points loaded from PLY for '%s'", label)
            return None

        centroid = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        scale = extent.max()
        if scale == 0:
            scale = 1.0
        points_norm = (points - centroid) / scale

        logger.info("PLY '%s': %d points, extent=%.1f", label, len(points), scale)

        def _export_mesh(mesh: "trimesh.Trimesh", method: str) -> dict[str, Any] | None:
            mesh.vertices = mesh.vertices * scale + centroid
            mesh.export(str(mesh_glb_path))
            mesh.export(str(mesh_obj_path))
            vc = len(mesh.vertices)
            logger.info("%s mesh for '%s': %d verts", method, label, vc)
            return {
                "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                "ply": ply_path, "vertex_count": vc, "method": method,
            }

        # Strategy 0: gsplat depth rendering -> TSDF (preferred, GPU-accelerated)
        try:
            import torch
            if torch.cuda.is_available():
                from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

                extractor = MeshExtractor(config=TSDFConfig(
                    target_faces=self.config.mesh.max_vertices // 2,
                ))
                previews_dir = self.job_dir / "previews"
                previews_dir.mkdir(parents=True, exist_ok=True)
                # Full-scene uses COLMAP cameras (better depth coverage for interiors).
                # Isolated objects from Hunyuan MV use orbit cameras.
                colmap_dir = None
                if label == "full_scene":
                    colmap_candidate = str(self.job_dir / "colmap")
                    if Path(colmap_candidate).exists():
                        colmap_dir = colmap_candidate
                mesh, color_images, cameras = extractor.extract_from_gsplat(
                    ply_path,
                    num_views=100,
                    render_size=1024,
                    target_faces=self.config.mesh.max_vertices // 2,
                    preview_dir=previews_dir,
                    colmap_dir=colmap_dir,
                )
                # gsplat returns mesh in world coordinates already, no rescale needed
                mesh.export(str(mesh_glb_path))
                mesh.export(str(mesh_obj_path))
                vc = len(mesh.vertices)
                fc = len(mesh.faces)
                logger.info("gsplat mesh for '%s': %d verts, %d faces", label, vc, fc)

                result_info = {
                    "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                    "ply": ply_path, "vertex_count": vc, "method": "gsplat",
                }

                # Bake UV texture only for small meshes where xatlas finishes
                # quickly (<60s). For larger meshes, rely on vertex colors in
                # the already-exported GLB.
                _TEXTURE_BAKE_FACE_LIMIT = 30_000
                if fc <= _TEXTURE_BAKE_FACE_LIMIT:
                    try:
                        from pipeline.texture_baker import TextureBaker, BakeConfig
                        safe_name = label.replace(" ", "_").replace("/", "_")[:50]
                        tex_dir = self.job_dir / "objects" / "meshes" / safe_name
                        tex_dir.mkdir(parents=True, exist_ok=True)
                        tex_path = tex_dir / f"{safe_name}_diffuse.png"
                        tex_obj_path = tex_dir / f"{safe_name}_textured.obj"

                        bake_cfg = BakeConfig(mcp_endpoint=self.config.mcp_endpoint)
                        baker = TextureBaker(config=bake_cfg)
                        textured_mesh, baked_tex = baker.bake_from_vertex_colors(
                            mesh, output_texture_path=tex_path,
                        )
                        textured_mesh.export(str(tex_obj_path))
                        result_info["textured_mesh"] = str(tex_obj_path)
                        result_info["texture"] = str(baked_tex)
                        logger.info("Baked texture for '%s' (%d faces) -> %s",
                                    label, fc, baked_tex)
                    except Exception as tex_exc:
                        logger.warning("Texture bake failed for '%s' (%d faces), "
                                       "keeping vertex-colored mesh: %s",
                                       label, fc, tex_exc)
                else:
                    logger.info("Skipping texture bake for '%s' (%d faces > %d limit), "
                                "vertex-colored GLB exported",
                                label, fc, _TEXTURE_BAKE_FACE_LIMIT)

                return result_info
        except Exception as exc:
            logger.warning("gsplat meshing failed for '%s': %s", label, exc)

        # Strategy 1: Hunyuan3D
        if self.config.hunyuan3d.enabled:
            try:
                from pipeline.hunyuan3d_client import Hunyuan3DClient
                import inspect

                h3d_kwargs = {
                    "comfyui_url": self.config.hunyuan3d.comfyui_url,
                    "api_url": self.config.hunyuan3d.api_url,
                    "quality": self.config.hunyuan3d.quality,
                    "turbo": self.config.hunyuan3d.turbo,
                    "timeout": self.config.hunyuan3d.timeout,
                    "seed": self.config.hunyuan3d.seed,
                }
                sig = inspect.signature(Hunyuan3DClient.__init__)
                if "multiview" in sig.parameters:
                    h3d_kwargs["multiview"] = self.config.hunyuan3d.multiview
                # Drop any kwargs the installed client version doesn't accept
                # (e.g. 'turbo' is config-only on some versions) to avoid a
                # TypeError that would crash this mesh strategy.
                h3d_kwargs = {k: v for k, v in h3d_kwargs.items() if k in sig.parameters}

                h3d = Hunyuan3DClient(**h3d_kwargs)
                result = h3d.reconstruct_from_gaussians(ply_path)
                if result.mesh is not None:
                    result.mesh.export(str(mesh_glb_path))
                    result.mesh.export(str(mesh_obj_path))
                    vc = len(result.mesh.vertices)
                    return {
                        "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                        "ply": ply_path, "vertex_count": vc, "method": "hunyuan3d",
                    }
            except Exception as exc:
                logger.warning("Hunyuan3D failed for '%s': %s", label, exc)

        # Strategy 2: TSDF
        try:
            from pipeline.mesh_extractor import MeshExtractor, TSDFConfig

            tsdf_cfg = TSDFConfig(
                target_faces=self.config.mesh.max_vertices // 2,
                mcp_endpoint=self.config.mcp_endpoint,
            )
            extractor = MeshExtractor(config=tsdf_cfg)
            mesh = extractor.extract_from_pointcloud(
                points_norm, colors=colors,
                target_faces=self.config.mesh.max_vertices // 2,
            )
            result = _export_mesh(mesh, "tsdf")
            if result:
                return result
        except Exception as exc:
            logger.warning("TSDF meshing failed for '%s': %s", label, exc)

        # Strategy 3: Point cloud marching cubes
        try:
            from pipeline.mesh_extractor import MeshExtractor

            extractor = MeshExtractor()
            mesh = extractor.extract_from_pointcloud(
                points_norm, colors=colors,
                target_faces=self.config.mesh.max_vertices // 2,
            )
            result = _export_mesh(mesh, "pointcloud")
            if result:
                return result
        except Exception as exc:
            logger.warning("Point-cloud meshing failed for '%s': %s", label, exc)

        # Strategy 4: Open3D Poisson
        try:
            _mesh_with_open3d(ply_path, str(mesh_glb_path))
            return {
                "label": label, "mesh": str(mesh_glb_path), "ply": ply_path,
                "vertex_count": 0, "method": "open3d",
            }
        except Exception as exc:
            logger.warning("Open3D fallback failed for '%s': %s", label, exc)

        # Strategy 5: Convex hull (last resort -- often produces garbage)
        try:
            pcd = trimesh.PointCloud(points)
            mesh = pcd.convex_hull
            if mesh is not None and len(mesh.vertices) > 0:
                if len(mesh.vertices) < 5000:
                    logger.error(
                        "Mesh extraction produced only %d vertices (convex hull fallback). "
                        "This indicates gsplat/TSDF failed. Check dependencies.",
                        len(mesh.vertices),
                    )
                    return None
                mesh.export(str(mesh_glb_path))
                mesh.export(str(mesh_obj_path))
                vc = len(mesh.vertices)
                return {
                    "label": label, "mesh": str(mesh_glb_path), "mesh_obj": str(mesh_obj_path),
                    "ply": ply_path, "vertex_count": vc, "method": "convex_hull",
                }
        except Exception as exc:
            logger.warning("Convex hull fallback failed for '%s': %s", label, exc)

        return None

    # ------------------------------------------------------------------
    # Stage 9: Texture bake
    # ------------------------------------------------------------------

    def texture_bake(self, meshes: list[dict[str, Any]] | str) -> StageResult:
        """Bake diffuse textures onto each object mesh.

        Args:
            meshes: List of mesh info dicts (from mesh_objects) or JSON string.

        Returns: {baked_count, textured_meshes}
        """
        if isinstance(meshes, str):
            meshes = json.loads(meshes)

        if not meshes:
            return StageResult(
                success=True, stage="texture_bake",
                metrics={"skipped": True, "reason": "no meshes"},
            )

        from pipeline.texture_baker import TextureBaker, BakeConfig
        import trimesh

        bake_cfg = BakeConfig(mcp_endpoint=self.config.mcp_endpoint)
        baker = TextureBaker(config=bake_cfg)
        baked_count = 0

        for mesh_info in meshes:
            label = mesh_info.get("label", "unknown")
            mesh_path = mesh_info.get("mesh_obj") or mesh_info.get("mesh")
            if not mesh_path or not Path(mesh_path).exists():
                continue

            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            texture_dir = self.job_dir / "objects" / "meshes" / safe_name
            texture_dir.mkdir(parents=True, exist_ok=True)
            texture_path = texture_dir / f"{safe_name}_diffuse.png"
            textured_mesh_path = texture_dir / f"{safe_name}_textured.obj"

            try:
                mesh = trimesh.load(mesh_path, process=False)
                has_vertex_colors = (
                    mesh.visual is not None
                    and hasattr(mesh.visual, "vertex_colors")
                    and mesh.visual.vertex_colors is not None
                    and len(mesh.visual.vertex_colors) > 0
                )

                if has_vertex_colors:
                    textured_mesh, tex_path = baker.bake_from_vertex_colors(
                        mesh, output_texture_path=texture_path,
                    )
                else:
                    textured_mesh, tex_path = baker.bake(
                        mesh, output_texture_path=texture_path,
                    )

                textured_mesh.export(str(textured_mesh_path))
                mesh_info["textured_mesh"] = str(textured_mesh_path)
                mesh_info["texture"] = str(tex_path)
                baked_count += 1
                logger.info("Baked texture for '%s' -> %s", label, tex_path)
            except Exception as exc:
                logger.warning("Texture baking failed for '%s': %s", label, exc)

        return StageResult(
            success=True, stage="texture_bake",
            metrics={"baked_count": baked_count, "total_meshes": len(meshes)},
            artifacts={
                "textured_meshes": json.dumps(meshes),
            },
        )

    # ------------------------------------------------------------------
    # Stage 10: Assemble USD
    # ------------------------------------------------------------------

    def _export_native_usd(self, out_path: Path) -> str:
        """Best-effort native USD export via LichtFeld MCP (scene.export_usd,
        v0.5.1+). Tries the resident training scene first, then a checkpoint
        reload. Returns the path on success, "" on any failure (the composed
        assembler remains the authority). Never raises.
        """
        try:
            from pipeline.mcp_client import McpClient, McpError, McpConnectionError
        except Exception:  # pragma: no cover - import guard
            return ""
        try:
            client = McpClient(self.config.mcp_endpoint)
            if not client.ping():
                logger.info("Native USD export skipped: LichtFeld MCP not reachable at %s",
                            self.config.mcp_endpoint)
                return ""
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                client.export_usd(str(out_path))
            except (McpError, McpConnectionError):
                # Scene may not be resident; reload the latest checkpoint/PLY and retry.
                candidates = sorted(self.job_dir.rglob("*.ckpt")) or sorted(self.job_dir.rglob("model/**/*.ply"))
                if not candidates:
                    return ""
                client.load_checkpoint(str(candidates[-1]))
                client.export_usd(str(out_path))
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info("Native LichtFeld USD export -> %s", out_path)
                return str(out_path)
            return ""
        except Exception as exc:  # pragma: no cover - best-effort
            logger.info("Native USD export failed (%s); using composed assembler", exc)
            return ""

    def assemble_usd(self, objects: dict | list | str, cameras: dict | str | None = None) -> StageResult:
        """Compose USD scene from meshes using the standalone assembler.

        The standalone script (scripts/assemble_usd_scene.py) runs under
        python3 (3.12) with usd-core installed directly. Falls back to a
        minimal USDA stub if the subprocess fails.

        Args:
            objects: Mesh results (from mesh_objects) as list/dict/JSON string.
            cameras: Optional camera data (unused in file-based mode).

        Returns: {usd_path}
        """
        if isinstance(objects, str):
            objects = json.loads(objects)
        if isinstance(objects, dict):
            objects = [objects]

        usd_dir = self.job_dir / "usd"
        usd_dir.mkdir(parents=True, exist_ok=True)
        usd_path = usd_dir / "scene.usda"

        # LichtFeld native USD export (scene.export_usd, v0.5.1+). Additive and
        # best-effort: produces an authoritative native scene USD alongside the
        # composed scene below. The composed assembler still supplies the
        # multi-object hierarchy + ADR-011 v2g:* metadata until the native
        # customData parity probe confirms native can carry it.
        native_usd = ""
        if getattr(self.config.export, "prefer_native_usd", False):
            native_usd = self._export_native_usd(usd_dir / "scene_native.usd")

        # Copy trained PLY as scene.ply (needed by the assembler)
        final_ply = self.job_dir / "scene.ply"
        model_plys = sorted(self.job_dir.rglob("model/**/*.ply"))
        if model_plys and not final_ply.exists():
            shutil.copy2(str(model_plys[-1]), str(final_ply))
            logger.info("Copied trained PLY as scene.ply")

        # Locate the standalone assembler script
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "assemble_usd_scene.py"
        if not script_path.exists():
            # Docker container path
            script_path = Path("/opt/gaussian-toolkit/scripts/assemble_usd_scene.py")

        assembled = False
        if script_path.exists():
            try:
                result = subprocess.run(
                    [
                        "python3",
                        str(script_path),
                        "--job-dir", str(self.job_dir),
                        "--output", str(usd_path),
                    ],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    assembled = True
                    logger.info("USD scene assembled via standalone script:\n%s", result.stdout)
                else:
                    logger.warning(
                        "Standalone USD assembler failed (rc=%d):\nstdout: %s\nstderr: %s",
                        result.returncode, result.stdout, result.stderr,
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("Standalone USD assembler error: %s", exc)

        if not assembled:
            logger.warning("Falling back to minimal USDA stub")
            meshes = [r for r in objects if Path(r.get("mesh", "")).exists()]
            _write_minimal_usda(usd_path, meshes)

        # ----- Blender-based assembly: import GLB, clean, materials, USD + previews -----
        blender_result = self._run_blender_assembler(usd_path)

        usd_size = usd_path.stat().st_size if usd_path.exists() else 0

        return StageResult(
            success=True, stage="assemble_usd",
            metrics={
                "mesh_count": len(objects),
                "usd_size_bytes": usd_size,
                "used_standalone_assembler": assembled,
                "native_usd_exported": bool(native_usd),
                "blender_assembled": blender_result.get("success", False),
                "blender_components_removed": blender_result.get("components_removed", 0),
                "preview_count": len(blender_result.get("renders", [])),
            },
            artifacts={
                "usd_path": str(usd_path),
                "native_usd_path": native_usd,
                "final_ply": str(final_ply) if final_ply.exists() else "",
                "previews_dir": str(self.job_dir / "previews"),
                "blender_renders": json.dumps(blender_result.get("renders", [])),
            },
        )

    # ------------------------------------------------------------------
    # Blender assembler helper
    # ------------------------------------------------------------------

    def _run_blender_assembler(self, usd_path: Path) -> dict:
        """Run the Blender-based scene assembler for GLB cleaning, materials, and previews.

        Locates the full_scene.glb, invokes Blender headlessly with the
        blender_assembler.py script, and parses JSON output.

        Returns the parsed JSON result dict or an error dict on failure.
        """
        # Locate the GLB input
        glb_candidates = [
            self.job_dir / "objects" / "meshes" / "full_scene" / "full_scene.glb",
            *(self.job_dir.rglob("objects/meshes/**/*.glb")),
        ]
        glb_path = None
        for candidate in glb_candidates:
            if isinstance(candidate, Path) and candidate.exists():
                glb_path = candidate
                break

        if glb_path is None:
            logger.info("No GLB mesh found for Blender assembler — skipping")
            return {"success": False, "error": "No GLB mesh found", "renders": []}

        # Locate the Blender assembler script
        script_path = Path(__file__).resolve().parent / "blender_assembler.py"
        if not script_path.exists():
            logger.warning("Blender assembler script not found at %s", script_path)
            return {"success": False, "error": "Blender assembler script not found", "renders": []}

        # Find Blender binary
        blender_bin = None
        for candidate_bin in ("/usr/local/bin/blender", "blender"):
            if candidate_bin == "blender" or Path(candidate_bin).exists():
                blender_bin = candidate_bin
                break

        if blender_bin is None:
            logger.warning("Blender binary not found — skipping Blender assembly")
            return {"success": False, "error": "Blender not found", "renders": []}

        previews_dir = self.job_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        colmap_dir = self.job_dir / "colmap"
        cmd = [
            blender_bin, "--background", "--python", str(script_path), "--",
            "--input", str(glb_path),
            "--output-usd", str(usd_path),
            "--output-renders", str(previews_dir),
            "--render-size", "1920x1080",
        ]
        if colmap_dir.exists():
            cmd.extend(["--colmap-dir", str(colmap_dir)])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                # Parse the JSON from stdout (last line)
                stdout_lines = proc.stdout.strip().splitlines()
                for line in reversed(stdout_lines):
                    line = line.strip()
                    if line.startswith("{"):
                        blender_result = json.loads(line)
                        logger.info(
                            "Blender assembler succeeded: %d components removed, %d renders",
                            blender_result.get("components_removed", 0),
                            len(blender_result.get("renders", [])),
                        )
                        return blender_result
                logger.warning("Blender assembler produced no JSON output")
                return {"success": False, "error": "No JSON in stdout", "renders": []}
            else:
                logger.warning(
                    "Blender assembler failed (rc=%d):\nstdout: %s\nstderr: %s",
                    proc.returncode, proc.stdout[:500], proc.stderr[:500],
                )
                return {"success": False, "error": f"Exit code {proc.returncode}", "renders": []}
        except subprocess.TimeoutExpired:
            logger.warning("Blender assembler timed out after 600s")
            return {"success": False, "error": "Timeout", "renders": []}
        except OSError as exc:
            logger.warning("Blender assembler OS error: %s", exc)
            return {"success": False, "error": str(exc), "renders": []}

    # ------------------------------------------------------------------
    # Stage 11: Validate
    # ------------------------------------------------------------------

    def validate(self) -> StageResult:
        """Final validation: check that key artifacts exist on disk.

        Returns: {usd_exists, ply_exists, mesh_count}
        """
        usd_path = self.job_dir / "usd" / "scene.usda"
        final_ply = self.job_dir / "scene.ply"
        meshes_dir = self.job_dir / "objects" / "meshes"

        mesh_count = 0
        if meshes_dir.exists():
            mesh_count = len(list(meshes_dir.rglob("*.glb"))) + len(list(meshes_dir.rglob("*.obj")))

        if not usd_path.exists() and mesh_count == 0:
            return StageResult(
                success=False, stage="validate",
                error="No USD scene or mesh files found",
            )

        return StageResult(
            success=True, stage="validate",
            metrics={
                "usd_exists": usd_path.exists(),
                "ply_exists": final_ply.exists(),
                "usd_size_mb": round(_get_file_size_mb(usd_path), 2),
                "ply_size_mb": round(_get_file_size_mb(final_ply), 2),
                "mesh_count": mesh_count,
            },
            artifacts={
                "usd_path": str(usd_path) if usd_path.exists() else "",
                "final_ply": str(final_ply) if final_ply.exists() else "",
            },
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Current job status for web UI.

        Reads the pipeline_status.json if it exists, otherwise inspects
        the directory tree to determine progress.
        """
        status_file = self.job_dir / self.config.status_file
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except Exception:
                pass

        # Infer status from directory structure
        stages_done = []
        if (self.job_dir / "frames").exists():
            stages_done.append("ingest")
        if (self.job_dir / "frames_selected").exists() or (self.job_dir / "frames_cleaned").exists():
            stages_done.append("select_frames")
        if (self.job_dir / "colmap").exists():
            stages_done.append("reconstruct")
        if (self.job_dir / "model").exists() and any(self.job_dir.rglob("model/**/*.ply")):
            stages_done.append("train")
        if (self.job_dir / "sam3_masks").exists():
            stages_done.append("segment")
        if (self.job_dir / "objects").exists():
            stages_done.append("extract_objects")
        if (self.job_dir / "objects" / "meshes").exists():
            stages_done.append("mesh_objects")
        if (self.job_dir / "usd" / "scene.usda").exists():
            stages_done.append("assemble_usd")

        return {
            "job_dir": str(self.job_dir),
            "stages_completed": stages_done,
            "progress": len(stages_done) / len(STAGE_NAMES),
        }
