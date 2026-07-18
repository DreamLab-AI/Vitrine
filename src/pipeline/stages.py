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
    "object_crops",
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

        # Pre-flight dependency check -- fail hard if critical deps are missing.
        # LFS_SKIP_PREFLIGHT=1 is a TEST-ONLY escape hatch so the pure-logic
        # stage methods (crops/isolation/placement bookkeeping) can be unit-
        # tested on a CPU CI runner without the torch/GPU stack. Production
        # never sets it, so the fail-fast behaviour is unchanged; any stage that
        # actually needs a GPU dep still raises loudly at its own import.
        if os.environ.get("LFS_SKIP_PREFLIGHT") == "1":
            self._preflight = {"skipped": True}
        else:
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

        Defaults to the transparent direct-COLMAP path (_run_colmap_direct):
        feature_extractor → matcher → mapper → image_undistorter. The bundled
        SplatReady plugin is available behind ``config.reconstruct.use_splatready``
        but is off by default due to two failure modes: (1) its status-file path
        is derived by string-replacing ``_run_config.json`` in the config path, so
        any config with a different suffix has the status path collide with the
        config itself, clobbering it mid-run; (2) the plugin exits 0 on failure,
        making the return code unreliable. The direct path validates output via the
        registration-rate gate (>30% of frames must register).

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

        # COLMAP SfM. Default to running COLMAP directly (feature_extractor → matcher →
        # mapper → image_undistorter, see _run_colmap_direct) rather than via the bundled
        # SplatReady plugin. SplatReady is fragile for our still-image path in two ways:
        #   1. Its runner derives its status file by string-replacing "_run_config.json"
        #      in the config path with "_progress.txt". A config NOT ending in
        #      "_run_config.json" makes that a no-op, so the status file path COLLIDES with
        #      the config path and the runner clobbers its own config mid-run (the reused
        #      config then fails json.load with "Extra data").
        #   2. The runner exits 0 even on failure, so returncode is not a reliable signal.
        # The direct path is transparent and its output is validated below (the
        # registration-rate gate). SplatReady stays available behind
        # config.reconstruct.use_splatready (video fast path); the naming bug is worked
        # around by giving the config the "_run_config.json" suffix it expects.
        use_splatready = getattr(self.config.reconstruct, "use_splatready", False)
        plugin_dir = Path.home() / ".lichtfeld" / "plugins" / "splat_ready"
        runner = plugin_dir / "core" / "runner.py"

        if use_splatready and runner.exists():
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
            # MUST end in "_run_config.json" so the runner's status file becomes a distinct
            # "_progress.txt" instead of overwriting this config (see note above).
            config_path = self.job_dir / "splatready_run_config.json"
            config_path.write_text(json.dumps(splatready_config, indent=2), encoding="utf-8")
            try:
                proc = subprocess.run(
                    ["python3", str(runner), str(config_path)],
                    capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    return StageResult(
                        success=False, stage="reconstruct",
                        error=f"SplatReady failed: {(proc.stderr or proc.stdout)[:500]}",
                    )
            except subprocess.TimeoutExpired:
                return StageResult(
                    success=False, stage="reconstruct",
                    error="COLMAP reconstruction timed out",
                )
        else:
            # Transparent, reliable path (default).
            try:
                self._run_colmap_direct(colmap_dir, frames_path)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
                return StageResult(
                    success=False, stage="reconstruct",
                    error=f"COLMAP failed: {exc} :: {stderr[:500]}",
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
        """Default COLMAP path: feature_extractor → matcher → mapper → image_undistorter.

        The image_undistorter step caps output resolution via ``--max_image_size``
        (``config.ingest.max_image_size``, default 2000px). Without this cap, raw
        inputs (e.g. 36MP DNGs decoded at full resolution) cause LichtFeld to
        CPU-downscale on every load iteration — that path is not disk-cached, costs
        ~5 s/image, and starves the GPU, turning a 30k-iter run into hours. With
        the cap, LichtFeld takes its fast no-resample path and stays GPU-bound
        (~15 min at ≥99% utilisation for a typical 30k igs+ run).
        """
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        colmap = self.config.reconstruct.colmap_exe

        feat_cmd = [
            colmap, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--ImageReader.single_camera", "1",
        ]
        # Robust-capture masking (scene_masking.py): exclude moving people + blown highlights from
        # SfM. COLMAP reads the mask for <image> at <mask_path>/<image>.png; 255=keep, 0=ignore.
        colmap_masks = output_dir / "colmap_masks"
        if colmap_masks.is_dir() and any(colmap_masks.iterdir()):
            feat_cmd += ["--ImageReader.mask_path", str(colmap_masks)]
            logger.info("COLMAP feature extraction using ignore-masks at %s", colmap_masks)
        subprocess.run(feat_cmd, check=True, capture_output=True, timeout=300)

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
        # Cap undistorted image size. LichtFeld downscales any image wider than its
        # --max-width (default 3840px) on the CPU on *every* load — the downscale path is
        # NOT disk-cached, so a 36MP source costs ~5s/image every iteration, starving the
        # GPU and turning a 30k-iter run into hours. Undistorting at a bounded size (COLMAP
        # bakes the correctly-rescaled intrinsics into cameras.bin) keeps images under that
        # threshold so LichtFeld takes its fast, no-resample path and stays GPU-bound.
        undistort_max = getattr(self.config.ingest, "max_image_size", 2000) or 2000
        subprocess.run([
            colmap, "image_undistorter",
            "--image_path", str(frame_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(undist_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(undistort_max),
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

        The LichtFeld binary is baked into the consolidated image at build time
        (``/opt/gaussian-toolkit/build/LichtFeld-Studio``); ``ldconfig`` indexes
        the full vcpkg lib tree so all shared objects resolve without a manual
        ``LD_LIBRARY_PATH`` prefix — this path is still set at runtime for host
        CUDA libs mounted at ``/opt/host-cuda-libs``.

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

        # Robust-capture masking (scene_masking.py): LichtFeld natively discovers
        # <data-path>/masks/<image> (MaskDirCache) and drops those pixels from the loss with
        # --mask-mode ignore — absorbing moving people + blown highlights without ghosting.
        # Masks are 255=keep / 0=ignore; set training.invert_masks if a build expects the reverse.
        masks_dir = dataset_dir / "masks"
        if masks_dir.is_dir() and any(masks_dir.glob("*.png")):
            train_cmd += ["--mask-mode", "ignore"]
            if getattr(self.config.training, "invert_masks", False):
                train_cmd += ["--invert-masks"]
            logger.info("LichtFeld training with native --mask-mode ignore (%s)", masks_dir)

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
        self._export_native_splat_formats(result, ply_path)
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
        self._export_native_splat_formats(stage_result, ply_for_opt)
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

    def _export_native_splat_formats(
        self,
        stage_result: StageResult,
        ply_path: Path | None = None,
    ) -> None:
        """Export extra LichtFeld-native splat formats requested in delivery config.

        Non-fatal everywhere: errors are logged as warnings, never raised.
        Scene-residency: if a ply_path is provided and McpClient exposes
        load_checkpoint(), the scene is loaded before exporting so the MCP
        session has the trained splat resident.  Docker-sidecar backends
        (MILo/CoMe/GaussianWrapping) do not share the MCP session, so this
        ensures exports work regardless of which backend was used.

        Supported formats and their dispatch:
          spz        -> client.export_spz
          sog        -> client.export_sog
          html       -> client.export_html
          rad        -> client.export_rad
          usdz_nurec -> client.export_usdz_nurec
            # NVIDIA NuRec -- proprietary EULA; do not enable for commercial
            # client deliverables without licence review.
        """
        extra_formats = self.config.delivery.lichtfeld_extra_formats
        if not extra_formats:
            return

        from pipeline.mcp_client import McpClient

        client = McpClient(
            endpoint=self.config.mcp_endpoint,
            timeout=self.config.mcp_timeout,
        )
        if not client.ping():
            logger.warning(
                "_export_native_splat_formats: MCP endpoint %s unreachable; "
                "skipping all extra format exports",
                self.config.mcp_endpoint,
            )
            return

        # Ensure the trained scene is resident in the MCP session.
        if ply_path is not None and ply_path.exists():
            try:
                client.load_checkpoint(str(ply_path))
                logger.info(
                    "_export_native_splat_formats: loaded checkpoint %s into MCP session",
                    ply_path,
                )
            except Exception as exc:
                logger.warning(
                    "_export_native_splat_formats: load_checkpoint failed (%s); "
                    "attempting export anyway (may fail if scene not resident)",
                    exc,
                )
        else:
            logger.info(
                "_export_native_splat_formats: no ply_path supplied; "
                "export requires scene already resident in MCP session"
            )

        dispatch: dict[str, Any] = {
            "spz": client.export_spz,
            "sog": client.export_sog,
            "html": client.export_html,
            "rad": client.export_rad,
            # NVIDIA NuRec -- proprietary EULA; do not enable for commercial
            # client deliverables without licence review.
            "usdz_nurec": client.export_usdz_nurec,
        }
        # Format key -> on-disk extension (usdz_nurec writes a .usdz container).
        ext_map: dict[str, str] = {
            "spz": "spz",
            "sog": "sog",
            "html": "html",
            "rad": "rad",
            "usdz_nurec": "usdz",
        }
        sh_degree = self.config.delivery.lichtfeld_extra_formats_sh_degree

        delivery_dir = self.job_dir / "delivery"
        delivery_dir.mkdir(parents=True, exist_ok=True)

        for fmt in extra_formats:
            exporter = dispatch.get(fmt)
            if exporter is None:
                logger.warning(
                    "_export_native_splat_formats: unknown format %r; skipping",
                    fmt,
                )
                continue
            out_path = delivery_dir / f"scene.{ext_map.get(fmt, fmt)}"
            try:
                exporter(str(out_path), sh_degree=sh_degree)
                stage_result.artifacts[f"delivery_splat_{fmt}"] = str(out_path)
                logger.info(
                    "_export_native_splat_formats: exported %s -> %s", fmt, out_path
                )
            except Exception as exc:
                logger.warning(
                    "_export_native_splat_formats: export %r failed (non-fatal): %s",
                    fmt, exc,
                )

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
        self._export_native_splat_formats(stage_result, ply_for_opt)
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
        self._export_native_splat_formats(stage_result, ply_for_opt)
        return stage_result

    # ------------------------------------------------------------------
    # Stage 5b: Render previews
    # ------------------------------------------------------------------

    def render_previews(self, ply_path: str, colmap_dir: str, num_views: int = 8) -> StageResult:
        """Render RGB + depth previews from COLMAP camera positions using gsplat.

        Saves to job_dir/previews/ for the web carousel.

        cameras.bin parsing uses the struct format ``<iiQQ`` (camera_id int32,
        model_id int32, width uint64, height uint64). The previous ``<iiii``
        format split the 64-bit width field into two int32s, making render_h=0
        and causing a SIGFPE in the gsplat rasterizer tile-grid division.

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
                        # COLMAP cameras.bin layout: camera_id(uint32), model_id(int32),
                        # width(uint64), height(uint64), then params(double*). Reading w/h
                        # as int32 previously split the width uint64 into (w, h=high32=0),
                        # yielding render_h=0 → SIGFPE in the rasterizer tile grid.
                        cam_id, model_id, w, h = struct.unpack('<iiQQ', f.read(24))
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

    def _resolve_segment_concepts(
        self, frames_dir: str, *, discovery_callback=None,
    ) -> tuple[list[str], list]:
        """Resolve the SAM3 concept list for the segment stage.

        Default path: the static ``sam3_concepts`` (back-compatible). When
        ``decompose.use_open_vocab_discovery`` is on, first run a *pluggable
        vision overseer* (object_discovery.py) over representative frames to
        enumerate the objects actually present and use THOSE as concepts.

        Returns ``(concepts, discoveries)`` where ``discoveries`` are the full
        DiscoveredObject records (empty list when static). Never raises: any
        discovery failure degrades to the static list so Stage 6 is unaffected.
        """
        decompose_cfg = self.config.decompose
        static_concepts = decompose_cfg.sam3_concepts or decompose_cfg.descriptions or [
            "paintings", "frames", "sculptures", "furniture",
            "walls", "floor", "ceiling", "fixtures", "doorways",
        ]
        if not getattr(decompose_cfg, "use_open_vocab_discovery", False):
            return static_concepts, []

        try:
            from pipeline import object_discovery as od

            overseer = od.select_overseer(
                getattr(decompose_cfg, "discovery_overseer", "static"),
                callback=discovery_callback,
            )
            concepts, discoveries = od.discover_objects(
                frames_dir,
                overseer=overseer,
                hint_concepts=static_concepts,
                num_frames=getattr(decompose_cfg, "discovery_num_frames", 8),
                min_confidence=getattr(decompose_cfg, "discovery_min_confidence", 0.3),
                max_objects=getattr(decompose_cfg, "discovery_max_objects", 24),
            )
            return (concepts or static_concepts), discoveries
        except Exception as exc:  # discovery is advisory — never block Stage 6
            logger.warning("open-vocab discovery failed (%s) — using static concepts",
                           exc)
            return static_concepts, []

    def _write_segment_v2g_metadata(self, objects, discoveries, masks_dir):
        """Write v2g.object.1 records from segment-stage results (best-effort).

        Merges the discovery proposals (label/confidence/description/aliases)
        with the per-object SAM mask paths produced this stage and any sizes
        already present in ``objects_fbx_scaled/object_meta.json``. Returns the
        written JSON path, or ``None`` when there were no discoveries (static
        path keeps the legacy behaviour and emits nothing new).
        """
        if not discoveries:
            return None
        try:
            from pipeline import object_discovery as od

            # Map mask paths by canonical label so build_object_records can join.
            label_to_mask = {}
            for obj in objects:
                key = od.canonical_key(obj.get("label", ""))
                mp = Path(masks_dir) / f"mask_{int(obj.get('object_id', 0)):04d}.npy"
                if key and mp.exists():
                    label_to_mask[key] = mp

            arts = od.discover_run_artifacts(self.job_dir)
            records = od.build_object_records(
                discoveries,
                sizes=arts.get("sizes"),
                crops=arts.get("crops"),
                masks=label_to_mask,
                recons=arts.get("recons"),
            )
            out_path = self.job_dir / "objects_v2g.json"
            frames = []
            for d in discoveries:
                frames.extend(getattr(d, "source_frames", []) or [])
            return od.write_objects_v2g(
                records, out_path,
                source_frames=sorted(set(frames)),
                overseer=getattr(self.config.decompose, "discovery_overseer",
                                 "static"),
            )
        except Exception as exc:
            logger.warning("v2g.object.1 metadata write failed (%s)", exc)
            return None

    def segment(self, ply_path: str, frames_dir: str,
                discovery_callback=None) -> StageResult:
        """SAM2/SAM3 segmentation + mask projection.

        ``discovery_callback`` (optional) is a vision-overseer callable
        ``(frame_paths) -> list[{label, confidence, description, ...}]`` used for
        open-vocab object discovery when ``decompose.use_open_vocab_discovery`` is
        enabled. DiffusionGemma is text-only, so the overseer is supplied by the
        orchestrator (the claude_code agent / web / MCP layer); absent a callback
        discovery degrades to the static concept list.

        RESOLVED 2026-07-09 (PRD v4 R1): the "coarse boxes" defect was an
        HWC/CHW layout bug in our Sam3Processor call — set_image's numpy
        branch read (height, width) = shape[-2:] = (W, 3), so every mask was
        interpolated to a Wx3 grid. sam3_segmentor now passes PIL images and
        SAM3 returns pixel-accurate full-resolution silhouettes (verified on
        rawcapdev: vessel/bottle/block, see sam3_fixed_overlay.jpg).

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
        concepts, discoveries = self._resolve_segment_concepts(
            frames_dir, discovery_callback=discovery_callback)

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
                        all_masks[concept].append(
                            (frame_path.name, mask[0] if mask.ndim == 3 else mask))

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

                masks_dir = self.job_dir / "sam3_masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                perframe_root = masks_dir / "perframe"

                for concept, frame_mask_list in all_masks.items():
                    # Union across frames -> the legacy merged mask, AND keep the
                    # per-frame masks for the depth-aware multi-view projection in
                    # extract_objects (ADR-010 D10).
                    merged = np.zeros((ref_h, ref_w), dtype=bool)
                    for _fname, m in frame_mask_list:
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
                        oid = next_id
                        objects.append({
                            "label": concept,
                            "object_id": oid,
                            "mask_pixels": mask_pixels,
                        })
                        merged_masks.append(merged)
                        merged_ids.append(oid)
                        np.save(str(masks_dir / f"mask_{oid:04d}.npy"), merged)
                        pf_dir = perframe_root / f"{oid:04d}"
                        pf_dir.mkdir(parents=True, exist_ok=True)
                        for fname, m in frame_mask_list:
                            np.save(str(pf_dir / f"{Path(fname).stem}.npy"), m.astype(bool))
                        next_id += 1

                id_to_concept = {obj["object_id"]: obj["label"] for obj in objects}

                seg.unload()

                if not objects:
                    objects = [{"label": "full_scene", "count": -1}]

                # Emit v2g.object.1 metadata when open-vocab discovery ran. This
                # revives the rich per-object record (label/confidence/description
                # + mask/crop paths) instead of the size-only object_meta.json.
                v2g_path = self._write_segment_v2g_metadata(
                    objects, discoveries, masks_dir)

                artifacts = {
                    "objects": json.dumps(objects),
                    "masks_dir": str(masks_dir),
                }
                if v2g_path is not None:
                    artifacts["objects_v2g"] = str(v2g_path)

                return StageResult(
                    success=True, stage="segment",
                    metrics={
                        "object_count": len(objects),
                        "method": "sam3",
                        "concepts": concepts,
                        "discovered": bool(discoveries),
                    },
                    artifacts=artifacts,
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
    # Stage 6b: Object crops (ADR-025 D1 / PRD v4 R3)
    # ------------------------------------------------------------------

    def object_crops(self, frames_dir: str,
                     objects: list | dict | str | None = None) -> StageResult:
        """Best-frame crop + matte per SAM3 object.

        The crops produced here are the ONLY conditioning input the 3D
        generator receives (ADR-025 D1) — the splat contributes pose/scale,
        never pixels. Every crop carries provenance (source frame, bbox,
        COLMAP camera pose, matting method, selection scores) in
        ``object_crops/crops.json``. Objects with no usable observation are
        reported failures, never silently skipped.

        Args:
            frames_dir: Directory of source frames the segment stage saw.
            objects: Object list from segment() (JSON string, list, or dict).

        Returns: {crops_manifest, object_crops: list[{label, object_id, crop}]}
        """
        from pipeline import object_crops as oc

        if isinstance(objects, str):
            objects = json.loads(objects)
        elif isinstance(objects, dict):
            objects = [objects]
        objects = [o for o in (objects or [])
                   if o.get("label") != "full_scene" and o.get("object_id") is not None]

        if not objects:
            return StageResult(
                success=True, stage="object_crops",
                metrics={"skipped": True, "reason": "no segmented objects"},
                artifacts={"object_crops": json.dumps([])},
            )

        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return StageResult(
                success=False, stage="object_crops",
                error=f"Frames directory not found: {frames_dir}",
            )

        cfg = self.config.object_crops
        perframe_root = self.job_dir / "sam3_masks" / "perframe"
        out_dir = self.job_dir / "object_crops"
        colmap_dir = self._find_colmap_dir()

        results: list[oc.CropResult] = []
        crop_failures: list[dict[str, Any]] = []
        for obj in objects:
            label = obj.get("label", "unknown")
            oid = int(obj["object_id"])
            perframe_dir = perframe_root / f"{oid:04d}"
            if not perframe_dir.exists():
                crop_failures.append({
                    "label": label, "object_id": oid,
                    "error": f"no per-frame masks at {perframe_dir}",
                })
                continue
            try:
                res = oc.extract_object_crop(
                    label, oid, perframe_dir, frames_path, out_dir,
                    colmap_dir=colmap_dir,
                    pad_frac=cfg.pad_frac,
                    out_size=cfg.out_size,
                    candidates=cfg.candidates,
                    min_mask_area_px=cfg.min_mask_area_px,
                    matting=cfg.matting,
                    boxlike_fill_threshold=cfg.boxlike_fill_threshold,
                )
            except Exception as exc:  # noqa: BLE001 — reported per object
                res = None
                crop_failures.append({"label": label, "object_id": oid,
                                      "error": f"crop extraction failed: {exc}"})
                continue
            if res is None:
                crop_failures.append({
                    "label": label, "object_id": oid,
                    "error": ("no usable observation (all frames below "
                              f"min_mask_area_px={cfg.min_mask_area_px} or unreadable)"),
                })
                continue
            if res.provenance.get("boxlike_mask"):
                logger.warning(
                    "Crop for '%s' built from a BOX-shaped mask (R1 defect); "
                    "matting=%s", label, res.provenance.get("matting"))
            results.append(res)

        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = oc.write_crops_manifest(
            results, crop_failures, out_dir / "crops.json")

        if not results:
            causes = "; ".join(f"{f['label']}: {f['error']}"
                               for f in crop_failures[:8])
            return StageResult(
                success=False, stage="object_crops",
                error=f"No object crops could be extracted. {causes}",
                metrics={"failed": len(crop_failures)},
                artifacts={"crops_manifest": str(manifest)},
            )

        crops_list = [
            {"label": r.label, "object_id": r.object_id, "crop": str(r.crop_path)}
            for r in results
        ]
        return StageResult(
            success=True, stage="object_crops",
            metrics={
                "crop_count": len(results),
                "failed": len(crop_failures),
                "boxlike_masks": sum(
                    1 for r in results if r.provenance.get("boxlike_mask")),
            },
            artifacts={
                "crops_manifest": str(manifest),
                "object_crops": json.dumps(crops_list),
                **{f"crop:{r.label}": str(r.crop_path) for r in results},
            },
        )

    # ------------------------------------------------------------------
    # Stage 7: Extract objects
    # ------------------------------------------------------------------

    def extract_objects(self, ply_path: str, labels: dict | list | str | None = None) -> StageResult:
        """Extract per-object PLY files from the trained model.

        Gaussians are isolated via depth-aware multi-view mask projection
        (ADR-010 D10) from the SAM3 per-frame masks + COLMAP cameras. Per
        ADR-025 there are NO silent fallbacks: an object that cannot be isolated
        (missing masks, no COLMAP, empty projection) is reported as a per-object
        failure with cause — never approximated by copying the whole scene.
        Only the literal ``full_scene`` label (the environment) copies the
        trained PLY, because the whole scene IS that object. Per-object PLYs are
        used downstream for pose/scale placement (ADR-025 D3); generator
        conditioning comes from the ``object_crops`` stage, never from these.

        Args:
            ply_path: Path to the trained gaussian PLY.
            labels: Object definitions from segment(). Can be a JSON string,
                    list of dicts, or None (full_scene fallback).

        Returns: {object_plys: list[str], object_failures: list[{label, error}]}
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
        # ADR-010 D10: COLMAP cameras + per-frame masks drive the depth-aware
        # multi-view projection. There is no camera-free fallback (ADR-025):
        # without them, isolation is a reported failure.
        colmap_dir = self._find_colmap_dir()
        perframe_avail = (masks_dir / "perframe").exists()

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
        failures: list[dict[str, Any]] = []

        for obj in objects:
            label = obj.get("label", "unknown")
            safe_name = label.replace(" ", "_").replace("/", "_")[:50]
            out_ply = objects_dir / f"{safe_name}.ply"

            # full_scene is the whole environment — the trained PLY IS the
            # object. Never keyness-ranked or dropped.
            if label == "full_scene":
                shutil.copy2(str(trained_ply), str(out_ply))
                extracted.append({"label": label, "ply": str(out_ply),
                                  "gaussian_count": -1, "keyness": float("inf")})
                logger.info("Copied trained PLY as '%s': %s", label, out_ply)
                continue

            # ADR-025: no silent fallbacks. Isolation requires SAM3 per-frame
            # masks + a COLMAP model for the depth-aware multi-view projection;
            # anything less is a reported per-object failure, not a fake asset.
            error: str | None = None
            if not has_masks:
                error = "no SAM3 masks (sam3_masks/ missing or empty)"
            elif obj.get("object_id") is None:
                error = "object has no object_id (segment stage emitted no mask)"
            elif colmap_dir is None:
                error = "no COLMAP sparse model for multi-view mask projection"
            elif not perframe_avail:
                error = "no per-frame masks (sam3_masks/perframe/ missing)"

            count = 0
            if error is None:
                try:
                    count = self._extract_with_mask_mv(
                        obj, trained_ply, out_ply, masks_dir, colmap_dir)
                    if count <= 0:
                        error = "multi-view mask projection assigned 0 gaussians"
                except Exception as exc:
                    error = f"multi-view mask projection failed: {exc}"

            if error is not None:
                logger.warning("Object isolation FAILED for '%s': %s", label, error)
                failures.append({"label": label, "error": error})
                out_ply.unlink(missing_ok=True)
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
            causes = "; ".join(f"{f['label']}: {f['error']}" for f in failures[:8])
            return StageResult(
                success=False, stage="extract_objects",
                error=(f"No objects could be isolated "
                       f"({len(failures)} failed, {len(dropped)} below "
                       f"min_object_gaussians={min_gaussians}). {causes}"),
                metrics={"failed": len(failures),
                         "dropped_below_threshold": len(dropped)},
                artifacts={"object_failures": json.dumps(failures)},
            )

        # FR-9: rank by keyness (descending) so downstream per-object hull
        # reconstruction processes the most significant items first.
        extracted.sort(key=lambda e: e.get("keyness", 0.0), reverse=True)

        return StageResult(
            success=True, stage="extract_objects",
            metrics={
                "extracted_count": len(extracted),
                "failed": len(failures),
                "dropped_below_threshold": len(dropped),
                "min_object_gaussians": min_gaussians,
                "ranking": [e["label"] for e in extracted],
            },
            artifacts={
                "object_plys": json.dumps([e["ply"] for e in extracted]),
                "object_ranking": json.dumps(extracted),
                "object_failures": json.dumps(failures),
                **{f"ply:{e['label']}": e["ply"] for e in extracted},
            },
        )

    def _find_colmap_dir(self) -> "Path | None":
        """Locate a COLMAP sparse model (text OR binary) in the job dir, for
        the depth-aware mask projection + crop pose provenance. The direct-
        COLMAP path emits binary-only models (e.g. rawcapdev), so both formats
        are accepted. Returns None if none is found."""
        from pipeline.colmap_parser import find_model_dir
        return find_model_dir(self.job_dir / "colmap")

    def _extract_with_mask_mv(
        self,
        obj: dict[str, Any],
        trained_ply: Path,
        output_path: Path,
        masks_dir: Path,
        colmap_dir: Path,
    ) -> int:
        """Depth-aware multi-view mask projection (ADR-010 D10).

        Projects every gaussian centre through each registered COLMAP camera that
        has a per-frame mask for this object, votes inside/outside the mask across
        views, and keeps gaussians seen inside the object in a majority of the
        views where they are visible. Multi-view consistency + parallax filter the
        occluded-background gaussians that a single-view projection would include.
        Returns the gaussian count (0 = isolation failed for this object).
        """
        import numpy as np
        object_id = obj.get("object_id")
        if object_id is None:
            return 0
        pf_dir = masks_dir / "perframe" / f"{int(object_id):04d}"
        if not pf_dir.exists():
            return 0
        try:
            from pipeline.colmap_parser import load_cameras, load_images
        except Exception:
            return 0
        cams = load_cameras(colmap_dir)
        images = load_images(colmap_dir)
        by_stem = {Path(im.name).stem: im for im in images}

        import trimesh
        pcd = trimesh.load(str(trained_ply))
        points = np.asarray(pcd.vertices) if hasattr(pcd, "vertices") else (
            np.asarray(pcd.points) if hasattr(pcd, "points") else None)
        if points is None or len(points) == 0:
            return 0
        P = points.astype(np.float64)
        N = len(P)
        votes_in = np.zeros(N, dtype=np.int32)
        votes_vis = np.zeros(N, dtype=np.int32)

        import cv2
        used = 0
        for mfile in sorted(pf_dir.glob("*.npy")):
            im = by_stem.get(mfile.stem)
            if im is None:
                continue
            cam = cams.get(im.camera_id)
            if cam is None:
                continue
            mask = np.load(str(mfile))
            mask = mask[0] if mask.ndim == 3 else mask
            if not mask.any():
                continue  # object not detected in this frame -> uninformative vote
            H, W = int(cam.height), int(cam.width)
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            mask = mask > 0
            # world -> camera (COLMAP world-to-camera): p_cam = R @ P + t
            qw, qx, qy, qz = im.quaternion
            nrm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
            if nrm == 0:
                continue
            qw, qx, qy, qz = qw / nrm, qx / nrm, qy / nrm, qz / nrm
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=np.float64)
            t = np.array(im.translation, dtype=np.float64)
            pc = P @ R.T + t
            z = pc[:, 2]
            front = z > 1e-6
            zsafe = np.where(front, z, 1.0)
            u = (cam.focal_x * pc[:, 0] / zsafe + cam.center_x)
            v = (cam.focal_y * pc[:, 1] / zsafe + cam.center_y)
            # Behind-camera / far-outside points can be inf/NaN or overflow
            # int32; clip before the cast (they are dropped by `vis` anyway).
            ui = np.nan_to_num(u, nan=-1.0, posinf=-1.0, neginf=-1.0) \
                .clip(-1, W).astype(np.int32)
            vi = np.nan_to_num(v, nan=-1.0, posinf=-1.0, neginf=-1.0) \
                .clip(-1, H).astype(np.int32)
            vis = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not vis.any():
                continue
            used += 1
            votes_vis[vis] += 1
            idx = np.where(vis)[0]
            inside = mask[vi[idx], ui[idx]]
            votes_in[idx[inside]] += 1

        if used == 0:
            return 0
        min_views = max(1, min(2, used))
        sel = (votes_vis >= min_views) & (votes_in >= 0.5 * np.maximum(votes_vis, 1))
        n_sel = int(sel.sum())
        if n_sel == 0:
            return 0
        # Save the gaussian-attribute SUBSET of the original PLY (opacity, scale,
        # rotation, SH) so downstream gsplat orbit-render / hull reconstruction
        # works — a bare xyz point cloud is not renderable as a splat.
        try:
            from plyfile import PlyData, PlyElement
            orig = PlyData.read(str(trained_ply))
            vdata = orig.elements[0].data
            if len(vdata) == N and len(vdata.dtype.names) > 3:
                PlyData([PlyElement.describe(vdata[sel], "vertex")],
                        text=False).write(str(output_path))
                logger.info("MV-projected %d/%d gaussians (full attrs) for '%s' across %d views",
                            n_sel, N, obj.get("label", "?"), used)
                return n_sel
        except Exception as exc:
            logger.warning("gaussian-subset save failed (%s); point-cloud fallback", exc)
        colors = None
        if hasattr(pcd, "visual") and hasattr(pcd.visual, "vertex_colors"):
            vc = np.asarray(pcd.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == N:
                colors = vc[sel]
        out = trimesh.PointCloud(points[sel])
        if colors is not None:
            out.colors = colors
        out.export(str(output_path))
        logger.info("MV-projected %d/%d gaussians for '%s' across %d views",
                    n_sel, N, obj.get("label", "?"), used)
        return n_sel

    # ------------------------------------------------------------------
    # Stage 8: Mesh objects
    # ------------------------------------------------------------------

    def mesh_objects(self, object_plys: list[str] | str,
                     crops: list | str | None = None) -> StageResult:
        """Generate meshes: environment from the splat, objects from crops.

        ADR-025: the 3D generator is conditioned on the object's best-frame
        crop (from the ``object_crops`` stage) — NEVER on renders of the
        per-object splat. The per-object PLY is kept alongside for pose/scale
        at assembly. ``full_scene`` (the environment) keeps the gsplat-TSDF
        geometry path. An object with no crop is a reported failure.

        Args:
            object_plys: List of PLY paths, or a JSON string of a list.
            crops: object_crops artifact — list of {label, object_id, crop}
                   (or its JSON string).

        Returns: {meshes: list[{label, mesh, ply, vertex_count, method, ...}],
                  mesh_failures: list[{label, error}]}
        """
        if isinstance(object_plys, str):
            object_plys = json.loads(object_plys)
        if isinstance(crops, str):
            crops = json.loads(crops)

        # Crops are keyed by the same safe-name used for the PLY stem.
        def _safe(name: str) -> str:
            return name.replace(" ", "_").replace("/", "_")[:50]
        crop_by_safe_label: dict[str, dict[str, Any]] = {
            _safe(c.get("label", "")): c for c in (crops or [])
        }
        crop_provenance = self._load_crop_provenance()

        meshes_dir = self.job_dir / "objects" / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        mesh_failures: list[dict[str, Any]] = []

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        for ply_path in object_plys:
            ply = Path(ply_path)
            if not ply.exists():
                logger.warning("PLY not found: %s", ply_path)
                mesh_failures.append({"label": ply.stem,
                                      "error": f"PLY not found: {ply_path}"})
                continue

            label = ply.stem
            obj_dir = meshes_dir / label
            obj_dir.mkdir(parents=True, exist_ok=True)
            mesh_glb = obj_dir / f"{label}.glb"
            mesh_obj = obj_dir / f"{label}.obj"
            crop_entry = crop_by_safe_label.get(label)

            # Run each object with a hard timeout to prevent the stage from
            # hanging in subprocess environments where SIGALRM doesn't propagate.
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._mesh_single, label, str(ply), obj_dir, mesh_obj,
                        mesh_glb, crop_entry, crop_provenance.get(label),
                    )
                    mesh_result = future.result(timeout=3600)
            except FuturesTimeout:
                logger.warning("Meshing timed out (3600s) for '%s'", label)
                mesh_result = None
                mesh_failures.append({"label": label, "error": "timed out (3600s)"})
            except Exception as exc:
                logger.warning("Meshing failed for '%s': %s", label, exc)
                mesh_result = None
                mesh_failures.append({"label": label, "error": str(exc)})

            if mesh_result is not None:
                results.append(mesh_result)
            elif not any(f["label"] == label for f in mesh_failures):
                mesh_failures.append({
                    "label": label,
                    "error": ("no generator produced a mesh "
                              + ("" if crop_entry else "(no object crop available)")).strip(),
                })

        if not results:
            causes = "; ".join(f"{f['label']}: {f['error']}" for f in mesh_failures[:8])
            return StageResult(
                success=False, stage="mesh_objects",
                error=f"No meshes could be generated. {causes}",
                metrics={"failed": len(mesh_failures)},
                artifacts={"mesh_failures": json.dumps(mesh_failures)},
            )

        return StageResult(
            success=True, stage="mesh_objects",
            metrics={"mesh_count": len(results), "failed": len(mesh_failures)},
            artifacts={
                "meshes": json.dumps(results),
                "mesh_failures": json.dumps(mesh_failures),
                **{f"mesh:{r['label']}": r["mesh"] for r in results},
            },
        )

    def _load_crop_provenance(self) -> dict[str, dict[str, Any]]:
        """Provenance per safe-label from the object_crops manifest (if any)."""
        manifest = self.job_dir / "object_crops" / "crops.json"
        if not manifest.exists():
            return {}
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            out: dict[str, dict[str, Any]] = {}
            for entry in data.get("crops", []):
                label = entry.get("label", "")
                safe = label.replace(" ", "_").replace("/", "_")[:50]
                out[safe] = entry
            return out
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read crops manifest: %s", exc)
            return {}

    def _mesh_single(
        self,
        label: str,
        ply_path: str,
        obj_dir: Path,
        mesh_obj_path: Path,
        mesh_glb_path: Path,
        crop_entry: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Mesh one entry: environment via geometry, objects via the generator.

        Non-``full_scene`` objects take the ADR-025 generator path — one matted
        crop into TRELLIS.2 (Hunyuan3D-2.1 single-image as fallback). There is
        no geometry-from-partial-splat fallback for objects: an orbit-TSDF of a
        partially-captured object splat is holey/noisy, and shipping it would
        be a silent quality lie. Failures raise (reported by mesh_objects).
        """
        import numpy as np
        import trimesh

        if label != "full_scene":
            return self._generate_object_from_crop(
                label, ply_path, mesh_glb_path, crop_entry, provenance)

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

        # Strategy 0: gsplat depth rendering -> TSDF. Correct for the ROOM
        # (full_scene, dense COLMAP cameras) but WRONG for isolated objects —
        # an orbit-TSDF of a partially-captured object splat is holey/noisy.
        # Gate it to full_scene so per-object PLYs go to TRELLIS.2 first (ADR-015
        # designated-primary hull, hull_e2e-validated). Fixes master-audit gap #8.
        try:
            import torch
            if label == "full_scene" and torch.cuda.is_available():
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

        # (The old TRELLIS.2/Hunyuan3D splat-render branches lived here. They
        # conditioned the generators on CPU re-renders of the segmented splat —
        # the anti-pattern ADR-025 retired. Objects now go through
        # _generate_object_from_crop above; only environment geometry follows.)

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

    def _object_placement_from_ply(self, ply_path: str) -> dict[str, Any] | None:
        """Scene-frame placement hints from the object's gaussian subset.

        The splat contributes pose/scale, never pixels (ADR-025 D3). Centroid
        + extent seed the R10 pose-solve at USD assembly.
        """
        try:
            points, _colors = _load_ply_points(ply_path)
            if points is None or len(points) == 0:
                return None
            return {
                "centroid": [float(v) for v in points.mean(axis=0)],
                "extent": [float(v) for v in (points.max(axis=0) - points.min(axis=0))],
                "gaussian_count": int(len(points)),
            }
        except Exception as exc:  # noqa: BLE001 — placement hints are best-effort
            logger.warning("Placement extraction failed for %s: %s", ply_path, exc)
            return None

    def _generate_object_from_crop(
        self,
        label: str,
        ply_path: str,
        mesh_glb_path: Path,
        crop_entry: dict[str, Any] | None,
        provenance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """ADR-025 object generation: ONE matted crop -> textured GLB.

        Generator chain: TRELLIS.2 single-image (primary, ADR-015 as amended)
        -> Hunyuan3D-2.1 single-image (the proven dreamlab graph). GLB bytes
        are persisted byte-identical with a recorded sha256 (PRD v4 R6); a
        ``<name>.lineage.json`` sits next to every asset. Raises on failure —
        mesh_objects reports it; nothing is faked.
        """
        crop_path = Path(crop_entry["crop"]) if crop_entry and crop_entry.get("crop") else None
        if crop_path is None or not crop_path.exists():
            raise RuntimeError(
                f"no object crop for '{label}' — the object_crops stage must "
                "produce one (ADR-025: generators are conditioned on crops, "
                "never splat renders)")

        placement = self._object_placement_from_ply(ply_path)

        def _persist(glb_data: bytes, method: str, lineage: dict[str, Any],
                     glb_low: bytes | None = None,
                     mesh: "Any | None" = None) -> dict[str, Any]:
            import hashlib
            mesh_glb_path.write_bytes(glb_data)
            sha = hashlib.sha256(glb_data).hexdigest()
            # The generated mesh's own normalized bbox — the denominator of the
            # R10 pose-solve scale ratio (real size / normalized size).
            glb_extent = None
            if mesh is not None and getattr(mesh, "extents", None) is not None:
                try:
                    glb_extent = [float(v) for v in mesh.extents]
                except (TypeError, ValueError):
                    glb_extent = None
            info: dict[str, Any] = {
                "label": label,
                "mesh": str(mesh_glb_path),
                "ply": ply_path,
                "vertex_count": 0 if mesh is None else len(mesh.vertices),
                "method": method,
                "generator": True,
                "glb_sha256": sha,
                "crop": str(crop_path),
                "placement": placement,
                "glb_extent": glb_extent,
                # R10 orientation solve reads the crop's COLMAP camera pose
                # (recorded in the object_crops manifest -> provenance here).
                "camera_pose": provenance.get("camera_pose") if provenance else None,
            }
            if glb_low is not None:
                low_path = mesh_glb_path.with_name(mesh_glb_path.stem + "_low.glb")
                low_path.write_bytes(glb_low)
                info["mesh_low"] = str(low_path)
            lineage_path = mesh_glb_path.with_suffix(".lineage.json")
            lineage_path.write_text(json.dumps({
                **lineage, "glb_sha256": sha, "placement": placement,
            }, indent=2, default=str), encoding="utf-8")
            info["lineage"] = str(lineage_path)
            return info

        errors: list[str] = []

        # PRD v4 R8: Pixal3D (MIT, TRELLIS.2 backbone) as an OPT-IN primary.
        # Default-off, eval-gated; on any failure the chain falls through to
        # TRELLIS.2 unchanged, so enabling only adds a candidate tier.
        pixal3d_cfg = getattr(self.config, "pixal3d", None)
        if pixal3d_cfg is not None and getattr(pixal3d_cfg, "enabled", False):
            try:
                from pipeline.pixal3d_client import Pixal3DClient
                px = Pixal3DClient.from_config(pixal3d_cfg)
                res = px.reconstruct_from_image(
                    crop_path, seed=getattr(pixal3d_cfg, "seed", 42),
                    label=label, provenance=provenance)
                if res.glb_data:
                    logger.info("Pixal3D object '%s': %d verts (%.0fs)",
                                label, res.vertex_count, res.duration_seconds)
                    return _persist(res.glb_data, "pixal3d", dict(res.lineage),
                                    glb_low=res.glb_low_data, mesh=res.mesh)
                errors.append(f"pixal3d: {res.error or 'no GLB'}")
            except Exception as exc:  # noqa: BLE001 — fall through to TRELLIS.2
                logger.warning("Pixal3D failed for '%s': %s", label, exc)
                errors.append(f"pixal3d: {exc}")

        trellis2_cfg = getattr(self.config, "trellis2", None)
        if trellis2_cfg is not None and getattr(trellis2_cfg, "enabled", False):
            try:
                from pipeline.trellis2_client import Trellis2Client
                from pipeline import object_candidate_score as ocs
                t2 = Trellis2Client.from_config(trellis2_cfg)

                n = max(1, int(getattr(trellis2_cfg, "best_of_n", 1)))
                seeds = ocs.derive_seeds(
                    getattr(trellis2_cfg, "seed", 42), n,
                    getattr(trellis2_cfg, "seeds", None))
                crop_aspect = ocs.crop_aspect_from_provenance(provenance)

                # Generate each seed; score by front-silhouette proportion +
                # mesh sanity (R7). best_of_n==1 -> single shot, unchanged.
                best = None            # (total, result, edit_lineage | None)
                scoreboard: list[dict[str, Any]] = []
                for seed in seeds:
                    res = t2.reconstruct_from_image(
                        crop_path, seed=seed, label=label, provenance=provenance)
                    if not res.glb_data:
                        errors.append(f"trellis2(seed={seed}): {res.error or 'no GLB'}")
                        continue
                    extent = list(res.mesh.extents) if (
                        res.mesh is not None and getattr(res.mesh, "extents", None)
                        is not None) else None
                    watertight = bool(getattr(res.mesh, "is_watertight", False)) \
                        if res.mesh is not None else False
                    cs = ocs.score_candidate(
                        seed, extent, crop_aspect, res.face_count, watertight)
                    scoreboard.append(cs.__dict__)
                    logger.info("TRELLIS.2 '%s' seed=%d: %d verts, score=%.3f "
                                "(prop=%.2f sanity=%.2f)", label, seed,
                                res.vertex_count, cs.total, cs.proportion, cs.sanity)
                    if best is None or cs.total > best[0]:
                        best = (cs.total, res, None)

                # Escalation rung (b), ADR-025 D4 / PRD v4 R7: the best seed
                # re-roll still scores poorly -> synthesize ONE image-edit
                # alternate view and add it as an additional single-image
                # candidate. Self-gating: a missing Qwen UNET degrades to
                # rung (a) silently-but-logged; a rung-(b) failure never kills
                # the seed-re-roll winner.
                threshold = float(getattr(trellis2_cfg, "escalation_threshold", 0.55))
                if (getattr(trellis2_cfg, "image_edit_escalation", False)
                        and (best is None or best[0] < threshold)):
                    try:
                        from pipeline.image_edit_view import ImageEditView
                        iev = ImageEditView.from_config(
                            getattr(self.config, "image_edit", None))
                        if iev.probe_edit_model() is None:
                            logger.info("image-edit escalation skipped for '%s': "
                                        "edit UNET not staged", label)
                        else:
                            edited = iev.edit_view(
                                crop_path, label=label, provenance=provenance,
                                output_path=mesh_glb_path.with_name(
                                    mesh_glb_path.stem + "__editview.png"))
                            if edited.image_path is None:
                                errors.append(f"image-edit: {edited.error}")
                            else:
                                res2 = t2.reconstruct_from_image(
                                    edited.image_path,
                                    seed=getattr(trellis2_cfg, "seed", 42),
                                    label=f"{label}_editview", provenance=provenance)
                                if not res2.glb_data:
                                    errors.append(f"trellis2(edit-view): "
                                                  f"{res2.error or 'no GLB'}")
                                else:
                                    extent2 = list(res2.mesh.extents) if (
                                        res2.mesh is not None and getattr(
                                            res2.mesh, "extents", None) is not None
                                    ) else None
                                    wt2 = bool(getattr(
                                        res2.mesh, "is_watertight", False)) \
                                        if res2.mesh is not None else False
                                    # A 180deg back view preserves the front
                                    # silhouette's width/height for most objects,
                                    # so proportion-vs-crop stays a fair score.
                                    cs2 = ocs.score_candidate(
                                        getattr(trellis2_cfg, "seed", 42), extent2,
                                        crop_aspect, res2.face_count, wt2)
                                    scoreboard.append(
                                        {"view": "image-edit-back", **cs2.__dict__})
                                    logger.info("TRELLIS.2 '%s' edit-view candidate:"
                                                " %d verts, score=%.3f", label,
                                                res2.vertex_count, cs2.total)
                                    if best is None or cs2.total > best[0]:
                                        best = (cs2.total, res2, edited.lineage)
                    except Exception as exc:  # noqa: BLE001 — rung (b) optional
                        logger.warning("image-edit escalation failed for '%s': %s",
                                       label, exc)
                        errors.append(f"image-edit: {exc}")

                if best is not None:
                    _, res, edit_lineage = best
                    lineage = dict(res.lineage)
                    if n > 1:
                        lineage["best_of_n"] = n
                        lineage["candidates"] = scoreboard
                        lineage["selected_seed"] = next(
                            (c["seed"] for c in scoreboard
                             if c["total"] == max(s["total"] for s in scoreboard)),
                            res.lineage.get("seed"))
                    if edit_lineage is not None:
                        # Winner came from the synthesized view — flag it: the
                        # entire conditioning image was model-inferred.
                        lineage["escalation"] = {"rung": "image-edit-view",
                                                 **edit_lineage}
                        lineage["surface"] = "image-edit-inferred"
                    logger.info("TRELLIS.2 object '%s': kept best of %d (%.0fs)",
                                label, len(scoreboard), res.duration_seconds)
                    return _persist(res.glb_data, "trellis2", lineage,
                                    glb_low=res.glb_low_data, mesh=res.mesh)
            except Exception as exc:  # noqa: BLE001 — try the fallback generator
                logger.warning("TRELLIS.2 failed for '%s': %s", label, exc)
                errors.append(f"trellis2: {exc}")

        if self.config.hunyuan3d.enabled:
            try:
                from pipeline.hunyuan3d_client import Hunyuan3DClient
                h3d = Hunyuan3DClient.from_config(self.config.hunyuan3d)
                res = h3d.reconstruct_from_image(
                    crop_path, seed=self.config.hunyuan3d.seed, label=label)
                if res.glb_data:
                    logger.info("Hy3D21 object '%s': %d verts (%.0fs)",
                                label, res.vertex_count, res.duration_seconds)
                    lineage = {
                        "conditioning": "single-image",
                        "crop": str(crop_path),
                        "generator": "Hunyuan3D-2.1",
                        "executor": res.backend,
                        "seed": self.config.hunyuan3d.seed,
                        "surface": "observed-front/inferred-back",
                        **({"source": provenance} if provenance else {}),
                    }
                    return _persist(res.glb_data, "hunyuan3d21", lineage,
                                    mesh=res.mesh)
                errors.append(f"hunyuan3d21: {res.error or 'no GLB'}")
            except Exception as exc:  # noqa: BLE001 — reported below
                logger.warning("Hy3D21 failed for '%s': %s", label, exc)
                errors.append(f"hunyuan3d21: {exc}")

        raise RuntimeError(
            f"all generators failed for '{label}': " + "; ".join(errors or ["none enabled"]))

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

        skipped_generator = 0
        for mesh_info in meshes:
            label = mesh_info.get("label", "unknown")
            # PRD v4 R6: generator GLBs (TRELLIS.2 / Hy3D21) already carry
            # baked PBR materials and are persisted byte-identical. Re-unwrap
            # + splat-projection here destroyed them (audit defect F7) —
            # splat-projection baking is for TSDF/environment meshes only.
            if mesh_info.get("generator") or mesh_info.get("method") in (
                    "trellis2", "hunyuan3d21"):
                skipped_generator += 1
                logger.info("texture_bake: skipping generator mesh '%s' "
                            "(PBR persisted verbatim)", label)
                continue
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
            metrics={"baked_count": baked_count, "total_meshes": len(meshes),
                     "skipped_generator_meshes": skipped_generator},
            artifacts={
                "textured_meshes": json.dumps(meshes),
            },
        )

    # ------------------------------------------------------------------
    # Stage 10: Assemble USD
    # ------------------------------------------------------------------

    def _export_native_usd(self, out_path: Path) -> str:
        """Best-effort native USD export via the LichtFeld CLI `convert`
        subcommand (the binary reads .ply and writes native .usd/.usda
        headlessly). Converts the trained scene gaussian field. Returns the path
        on success, "" on any failure (the composed assembler stays authoritative).
        Never raises.
        """
        try:
            # Source = the latest trained gaussian PLY for this job.
            candidates = sorted(self.job_dir.rglob("model/**/*.ply"))
            if not candidates:
                candidates = sorted(p for p in self.job_dir.rglob("*.ply")
                                    if p.name not in ("scene.ply",))
            if not candidates:
                logger.info("Native USD export skipped: no trained PLY found")
                return ""
            src_ply = candidates[-1]

            # Resolve the LichtFeld binary (same candidate list as train()).
            lfs_binary = self.config.training.lichtfeld_binary
            for cand in (
                lfs_binary,
                "/opt/gaussian-toolkit/build/LichtFeld-Studio",
                str(Path.home() / "workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio"),
            ):
                if cand and Path(cand).exists():
                    lfs_binary = cand
                    break
            if not lfs_binary or not Path(lfs_binary).exists():
                logger.info("Native USD export skipped: LichtFeld binary not found")
                return ""

            # `convert` accepts .usd/.usda/.usdc (not .usdz).
            if out_path.suffix not in (".usd", ".usda", ".usdc"):
                out_path = out_path.with_suffix(".usda")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            blib = Path(lfs_binary).parent
            ld = ":".join(p for p in [
                str(blib),
                str(blib / "vcpkg_installed" / "x64-linux" / "debug" / "lib"),
                str(blib / "vcpkg_installed" / "x64-linux" / "lib"),
                "/opt/host-cuda-libs",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ] if p)
            proc = subprocess.run(
                [str(lfs_binary), "convert", str(src_ply), str(out_path)],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "LD_LIBRARY_PATH": ld},
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info("Native LichtFeld USD export -> %s (%d bytes)",
                            out_path, out_path.stat().st_size)
                return str(out_path)
            logger.info("Native USD export produced no output (rc=%d): %s",
                        proc.returncode, (proc.stderr or "")[-200:])
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

        # R10 pose-solve: emit per-object placement (world position + uniform
        # scale from the object's Gaussian subset) for the standalone assembler
        # to apply, so generated objects land at their real scene location/size
        # instead of piled at the origin at unit scale. Orientation is flagged
        # unsolved (ADR-025 D3). Best-effort — never blocks assembly.
        try:
            from pipeline.object_placement import build_placements
            placements = build_placements(objects)
            if placements:
                (usd_dir / "placements.json").write_text(
                    json.dumps(placements, indent=2), encoding="utf-8")
                logger.info("R10: wrote placements for %d object(s)", len(placements))
        except Exception as exc:  # noqa: BLE001 — placement is additive
            logger.warning("R10 placement solve failed: %s", exc)

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
