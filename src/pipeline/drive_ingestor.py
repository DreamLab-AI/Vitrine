# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bulk capture ingestor — pull → process → push → purge over a cloud remote.

Replaces browser drag-and-drop as the primary path for bulk capture sets that
live on a Google Cloud / Google Drive remote. Designed for the room-orbit
capture style documented in ``research/decisions/video-ingestion-plan.md``:

* **Flat folder per capture** — each directory under the configured base path
  is one *session*; every video file inside it is a *set* of that session.
* **Same lens across sets** — sets of one session are pooled into a single
  combined COLMAP reconstruction (``single_camera=True``), which is the biggest
  quality lever for orbit-at-different-heights captures.
* **Outputs back to the same folder** — results are pushed to
  ``<remote>:<base>/<session>/outputs/`` using the same service-account creds.
* **One-time batch** — processes every discovered session sequentially then
  exits. A small SQLite ledger makes the batch resumable (skip-completed).
* **Delete-local-only retention** — after a *verified* upload the local raw +
  heavy scratch are purged; the raw stays on the remote as source of truth.

All cloud I/O goes through ``rclone`` (configured with a service-account file,
remote-type-agnostic — works for ``drive:`` or ``gcs:`` remotes). Credentials
are never interpolated into command strings; the remote is expected to be
pre-configured in an rclone config file (see ``--rclone-config``) per the
plan's security section (service-account file, not env-var secrets).

The heavy COLMAP/CUDA stages run via the existing :class:`PipelineStages`; this
module only orchestrates the data movement and stage sequencing around them.
Heavy imports are deferred so the module loads (and ``ast.parse``s) on hosts
without rclone, CUDA, or the pipeline's runtime deps installed.

CLI::

    python -m pipeline.drive_ingestor run \\
        --remote gcs:my-bucket/captures \\
        --rclone-config /run/secrets/rclone.conf \\
        --scratch /data/scratch --raw /data/raw

    python -m pipeline.drive_ingestor run --remote gdrive:Captures --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Video extensions treated as capture "sets" within a session folder.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

# Output artefacts uploaded back to the remote. Heavy, reproducible
# intermediates (raw frames, COLMAP undistorted images, databases) are NOT
# uploaded — only the final/curated geometry + a manifest + small previews.
DEFAULT_LEDGER_NAME = "drive_ingest_ledger.sqlite"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DriveIngestConfig:
    """Configuration for the bulk-capture ingestor.

    Attributes:
        remote: rclone remote + base path, e.g. ``gcs:bucket/captures`` or
            ``gdrive:Captures``. Each sub-folder is one session.
        rclone_config: Path to an rclone config file holding the
            service-account-authenticated remote. ``None`` uses rclone's
            default config discovery.
        local_raw_dir: Host NVMe path where the active session's raw videos are
            copied (purged after a verified upload).
        local_scratch_dir: Host NVMe path for the pipeline working set (frames,
            COLMAP, model, meshes) — purged after a verified upload.
        output_subfolder: Sub-folder name created inside each session folder on
            the remote to receive outputs.
        ledger_path: SQLite ledger location (defaults under ``local_scratch_dir``
            so it survives session purges but not a full scratch wipe; point it
            at a persistent volume for cross-run resume).
        fps: Frame-extraction rate per set (slow orbit → low fps is enough).
        target_frames: Pooled selected-frame target across all sets of a session.
        max_image_size: COLMAP/SfM downscale cap (geometry doesn't need 4K).
        mesh_method: Mesh backend — ``auto`` applies the ADR-003 policy.
        scene_preset: Training preset (``indoor_reflective`` suits a small room).
        enable_splat_optimize: Produce a compressed web splat (delivery).
        use_fibonacci_coverage: Blend Fibonacci-sphere viewpoint coverage into
            frame selection (ADR-007).
        run_decomposition: Run the segment→objects→USD assembly stages after
            geometry. When False, stops after train + previews (geometry only).
        copy_transfers: rclone parallel transfer count.
        copy_timeout: Per-session download/upload timeout (seconds).
        keep_local_on_success: When True, skip the purge step (debugging).
    """
    remote: str = ""
    rclone_config: Optional[str] = None
    local_raw_dir: str = "/data/raw"
    local_scratch_dir: str = "/data/scratch"
    output_subfolder: str = "outputs"
    ledger_path: Optional[str] = None

    # Pipeline knobs (the §5.4 room-orbit starting point)
    fps: float = 2.0
    target_frames: int = 600
    max_image_size: int = 3200
    mesh_method: str = "auto"
    scene_preset: str = "indoor_reflective"
    enable_splat_optimize: bool = True
    use_fibonacci_coverage: bool = True
    coverage_weight: float = 0.4
    run_decomposition: bool = True

    # rclone / IO behaviour
    copy_transfers: int = 8
    copy_timeout: int = 14400  # 4 h per session for the big 4K pulls/pushes
    keep_local_on_success: bool = False

    def resolved_ledger_path(self) -> Path:
        if self.ledger_path:
            return Path(self.ledger_path)
        return Path(self.local_scratch_dir) / DEFAULT_LEDGER_NAME


# ---------------------------------------------------------------------------
# Resumable ledger (SQLite)
# ---------------------------------------------------------------------------

class Ledger:
    """Tiny SQLite ledger for skip-completed / resume across a batch run.

    Statuses: ``pending`` → ``processing`` → ``done`` | ``failed``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                remote_path  TEXT,
                status       TEXT NOT NULL,
                n_sets       INTEGER,
                output_remote TEXT,
                error        TEXT,
                started_at   REAL,
                finished_at  REAL,
                updated_at   REAL
            )
            """
        )
        self._conn.commit()

    def upsert(self, session_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        existing = self.get(session_id)
        if existing is None:
            cols = ["session_id"] + list(fields.keys())
            placeholders = ", ".join("?" for _ in cols)
            self._conn.execute(
                f"INSERT INTO sessions ({', '.join(cols)}) VALUES ({placeholders})",
                [session_id, *fields.values()],
            )
        else:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            self._conn.execute(
                f"UPDATE sessions SET {assignments} WHERE session_id = ?",
                [*fields.values(), session_id],
            )
        self._conn.commit()

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def is_done(self, session_id: str) -> bool:
        rec = self.get(session_id)
        return bool(rec and rec.get("status") == "done")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# rclone wrappers (no shell, no credential interpolation)
# ---------------------------------------------------------------------------

def _rclone_base(cfg: DriveIngestConfig) -> list[str]:
    cmd = ["rclone"]
    if cfg.rclone_config:
        cmd += ["--config", cfg.rclone_config]
    return cmd


def rclone_available(cfg: Optional[DriveIngestConfig] = None) -> bool:
    """True if the rclone binary is callable."""
    try:
        proc = subprocess.run(
            ["rclone", "version"], capture_output=True, text=True, timeout=15
        )
        return proc.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def list_sessions(cfg: DriveIngestConfig) -> list[dict[str, Any]]:
    """Enumerate session folders under the remote base path (flat layout).

    Returns a list of ``{"session_id": <folder name>, "remote_path": <full>}``.
    """
    cmd = _rclone_base(cfg) + ["lsjson", cfg.remote, "--dirs-only"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed: {proc.stderr.strip()[:500]}")

    entries = json.loads(proc.stdout or "[]")
    sessions = []
    for e in entries:
        if not e.get("IsDir"):
            continue
        name = e.get("Name") or e.get("Path")
        if not name:
            continue
        sessions.append(
            {"session_id": name, "remote_path": f"{cfg.remote}/{name}"}
        )
    sessions.sort(key=lambda s: s["session_id"])
    return sessions


def count_sets(cfg: DriveIngestConfig, remote_path: str) -> int:
    """Count video files (sets) directly inside a session folder."""
    cmd = _rclone_base(cfg) + ["lsjson", remote_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return 0
    entries = json.loads(proc.stdout or "[]")
    return sum(
        1
        for e in entries
        if not e.get("IsDir")
        and Path(e.get("Name", "")).suffix.lower() in VIDEO_EXTS
    )


def copy_down(cfg: DriveIngestConfig, remote_path: str, local_dir: Path) -> int:
    """Copy a session's video files from the remote to local NVMe."""
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = _rclone_base(cfg) + [
        "copy", remote_path, str(local_dir),
        "--checksum",
        "--transfers", str(cfg.copy_transfers),
    ]
    for ext in VIDEO_EXTS:
        cmd += ["--include", f"*{ext}", "--include", f"*{ext.upper()}"]
    logger.info("rclone copy down: %s -> %s", remote_path, local_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.copy_timeout)
    if proc.returncode != 0:
        logger.error("rclone copy down failed: %s", proc.stderr.strip()[-500:])
    return proc.returncode


def copy_up(cfg: DriveIngestConfig, local_dir: Path, remote_dest: str) -> int:
    """Push local outputs to the remote destination sub-folder."""
    cmd = _rclone_base(cfg) + [
        "copy", str(local_dir), remote_dest,
        "--checksum",
        "--transfers", str(cfg.copy_transfers),
    ]
    logger.info("rclone copy up: %s -> %s", local_dir, remote_dest)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.copy_timeout)
    if proc.returncode != 0:
        logger.error("rclone copy up failed: %s", proc.stderr.strip()[-500:])
    return proc.returncode


def verify_upload(cfg: DriveIngestConfig, local_dir: Path, remote_dest: str) -> bool:
    """Verify every local output exists on the remote (one-way check)."""
    cmd = _rclone_base(cfg) + [
        "check", str(local_dir), remote_dest, "--one-way", "--checksum",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.copy_timeout)
    ok = proc.returncode == 0
    if not ok:
        logger.error("rclone check failed (upload not verified): %s",
                     (proc.stderr or proc.stdout).strip()[-500:])
    return ok


# ---------------------------------------------------------------------------
# Pooled multi-set frame extraction
# ---------------------------------------------------------------------------

def _list_local_videos(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def _extract_pooled_frames(videos: list[Path], frames_dir: Path, fps: float) -> int:
    """Extract frames from every set into one pooled dir with per-set prefixes.

    Per-set prefixes (``set00_``, ``set01_``…) keep filenames unique so the
    sets share a single frame directory — the basis for one combined COLMAP
    reconstruction. Returns the total pooled frame count.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, video in enumerate(videos):
        prefix = f"set{i:02d}_"
        pattern = str(frames_dir / f"{prefix}frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            pattern,
        ]
        logger.info("Extracting set %d/%d: %s (fps=%.2f)", i + 1, len(videos), video.name, fps)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except (FileNotFoundError, OSError) as exc:
            logger.error("ffmpeg could not launch for %s: %s", video.name, exc)
            continue
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out on %s", video.name)
            continue
        if proc.returncode != 0:
            logger.error("ffmpeg failed on %s: %s", video.name, proc.stderr[-400:])
            continue
        n = len(list(frames_dir.glob(f"{prefix}*.jpg")))
        logger.info("  set %d → %d frames", i, n)
        total += n
    return total


# ---------------------------------------------------------------------------
# Config bridge → PipelineConfig
# ---------------------------------------------------------------------------

def build_room_config(cfg: DriveIngestConfig) -> "Any":
    """Build a PipelineConfig tuned for combined room-orbit reconstruction.

    Mirrors the §5.4 recommended starting config from the ingestion plan.
    Imported lazily so this module loads without the pipeline runtime.
    """
    from pipeline.config import PipelineConfig

    pcfg = PipelineConfig()
    pcfg.ingest.fps = cfg.fps
    pcfg.ingest.target_frames = cfg.target_frames
    pcfg.ingest.max_image_size = cfg.max_image_size
    pcfg.ingest.use_fibonacci_coverage = cfg.use_fibonacci_coverage
    pcfg.ingest.coverage_weight = cfg.coverage_weight

    # Combined reconstruction across pooled sets: vocab-tree spatial matching,
    # one shared camera model (same lens), undistort for training.
    pcfg.reconstruct.matcher = "vocab_tree"
    pcfg.reconstruct.single_camera = True

    pcfg.training.scene_preset = cfg.scene_preset
    pcfg.training.mesh_method = cfg.mesh_method
    if cfg.mesh_method == "auto":
        pcfg.training.mesh_backend_auto = True

    pcfg.delivery.enable_splat_optimize = cfg.enable_splat_optimize
    return pcfg


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------

def _collect_outputs(job_dir: Path, artifacts: dict[str, str], metrics: dict[str, Any]) -> Path:
    """Stage the curated output set into ``job_dir/_upload`` for upload.

    Uploads geometry + delivery + previews + a manifest — never the raw frames,
    COLMAP database, or undistorted images (large and reproducible).
    """
    upload_dir = job_dir / "_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    def _stage(src: Optional[str], rel: str) -> None:
        if not src:
            return
        sp = Path(src)
        if not sp.exists():
            return
        dst = upload_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if sp.is_dir():
            shutil.copytree(sp, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(sp, dst)

    # Trained splat + scene mesh + delivery splat from the train stage artifacts.
    _stage(artifacts.get("ply_path"), f"splat/{Path(artifacts.get('ply_path', 'model.ply')).name}")
    _stage(artifacts.get("scene_mesh_glb"), "mesh/full_scene.glb")
    for k in ("milo_mesh_path", "come_mesh_path", "gaussianwrapping_mesh_path"):
        _stage(artifacts.get(k), f"mesh/{Path(artifacts[k]).name}" if artifacts.get(k) else "")
    _stage(artifacts.get("delivery_splat"), f"delivery/{Path(artifacts.get('delivery_splat', 'scene.ksplat')).name}")

    # Whole sub-trees when present.
    for sub in ("previews", "delivery", "objects", "usd"):
        d = job_dir / sub
        if d.exists():
            _stage(str(d), sub)

    # USD scene files at the job root.
    for usd in list(job_dir.glob("*.usd*")):
        _stage(str(usd), f"usd/{usd.name}")

    # Manifest of what ran.
    manifest = {
        "generated_at": time.time(),
        "artifacts": artifacts,
        "metrics": metrics,
    }
    (upload_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return upload_dir


def _purge(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Purged %s", path)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not purge %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Per-session processing (pull → process → push → purge)
# ---------------------------------------------------------------------------

def process_session(
    cfg: DriveIngestConfig,
    session: dict[str, Any],
    ledger: Ledger,
) -> dict[str, Any]:
    """Run the full pull→process→push→purge cycle for one session.

    Never raises for an expected failure — records ``failed`` in the ledger,
    leaves the raw in place, and returns a result dict. Reconstruct + train are
    hard-required; decomposition/preview stages are best-effort.
    """
    sid = session["session_id"]
    remote_path = session["remote_path"]
    result: dict[str, Any] = {"session_id": sid, "success": False, "error": None}

    local_raw = Path(cfg.local_raw_dir) / sid
    job_dir = Path(cfg.local_scratch_dir) / sid
    ledger.upsert(sid, remote_path=remote_path, status="processing", started_at=time.time())

    try:
        from pipeline.stages import PipelineStages
    except Exception as exc:  # pragma: no cover - runtime dep guard
        msg = f"pipeline runtime unavailable: {exc}"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result

    # -- 1. Pull -----------------------------------------------------------
    if copy_down(cfg, remote_path, local_raw) != 0:
        msg = "download failed"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result

    videos = _list_local_videos(local_raw)
    if not videos:
        msg = "no video sets found in session folder"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result
    ledger.upsert(sid, n_sets=len(videos))

    # -- 2. Pooled frame extraction ---------------------------------------
    frames_dir = job_dir / "frames"
    pooled = _extract_pooled_frames(videos, frames_dir, cfg.fps)
    pcfg = build_room_config(cfg)
    if pooled < pcfg.ingest.min_frames:
        msg = f"only {pooled} pooled frames (need {pcfg.ingest.min_frames})"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result

    # -- 3. Run the pipeline stages on the pooled frames -------------------
    stages = PipelineStages(str(job_dir), config=pcfg)
    artifacts: dict[str, str] = {}
    metrics: dict[str, Any] = {"sets": len(videos), "pooled_frames": pooled}
    cur = str(frames_dir)

    # remove_people (best-effort; auto-skips if disabled in config)
    r = stages.remove_people(cur)
    if r.success:
        cur = r.artifacts.get("cleaned_frames_dir", cur)
        artifacts.update(r.artifacts)

    # select_frames (core)
    r = stages.select_frames(cur, target=cfg.target_frames)
    if not r.success:
        msg = f"select_frames failed: {r.error}"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result
    cur = r.artifacts.get("selected_frames_dir", cur)
    metrics["selected_frames"] = r.metrics.get("selected")

    # reconstruct (core — combined COLMAP)
    r = stages.reconstruct(cur)
    if not r.success:
        msg = f"reconstruct failed: {r.error}"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result
    colmap_dir = r.artifacts["colmap_dir"]
    metrics["colmap"] = r.metrics

    # train (core — produces splat + scene mesh for milo/come/gw backends)
    r = stages.train(colmap_dir, iterations=pcfg.training.iterations)
    if not r.success:
        msg = f"train failed: {r.error}"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result
    artifacts.update(r.artifacts)
    metrics["train"] = r.metrics
    ply_path = r.artifacts.get("ply_path")

    # render_previews (best-effort)
    if ply_path:
        try:
            pr = stages.render_previews(ply_path, colmap_dir)
            artifacts.update(pr.artifacts)
        except Exception as exc:
            logger.warning("render_previews failed (non-fatal): %s", exc)

    # decomposition → USD assembly (best-effort; geometry already exists)
    if cfg.run_decomposition and ply_path:
        _run_decomposition(stages, ply_path, cur, artifacts, metrics)

    # -- 4. Collect + push outputs ----------------------------------------
    upload_dir = _collect_outputs(job_dir, artifacts, metrics)
    remote_dest = f"{remote_path}/{cfg.output_subfolder}"
    ledger.upsert(sid, status="uploading", output_remote=remote_dest)

    if copy_up(cfg, upload_dir, remote_dest) != 0:
        msg = "upload failed (raw + scratch kept for retry)"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result

    if not verify_upload(cfg, upload_dir, remote_dest):
        msg = "upload verification failed (raw + scratch kept for retry)"
        ledger.upsert(sid, status="failed", error=msg, finished_at=time.time())
        result["error"] = msg
        return result

    # -- 5. Purge local (delete-local-only; raw safe on remote) -----------
    if not cfg.keep_local_on_success:
        _purge(local_raw)
        # Keep the small per-session log + manifest; drop heavy scratch.
        for heavy in ("frames", "frames_cleaned", "frames_selected", "colmap"):
            _purge(job_dir / heavy)

    ledger.upsert(sid, status="done", finished_at=time.time(), error=None)
    result["success"] = True
    result["output_remote"] = remote_dest
    result["metrics"] = metrics
    logger.info("Session %s done → %s", sid, remote_dest)
    return result


def _run_decomposition(
    stages: "Any",
    ply_path: str,
    frames_dir: str,
    artifacts: dict[str, str],
    metrics: dict[str, Any],
) -> None:
    """Best-effort segment→objects→USD assembly. Failures are recorded, not raised."""
    try:
        seg = stages.segment(ply_path, frames_dir)
        if not seg.success:
            metrics["decomposition"] = f"segment failed: {seg.error}"
            return
        objects_json = seg.artifacts.get("objects", "[]")

        eo = stages.extract_objects(ply_path, labels=objects_json)
        object_plys = eo.artifacts.get("object_plys", "[]") if eo.success else "[]"

        mo = stages.mesh_objects(object_plys)
        meshes_json = mo.artifacts.get("meshes", "[]") if mo.success else "[]"

        tb = stages.texture_bake(meshes_json)
        meshes_json = tb.artifacts.get("textured_meshes", meshes_json)

        au = stages.assemble_usd(meshes_json)
        artifacts.update(au.artifacts)
        metrics["decomposition"] = "ok" if au.success else f"assemble_usd: {au.error}"
    except Exception as exc:
        logger.warning("Decomposition stage failed (non-fatal): %s", exc)
        metrics["decomposition"] = f"exception: {exc}"


# ---------------------------------------------------------------------------
# Batch driver (one-time)
# ---------------------------------------------------------------------------

def run_batch(cfg: DriveIngestConfig, dry_run: bool = False) -> dict[str, Any]:
    """Process every discovered session once, then return a summary.

    Skips sessions already marked ``done`` in the ledger (resume-safe).
    """
    if not cfg.remote:
        raise ValueError("DriveIngestConfig.remote is required (e.g. 'gcs:bucket/captures')")
    if not rclone_available(cfg):
        raise RuntimeError("rclone not found on PATH; install it in the container (Phase 1 plumbing)")

    sessions = list_sessions(cfg)
    ledger = Ledger(cfg.resolved_ledger_path())
    summary: dict[str, Any] = {
        "remote": cfg.remote,
        "discovered": len(sessions),
        "processed": [],
        "skipped": [],
        "failed": [],
    }

    logger.info("Discovered %d session(s) under %s", len(sessions), cfg.remote)

    for s in sessions:
        sid = s["session_id"]
        if ledger.is_done(sid):
            logger.info("Skip %s (already done)", sid)
            summary["skipped"].append(sid)
            continue

        n = count_sets(cfg, s["remote_path"])
        if dry_run:
            logger.info("[dry-run] would process %s (%d sets) → %s/%s",
                        sid, n, s["remote_path"], cfg.output_subfolder)
            ledger.upsert(sid, remote_path=s["remote_path"], status="pending", n_sets=n)
            summary["processed"].append({"session_id": sid, "n_sets": n, "dry_run": True})
            continue

        res = process_session(cfg, s, ledger)
        if res["success"]:
            summary["processed"].append({"session_id": sid, "output": res.get("output_remote")})
        else:
            summary["failed"].append({"session_id": sid, "error": res.get("error")})

    ledger.close()
    logger.info(
        "Batch complete: %d processed, %d skipped, %d failed",
        len(summary["processed"]), len(summary["skipped"]), len(summary["failed"]),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="drive_ingestor",
        description="Bulk capture ingestor (pull→process→push→purge) over a cloud remote",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Process all sessions under the remote base path")
    run.add_argument("--remote", required=True,
                     help="rclone remote + base path, e.g. gcs:bucket/captures or gdrive:Captures")
    run.add_argument("--rclone-config", help="Path to rclone config holding the service-account remote")
    run.add_argument("--scratch", default="/data/scratch", help="Local NVMe scratch dir")
    run.add_argument("--raw", default="/data/raw", help="Local NVMe raw-cache dir")
    run.add_argument("--output-subfolder", default="outputs")
    run.add_argument("--ledger", help="SQLite ledger path (persistent volume recommended)")
    run.add_argument("--fps", type=float, default=2.0)
    run.add_argument("--target-frames", type=int, default=600)
    run.add_argument("--mesh-method", default="auto",
                     choices=["auto", "tsdf", "milo", "come", "gaussianwrapping"])
    run.add_argument("--scene-preset", default="indoor_reflective",
                     choices=["default", "indoor_reflective"])
    run.add_argument("--no-decomposition", action="store_true",
                     help="Stop after geometry (train + previews); skip USD assembly")
    run.add_argument("--no-splat-optimize", action="store_true")
    run.add_argument("--keep-local", action="store_true", help="Do not purge local on success")
    run.add_argument("--dry-run", action="store_true", help="List what would be processed; no copy/compute")
    run.add_argument("--verbose", "-v", action="store_true")

    ls = sub.add_parser("list", help="List discovered sessions and their set counts")
    ls.add_argument("--remote", required=True)
    ls.add_argument("--rclone-config")
    ls.add_argument("--verbose", "-v", action="store_true")

    return p


def _cfg_from_args(args) -> DriveIngestConfig:
    return DriveIngestConfig(
        remote=args.remote,
        rclone_config=getattr(args, "rclone_config", None),
        local_raw_dir=getattr(args, "raw", "/data/raw"),
        local_scratch_dir=getattr(args, "scratch", "/data/scratch"),
        output_subfolder=getattr(args, "output_subfolder", "outputs"),
        ledger_path=getattr(args, "ledger", None),
        fps=getattr(args, "fps", 2.0),
        target_frames=getattr(args, "target_frames", 600),
        mesh_method=getattr(args, "mesh_method", "auto"),
        scene_preset=getattr(args, "scene_preset", "indoor_reflective"),
        enable_splat_optimize=not getattr(args, "no_splat_optimize", False),
        run_decomposition=not getattr(args, "no_decomposition", False),
        keep_local_on_success=getattr(args, "keep_local", False),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "list":
        cfg = DriveIngestConfig(remote=args.remote, rclone_config=args.rclone_config)
        if not rclone_available(cfg):
            print("rclone not found on PATH", flush=True)
            return 1
        for s in list_sessions(cfg):
            n = count_sets(cfg, s["remote_path"])
            print(f"{s['session_id']}\t{n} set(s)\t{s['remote_path']}")
        return 0

    if args.command == "run":
        cfg = _cfg_from_args(args)
        try:
            summary = run_batch(cfg, dry_run=args.dry_run)
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", flush=True)
            return 1
        print(json.dumps(summary, indent=2, default=str))
        return 0 if not summary["failed"] else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
