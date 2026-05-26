# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Flask web application for the LichtFeld video-to-scene pipeline.

Provides upload, monitoring, log streaming (SSE), preview, and download
endpoints. Runs on port 7860 by default.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

# Ensure src/ is on the path so pipeline imports resolve
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from web.job_manager import (
    INPUT_DIR,
    JobState,
    cleanup_old_jobs,
    create_job,
    delete_job,
    get_job,
    list_jobs,
    update_job,
)
from web.pipeline_runner import (
    cancel_pipeline,
    complete_job,
    complete_stage,
    is_running,
    start_pipeline,
    update_stage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key storage
# ---------------------------------------------------------------------------

API_KEY_PATH = Path(os.environ.get("LFS_API_KEY_PATH", "/data/.anthropic_key"))


def _read_api_key() -> str | None:
    """Read stored Anthropic API key from persistent volume. Returns None if absent."""
    if API_KEY_PATH.exists():
        key = API_KEY_PATH.read_text().strip()
        return key if key else None
    return None


def _save_api_key(key: str) -> None:
    """Save Anthropic API key to persistent volume with restricted permissions."""
    API_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_PATH.write_text(key.strip())
    try:
        API_KEY_PATH.chmod(0o600)
    except OSError:
        pass  # may not own the file in all container setups


def _redact_key(key: str) -> str:
    """Return a redacted version of the key for display: sk-ant-...last4."""
    if not key or len(key) < 12:
        return "****"
    return key[:7] + "..." + key[-4:]


def validate_api_key(key: str) -> tuple[bool, str]:
    """Test an Anthropic API key by running a minimal Claude Code invocation.

    Returns (success, message).
    """
    if not key or not key.strip():
        return False, "Key is empty"

    env = {**os.environ, "ANTHROPIC_API_KEY": key.strip(), "HOME": "/home/ubuntu"}
    try:
        result = subprocess.run(
            ["claude", "-p", "respond with OK", "--max-turns", "1"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "OK" in result.stdout:
            return True, "Key is valid — Claude Code responded successfully"
        # Check for common error patterns
        combined = result.stdout + result.stderr
        if "Invalid API key" in combined or "invalid_api_key" in combined:
            return False, "Invalid API key"
        if "permission" in combined.lower():
            return False, "API key lacks required permissions"
        return False, f"Unexpected response: {combined[:200]}"
    except FileNotFoundError:
        return False, "Claude Code binary not found — is it installed?"
    except subprocess.TimeoutExpired:
        return False, "Validation timed out (30s)"
    except Exception as exc:
        return False, f"Validation error: {exc}"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# 2 GB upload limit
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Serve the single-page upload/monitoring UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload() -> tuple[Response, int]:
    """Accept a video file upload, create a job, and start the pipeline."""
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file = request.files["video"]
    if file.filename is None or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    filename = secure_filename(file.filename)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use timestamp prefix to avoid collisions
    ts = int(time.time() * 1000)
    save_name = f"{ts}_{filename}"
    save_path = INPUT_DIR / save_name

    try:
        file.save(str(save_path))
    except Exception as exc:
        logger.error("File save failed: %s", exc)
        return jsonify({"error": "Failed to save file"}), 500

    file_size = save_path.stat().st_size
    job = create_job(
        filename=filename,
        input_video_path=str(save_path),
        file_size_bytes=file_size,
    )

    started = start_pipeline(job.job_id)
    if not started:
        return jsonify({"error": "Failed to start pipeline"}), 500

    return jsonify({
        "job_id": job.job_id,
        "filename": filename,
        "file_size_bytes": file_size,
        "state": job.state,
    }), 201


# ---------------------------------------------------------------------------
# Bulk capture ingestion from a cloud remote (Google Cloud / Drive)
#
# Replaces drag-and-drop as the primary path for the ~70 GB room-orbit
# captures. Credentials come from a mounted rclone config (service account),
# never an env-var token. See src/pipeline/drive_ingestor.py and
# research/decisions/video-ingestion-plan.md.
# ---------------------------------------------------------------------------

DRIVE_INGEST_REMOTE = os.environ.get("DRIVE_INGEST_REMOTE", "")
RCLONE_CONFIG = os.environ.get("RCLONE_CONFIG") or None


@app.route("/api/ingest/drive/sessions")
def drive_sessions() -> tuple[Response, int]:
    """List capture sessions discovered under the configured remote.

    Query param ``remote`` overrides ``DRIVE_INGEST_REMOTE``.
    """
    remote = request.args.get("remote") or DRIVE_INGEST_REMOTE
    if not remote:
        return jsonify({"error": "No remote configured (set DRIVE_INGEST_REMOTE)"}), 400

    try:
        from pipeline.drive_ingestor import (
            DriveIngestConfig,
            list_sessions,
            count_sets,
            rclone_available,
        )
    except Exception as exc:  # pragma: no cover - runtime dep guard
        return jsonify({"error": f"ingestor unavailable: {exc}"}), 500

    cfg = DriveIngestConfig(remote=remote, rclone_config=RCLONE_CONFIG)
    if not rclone_available(cfg):
        return jsonify({"error": "rclone not installed in container"}), 503

    try:
        sessions = list_sessions(cfg)
        for s in sessions:
            s["n_sets"] = count_sets(cfg, s["remote_path"])
    except Exception as exc:
        return jsonify({"error": f"discovery failed: {exc}"}), 502

    return jsonify({"remote": remote, "sessions": sessions}), 200


@app.route("/api/ingest/drive", methods=["POST"])
def drive_ingest() -> tuple[Response, int]:
    """Launch the one-time bulk ingestion batch as a detached background run.

    Form/JSON params (all optional):
        remote: rclone remote+base path (defaults to DRIVE_INGEST_REMOTE)
        dry_run: "true" to list-only without copy/compute
        mesh_method, scene_preset, target_frames, fps: pipeline overrides

    Returns the log path; progress is written there and to the ledger. Long
    runs are intentionally detached (one-time batch over many sessions).
    """
    data = request.get_json(silent=True) or request.form
    remote = data.get("remote") or DRIVE_INGEST_REMOTE
    if not remote:
        return jsonify({"error": "No remote configured (set DRIVE_INGEST_REMOTE)"}), 400

    dry_run = str(data.get("dry_run", "")).lower() in ("1", "true", "yes", "on")

    cmd = [
        sys.executable, "-m", "pipeline.drive_ingestor", "run",
        "--remote", remote,
        "--scratch", os.environ.get("DRIVE_SCRATCH_DIR", "/data/scratch"),
        "--raw", os.environ.get("DRIVE_RAW_DIR", "/data/raw"),
        "--ledger", "/data/output/drive_ingest_ledger.sqlite",
    ]
    if RCLONE_CONFIG:
        cmd += ["--rclone-config", RCLONE_CONFIG]
    if dry_run:
        cmd.append("--dry-run")
    for key, flag in (
        ("mesh_method", "--mesh-method"),
        ("scene_preset", "--scene-preset"),
    ):
        val = data.get(key)
        if val:
            cmd += [flag, str(val)]
    for key, flag in (("target_frames", "--target-frames"), ("fps", "--fps")):
        val = data.get(key)
        if val:
            cmd += [flag, str(val)]

    log_dir = Path(os.environ.get("LFS_OUTPUT_DIR", "/data/output"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"drive_ingest_{int(time.time())}.log"

    env = {
        **os.environ,
        "PYTHONPATH": _src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    try:
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=_src_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        logger.error("Failed to launch drive ingestor: %s", exc)
        return jsonify({"error": f"launch failed: {exc}"}), 500

    logger.info("Launched drive ingestor (pid=%d) remote=%s dry_run=%s log=%s",
                proc.pid, remote, dry_run, log_path)
    return jsonify({
        "started": True,
        "pid": proc.pid,
        "remote": remote,
        "dry_run": dry_run,
        "log": str(log_path),
    }), 202


@app.route("/status/<job_id>")
def status(job_id: str) -> tuple[Response, int]:
    """Return full job status as JSON."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job.to_dict()), 200


@app.route("/stream/<job_id>")
def stream(job_id: str) -> Response:
    """Server-Sent Events endpoint for real-time log and status updates."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    def generate():
        last_log_index = 0
        last_state = ""
        last_progress = -1.0

        while True:
            job = get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break

            # Send state changes
            if job.state != last_state:
                last_state = job.state
                yield f"data: {json.dumps({'type': 'state', 'state': job.state, 'current_stage': job.current_stage})}\n\n"

            # Send progress changes
            if job.progress != last_progress:
                last_progress = job.progress
                yield f"data: {json.dumps({'type': 'progress', 'progress': job.progress})}\n\n"

            # Send new log lines
            if len(job.logs) > last_log_index:
                new_lines = job.logs[last_log_index:]
                last_log_index = len(job.logs)
                for line in new_lines:
                    yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            # Send preview updates
            if job.previews:
                yield f"data: {json.dumps({'type': 'previews', 'previews': job.previews})}\n\n"

            # Terminal states
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                yield f"data: {json.dumps({'type': 'done', 'state': job.state, 'error': job.error, 'result_archive': job.result_archive})}\n\n"
                break

            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/download/<job_id>")
def download(job_id: str) -> Response:
    """Download a zip archive of the job output."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    output_dir = Path(job.output_dir) if hasattr(job, "output_dir") and job.output_dir else None

    # Fall back to pre-built archive if output_dir is unavailable
    if output_dir is None or not output_dir.exists():
        if job.result_archive and Path(job.result_archive).exists():
            return send_file(
                job.result_archive,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{job.filename.rsplit('.', 1)[0]}_result.zip",
            )
        abort(404, description="Output directory not found")

    # Create zip on the fly - include meshes, USD, previews, PLY (skip frames)
    import io
    import zipfile

    zip_buffer = io.BytesIO()
    base_name = job.filename.rsplit(".", 1)[0] if job.filename else job.job_id

    include_dirs = {"objects", "usd", "previews", "model"}
    include_extensions = {
        ".glb", ".obj", ".ply", ".usda", ".usdc", ".usdz",
        ".jpg", ".png", ".json", ".mtl",
    }

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir)
            # Skip frames directory (too large), skip input
            top_dir = rel.parts[0] if rel.parts else ""
            if top_dir in ("frames", "frames_selected", "input", "colmap"):
                continue
            if path.suffix.lower() in include_extensions or top_dir in include_dirs:
                zf.write(path, arcname=str(rel))

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{base_name}_scene.zip",
    )


@app.route("/viewer/<job_id>")
def viewer(job_id: str) -> str:
    """Serve the 3D viewer page for a completed job."""
    job = get_job(job_id)
    if job is None:
        abort(404)
    return render_template("viewer.html", job_id=job_id, filename=job.filename or job_id)


@app.route("/mesh/<job_id>")
def serve_mesh(job_id: str) -> Response:
    """Serve the GLB mesh for the 3D viewer."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    output_dir = Path(job.output_dir) if hasattr(job, "output_dir") and job.output_dir else None
    if output_dir is None or not output_dir.exists():
        abort(404, description="Output directory not found")

    # Prefer Blender-assembled USD GLB (has materials), then MILo, then raw TSDF GLB
    glb_candidates = [
        output_dir / "usd" / "scene.glb",  # Blender export
        output_dir / "model_milo" / "mesh_learnable_sdf.glb",  # MILo mesh
    ]
    # Then look in standard locations
    glb_candidates += sorted(output_dir.glob("objects/meshes/**/*.glb"))
    glb_candidates += sorted(output_dir.glob("**/*.glb"))

    glb_file = None
    for candidate in glb_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            glb_file = candidate
            break

    if glb_file is None:
        abort(404, description="No GLB mesh found")

    return send_file(str(glb_file), mimetype="model/gltf-binary")


@app.route("/preview/<job_id>/<stage>")
def preview(job_id: str, stage: str) -> Response:
    """Serve a Blender render / preview image for a specific pipeline stage."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    preview_path = job.previews.get(stage)
    if not preview_path or not Path(preview_path).exists():
        abort(404, description=f"No preview for stage {stage}")

    # Determine mimetype from extension
    ext = Path(preview_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".exr": "image/x-exr"}
    mime = mime_map.get(ext, "application/octet-stream")

    return send_file(preview_path, mimetype=mime)


@app.route("/jobs")
def jobs() -> tuple[Response, int]:
    """List all jobs with summary info."""
    return jsonify(list_jobs()), 200


@app.route("/job/<job_id>", methods=["DELETE"])
def remove_job(job_id: str) -> tuple[Response, int]:
    """Cancel a running job or delete a finished one."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    # Cancel if running
    if job.state in (JobState.RUNNING, JobState.QUEUED) or job.state.startswith("stage_"):
        cancel_pipeline(job_id)
        update_job(
            job_id,
            state=JobState.CANCELLED,
            finished_at=time.time(),
            error="Cancelled by user",
        )

    deleted = delete_job(job_id)
    if deleted:
        return jsonify({"status": "deleted", "job_id": job_id}), 200
    return jsonify({"error": "Delete failed"}), 500


@app.route("/health")
def health() -> tuple[Response, int]:
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": time.time()}), 200


# ---------------------------------------------------------------------------
# API key / configuration
# ---------------------------------------------------------------------------


@app.route("/setup")
def setup() -> str:
    """Dedicated setup page — redirects to index if key already configured."""
    key = _read_api_key()
    if key:
        return redirect(url_for("index"))
    return render_template("index.html")


def _check_oauth_session() -> bool:
    """Check if Claude Code has an active OAuth session (subscription auth).

    Looks for .credentials.json in the ubuntu user's .claude directory.
    This is fast (file check only, no subprocess).
    """
    for creds_path in [
        Path("/home/ubuntu/.claude/.credentials.json"),
        Path.home() / ".claude" / ".credentials.json",
    ]:
        if creds_path.exists() and creds_path.stat().st_size > 10:
            return True
    return False


@app.route("/api/config", methods=["GET"])
def get_config() -> tuple[Response, int]:
    """Return current configuration (API key or OAuth session)."""
    key = _read_api_key()
    oauth = _check_oauth_session() if not key else False
    return jsonify({
        "api_key_configured": key is not None,
        "api_key_redacted": _redact_key(key) if key else None,
        "oauth_session_active": oauth,
        "claude_ready": (key is not None) or oauth,
    }), 200


@app.route("/api/config", methods=["POST"])
def save_config() -> tuple[Response, int]:
    """Save configuration. Body: {"api_key": "sk-ant-..."}"""
    data = request.get_json(silent=True) or {}
    key = data.get("api_key", "").strip()

    if not key:
        return jsonify({"error": "api_key is required"}), 400

    # Basic format check
    if not key.startswith("sk-"):
        return jsonify({"error": "Key should start with sk-"}), 400

    _save_api_key(key)
    logger.info("API key saved (redacted: %s)", _redact_key(key))

    return jsonify({
        "status": "saved",
        "api_key_redacted": _redact_key(key),
    }), 200


@app.route("/api/config/test", methods=["POST"])
def test_config() -> tuple[Response, int]:
    """Test the currently stored API key (or one provided in the body).

    Body (optional): {"api_key": "sk-ant-..."} — if omitted, tests stored key.
    """
    data = request.get_json(silent=True) or {}
    key = data.get("api_key", "").strip() or _read_api_key()

    if not key:
        return jsonify({"valid": False, "message": "No API key configured"}), 200

    valid, message = validate_api_key(key)
    return jsonify({"valid": valid, "message": message}), 200


# ---------------------------------------------------------------------------
# Claude Code orchestration API
# ---------------------------------------------------------------------------


@app.route("/api/job/<job_id>/stage", methods=["POST"])
def api_update_stage(job_id: str) -> tuple[Response, int]:
    """Called by Claude Code to report stage progress.

    Body: {"stage": "train", "progress": 0.5, "message": "30k iter, loss 0.02"}
    """
    data = request.get_json(silent=True) or {}
    stage = data.get("stage", "")
    progress = float(data.get("progress", 0.0))
    message = data.get("message", "")

    if not stage:
        return jsonify({"error": "stage is required"}), 400

    ok = update_stage(job_id, stage=stage, progress=progress, message=message)
    if not ok:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": "updated", "job_id": job_id, "stage": stage}), 200


@app.route("/api/job/<job_id>/stage/complete", methods=["POST"])
def api_complete_stage(job_id: str) -> tuple[Response, int]:
    """Mark a stage as completed or failed.

    Body: {"stage": "train", "success": true, "error": ""}
    """
    data = request.get_json(silent=True) or {}
    stage = data.get("stage", "")
    success = data.get("success", True)
    error = data.get("error", "")

    if not stage:
        return jsonify({"error": "stage is required"}), 400

    ok = complete_stage(job_id, stage=stage, success=success, error=error)
    if not ok:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": "stage_completed" if success else "stage_failed", "stage": stage}), 200


@app.route("/api/job/<job_id>/complete", methods=["POST"])
def api_complete_job(job_id: str) -> tuple[Response, int]:
    """Mark the entire job as completed or failed. Creates download archive.

    Body: {"success": true} or {"success": false, "error": "reason"}
    """
    data = request.get_json(silent=True) or {}
    success = data.get("success", True)
    error = data.get("error", "")

    ok = complete_job(job_id, success=success, error=error)
    if not ok:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": "completed" if success else "failed", "job_id": job_id}), 200


# ---------------------------------------------------------------------------
# Preview image carousel endpoints
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(os.environ.get("LFS_OUTPUT_DIR", "/data/output"))


@app.route("/api/job/<job_id>/previews")
def get_previews(job_id: str) -> tuple[Response, int]:
    """Return list of preview images found in the job output directory."""
    safe_id = secure_filename(job_id)
    job_dir = OUTPUT_DIR / safe_id
    if not job_dir.is_dir():
        return jsonify([]), 200

    previews: list[dict] = []
    for pattern in ("**/*.png", "**/*.jpg", "**/*.jpeg"):
        for img in job_dir.glob(pattern):
            try:
                rel = img.relative_to(job_dir)
            except ValueError:
                continue
            stage = rel.parts[0] if len(rel.parts) > 1 else "output"
            try:
                size_kb = img.stat().st_size // 1024
            except OSError:
                size_kb = 0
            previews.append({
                "path": str(rel),
                "url": f"/api/job/{safe_id}/file/{rel}",
                "stage": stage,
                "size_kb": size_kb,
            })

    previews.sort(key=lambda p: p["path"])
    return jsonify(previews), 200


@app.route("/api/job/<job_id>/file/<path:filepath>")
def serve_job_file(job_id: str, filepath: str) -> Response:
    """Serve a file from a job's output directory."""
    safe_id = secure_filename(job_id)
    job_dir = OUTPUT_DIR / safe_id

    # Prevent directory traversal
    resolved = (job_dir / filepath).resolve()
    if not str(resolved).startswith(str(job_dir.resolve())):
        abort(403)

    if not resolved.is_file():
        abort(404)

    return send_from_directory(str(job_dir), filepath)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 2 GB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Application factory for WSGI servers."""
    # Run cleanup of old jobs on startup
    try:
        removed = cleanup_old_jobs()
        if removed:
            logger.info("Cleaned up %d old jobs", removed)
    except Exception as exc:
        logger.warning("Job cleanup failed: %s", exc)

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("LFS_WEB_PORT", "7860"))
    application.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("LFS_DEBUG", "").lower() in ("1", "true"),
        threaded=True,
    )
