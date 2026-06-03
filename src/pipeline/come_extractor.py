# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CoMe mesh extraction -- confidence-based marching-tetrahedra from 3D Gaussians.

Runs CoMe (github.com/r4dl/CoMe, arXiv:2603.24725, code released 2026-04-22) in a
sidecar Docker container (Ubuntu 22.04, Python 3.10, CUDA 12.1). Produces
geometry-only PLY meshes via confidence-weighted Gaussian training followed by
marching tetrahedra (unbounded) or TSDF (bounded) extraction.

CoMe cannot run in our main container (Python 3.12, CUDA 12.8, Ubuntu 24.04)
because its SOF-derived CUDA extensions require Python 3.10 and CUDA 12.1.  It
also cannot share the MILo sidecar (CUDA 11.8 -- incompatible toolkit level).
Instead it runs in a dedicated ``come`` sidecar container via ``docker exec``.

Falls back to a conda environment (COME_CONDA_ENV, default "come") if the
sidecar is not available.

LICENSING NOTE (ADR-004):
    Verified 2026-05-26 against the live repo: CoMe ships ``LICENSE.md`` — the
    **Inria/MPII "Gaussian-Splatting License"** (plus ``NOTICE.md`` covering
    SOF and StopThePop).  This is a **non-commercial research** licence, NOT a
    permissive one: it permits research/evaluation use only and prohibits
    commercial use/distribution without a separate agreement with Inria.
    ``INSTALL_COME`` therefore remains an explicit opt-in build arg, and
    ``is_come_available()`` emits a WARNING reminding callers of the
    non-commercial restriction (suppress with ``COME_DEV_ENVIRONMENT=1``).

CLI (verified 2026-05-26 against github.com/r4dl/CoMe):
    Train:        ``train.py --splatting_config configs/hierarchical.json
                  -s <dataset> -m <output>``  (``-s``/``-m`` are the standard
                  3DGS ParamGroup short flags; iterations etc. come from the
                  splatting-config JSON).
    Extract real: ``extract_mesh_tets.py -m <output>``   (marching tetrahedra)
    Extract synth:``extract_mesh_tsdf.py -m <output>``   (TSDF)
    Only ``configs/hierarchical.json`` ships in the repo.  Script names and
    flags are module-level constants below so they live in one place.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-driven installation paths
# ---------------------------------------------------------------------------
COME_DIR = Path(os.environ.get("COME_DIR", "/opt/come"))
COME_CONDA_ENV = os.environ.get("COME_CONDA_ENV", "come")

# ---------------------------------------------------------------------------
# Script / flag constants -- adjust here once verified against the CoMe repo
# ---------------------------------------------------------------------------
#: Training entry-point (SOF-derived; may need updating after code review)
COME_TRAIN_SCRIPT = "train.py"
#: Mesh extraction for *unbounded* scenes via marching tetrahedra (SOF pattern)
COME_EXTRACT_TETS_SCRIPT = "extract_mesh_tets.py"
#: Mesh extraction for *bounded* scenes via TSDF (SOF pattern)
COME_EXTRACT_TSDF_SCRIPT = "extract_mesh_tsdf.py"

#: CLI flag: path to JSON splatting config passed to train.py
COME_TRAIN_FLAG_CONFIG = "--splatting_config"
#: CLI flag: dataset source root (images/ + sparse/) passed to train.py
COME_TRAIN_FLAG_SOURCE = "-s"
#: CLI flag: model/output directory passed to train.py
COME_TRAIN_FLAG_MODEL = "-m"
#: CLI flag: model directory passed to extract_mesh_*.py
COME_EXTRACT_FLAG_MODEL = "-m"

# ---------------------------------------------------------------------------
# CoMe environment detection
# ---------------------------------------------------------------------------

_DOCKER_CONTAINER_NAME = "come"
#: Working directory inside the come sidecar (where the repo is installed)
_CONTAINER_COME_DIR = "/opt/come"


def _run_cwd(exec_prefix: list[str]) -> Optional[str]:
    """Return the host-side cwd for subprocess.run.

    For docker-exec mode the working directory lives *inside* the container
    (set via ``-w``), so the host cwd must be ``None`` -- passing a container
    path that does not exist on the host raises FileNotFoundError before the
    process starts. For conda/local mode COME_DIR exists on the host.
    """
    if exec_prefix and exec_prefix[0] == "docker":
        return None
    return str(COME_DIR)


@dataclass
class CoMeConfig:
    """Configuration for CoMe mesh extraction.

    Attributes:
        splatting_config: Name of the JSON training config passed to
            ``train.py --splatting_config``.  The value is relative to the
            CoMe installation directory (e.g. ``configs/unbounded.json``).
            Adjust once the released CoMe repo structure is confirmed.
        scene_type: ``"unbounded"`` uses marching tetrahedra
            (``extract_mesh_tets.py``); ``"bounded"`` uses TSDF
            (``extract_mesh_tsdf.py``).
        iterations: Training iterations forwarded in the splatting config.
            CoMe default is 30 000.  This field is reserved for future use
            when CoMe exposes a direct ``--iterations`` flag.
        train_timeout: Maximum seconds for the training subprocess (default
            2400 s -- ~40 min; paper reports ~18 min on RTX 4090).
        extract_timeout: Maximum seconds for mesh extraction (default 600 s
            -- 10 min; paper reports ~7 min on RTX 4090).
    """

    splatting_config: str = "configs/hierarchical.json"
    scene_type: str = "unbounded"
    iterations: int = 30_000
    train_timeout: int = 2400
    extract_timeout: int = 600


# ---------------------------------------------------------------------------
# Execution prefix helpers
# ---------------------------------------------------------------------------


def _come_exec_prefix() -> list[str]:
    """Return the command prefix for running CoMe Python scripts.

    Probes the ``come`` Docker sidecar first; falls back to the conda
    environment named by ``COME_CONDA_ENV``.  Returns an empty list if
    neither is reachable.
    """
    # -- Docker sidecar probe ------------------------------------------------
    try:
        result = subprocess.run(
            ["docker", "exec", _DOCKER_CONTAINER_NAME,
             "python3", "-c", "import torch; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # ``-w`` sets the working directory *inside* the container so
            # relative paths (e.g. configs/) resolve, without the host ever
            # needing COME_DIR to exist locally (avoids FileNotFoundError on
            # subprocess cwd=). See _run_cwd().
            return ["docker", "exec", "-w", _CONTAINER_COME_DIR,
                    _DOCKER_CONTAINER_NAME, "python3"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # -- Conda environment fallback ------------------------------------------
    try:
        result = subprocess.run(
            ["conda", "run", "-n", COME_CONDA_ENV,
             "python", "-c", "import torch; print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return ["conda", "run", "--no-capture-output", "-n", COME_CONDA_ENV, "python"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


def is_come_available() -> bool:
    """Return True if CoMe is reachable via docker sidecar or conda env.

    Emits a WARNING reminding callers of the non-commercial licence (ADR-004)
    when the sidecar is found running outside a recognised development
    environment (``COME_DEV_ENVIRONMENT=1`` suppresses it).
    """
    prefix = _come_exec_prefix()
    if not prefix:
        logger.debug("CoMe not available (no docker sidecar or conda env)")
        return False

    logger.debug("CoMe available via: %s", " ".join(prefix[:3]))

    dev_env = os.environ.get("COME_DEV_ENVIRONMENT", "0").strip()
    if dev_env not in ("1", "true", "yes"):
        logger.warning(
            "CoMe backend is available but is licensed under the Inria/MPII "
            "Gaussian-Splatting License — NON-COMMERCIAL research use only. "
            "Do NOT use its outputs in commercial products/distribution without "
            "a separate agreement with Inria. See ADR-004. Set "
            "COME_DEV_ENVIRONMENT=1 to suppress this warning."
        )

    return True


# ---------------------------------------------------------------------------
# COLMAP dataset validation helpers (mirrored from milo_extractor)
# ---------------------------------------------------------------------------


def _find_sparse_dir(colmap_path: Path) -> Optional[Path]:
    """Locate the COLMAP sparse model directory within a dataset."""
    candidates = [
        colmap_path / "sparse" / "0",
        colmap_path / "sparse",
        colmap_path / "undistorted" / "sparse" / "0",
        colmap_path / "undistorted" / "sparse",
    ]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    return None


def _find_dataset_root(sparse_dir: Path) -> Path:
    """Derive the COLMAP dataset root from the sparse model directory.

    CoMe (like MILo) expects the root directory that contains both
    ``sparse/`` and ``images/``.
    """
    if sparse_dir.name == "0":
        return sparse_dir.parent.parent
    return sparse_dir.parent


# ---------------------------------------------------------------------------
# Script path resolution
# ---------------------------------------------------------------------------


def _resolve_script(prefix: list[str], script_name: str) -> str:
    """Return the absolute path to a CoMe script appropriate for the prefix.

    When running via docker exec, scripts live at ``/opt/come/<name>``
    inside the container regardless of the host-side ``COME_DIR``.
    """
    if prefix and prefix[0] == "docker":
        return f"/opt/come/{script_name}"
    return str(COME_DIR / script_name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_come(
    colmap_dir: str,
    output_dir: str,
    config: Optional[CoMeConfig] = None,
) -> dict[str, Any]:
    """Run CoMe training + mesh extraction on a COLMAP dataset.

    Executes two subprocesses sequentially:

    1. ``train.py --splatting_config <cfg> -s <dataset_root> -m <output>``
    2. ``extract_mesh_tets.py -m <output>``  (unbounded)
       or ``extract_mesh_tsdf.py -m <output>``  (bounded)

    On completion, locates the newest mesh PLY in the output directory,
    converts it to GLB via trimesh, and returns all artifact paths.

    Args:
        colmap_dir: Path to COLMAP dataset (must contain ``sparse/0/``
            with ``cameras.bin``/``.txt`` and an ``images/`` directory).
        output_dir: Destination for CoMe checkpoints and mesh output.
        config: ``CoMeConfig`` instance.  Uses defaults when ``None``.

    Returns:
        Dict with keys:

        - ``success`` (bool): Whether the full pipeline completed.
        - ``mesh_path`` (str | None): Path to the extracted mesh PLY.
        - ``ply_path`` (str | None): Path to the trained gaussian point
          cloud PLY (``point_cloud/<iter>/point_cloud.ply``).
        - ``glb_path`` (str | None): Path to the GLB conversion of the
          mesh (populated on successful trimesh conversion).
        - ``duration`` (float): Total wall-clock seconds.
        - ``error`` (str | None): Error message; ``None`` on success.

    Notes:
        This function never raises to the caller.  All errors are captured
        in the returned dict.
    """
    cfg = config or CoMeConfig()
    result: dict[str, Any] = {
        "success": False,
        "mesh_path": None,
        "ply_path": None,
        "glb_path": None,
        "duration": 0.0,
        "error": None,
    }

    colmap_path = Path(colmap_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -- Validate COLMAP dataset structure -----------------------------------
    sparse_dir = _find_sparse_dir(colmap_path)
    if sparse_dir is None:
        result["error"] = f"No COLMAP sparse model found in {colmap_dir}"
        logger.error(result["error"])
        return result

    dataset_root = _find_dataset_root(sparse_dir)

    t_start = time.time()

    # -- Resolve execution prefix --------------------------------------------
    exec_prefix = _come_exec_prefix()
    if not exec_prefix:
        result["error"] = "CoMe not available (no docker sidecar or conda env)"
        logger.error(result["error"])
        return result

    # -- Step 1: CoMe training -----------------------------------------------
    train_script = _resolve_script(exec_prefix, COME_TRAIN_SCRIPT)
    train_cmd = exec_prefix + [
        train_script,
        COME_TRAIN_FLAG_CONFIG, cfg.splatting_config,
        COME_TRAIN_FLAG_SOURCE, str(dataset_root),
        COME_TRAIN_FLAG_MODEL, str(output_path),
    ]

    logger.info("CoMe training: %s", " ".join(train_cmd))

    try:
        proc = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=cfg.train_timeout,
            cwd=_run_cwd(exec_prefix),
        )
        if proc.returncode != 0:
            result["error"] = (
                f"CoMe training failed (rc={proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
            logger.error("CoMe training stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = f"CoMe training timed out ({cfg.train_timeout}s)"
        logger.error(result["error"])
        return result
    except (FileNotFoundError, OSError) as exc:
        result["error"] = f"CoMe training could not launch: {exc}"
        logger.error(result["error"])
        return result

    train_duration = time.time() - t_start
    logger.info("CoMe training completed in %.0fs", train_duration)

    # -- Step 2: Mesh extraction ---------------------------------------------
    extract_script_name = (
        COME_EXTRACT_TETS_SCRIPT
        if cfg.scene_type == "unbounded"
        else COME_EXTRACT_TSDF_SCRIPT
    )
    extract_script = _resolve_script(exec_prefix, extract_script_name)
    extract_cmd = exec_prefix + [
        extract_script,
        COME_EXTRACT_FLAG_MODEL, str(output_path),
    ]

    logger.info(
        "CoMe mesh extraction (%s): %s",
        extract_script_name, " ".join(extract_cmd),
    )

    try:
        proc = subprocess.run(
            extract_cmd,
            capture_output=True,
            text=True,
            timeout=cfg.extract_timeout,
            cwd=_run_cwd(exec_prefix),
        )
        if proc.returncode != 0:
            result["error"] = (
                f"CoMe mesh extraction failed (rc={proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
            logger.error("CoMe extraction stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = f"CoMe mesh extraction timed out ({cfg.extract_timeout}s)"
        logger.error(result["error"])
        return result
    except (FileNotFoundError, OSError) as exc:
        result["error"] = f"CoMe mesh extraction could not launch: {exc}"
        logger.error(result["error"])
        return result

    total_duration = time.time() - t_start

    # -- Locate output artifacts ---------------------------------------------
    # CoMe (SOF pattern) writes mesh PLYs into the output directory root.
    # Prefer files whose name contains "mesh"; fall back to any PLY.
    mesh_candidates = sorted(
        output_path.glob("mesh_*.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mesh_candidates:
        mesh_candidates = sorted(
            output_path.glob("*.ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not mesh_candidates:
        result["error"] = "No mesh PLY found in CoMe output directory"
        logger.error(result["error"])
        return result

    mesh_path = mesh_candidates[0]

    # Locate the trained gaussian point cloud
    ply_candidates = sorted(output_path.rglob("point_cloud/*/point_cloud.ply"))
    ply_path = ply_candidates[-1] if ply_candidates else None

    result["success"] = True
    result["mesh_path"] = str(mesh_path)
    result["ply_path"] = str(ply_path) if ply_path else None
    result["duration"] = total_duration

    # -- GLB conversion for the web viewer -----------------------------------
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), force="mesh")
        glb_path = mesh_path.with_suffix(".glb")
        mesh.export(str(glb_path))
        result["glb_path"] = str(glb_path)
        logger.info(
            "Converted CoMe mesh to GLB: %s (%d verts)",
            glb_path.name, len(mesh.vertices),
        )
    except Exception as conv_exc:
        logger.warning("Failed to convert CoMe PLY to GLB: %s", conv_exc)

    logger.info(
        "CoMe complete: mesh=%s (%d bytes), duration=%.0fs",
        mesh_path.name, mesh_path.stat().st_size, total_duration,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_come_mesh(mesh_path: str) -> Any:
    """Load a CoMe mesh PLY and return as a trimesh.Trimesh.

    Args:
        mesh_path: Path to the PLY mesh file produced by CoMe.

    Returns:
        A ``trimesh.Trimesh`` instance.

    Raises:
        ImportError: If trimesh is not installed.
        ValueError: If the file cannot be loaded as a mesh.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    logger.info(
        "Loaded CoMe mesh: %d vertices, %d faces",
        len(mesh.vertices), len(mesh.faces),
    )
    return mesh
