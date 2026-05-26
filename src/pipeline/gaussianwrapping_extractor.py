# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GaussianWrapping mesh extraction -- stochastic oriented surface elements.

Runs GaussianWrapping (github.com/diego1401/GaussianWrapping, 187 stars) inside
the existing MILo sidecar container (Ubuntu 22.04, Python 3.9, CUDA 11.8).
GaussianWrapping reinterprets 3D Gaussians as stochastic oriented surface
elements and extracts watertight, textured meshes that capture extremely thin
structures -- bicycle spokes, wires, fences, railings, thin architectural
elements -- where TSDF / marching-cubes methods fail.

GaussianWrapping shares the MILo sidecar because both tools require exactly the
same runtime (CUDA 11.8 / Python 3.9 / Ubuntu 22.04), avoiding a third
independent sidecar container.  It is installed at ``/opt/gaussianwrapping``
inside the container; ``is_gaussianwrapping_available()`` checks for that path
before reporting availability.

**Licensing note** (ADR-005): GaussianWrapping has no formal LICENSE file as of
2026-05-26.  ``is_gaussianwrapping_available()`` logs a WARNING referencing
ADR-005 when called outside development environments to remind operators not to
include the backend in commercial distribution images without legal review.

**CLI flags** (inferred from the repository structure; verify against
``/opt/gaussianwrapping`` after Dockerfile.milo rebuild):
    Training entry-point : ``train.py``
    Dataset flag         : ``-s <colmap_dataset_root>``
    Output model flag    : ``-m <output_dir>``
    Rasterizer flag      : ``--rasterizer radegs|median_depth``
    Iterations flag      : ``--iterations <int>``
    Adaptive meshing     : ``--adaptive_meshing`` (Primal Adaptive Meshing pass)
    Mesh-extract entry   : ``extract_mesh.py`` (exact name needs verification)
    Mesh flag            : ``--mesh_path <output_ply>``

Both entry-point script names (``GW_TRAIN_SCRIPT``, ``GW_EXTRACT_SCRIPT``) and
flag names are centralised as module-level constants below so that corrections
only need to be made in one place once the repo interface is confirmed.
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
# Environment configuration
# ---------------------------------------------------------------------------

GW_DIR = Path(os.environ.get("GW_DIR", "/opt/gaussianwrapping"))
GW_CONDA_ENV = os.environ.get("GW_CONDA_ENV", "milo")
#: Working directory inside the milo sidecar where GaussianWrapping is installed
_CONTAINER_GW_DIR = "/opt/gaussianwrapping"

# ---------------------------------------------------------------------------
# CLI constants
# NOTE: These names are inferred from the GaussianWrapping repository structure.
#       Verify against the installed version after docker/Dockerfile.milo rebuild.
# ---------------------------------------------------------------------------

GW_TRAIN_SCRIPT = "train.py"
GW_EXTRACT_SCRIPT = "extract_mesh.py"

GW_FLAG_SOURCE = "-s"
GW_FLAG_MODEL = "-m"
GW_FLAG_RASTERIZER = "--rasterizer"
GW_FLAG_ITERATIONS = "--iterations"
GW_FLAG_ADAPTIVE_MESHING = "--adaptive_meshing"
GW_FLAG_MESH_PATH = "--mesh_path"

GW_RASTERIZER_QUALITY = "radegs"
GW_RASTERIZER_SPEED = "median_depth"

# Environment variable used to detect non-development deployments for the
# ADR-005 licensing warning.
_PROD_ENV_VAR = "LICHTFELD_ENV"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class GWConfig:
    """Configuration for a GaussianWrapping mesh extraction run.

    Attributes:
        rasterizer: Rasteriser backend.  ``"radegs"`` gives higher quality;
            ``"median_depth"`` is faster and suitable for previews.
        iterations: Training iterations (gaussian-splatting default is 30 000).
        adaptive_meshing: Enable Primal Adaptive Meshing refinement, which
            targets high-resolution extraction in user-specified or
            automatically identified regions.
        train_timeout: Maximum wall-clock seconds allowed for the training
            subprocess before it is killed.
        extract_timeout: Maximum wall-clock seconds allowed for the mesh
            extraction subprocess before it is killed.
    """

    rasterizer: str = GW_RASTERIZER_QUALITY
    iterations: int = 30_000
    adaptive_meshing: bool = False
    train_timeout: int = 3000
    extract_timeout: int = 900


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gw_exec_prefix() -> list[str]:
    """Return the subprocess command prefix for executing GaussianWrapping.

    Tries ``docker exec milo python3`` first (the sidecar container), then
    falls back to ``conda run -n <GW_CONDA_ENV> python``.  Returns an empty
    list when neither is available.
    """
    # 1. Docker sidecar: verify the milo container is up AND gaussianwrapping
    #    is installed inside it (guards against partially updated images).
    #    Use ``test -d`` (no Python interpreter) so GW_DIR is passed as a
    #    discrete argv element and can never be interpreted as code. Quoting a
    #    user/env-controlled path into ``python -c`` is a code-injection vector.
    try:
        probe = subprocess.run(
            ["docker", "exec", "milo", "test", "-d", str(GW_DIR)],
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode == 0:
            # ``-w`` sets the in-container working dir so relative paths
            # resolve; the host never needs /opt/gaussianwrapping to exist
            # (subprocess cwd= must stay None for docker — see run_cwd below).
            return ["docker", "exec", "-w", _CONTAINER_GW_DIR, "milo", "python3"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. Conda env fallback (development / CI usage). Probe the directory with
    #    ``test -d`` for the same injection-safety reason as above.
    try:
        probe = subprocess.run(
            ["conda", "run", "-n", GW_CONDA_ENV, "test", "-d", str(GW_DIR)],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0:
            return ["conda", "run", "--no-capture-output", "-n", GW_CONDA_ENV, "python"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


def _gw_script(script_name: str, exec_prefix: list[str]) -> str:
    """Resolve the absolute path of a GaussianWrapping script.

    When running via docker exec the path is always inside the container
    (``/opt/gaussianwrapping/``); when running via conda it uses the local
    ``GW_DIR`` environment variable.
    """
    if exec_prefix and exec_prefix[0] == "docker":
        return f"/opt/gaussianwrapping/{script_name}"
    return str(GW_DIR / script_name)


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

    GaussianWrapping expects the root that contains both ``sparse/`` and
    ``images/``, following the same gaussian-splatting convention as MILo.
    """
    if sparse_dir.name == "0":
        return sparse_dir.parent.parent
    return sparse_dir.parent


def _warn_license_if_production() -> None:
    """Emit a licensing warning when running outside a development environment.

    ADR-005 mandates this check because GaussianWrapping has no formal LICENSE
    file and must not be shipped in commercial distribution images without legal
    review.
    """
    env = os.environ.get(_PROD_ENV_VAR, "development").lower()
    if env not in {"development", "dev", "test", "ci"}:
        logger.warning(
            "GaussianWrapping has no formal LICENSE file (ADR-005). "
            "Do not include this backend in commercial distribution images "
            "until the license is reviewed.  "
            "See research/decisions/adr-005-gaussianwrapping-milo-sidecar.md."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_gaussianwrapping_available() -> bool:
    """Return True if GaussianWrapping is reachable via docker sidecar or conda.

    Checks that:
    1. The ``milo`` sidecar container is running (or the ``GW_CONDA_ENV`` conda
       env exists), **and**
    2. ``/opt/gaussianwrapping`` (or local ``GW_DIR``) is present -- this
       guards against a healthy container whose CUDA extension build failed.

    Always returns ``False`` cleanly when neither condition is met; never
    raises.
    """
    _warn_license_if_production()
    prefix = _gw_exec_prefix()
    if not prefix:
        logger.debug(
            "GaussianWrapping not available "
            "(no docker sidecar with GW_DIR=%s or conda env '%s')",
            GW_DIR, GW_CONDA_ENV,
        )
        return False
    logger.debug("GaussianWrapping available via: %s", " ".join(prefix[:3]))
    return True


def run_gaussianwrapping(
    colmap_dir: str,
    output_dir: str,
    config: Optional[GWConfig] = None,
) -> dict[str, Any]:
    """Run GaussianWrapping training + mesh extraction on a COLMAP dataset.

    GaussianWrapping is particularly effective for thin-structure scenes
    (bicycle spokes, wires, fences, railings) where TSDF and marching-cubes
    methods produce incomplete meshes.

    Args:
        colmap_dir: Path to COLMAP dataset.  Must contain a ``sparse/0/``
            directory with ``cameras.bin``/``cameras.txt`` and an ``images/``
            directory.
        output_dir: Directory where GaussianWrapping writes its output
            (checkpoints and extracted mesh).  Created if absent.
        config: GaussianWrapping configuration.  Uses ``GWConfig`` defaults
            when ``None``.

    Returns:
        A dict with the following keys:

        - ``success`` (bool): ``True`` when the full train + extract pipeline
          completed without error.
        - ``mesh_path`` (str | None): Absolute path to the extracted mesh PLY.
        - ``ply_path`` (str | None): Absolute path to the trained gaussian PLY
          (point cloud), or ``None`` if not found.
        - ``glb_path`` (str | None): Absolute path to the GLB-converted mesh,
          or ``None`` when trimesh conversion failed.
        - ``duration`` (float): Total wall-clock seconds for train + extract.
        - ``error`` (str | None): Human-readable error message on failure;
          ``None`` on success.

    Never raises; all exceptions are caught and reported via ``"error"``.
    """
    cfg = config or GWConfig()
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
        logger.error("GaussianWrapping: %s", result["error"])
        return result

    dataset_root = _find_dataset_root(sparse_dir)

    t_start = time.time()

    # -- Resolve execution environment ---------------------------------------
    exec_prefix = _gw_exec_prefix()
    if not exec_prefix:
        result["error"] = (
            "GaussianWrapping not available "
            f"(no docker sidecar with GW_DIR={GW_DIR} or conda env '{GW_CONDA_ENV}')"
        )
        logger.error("GaussianWrapping: %s", result["error"])
        return result

    # Host-side cwd for subprocess.run: None under docker (the working dir is
    # set inside the container via ``-w``; a container path passed as host cwd
    # would raise FileNotFoundError before the process starts), the local
    # install dir under conda.
    run_cwd = None if exec_prefix[0] == "docker" else str(GW_DIR)

    # -- Step 1: GaussianWrapping training -----------------------------------
    train_script = _gw_script(GW_TRAIN_SCRIPT, exec_prefix)
    train_cmd = exec_prefix + [
        train_script,
        GW_FLAG_SOURCE, str(dataset_root),
        GW_FLAG_MODEL, str(output_path),
        GW_FLAG_RASTERIZER, cfg.rasterizer,
        GW_FLAG_ITERATIONS, str(cfg.iterations),
    ]
    if cfg.adaptive_meshing:
        train_cmd.append(GW_FLAG_ADAPTIVE_MESHING)

    logger.info("GaussianWrapping training: %s", " ".join(train_cmd))

    try:
        proc = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=cfg.train_timeout,
            cwd=run_cwd,
        )
        if proc.returncode != 0:
            result["error"] = (
                f"GaussianWrapping training failed (rc={proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
            logger.error("GaussianWrapping training stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = (
            f"GaussianWrapping training timed out ({cfg.train_timeout}s)"
        )
        logger.error("GaussianWrapping: %s", result["error"])
        return result
    except (FileNotFoundError, OSError) as exc:
        result["error"] = f"GaussianWrapping training could not launch: {exc}"
        logger.error("GaussianWrapping: %s", result["error"])
        return result

    train_duration = time.time() - t_start
    logger.info("GaussianWrapping training completed in %.0fs", train_duration)

    # -- Step 2: Mesh extraction ---------------------------------------------
    mesh_ply_path = output_path / "gaussianwrapping_mesh.ply"
    extract_script = _gw_script(GW_EXTRACT_SCRIPT, exec_prefix)
    extract_cmd = exec_prefix + [
        extract_script,
        GW_FLAG_SOURCE, str(dataset_root),
        GW_FLAG_MODEL, str(output_path),
        GW_FLAG_RASTERIZER, cfg.rasterizer,
        GW_FLAG_MESH_PATH, str(mesh_ply_path),
    ]

    logger.info("GaussianWrapping mesh extraction: %s", " ".join(extract_cmd))

    try:
        proc = subprocess.run(
            extract_cmd,
            capture_output=True,
            text=True,
            timeout=cfg.extract_timeout,
            cwd=run_cwd,
        )
        if proc.returncode != 0:
            result["error"] = (
                f"GaussianWrapping mesh extraction failed (rc={proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
            logger.error("GaussianWrapping extraction stderr: %s", proc.stderr[-2000:])
            return result
    except subprocess.TimeoutExpired:
        result["error"] = (
            f"GaussianWrapping mesh extraction timed out ({cfg.extract_timeout}s)"
        )
        logger.error("GaussianWrapping: %s", result["error"])
        return result
    except (FileNotFoundError, OSError) as exc:
        result["error"] = f"GaussianWrapping mesh extraction could not launch: {exc}"
        logger.error("GaussianWrapping: %s", result["error"])
        return result

    total_duration = time.time() - t_start

    # -- Locate mesh output artifact -----------------------------------------
    # Primary: the explicit output path we passed to extract_mesh.py.
    # Fallback: glob for any PLY written by the extraction step.
    mesh_path: Optional[Path] = None
    if mesh_ply_path.exists():
        mesh_path = mesh_ply_path
    else:
        candidates = sorted(
            output_path.glob("*.ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            mesh_path = candidates[0]

    if mesh_path is None:
        result["error"] = "No mesh PLY found in GaussianWrapping output directory"
        logger.error("GaussianWrapping: %s", result["error"])
        return result

    # -- Locate trained gaussian PLY (point_cloud/<iteration>/point_cloud.ply) --
    ply_candidates = sorted(output_path.rglob("point_cloud/*/point_cloud.ply"))
    ply_path: Optional[Path] = ply_candidates[-1] if ply_candidates else None

    result["success"] = True
    result["mesh_path"] = str(mesh_path)
    result["ply_path"] = str(ply_path) if ply_path else None
    result["duration"] = total_duration

    # -- GLB conversion for web viewer ---------------------------------------
    try:
        import trimesh  # type: ignore[import]

        mesh = trimesh.load(str(mesh_path), force="mesh")
        glb_path = mesh_path.with_suffix(".glb")
        mesh.export(str(glb_path))
        result["glb_path"] = str(glb_path)
        logger.info(
            "Converted GaussianWrapping mesh to GLB: %s (%d verts)",
            glb_path.name, len(mesh.vertices),
        )
    except Exception as conv_exc:
        logger.warning("Failed to convert GaussianWrapping PLY to GLB: %s", conv_exc)

    logger.info(
        "GaussianWrapping complete: mesh=%s (%d bytes), duration=%.0fs",
        mesh_path.name, mesh_path.stat().st_size, total_duration,
    )
    return result


def load_gaussianwrapping_mesh(mesh_path: str) -> Any:
    """Load a GaussianWrapping mesh PLY and return as a trimesh.Trimesh.

    Args:
        mesh_path: Path to the PLY mesh file produced by GaussianWrapping.

    Returns:
        A ``trimesh.Trimesh`` instance.

    Raises:
        ImportError: If trimesh is not installed.
        ValueError: If the file cannot be loaded as a mesh.
    """
    import trimesh  # type: ignore[import]

    mesh = trimesh.load(mesh_path, force="mesh")
    logger.info(
        "Loaded GaussianWrapping mesh: %d vertices, %d faces",
        len(mesh.vertices), len(mesh.faces),
    )
    return mesh
