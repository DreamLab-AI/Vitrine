# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-flight dependency verification for the Gaussian Toolkit pipeline.

Called at the START of every pipeline run, BEFORE any processing.
Fails HARD (raises RuntimeError) if any critical dependency is missing.

Usage::

    from pipeline.preflight import check_all, check_dependencies

    # Full check (called by PipelineStages.__init__)
    results = check_all()

    # Granular check returning structured dict
    deps = check_dependencies()
    for name, info in deps.items():
        print(f"{name}: available={info['available']}, version={info.get('version')}")
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_PYTHON = (3, 12)


def _env_flag(name: str) -> bool:
    """True when an env var is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

# Dependencies classified by criticality
CRITICAL_PYTHON_DEPS = [
    "torch",
    "gsplat",
    "trimesh",
    "xatlas",
    "plyfile",
    "numpy",
]

IMPORTANT_PYTHON_DEPS = [
    "PIL",           # Pillow -- texture I/O
    "scipy",         # texture dilation, spatial transforms
    "skimage",       # marching cubes
    "cv2",           # frame extraction, image processing
]

OPTIONAL_PYTHON_DEPS = [
    "pxr",           # OpenUSD -- USD assembly (fallback exists)
    "open3d",        # alternative mesh extraction
    "sam2",          # SAM2 segmentation
]

CRITICAL_BINARIES = [
    "colmap",
]

IMPORTANT_BINARIES = [
    "ffmpeg",
    "blender",
]

OPTIONAL_BINARIES = [
    "lfs-mcp",
]


def check_dependencies() -> dict[str, dict[str, Any]]:
    """Verify all critical dependencies are available and correct.

    Returns dict of {dep: {available: bool, version: str, path: str, critical: bool}}
    Fails HARD (raises RuntimeError) if any critical dep is missing.
    """
    checks: dict[str, dict[str, Any]] = {}

    # ── Python version ──────────────────────────────────────────────
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = {
        "available": True,
        "version": py_ver,
        "path": sys.executable,
        "critical": True,
    }
    if sys.version_info[:2] < REQUIRED_PYTHON:
        raise RuntimeError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
            f"got {py_ver}"
        )

    # ── torch + CUDA ────────────────────────────────────────────────
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if cuda_available
            else None
        )
        checks["torch"] = {
            "available": True,
            "version": torch.__version__,
            "path": str(Path(torch.__file__).parent),
            "cuda": cuda_available,
            "gpu": gpu_name,
            "vram_gb": vram_gb,
            "critical": True,
        }
        if not cuda_available:
            raise RuntimeError(
                "CRITICAL: torch has no CUDA. GPU operations will fail. "
                "Install torch with CUDA support."
            )
    except ImportError:
        raise RuntimeError(
            "CRITICAL: torch not installed. Pipeline cannot run. "
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    # ── gsplat ──────────────────────────────────────────────────────
    try:
        import gsplat

        checks["gsplat"] = {
            "available": True,
            "version": gsplat.__version__,
            "path": str(Path(gsplat.__file__).parent),
            "critical": True,
        }
    except ImportError:
        raise RuntimeError(
            "CRITICAL: gsplat not installed. Mesh extraction will produce garbage. "
            "Install: pip install gsplat"
        )

    # ── trimesh ─────────────────────────────────────────────────────
    try:
        import trimesh

        checks["trimesh"] = {
            "available": True,
            "version": trimesh.__version__,
            "path": str(Path(trimesh.__file__).parent),
            "critical": True,
        }
    except ImportError:
        raise RuntimeError("CRITICAL: trimesh not installed. pip install trimesh")

    # ── xatlas ──────────────────────────────────────────────────────
    try:
        import xatlas

        checks["xatlas"] = {
            "available": True,
            "version": getattr(xatlas, "__version__", "unknown"),
            "critical": True,
        }
    except ImportError:
        raise RuntimeError("CRITICAL: xatlas not installed. pip install xatlas")

    # ── plyfile ─────────────────────────────────────────────────────
    try:
        from plyfile import PlyData  # noqa: F401

        checks["plyfile"] = {
            "available": True,
            "version": "installed",
            "critical": True,
        }
    except ImportError:
        raise RuntimeError("CRITICAL: plyfile not installed. pip install plyfile")

    # ── numpy ───────────────────────────────────────────────────────
    try:
        import numpy as np

        checks["numpy"] = {
            "available": True,
            "version": np.__version__,
            "critical": True,
        }
    except ImportError:
        raise RuntimeError("CRITICAL: numpy not installed.")

    # ── Important (non-fatal but degrade quality) ───────────────────
    for mod_name, pkg_name, install_hint in [
        ("PIL", "Pillow", "pip install Pillow"),
        ("scipy", "scipy", "pip install scipy"),
        ("skimage", "scikit-image", "pip install scikit-image"),
        ("cv2", "OpenCV", "pip install opencv-python-headless"),
    ]:
        try:
            mod = __import__(mod_name)
            checks[pkg_name] = {
                "available": True,
                "version": getattr(mod, "__version__", "unknown"),
                "critical": False,
            }
        except ImportError:
            logger.warning(
                "%s not installed (%s). Some pipeline features degraded.",
                pkg_name,
                install_hint,
            )
            checks[pkg_name] = {
                "available": False,
                "version": None,
                "critical": False,
                "install": install_hint,
            }

    # ── Optional Python deps ────────────────────────────────────────
    for mod_name, pkg_name, desc in [
        ("pxr", "usd-core", "USD assembly -- fallback to minimal USDA if missing"),
        ("open3d", "open3d", "Alternative mesh extraction"),
    ]:
        try:
            mod = __import__(mod_name)
            checks[pkg_name] = {
                "available": True,
                "version": getattr(mod, "__version__", "unknown"),
                "critical": False,
                "optional": True,
            }
        except ImportError:
            checks[pkg_name] = {
                "available": False,
                "version": None,
                "critical": False,
                "optional": True,
                "description": desc,
            }

    # ── SAM2 / SAM3 ────────────────────────────────────────────────
    for mod_name, pkg_name in [("sam2", "sam2"), ("sam3", "sam3")]:
        try:
            __import__(mod_name)
            checks[pkg_name] = {"available": True, "critical": False, "optional": True}
        except ImportError:
            checks[pkg_name] = {
                "available": False,
                "critical": False,
                "optional": True,
                "description": f"{pkg_name} segmentation model",
            }

    # ── Critical binaries ───────────────────────────────────────────
    for binary in CRITICAL_BINARIES:
        path = shutil.which(binary)
        if path is None:
            raise RuntimeError(
                f"CRITICAL: {binary} not found in PATH. "
                f"Install COLMAP: https://colmap.github.io/install.html"
            )
        version = _get_binary_version(binary)
        checks[binary] = {
            "available": True,
            "version": version,
            "path": path,
            "critical": True,
        }

    # ── Important binaries ──────────────────────────────────────────
    for binary in IMPORTANT_BINARIES:
        path = shutil.which(binary)
        version = _get_binary_version(binary) if path else None
        if path is None:
            logger.warning(
                "%s not found in PATH. Some features unavailable.", binary
            )
        checks[binary] = {
            "available": path is not None,
            "version": version,
            "path": path,
            "critical": False,
        }

    # ── Optional binaries ───────────────────────────────────────────
    for binary in OPTIONAL_BINARIES:
        path = shutil.which(binary)
        checks[binary] = {
            "available": path is not None,
            "path": path,
            "critical": False,
            "optional": True,
        }

    # ── LichtFeld Studio binary ─────────────────────────────────────
    lfs_paths = [
        "/opt/gaussian-toolkit/build/LichtFeld-Studio",
        shutil.which("LichtFeld-Studio"),
    ]
    lfs_path = next((p for p in lfs_paths if p and Path(p).exists()), None)
    checks["lichtfeld-studio"] = {
        "available": lfs_path is not None,
        "path": lfs_path,
        "critical": False,
        "optional": True,
        "description": "LichtFeld Studio native binary for MCP rendering",
    }

    # ── VRAM check ──────────────────────────────────────────────────
    if checks.get("torch", {}).get("cuda"):
        import torch

        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        total_vram = torch.cuda.mem_get_info()[1] / 1e9
        checks["gpu_memory"] = {
            "available": True,
            "free_gb": round(free_vram, 1),
            "total_gb": round(total_vram, 1),
            "critical": False,
        }
        if free_vram < 4.0:
            logger.warning(
                "Low GPU memory: %.1f GB free of %.1f GB. "
                "Large models may OOM during rendering.",
                free_vram,
                total_vram,
            )

    return checks


def check_all() -> dict:
    """Verify all pipeline dependencies. Raises RuntimeError on critical failure.

    Returns a flat summary dict for backward compatibility.
    """
    full = check_dependencies()

    # Build flat summary for legacy callers
    summary: dict[str, Any] = {}
    for name, info in full.items():
        if info.get("version"):
            summary[name] = info["version"]
        elif info.get("path"):
            summary[name] = info["path"]
        elif info["available"]:
            summary[name] = True
        else:
            summary[name] = False

    critical_missing = [
        name
        for name, info in full.items()
        if info.get("critical") and not info["available"]
    ]
    if critical_missing:
        raise RuntimeError(
            f"CRITICAL: Missing dependencies: {', '.join(critical_missing)}"
        )

    # SOTA registry idiot-check (advisory unless SOTA_STRICT). Validates the
    # host against the model registry: checkpoints staged, VRAM fit, licence
    # posture, pinning, known caveats. Raises only under strict + overall=FAIL.
    sota = check_sota_environment()
    if sota is not None:
        summary["sota_overall"] = sota.get("overall")

    logger.info("Preflight check PASSED: %s", summary)
    return summary


def check_sota_environment(
    *, strict: bool | None = None, commercial: bool | None = None
) -> dict | None:
    """Validate the host against the SOTA model registry (idiot-check).

    Delegates to ``sota_registry.check_environment``: are the chosen checkpoints
    staged, does each model fit GPU VRAM, is the licence posture acceptable, are
    ComfyUI nodes pinned, and surfaces known wiring caveats.

    Advisory by default — logs the formatted report and never blocks a dev run
    on a partially-staged host. Set ``strict`` (or the ``SOTA_STRICT`` env var)
    to raise ``RuntimeError`` when ``overall == "FAIL"``. ``commercial`` (or
    ``SOTA_COMMERCIAL``) switches posture so non-commercial models FAIL instead
    of WARN.

    Returns the structured report dict, or ``None`` if the registry cannot be
    imported (a registry import error must never break preflight).
    """
    if strict is None:
        strict = _env_flag("SOTA_STRICT")
    if commercial is None:
        commercial = _env_flag("SOTA_COMMERCIAL")

    try:
        from pipeline.sota_registry import check_environment, format_report
    except Exception as e:  # noqa: BLE001 — registry optional; never break preflight
        logger.warning("SOTA registry unavailable, skipping idiot-check: %s", e)
        return None

    report = check_environment(commercial_use=bool(commercial))
    overall = report.get("overall", "PASS")
    if overall == "FAIL":
        text = format_report(report)
        if strict:
            raise RuntimeError(
                "SOTA idiot-check FAILED (strict posture). Report:\n" + text
            )
        logger.warning("SOTA idiot-check overall=FAIL (advisory):\n%s", text)
    elif overall == "WARN":
        logger.info("SOTA idiot-check overall=WARN:\n%s", format_report(report))
    else:
        logger.info("SOTA idiot-check overall=PASS")
    return report


def print_report() -> None:
    """Print a human-readable dependency report to stdout."""
    try:
        deps = check_dependencies()
    except RuntimeError as e:
        print(f"PREFLIGHT FAILED: {e}")
        return

    print("=" * 60)
    print("LichtFeld Pipeline -- Dependency Report")
    print("=" * 60)
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python:   {sys.version}")
    print()

    for name, info in deps.items():
        status = "OK" if info["available"] else "MISSING"
        critical = " [CRITICAL]" if info.get("critical") else ""
        optional = " (optional)" if info.get("optional") else ""
        version = f" v{info['version']}" if info.get("version") else ""
        path = f"  @ {info['path']}" if info.get("path") else ""
        extra = ""
        if name == "torch" and info.get("cuda"):
            extra = f"  CUDA={info.get('gpu', '?')} VRAM={info.get('vram_gb', '?')}GB"
        if name == "gpu_memory":
            extra = f"  {info.get('free_gb', '?')}GB free / {info.get('total_gb', '?')}GB total"

        marker = "+" if info["available"] else "-"
        print(f"  [{marker}] {name}{version}{critical}{optional}{path}{extra}")

    print()
    missing_optional = [
        name
        for name, info in deps.items()
        if not info["available"] and info.get("optional")
    ]
    if missing_optional:
        print(f"  Optional deps not installed: {', '.join(missing_optional)}")
        print("  These are non-blocking but may limit functionality.")
    print("=" * 60)

    # SOTA registry idiot-check (advisory in the report view).
    try:
        from pipeline.sota_registry import check_environment, format_report

        print()
        print(format_report(check_environment()))
    except Exception as e:  # noqa: BLE001 — advisory only
        print(f"  (SOTA idiot-check skipped: {e})")


def _get_binary_version(binary: str) -> str | None:
    """Try to extract version string from a binary."""
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout or result.stderr).strip()
        # Return first line
        return output.split("\n")[0][:100] if output else None
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None


if __name__ == "__main__":
    print_report()
