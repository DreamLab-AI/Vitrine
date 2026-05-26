# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Splat optimisation -- PlayCanvas splat-transform CLI wrapper.

Wraps the ``@playcanvas/splat-transform`` npm CLI to compress, crop,
filter, sort, and convert trained Gaussian splat PLY files for web
delivery.  The tool is invoked via ``npx @playcanvas/splat-transform``
so no global install is required beyond having Node.js / npx on the
PATH.

ADR Reference: ADR-006 (splat-transform web-delivery decision).
PRD Reference: Section 3.1.2, Section 6.1.

Typical output sizes (relative):
    Raw PLY   ~100+ MB
    ksplat    < 20 MB   (after compress + crop + filter)

Integration point: post-3DGS training, before web delivery.
The original ``.ply`` is always retained as the source-of-truth for
downstream mesh extraction backends.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Package name as published on npm.  Pin this in the Dockerfile install.
_NPX_PACKAGE = "@playcanvas/splat-transform"

# Supported output formats.  "ksplat" is the PlayCanvas native compressed
# format; "ply" and "compressed-ply" are intermediate / full-fidelity forms.
_VALID_FORMATS = frozenset({"ksplat", "sog", "glb", "ply", "compressed-ply"})


@dataclass
class SplatOptConfig:
    """Configuration for the splat-transform optimisation stage.

    Attributes:
        crop_box: Optional 6-tuple ``(x_min, y_min, z_min, x_max, y_max,
            z_max)`` in world-space units.  Gaussians outside the box are
            removed before further processing.
        opacity_min_threshold: Gaussians with opacity below this value are
            discarded (removes floaters).  Range [0, 1]; default 0.05.
        max_scale: If set, Gaussians whose maximum axis scale exceeds this
            value are discarded (removes large background artefacts).
        sort: Reorder Gaussians in Morton (Z-order) sort for optimal
            front-to-back rendering.  Recommended True.
        output_format: Target delivery format.  Must be one of
            ``"ksplat"``, ``"sog"``, ``"glb"``, ``"ply"``, or
            ``"compressed-ply"``.
        generate_html_viewer: Emit a self-contained HTML file alongside
            the output that embeds a minimal PlayCanvas splat viewer.
        timeout: Maximum seconds to wait for the splat-transform process.
    """

    crop_box: Optional[tuple[float, float, float, float, float, float]] = None
    opacity_min_threshold: float = 0.05
    max_scale: Optional[float] = None
    sort: bool = True
    output_format: str = "ksplat"
    generate_html_viewer: bool = False
    timeout: int = 300


def is_splat_transform_available() -> bool:
    """Return True if the splat-transform CLI can be invoked.

    Executes ``npx @playcanvas/splat-transform --help`` with a short
    timeout.  Returns False on ``FileNotFoundError`` (npx/node not on
    PATH), ``TimeoutExpired``, or non-zero exit code.
    """
    try:
        result = subprocess.run(
            ["npx", "--yes", _NPX_PACKAGE, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        available = result.returncode == 0
        if available:
            logger.debug("splat-transform is available via npx")
        else:
            logger.debug(
                "splat-transform --help returned rc=%d; stderr: %s",
                result.returncode,
                result.stderr[:200],
            )
        return available
    except FileNotFoundError:
        logger.debug("npx not found; splat-transform unavailable")
        return False
    except subprocess.TimeoutExpired:
        logger.debug("splat-transform availability check timed out")
        return False


def _build_cli_args(
    input_ply: Path,
    output_path: Path,
    config: SplatOptConfig,
) -> list[str]:
    """Construct the splat-transform CLI argument list.

    Builds a single compound command that applies: crop (if requested),
    opacity / scale filtering, sort, and format conversion / compression
    in one invocation where the CLI supports it.  For CLIs that require
    chained invocations the orchestrating function is responsible for
    calling this in stages; this helper targets the single-pass form.

    Returns:
        List of strings suitable for :func:`subprocess.run`.
    """
    cmd: list[str] = ["npx", "--yes", _NPX_PACKAGE]

    # Primary subcommand: compress handles quantisation + format conversion.
    # Additional flags enable crop, filter, and sort within the same pass.
    cmd.append("compress")
    cmd.extend([str(input_ply), "-o", str(output_path)])

    if config.crop_box is not None:
        box_str = ",".join(str(v) for v in config.crop_box)
        cmd.extend(["--box", box_str])

    if config.opacity_min_threshold > 0:
        cmd.extend(["--alpha-min", str(config.opacity_min_threshold)])

    if config.max_scale is not None:
        cmd.extend(["--scale-max", str(config.max_scale)])

    if config.sort:
        cmd.append("--sort")

    if config.generate_html_viewer:
        cmd.append("--html")

    return cmd


def optimize(
    input_ply: str,
    output_dir: str,
    config: Optional[SplatOptConfig] = None,
) -> dict[str, Any]:
    """Run splat-transform to optimise a trained Gaussian PLY for delivery.

    Applies crop -> filter -> sort -> compress -> convert in a single
    subprocess call where supported by the CLI.  The original PLY is
    never modified.

    Args:
        input_ply: Absolute path to the trained Gaussian PLY file.
        output_dir: Directory where the compressed output will be written.
            Created if it does not exist.
        config: Optimisation settings.  Defaults are used if ``None``.

    Returns:
        Dictionary with the following keys:

        - ``success`` (bool): Whether the optimisation completed.
        - ``output_path`` (str | None): Path to the output file.
        - ``input_size_mb`` (float): Input PLY size in megabytes.
        - ``output_size_mb`` (float): Output file size in megabytes (0.0
          on failure).
        - ``compression_ratio`` (float): ``input / output`` ratio (1.0 on
          failure).
        - ``duration`` (float): Wall-clock seconds for the subprocess.
        - ``error`` (str | None): Human-readable error message on failure.
    """
    cfg = config or SplatOptConfig()

    result: dict[str, Any] = {
        "success": False,
        "output_path": None,
        "input_size_mb": 0.0,
        "output_size_mb": 0.0,
        "compression_ratio": 1.0,
        "duration": 0.0,
        "error": None,
    }

    if cfg.output_format not in _VALID_FORMATS:
        result["error"] = (
            f"Invalid output_format '{cfg.output_format}'. "
            f"Must be one of: {sorted(_VALID_FORMATS)}"
        )
        return result

    input_path = Path(input_ply)
    if not input_path.exists():
        result["error"] = f"Input PLY not found: {input_ply}"
        return result

    input_size_bytes = input_path.stat().st_size
    result["input_size_mb"] = input_size_bytes / (1024 * 1024)

    if not is_splat_transform_available():
        result["error"] = "splat-transform not available (npx / Node.js missing)"
        return result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename: same stem, new extension.
    ext_map = {
        "ksplat": ".ksplat",
        "sog": ".sog",
        "glb": ".glb",
        "ply": ".ply",
        "compressed-ply": ".ply",
    }
    output_ext = ext_map[cfg.output_format]
    output_stem = input_path.stem + "_optimized"
    output_path = out_dir / (output_stem + output_ext)

    cmd = _build_cli_args(input_path, output_path, cfg)

    logger.info("splat-transform command: %s", " ".join(cmd))

    t_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
        )
        result["duration"] = time.time() - t_start

        if proc.returncode != 0:
            result["error"] = (
                f"splat-transform failed (rc={proc.returncode}): "
                f"{proc.stderr[-1000:]}"
            )
            logger.error(
                "splat-transform stderr: %s",
                proc.stderr[-2000:],
            )
            return result

    except subprocess.TimeoutExpired:
        result["duration"] = time.time() - t_start
        result["error"] = f"splat-transform timed out after {cfg.timeout}s"
        logger.error("splat-transform timed out (%ds)", cfg.timeout)
        return result
    except OSError as exc:
        result["duration"] = time.time() - t_start
        result["error"] = f"Failed to launch splat-transform: {exc}"
        logger.error("splat-transform OSError: %s", exc)
        return result

    if not output_path.exists():
        result["error"] = (
            f"splat-transform succeeded but output not found: {output_path}"
        )
        return result

    output_size_bytes = output_path.stat().st_size
    output_size_mb = output_size_bytes / (1024 * 1024)
    compression_ratio = (
        input_size_bytes / output_size_bytes if output_size_bytes > 0 else 1.0
    )

    result["success"] = True
    result["output_path"] = str(output_path)
    result["output_size_mb"] = output_size_mb
    result["compression_ratio"] = compression_ratio

    logger.info(
        "splat-transform complete: %s -> %s (%.1f MB -> %.1f MB, %.1fx, %.1fs)",
        input_path.name,
        output_path.name,
        result["input_size_mb"],
        output_size_mb,
        compression_ratio,
        result["duration"],
    )

    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Optimise a trained Gaussian PLY via PlayCanvas splat-transform"
    )
    parser.add_argument("--input-ply", required=True, help="Path to trained PLY file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--format",
        default="ksplat",
        choices=sorted(_VALID_FORMATS),
        help="Output format (default: ksplat)",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.05,
        help="Minimum opacity threshold for filtering (default: 0.05)",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=None,
        help="Maximum Gaussian scale; larger splats are removed (default: none)",
    )
    parser.add_argument(
        "--crop-box",
        default=None,
        help="Crop bounding box as 'xmin,ymin,zmin,xmax,ymax,zmax'",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable Morton-order sort",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate an HTML viewer alongside the output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Subprocess timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    crop_box = None
    if args.crop_box:
        parts = [float(v) for v in args.crop_box.split(",")]
        if len(parts) != 6:
            parser.error("--crop-box must have exactly 6 comma-separated values")
        crop_box = tuple(parts)  # type: ignore[assignment]

    cfg = SplatOptConfig(
        crop_box=crop_box,
        opacity_min_threshold=args.opacity_threshold,
        max_scale=args.max_scale,
        sort=not args.no_sort,
        output_format=args.format,
        generate_html_viewer=args.html,
        timeout=args.timeout,
    )

    result = optimize(args.input_ply, args.output_dir, cfg)
    import json
    print(json.dumps(result, indent=2))
