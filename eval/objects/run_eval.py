#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Object-quality eval harness (PRD v4 R9).

Runs the ADR-025 single-image generator over a fixed crop set and grades the
results against a committed reference, so generator/service/executor changes
are regression-gated instead of eyeballed. Also exposes a --stats-only mode
to grade an existing GLB (e.g. the validated 2026-07-02 dreamlab assets).

Usage (from the gaussian-toolkit container, repo root):

    # Full eval: generate per crop + compare to references
    python3 eval/objects/run_eval.py --crops /data/output/rawcapdev/object_crops \
        --out /data/output/eval_objects

    # Grade one existing GLB (no GPU needed)
    python3 eval/objects/run_eval.py --stats-only path/to/object.glb

    # Refresh the committed reference after an ACCEPTED quality change
    python3 eval/objects/run_eval.py --crops ... --out ... --write-references

Metrics per object: generation wall time, GLB byte size + sha256, vertex /
face counts, watertightness, bbox extents, PBR material presence. Turntable
renders (via headless Blender when available) land next to the metrics for
the human sign-off pass — the numeric gate is necessary, not sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

REFERENCES = Path(__file__).parent / "references.json"

# Regression gates vs reference (fractional tolerance where applicable).
FACE_COUNT_TOLERANCE = 0.30      # ±30% faces vs reference
DURATION_TOLERANCE = 1.5         # ≤1.5x reference wall time


def mesh_stats(glb_path: Path) -> dict:
    """Numeric geometry/material stats for one GLB (no GPU required)."""
    import trimesh

    data = glb_path.read_bytes()
    scene = trimesh.load(str(glb_path), file_type="glb", force="scene")
    meshes = ([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
              if isinstance(scene, trimesh.Scene) else [scene])
    if not meshes:
        return {"error": "no meshes in GLB", "bytes": len(data)}
    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    extents = [round(float(v), 4) for v in mesh.extents] if mesh.vertices.size else []
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bbox_extents": extents,
        "has_material": material is not None,
        "material_type": type(material).__name__ if material is not None else None,
        "has_uv": bool(getattr(visual, "uv", None) is not None
                       and len(getattr(visual, "uv", [])) > 0),
    }


def render_turntable(glb_path: Path, out_dir: Path, views: int = 8) -> list[str]:
    """Headless-Blender turntable for the human review pass. Best-effort."""
    blender = shutil.which("blender")
    if blender is None:
        return []
    script = Path(__file__).parent / "blender_turntable.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [blender, "-b", "-noaudio", "-P", str(script), "--",
             str(glb_path), str(out_dir), str(views)],
            capture_output=True, timeout=600, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  turntable render skipped ({exc})")
        return []
    return sorted(str(p) for p in out_dir.glob("view_*.png"))


def grade(name: str, stats: dict, reference: dict | None) -> list[str]:
    """Compare one object's stats to its reference; return regression strings."""
    problems: list[str] = []
    if stats.get("error"):
        return [f"{name}: {stats['error']}"]
    if reference is None:
        return []
    if reference.get("watertight") and not stats.get("watertight"):
        problems.append(f"{name}: watertight -> NOT watertight")
    if reference.get("has_material") and not stats.get("has_material"):
        problems.append(f"{name}: PBR material LOST")
    ref_faces = reference.get("faces", 0)
    if ref_faces:
        delta = abs(stats.get("faces", 0) - ref_faces) / ref_faces
        if delta > FACE_COUNT_TOLERANCE:
            problems.append(
                f"{name}: face count {stats.get('faces')} vs ref {ref_faces} "
                f"({delta:+.0%} > ±{FACE_COUNT_TOLERANCE:.0%})")
    ref_t = reference.get("duration_s", 0)
    if ref_t and stats.get("duration_s", 0) > ref_t * DURATION_TOLERANCE:
        problems.append(f"{name}: {stats['duration_s']:.0f}s vs ref {ref_t:.0f}s "
                        f"(> {DURATION_TOLERANCE}x)")
    return problems


def run_generation(crops_dir: Path, out_dir: Path, generator: str, seed: int) -> dict:
    """Generate one GLB per crop via the configured ADR-025 generator."""
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    results: dict[str, dict] = {}
    crops = sorted(p for p in crops_dir.glob("*.png") if "_mask" not in p.stem
                   and "_thumb" not in p.stem)
    if not crops:
        raise SystemExit(f"No crops found in {crops_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for crop in crops:
        name = crop.stem
        print(f"== {name} ({generator}) ==")
        t0 = time.monotonic()
        try:
            if generator == "trellis2":
                from pipeline.trellis2_client import Trellis2Client
                client = Trellis2Client.from_config(cfg.trellis2)
                res = client.reconstruct_from_image(crop, seed=seed, label=name)
                glb = res.glb_data
            elif generator == "pixal3d":
                # R8 head-to-head: constructs the client directly, so the eval
                # never needs the pipeline flag flipped (cfg.pixal3d.enabled).
                from pipeline.pixal3d_client import Pixal3DClient
                client = Pixal3DClient.from_config(cfg.pixal3d)
                res = client.reconstruct_from_image(crop, seed=seed, label=name)
                glb = res.glb_data
            else:
                from pipeline.hunyuan3d_client import Hunyuan3DClient
                client = Hunyuan3DClient.from_config(cfg.hunyuan3d)
                res = client.reconstruct_from_image(crop, seed=seed, label=name)
                glb = res.glb_data
        except Exception as exc:  # noqa: BLE001 — record and continue the sweep
            results[name] = {"error": str(exc)}
            print(f"  FAILED: {exc}")
            continue
        duration = time.monotonic() - t0
        if not glb:
            results[name] = {"error": getattr(res, "error", "no GLB")}
            print(f"  FAILED: {results[name]['error']}")
            continue
        glb_path = out_dir / f"{name}.glb"
        glb_path.write_bytes(glb)                       # verbatim, always
        stats = mesh_stats(glb_path)
        stats["duration_s"] = round(duration, 1)
        stats["turntable"] = render_turntable(glb_path, out_dir / f"{name}_turntable")
        results[name] = stats
        print(f"  {stats.get('faces', '?')} faces, {duration:.0f}s, "
              f"material={stats.get('material_type')}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crops", type=Path, help="Directory of object crops (RGBA PNGs)")
    ap.add_argument("--out", type=Path, help="Output directory for GLBs + metrics")
    ap.add_argument("--generator", choices=["trellis2", "hy3d21", "pixal3d"],
                    default="trellis2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", type=Path, metavar="GLB",
                    help="Grade one existing GLB and exit (no generation)")
    ap.add_argument("--write-references", action="store_true",
                    help="Accept this run as the new committed reference")
    args = ap.parse_args()

    if args.stats_only:
        stats = mesh_stats(args.stats_only)
        print(json.dumps(stats, indent=2))
        return 0 if not stats.get("error") else 1

    if not args.crops or not args.out:
        ap.error("--crops and --out are required (or use --stats-only)")

    results = run_generation(args.crops, args.out, args.generator, args.seed)
    (args.out / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    references = (json.loads(REFERENCES.read_text())
                  if REFERENCES.exists() else {})
    problems: list[str] = []
    for name, stats in results.items():
        problems += grade(name, stats, references.get(name))

    if args.write_references:
        REFERENCES.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"References updated: {REFERENCES}")

    generated = sum(1 for s in results.values() if not s.get("error"))
    print(f"\n{generated}/{len(results)} objects generated; "
          f"{len(problems)} regression(s)")
    for p in problems:
        print(f"  REGRESSION: {p}")
    return 1 if problems or generated < len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
