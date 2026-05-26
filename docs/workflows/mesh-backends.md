# Mesh Extraction Backends

Gaussian Toolkit v2 provides four mesh extraction backends. This guide explains how to choose one, what sidecar services each requires, and how to enable the splat optimisation and Fibonacci frame selection features introduced in v2.

---

## Quick Reference

| Backend | Config value | Container required | Speed (RTX 4090) | Best for |
|---------|-------------|-------------------|------------------|----------|
| TSDF | `"tsdf"` | main (always running) | ~5 min | Previews; fast iteration |
| MILo | `"milo"` | `milo` sidecar | ~69 min | General high-quality scenes |
| CoMe | `"come"` | `come` sidecar | ~25 min | Speed + quality balance |
| GaussianWrapping | `"gaussianwrapping"` | `milo` sidecar | ~30–50 min | Thin structures |
| Auto | `"auto"` | varies | varies | Let the pipeline decide |

Set the backend in your pipeline configuration:

```python
from pipeline.config import PipelineConfig, TrainingConfig

config = PipelineConfig(
    training=TrainingConfig(mesh_method="come")  # or "tsdf", "milo", "gaussianwrapping", "auto"
)
```

Or via JSON config file (`config.json` in your job directory):

```json
{
  "training": {
    "mesh_method": "come"
  }
}
```

---

## Backend Details

### TSDF — Fast Preview

**Module**: `mesh_extractor.py`
**Container**: main (CUDA 12.8, Python 3.12)
**Dependencies**: Open3D, gsplat

TSDF (Truncated Signed Distance Function) fusion renders depth maps from the trained Gaussian splat using gsplat, then fuses them into a triangle mesh via Open3D's `ScalableTSDFVolume` followed by marching cubes. This runs entirely in the main container without any sidecar.

**Strengths**: Very fast; always available; no extra container required.

**Weaknesses**: Lower geometric detail at object boundaries; poor on thin structures; noisy on reflective surfaces.

**When to use**: Previews, quick iteration, single-GPU deployments without sidecar capacity.

---

### MILo — General High Quality

**Module**: `milo_extractor.py`
**Container**: `milo` sidecar (CUDA 11.8, Python 3.10)
**Sidecar check**: `is_milo_available()` probes `docker exec milo python3 -c "import torch"`.

MILo (Mesh-In-the-Loop, SIGGRAPH Asia 2025) jointly trains a Gaussian model and a triangle mesh, using Delaunay triangulation and a learned SDF. The mesh participates in the Gaussian loss, so training quality and mesh quality are co-optimised.

**Strengths**: High-quality surface reconstruction; handles most scene types well; textured output (xatlas UV + multi-view reprojection).

**Weaknesses**: Slow (~69 min); requires the milo sidecar; moderate performance on thin structures.

**When to use**: Production-quality scenes where timing is not critical.

**Starting the sidecar**:
```bash
docker compose -f docker-compose.consolidated.yml up -d milo
```

---

### CoMe — Speed and Quality Balance

**Module**: `come_extractor.py`
**Container**: `come` sidecar (CUDA 12.1, Python 3.10)
**Sidecar check**: `is_come_available()` probes `docker exec come python3 -c "import torch; print('ok')"`.

CoMe (Confidence-based Mesh Extraction, github.com/r4dl/CoMe) augments each Gaussian with a per-Gaussian confidence value during training, then extracts a mesh via marching tetrahedra (`scene_type="unbounded"`) or TSDF (`scene_type="bounded"`). Training takes approximately 18 minutes and extraction approximately 7 minutes on an RTX 4090.

Reported F1 scores: 0.521 on Tanks & Temples, 0.662 on ScanNet++.

**Strengths**: ~3x faster than MILo at comparable quality; confidence mechanism suppresses floater artefacts.

**Weaknesses**: Produces geometry-only PLY; a separate texturing pass (xatlas + multi-view reprojection) is needed for textured output, adding latency. No formal LICENSE file as of 2026-05-26.

**Licensing gate**: CoMe must not be used in commercial distribution images until a permissive licence is published and reviewed. The container is built with `INSTALL_COME=0` by default.

```bash
# Build with CoMe enabled (development only):
docker compose -f docker-compose.consolidated.yml build --build-arg INSTALL_COME=1 come

# Start the sidecar:
docker compose -f docker-compose.consolidated.yml up -d come
```

**CLI flag notice**: The script names and CLI flags in `come_extractor.py` (`COME_TRAIN_SCRIPT`, `COME_EXTRACT_TETS_SCRIPT`, etc.) are inferred from the SOF codebase on which CoMe is built. They have not been verified against the released CoMe source. If CoMe fails with a "script not found" or "unrecognised flag" error, check the constants at the top of `src/pipeline/come_extractor.py` and update them to match the installed version.

**Configuration**:
```python
from pipeline.come_extractor import CoMeConfig

config = CoMeConfig(
    splatting_config="configs/come_unbounded.json",  # relative to /opt/come
    scene_type="unbounded",   # "unbounded" (marching tet) or "bounded" (TSDF)
    iterations=30_000,
    train_timeout=2400,       # seconds; paper: ~18 min on RTX 4090
    extract_timeout=600,      # seconds; paper: ~7 min
)
```

Set `COME_DEV_ENVIRONMENT=1` in your shell environment to suppress the licensing warning during development.

---

### GaussianWrapping — Thin Structures

**Module**: `gaussianwrapping_extractor.py`
**Container**: `milo` sidecar (CUDA 11.8, Python 3.10; GaussianWrapping installed at `/opt/gaussianwrapping`)
**Sidecar check**: `is_gaussianwrapping_available()` verifies both that the `milo` container is running and that `/opt/gaussianwrapping` exists inside it.

GaussianWrapping (github.com/diego1401/GaussianWrapping) reinterprets 3D Gaussians as stochastic oriented surface elements and wraps a watertight mesh around the Gaussian cloud. The method excels at capturing thin structures (bicycle spokes, chain-link fences, wires, railings, thin architectural elements) where TSDF and marching cubes produce incomplete geometry.

Two rasterisation backends are available: `"radegs"` (RaDeGS, higher quality, default) and `"median_depth"` (faster, suitable for previews). An optional Primal Adaptive Meshing pass (`adaptive_meshing=True`) targets high-resolution extraction in complex regions.

**Strengths**: Only backend that reliably handles thin structures; shares the existing milo sidecar (no new container needed); produces textured output.

**Weaknesses**: Research code; no formal LICENSE file as of 2026-05-26; CLI flags are inferred and need verification.

**Licensing gate**: Same as CoMe — must not be in commercial distribution images without legal review. Gated behind `--build-arg INSTALL_GAUSSIANWRAPPING=1`.

```bash
# Build with GaussianWrapping enabled (development only):
docker compose -f docker-compose.consolidated.yml build --build-arg INSTALL_GAUSSIANWRAPPING=1 milo

# The milo sidecar is then started as usual:
docker compose -f docker-compose.consolidated.yml up -d milo
```

**CLI flag notice**: Script names and flags in `gaussianwrapping_extractor.py` (`GW_TRAIN_SCRIPT`, `GW_EXTRACT_SCRIPT`, etc.) are inferred from the GaussianWrapping repository structure. Verify against `/opt/gaussianwrapping` after the Dockerfile.milo rebuild.

**Configuration**:
```python
from pipeline.gaussianwrapping_extractor import GWConfig

config = GWConfig(
    rasterizer="radegs",       # "radegs" (quality) or "median_depth" (speed)
    iterations=30_000,
    adaptive_meshing=False,    # True = Primal Adaptive Meshing refinement
    train_timeout=3000,        # seconds
    extract_timeout=900,       # seconds
)
```

Set `LICHTFELD_ENV=development` (or `dev`, `test`, `ci`) to suppress the licensing warning.

---

## Auto-Selection Policy

When `mesh_method = "auto"`, the pipeline applies the following decision table in `stages._select_mesh_backend()`:

| Condition | Selected backend |
|-----------|-----------------|
| `preview=True` or speed priority | `tsdf` |
| Scene contains thin structures (SAM label heuristic) and GaussianWrapping is available | `gaussianwrapping` |
| CoMe sidecar is available | `come` |
| MILo sidecar is available | `milo` |
| No sidecar available | `tsdf` (always-available fallback) |

The thin-structure heuristic currently uses SAM label names (bicycle, fence, wire, railing, gate) from the segmentation stage. If segmentation has not run yet or the labels do not match, the heuristic falls back to CoMe or MILo.

**Override**: Set `mesh_method` to a concrete backend name to bypass auto-selection for reproducible runs.

---

## Splat Optimisation (post-training)

After 3DGS training, `splat_optimizer.py` can compress the trained Gaussian PLY for web delivery using the PlayCanvas `@playcanvas/splat-transform` npm package.

**Enable**:
```python
from pipeline.config import DeliveryConfig

config = PipelineConfig(
    delivery=DeliveryConfig(
        enable_splat_optimize=True,
        output_format="ksplat",          # "ksplat" | "sog" | "glb" | "ply" | "compressed-ply"
        opacity_min_threshold=0.05,      # discard Gaussians below this opacity
        max_scale=None,                  # discard oversized Gaussians (None = no limit)
        sort=True,                       # Morton-order sort for front-to-back rendering
    )
)
```

**Typical size reduction**: 100+ MB raw PLY → <20 MB `.ksplat` (after crop + filter + sort + compress).

**Requirements**: Node.js and `npx` must be installed in the main container. The Dockerfile.consolidated includes `apt-get install -y nodejs npm`.

**Availability check**:
```python
from pipeline.splat_optimizer import is_splat_transform_available
print(is_splat_transform_available())
```

The original `.ply` is always kept alongside the compressed output. The `.ksplat` is a delivery artefact; all mesh extraction backends use the original `.ply` as their source of truth.

**Direct invocation** (standalone):
```bash
python -m pipeline.splat_optimizer \
  --input-ply /data/output/JOB/model/point_cloud.ply \
  --output-dir /data/output/JOB/delivery/ \
  --format ksplat \
  --opacity-threshold 0.05 \
  --sort
```

---

## Fibonacci Frame Selection (post-COLMAP)

After COLMAP SfM, `fibonacci_sampler.py` can improve frame selection by scoring frames by their viewpoint coverage of a Fibonacci-sphere distribution — a near-optimal uniform arrangement on the unit sphere.

**Enable**:
```python
config = PipelineConfig(
    ingest=IngestConfig(
        use_fibonacci_coverage=True,
        coverage_weight=0.4,   # 0.4 coverage / 0.6 quality (ADR-007 default)
    )
)
```

**How it works**: Camera positions from `cameras.bin` are normalised to the unit sphere around their centroid. Each camera is scored by how closely it covers an under-represented Fibonacci-sphere direction. The combined score is:

```
score = (1 - coverage_weight) * quality_score + coverage_weight * fibonacci_score
```

Frames that fill angular coverage gaps score higher; frames that duplicate already-represented directions score lower.

**Fallback**: If COLMAP camera positions are unavailable (e.g., during the pre-SfM frame selection pass), the scorer falls back silently to the v1 quality-only path.

**Direct invocation** (standalone):
```python
from pipeline.fibonacci_sampler import select_frames_by_coverage
import numpy as np

# positions: (N, 3) COLMAP camera centres, quality: (N,) blur+exposure scores
selected_indices = select_frames_by_coverage(positions, quality, n_select=150)
```

---

## Troubleshooting

### CoMe sidecar not found

```
CoMe not available (no docker sidecar or conda env)
```

Check that the `come` container is running:
```bash
docker ps | grep come
```

If the container is absent, start it:
```bash
docker compose -f docker-compose.consolidated.yml up -d come
```

If the container starts but CoMe is not installed, rebuild with `INSTALL_COME=1` after reviewing the licence situation.

### GaussianWrapping: `is_gaussianwrapping_available()` returns False despite milo running

The availability check tests for `/opt/gaussianwrapping` inside the container:
```bash
docker exec milo test -d /opt/gaussianwrapping && echo "installed" || echo "not installed"
```

If not installed, rebuild the milo sidecar with `INSTALL_GAUSSIANWRAPPING=1`.

### CoMe or GaussianWrapping CLI error: script not found

The script paths and CLI flags in `come_extractor.py` and `gaussianwrapping_extractor.py` are inferred and may not match the released code exactly. Edit the module-level constants (`COME_TRAIN_SCRIPT`, `GW_EXTRACT_SCRIPT`, etc.) to match the actual filenames found in `/opt/come` or `/opt/gaussianwrapping` inside the respective containers.

### splat-transform not available

```
splat-transform not available (npx / Node.js missing)
```

Ensure Node.js and npm are installed in the main container. The optimisation stage is non-fatal -- if splat-transform is unavailable, the pipeline continues without the `.ksplat` delivery artefact.

### Fallback to TSDF when a sidecar backend was requested

If an explicit backend (`"come"`, `"milo"`, `"gaussianwrapping"`) is requested but the sidecar is unreachable, `stages.py` logs an error and falls back to TSDF rather than failing the job. To force a hard failure instead, set `config.training.mesh_backend_auto = False` and ensure the sidecar is healthy before submitting a job.
