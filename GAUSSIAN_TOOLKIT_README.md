# Gaussian Toolkit

**Video-to-3D-scene pipeline built on LichtFeld Studio.**

Gaussian Toolkit is our fork of [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (MrNeRF) that adds a complete video-to-structured-3D pipeline. LichtFeld Studio is the upstream product -- a native workstation for 3D Gaussian Splatting training, editing, and export. Gaussian Toolkit extends it with automated video ingestion, object segmentation, mesh extraction, USD scene assembly, and compressed-splat web delivery, all running inside a consolidated Docker deployment on a dedicated GPU workstation.

We do not modify upstream LichtFeld code. Our additions live in separate directories (`src/pipeline/`, `src/web/`, `docker/`, `scripts/`, `research/`). See [BOUNDARIES.md](BOUNDARIES.md) for the full separation policy.

Upstream sync is **one-way pull only**: we merge from upstream; we never push code or open pull requests against the upstream repository (ADR-002).

---

## Quick Start

```bash
# 1. Clone and checkout the feat/v2-upgrade-swarm branch
git clone <repo-url> && cd LichtFeld-Studio
git checkout feat/v2-upgrade-swarm

# 2. Set your HuggingFace token (needed for model downloads)
export HF_TOKEN=hf_your_token_here

# 3. Build and start the containers
docker compose -f docker-compose.consolidated.yml up -d

# 4. Open the web interface
#    http://localhost:7860
```

Services exposed by the containers:

| Port  | Service              |
|-------|----------------------|
| 7860  | Web upload interface |
| 8188  | ComfyUI              |
| 45677 | LichtFeld MCP server |
| 5901  | VNC remote desktop   |

Upload a video at `:7860`, and the pipeline will produce a structured USD scene with per-object meshes and a compressed `.ksplat` for web delivery.

---

## What This Fork Adds

LichtFeld Studio provides 3DGS training, visualisation, editing, and export. Gaussian Toolkit adds everything needed to go from a raw video file to a fully decomposed 3D scene:

### Pipeline Modules (32 files in `src/pipeline/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `orchestrator.py` | End-to-end pipeline driver (PyAV frame extraction, COLMAP orchestration) | Working |
| `cli.py` / `__main__.py` | CLI entry point | Working |
| `config.py` | Typed pipeline configuration with JSON persistence | Working |
| `mcp_client.py` | LichtFeld MCP client for training control | Working |
| `stages.py` | Independent, stateless stage functions; backend dispatch | Working |
| `preflight.py` | Pre-run dependency and sidecar health checks | Working |
| `sam2_segmentor.py` | SAM2 2D segmentation (grid-point prompts) | Working |
| `sam3_segmentor.py` | SAM3 segmentation wrapper | In Progress |
| `sam3d_client.py` | SAM3D concept segmentation client (4M concepts) | In Progress |
| `mask_projector.py` | 2D mask to 3D Gaussian label projection (98.3% coverage) | Working |
| `mesh_extractor.py` | TSDF mesh extraction (Open3D depth fusion) | Working |
| `milo_extractor.py` | MILo sidecar client (SIGGRAPH Asia 2025) | Working |
| `come_extractor.py` | CoMe sidecar client -- confidence-based marching tetrahedra | New (v2); CLI flags inferred, needs verification |
| `gaussianwrapping_extractor.py` | GaussianWrapping sidecar client -- thin-structure specialist | New (v2); CLI flags inferred, needs verification |
| `splat_optimizer.py` | PlayCanvas splat-transform CLI wrapper -- compress/crop/sort PLY | New (v2); requires Node.js + npx |
| `fibonacci_sampler.py` | Fibonacci-sphere viewpoint coverage scoring for frame selection | New (v2) |
| `mesh_cleaner.py` | Decimation, hole filling, manifold repair | Working |
| `texture_baker.py` | UV unwrapping + texture bake (xatlas) | In Progress |
| `material_assigner.py` | PBR material assignment | In Progress |
| `usd_assembler.py` | USD scene assembly (variant sets: Gaussian + Mesh) | Working |
| `multiview_renderer.py` | Camera orbit renders for Hunyuan3D input | Working |
| `hunyuan3d_client.py` | Hunyuan3D 2.0 multi-view to textured mesh | Working |
| `comfyui_inpainter.py` | FLUX inpainting via ComfyUI for background recovery | Working |
| `person_remover.py` | Person removal from training views | Working |
| `frame_selector.py` | Keyframe selection with quality + coverage scoring | Working |
| `frame_quality.py` | Blur/exposure quality scoring | Working |
| `coordinate_transform.py` | COLMAP / 3DGS / USD coordinate transforms | Working |
| `colmap_parser.py` | COLMAP binary model reader | Working (needs hardening) |
| `quality_gates.py` | Per-stage pass/fail quality checks | Working |
| `gsplat_trainer.py` | Direct gsplat Python API training (alternative to LichtFeld MCP) | Working |

### Web Interface (`src/web/`)

Flask application on port 7860 with video upload, job tracking, and result download.

### Deployment (`docker/`, `docker-compose.consolidated.yml`)

Three-container Docker deployment running all services under supervisord. Designed for a dedicated GPU workstation (tested on dual RTX 6000 Ada, 96GB VRAM, 251GB RAM, Threadripper PRO 48-core).

| Container | Image | Purpose |
|-----------|-------|---------|
| `gaussian-toolkit` | Ubuntu 24.04, CUDA 12.8, Python 3.12 | COLMAP, LichtFeld, SAM3, Blender, web UI, pipeline |
| `milo` | Ubuntu 22.04, CUDA 11.8, Python 3.10 | MILo mesh extraction + optional GaussianWrapping |
| `come` | Ubuntu 22.04, CUDA 12.1, Python 3.10 | CoMe confidence-based mesh extraction (dev/opt-in only) |

The CoMe container is excluded from production images by default (`INSTALL_COME=0`) pending licence review (ADR-004). GaussianWrapping in the `milo` container is similarly gated (`INSTALL_GAUSSIANWRAPPING=0`, ADR-005).

### Research (`research/`)

Research documents covering tool landscape, pipeline architecture decisions, segmentation methods, mesh extraction approaches, and the v2 upgrade architecture (ADRs 001–008, DDD domain model). This is research context, not product documentation.

### Utility Scripts (`scripts/`)

Pipeline runners, test harnesses, MCP bridge, hardware tracing, gallery assembly.

---

## Pipeline Architecture

```
Video (.mp4/.mov) or Web Upload (:7860)
    |
    v [Stage 1] Frame extraction (PyAV)
JPEG Frames
    |
    v [Stage 2] COLMAP SfM (feature extract -> match -> sparse -> undistort)
COLMAP Dataset (+ camera positions for Fibonacci scoring post-SfM)
    |
    v [Stage 3] 3DGS Training (LichtFeld MCP, MRNF densification)
Trained Gaussian Splat (~1M gaussians)
    |
    v [Stage 3b] Splat optimisation (splat-transform: crop/filter/sort/compress)
Compressed .ksplat for web delivery  [NEW in v2]
    |
    v [Stage 4] SAM2/SAM3 segmentation on training views
2D Object Masks
    |
    v [Stage 5] Mask projection to 3D Gaussians (98.3% coverage)
Per-Object PLY Files
    |
    v [Stage 6] Mesh extraction (TSDF | MILo | CoMe | GaussianWrapping)
Polygonal Meshes (GLB + OBJ)
    |
    v [Stage 7] Background inpainting (FLUX via ComfyUI)
Clean Background Views
    |
    v [Stage 8] USD scene assembly (variant sets: Gaussian + Mesh)
USD Scene + .ksplat + GLB exports
```

---

## Mesh Extraction Backends

Four backends are available, selected via `config.training.mesh_method`. See [docs/workflows/mesh-backends.md](docs/workflows/mesh-backends.md) for the user guide.

| Backend | Module | Container | CUDA | Speed | Best For |
|---------|--------|-----------|------|-------|----------|
| `tsdf` | `mesh_extractor.py` | main (CUDA 12.8) | 12.8 | ~5 min | Previews, fast iteration |
| `milo` | `milo_extractor.py` | `milo` sidecar (CUDA 11.8) | 11.8 | ~69 min | General high-quality scenes |
| `come` | `come_extractor.py` | `come` sidecar (CUDA 12.1) | 12.1 | ~25 min | Speed + quality; dev/opt-in only |
| `gaussianwrapping` | `gaussianwrapping_extractor.py` | `milo` sidecar (CUDA 11.8) | 11.8 | ~30-50 min | Thin structures (wires, fences, spokes) |
| `auto` | (dispatch in `stages.py`) | varies | varies | varies | Let the pipeline decide |

**Auto-selection policy** (ADR-003): thin-structure hint → GaussianWrapping; speed priority → CoMe (if available); default quality → MILo; fallback → TSDF.

**CLI flag notice**: CoMe (`come_extractor.py`) and GaussianWrapping (`gaussianwrapping_extractor.py`) CLI flags are inferred from their upstream repositories and have not been verified against the released code. Script names and flag constants are centralised in their respective modules for easy correction.

---

## Splat-Transform Delivery Stage

After 3DGS training, `splat_optimizer.py` wraps the PlayCanvas `@playcanvas/splat-transform` npm CLI to:
- Crop Gaussians outside the scene bounding box (removes sky/ground noise)
- Filter by opacity threshold (removes floaters)
- Sort in Morton order for optimal front-to-back rendering
- Compress to `.ksplat` format (typically <20 MB vs. 100+ MB raw PLY)

Enable via `config.delivery.enable_splat_optimize = True`. The original `.ply` is always kept alongside the compressed form.

---

## Fibonacci Frame Selection

`fibonacci_sampler.py` provides viewpoint coverage scoring for `frame_selector.py`. After COLMAP SfM, camera positions are scored by their coverage of a Fibonacci-sphere distribution — a near-optimal uniform arrangement of viewpoints. The combined frame score is:

```
score = 0.6 * quality_score + 0.4 * fibonacci_coverage_score
```

Enable via `config.ingest.use_fibonacci_coverage = True` (default: False for backward compatibility). The `coverage_weight` is configurable. Falls back to the v1 quality-only path if COLMAP camera positions are unavailable.

---

## Directory Boundaries

```
LichtFeld-Studio/
  src/
    core/          # UPSTREAM - LichtFeld core (do not modify)
    app/           # UPSTREAM - LichtFeld application
    mcp/           # UPSTREAM - LichtFeld MCP server
    rendering/     # UPSTREAM - LichtFeld rendering
    training/      # UPSTREAM - LichtFeld training
    pipeline/      # OURS - 32 pipeline modules
    web/           # OURS - Flask web interface
  research/        # OURS - Research documents, ADRs, DDD model
  docker/          # OURS - Docker configuration (Dockerfile.milo, Dockerfile.come)
  scripts/         # OURS - Utility scripts
  docker-compose.consolidated.yml  # OURS (three-container deployment)
  README.md        # OURS (rewritten for the fork)
```

See [BOUNDARIES.md](BOUNDARIES.md) for the complete policy.

Upstream sync is one-way pull. We merged to the **v0.5.2 stable tag** (2026-04-21) and deferred the v0.5.3 Vulkan migration to a separate, gated decision (ADR-008).

---

## Feature Status Summary

| Category | Feature | Status |
|----------|---------|--------|
| Ingestion | Video frame extraction | Working |
| Ingestion | Web upload interface | Working |
| Ingestion | Fibonacci-sphere viewpoint scoring | New (v2); opt-in |
| Reconstruction | COLMAP SfM | Working |
| Reconstruction | 3DGS training via MCP (MRNF densification) | Working |
| Segmentation | SAM2 (grid-point prompts) | Working |
| Segmentation | SAM3 (text+visual, 4M concepts) | In Progress |
| Mesh | TSDF extraction (Open3D) | Working |
| Mesh | MILo (SIGGRAPH Asia 2025, high quality) | Working |
| Mesh | CoMe (confidence-based, ~25 min) | New (v2); dev/opt-in only; CLI flags unverified |
| Mesh | GaussianWrapping (thin structures) | New (v2); dev/opt-in only; CLI flags unverified |
| Mesh | Per-object Hunyuan3D 2.0 | Working |
| Mesh | Texture baking (xatlas) | In Progress |
| Mesh | Material assignment | In Progress |
| Delivery | Splat optimisation (splat-transform .ksplat) | New (v2); requires Node.js |
| Scene | USD assembly with variant sets | Working |
| Scene | Background inpainting (FLUX) | Working |
| Infra | Three-container Docker (main + milo + come) | New (v2) |
| Infra | MCP bridge | Working |
| Infra | Quality gates | Working |

---

## Known Limitations

1. **COLMAP is the bottleneck** -- ~20 min on 32 cores for 15 frames. No GPU BA solver available.
2. **SAM2 prompt quality varies** -- Grid-point prompts need tuning per scene. SAM3 upgrade will eliminate this.
3. **No UV texture baking yet** -- Meshes are vertex-coloured. xatlas integration is in progress.
4. **USD Gaussian variants are path references** -- Not embedded splat data.
5. **Single-machine deployment only** -- The consolidated container assumes all GPUs are local.

---

## MCP Integration

The pipeline can be driven programmatically through LichtFeld's MCP server (70+ tools on port 45677). See [AGENTS.md](AGENTS.md) for the agent operating guide.

```bash
# Example: start training from CLI
lfs-mcp call scene.load_dataset '{"path": "/opt/output/colmap"}'
lfs-mcp call training.start
lfs-mcp call training.get_state
```

---

## Contributing

Pipeline, web, Docker, and research changes live in this repository on the `feat/v2-upgrade-swarm` branch (merged into `main` after review). Upstream code (`src/core/`, `src/app/`, etc.) is never modified here; upstream improvements are synced inbound via `git merge upstream/vX.Y.Z`. We never push to or open PRs against the upstream repository.

For architecture decisions, see `research/decisions/adr-*.md`. For the domain model, see `research/ddd/`.

For the mesh backend user guide, see [docs/workflows/mesh-backends.md](docs/workflows/mesh-backends.md).
