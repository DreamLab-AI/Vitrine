# Gaussian Toolkit

Video-to-3D-scene pipeline built on [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio). Upload a video, get a textured polygonal mesh, a USD scene, and a compressed Gaussian splat for web delivery.

Upstream sync is one-way pull only. We never push to or open PRs against the upstream repository.

## Architecture

```
Video Upload → Frame Extraction (Fibonacci viewpoint scoring) →
  → COLMAP SfM → 3DGS Training (MRNF densification) →
  → Splat Optimisation (splat-transform → .ksplat) →
  → Object Segmentation (SAM3) →
  → Mesh Extraction (TSDF | MILo | CoMe | GaussianWrapping) →
  → Blender Assembly + Texture Bake → USD Scene + .ksplat + Web Viewer
```

Three-container Docker deployment:

| Container | Base | GPU | Purpose |
|-----------|------|-----|---------|
| `gaussian-toolkit` | Ubuntu 24.04, CUDA 12.8, Python 3.12 | GPU 0 | COLMAP, LichtFeld 3DGS, web UI, Blender, SAM3 |
| `milo` | Ubuntu 22.04, CUDA 11.8, Python 3.10 | GPU 1 | MILo + optional GaussianWrapping mesh extraction |
| `come` | Ubuntu 22.04, CUDA 12.1, Python 3.10 | GPU 1 | CoMe mesh extraction (dev/opt-in; licence pending) |

## Quick Start

```bash
# Set HuggingFace token (needed for SAM3 segmentation models)
export HF_TOKEN=hf_your_token_here

# Start both containers
docker compose -f docker-compose.consolidated.yml up -d

# Open the web interface
# http://localhost:7860
```

Upload a video at the web UI. The pipeline runs autonomously via Claude Code inside the container.

## Pipeline Stages

| Stage | Tool | Output |
|-------|------|--------|
| Frame extraction | PyAV | JPEG frames |
| Viewpoint scoring (optional) | `fibonacci_sampler.py` | Per-frame coverage scores |
| Structure-from-Motion | COLMAP 4.1.0 | Camera poses + sparse point cloud |
| 3DGS training | LichtFeld Studio (MCP, MRNF densification) | Trained gaussian PLY (~1M splats) |
| Splat optimisation | `splat_optimizer.py` + splat-transform | Compressed `.ksplat` for web |
| Object segmentation | SAM3 (4M concepts, text+visual) | Per-object 2D masks |
| Mask projection | Custom (ray casting) | Per-object 3D gaussian labels |
| Mesh extraction | TSDF / MILo / CoMe / GaussianWrapping | GLB meshes |
| Scene assembly | Blender (Cycles GPU) | Texture-baked USD scene |
| Web delivery | Flask + model-viewer | Preview carousel, download ZIP |

### Mesh Extraction Backends

Four backends are available. Set `config.training.mesh_method` to one of the values below, or use `"auto"` to let the pipeline apply the ADR-003 selection policy.

| Backend | Value | Container | Speed | Best For |
|---------|-------|-----------|-------|----------|
| TSDF | `"tsdf"` | main | ~5 min | Previews, fast iteration |
| MILo | `"milo"` | `milo` sidecar | ~69 min | General high-quality scenes |
| CoMe | `"come"` | `come` sidecar | ~25 min | Speed + quality balance (dev/opt-in; licence pending) |
| GaussianWrapping | `"gaussianwrapping"` | `milo` sidecar | ~30-50 min | Thin structures: bicycle spokes, wires, fences, railings |

**Auto-selection policy** (`"auto"`): thin-structure hint → GaussianWrapping; CoMe available → CoMe; MILo available → MILo; fallback → TSDF.

**Licensing note**: CoMe and GaussianWrapping have no formal LICENSE files as of 2026-05-26. Both are gated behind Docker build args (`INSTALL_COME=1`, `INSTALL_GAUSSIANWRAPPING=1`) and must not be included in commercial distribution images without legal review. See ADR-004 and ADR-005.
Windows binaries are now available through the Lichtfeld Portal. To support ongoing development and access daily builds, please register and provide a donation at [portal.lichtfeld.io](https://portal.lichtfeld.io/). Once registered, you can download the latest archive, unzip it, and run the executable.

**CLI flag notice**: CoMe and GaussianWrapping CLI flags are inferred from their upstream repositories and have not been verified against the released source. All script names and flag constants are defined as module-level constants in `come_extractor.py` and `gaussianwrapping_extractor.py` for easy correction once verified.

See [docs/workflows/mesh-backends.md](docs/workflows/mesh-backends.md) for the full usage guide.

## Web Interface

The Flask app on port 7860 provides:

- Video upload with drag-and-drop
- Real-time pipeline progress (SSE log streaming)
- 3D preview via `<model-viewer>` (Google)
- Preview image carousel from Blender renders
- Download ZIP of all outputs (mesh, USD, previews)
- Anthropic API key management for Claude Code orchestration

## Services

| Port | Service |
|------|---------|
| 7860 | Web UI (Flask) |
| 7681 | Web terminal (ttyd / Claude Code) |
| 8188 | ComfyUI |
| 45677 | LichtFeld MCP server (70+ tools) |
| 5901 | VNC (Blender remote desktop) |

## Pipeline Modules

32 Python modules in `src/pipeline/`:

| Category | Modules |
|----------|---------|
| Core | `stages.py`, `orchestrator.py`, `cli.py`, `config.py`, `preflight.py` |
| Reconstruction | `colmap_parser.py`, `coordinate_transform.py`, `frame_selector.py`, `frame_quality.py` |
| Ingestion | `fibonacci_sampler.py` (viewpoint coverage scoring — new v2) |
| Segmentation | `sam2_segmentor.py`, `sam3_segmentor.py`, `sam3d_client.py`, `mask_projector.py` |
| Mesh | `mesh_extractor.py` (TSDF), `milo_extractor.py` (MILo), `come_extractor.py` (CoMe — new v2), `gaussianwrapping_extractor.py` (GaussianWrapping — new v2), `mesh_cleaner.py` |
| Delivery | `splat_optimizer.py` (splat-transform wrapper — new v2) |
| Texturing | `texture_baker.py`, `material_assigner.py` |
| Scene | `blender_assembler.py`, `usd_assembler.py` |
| Rendering | `multiview_renderer.py`, `gsplat_trainer.py`, `hunyuan3d_client.py`, `comfyui_inpainter.py` |
| Utilities | `mcp_client.py`, `quality_gates.py`, `person_remover.py` |

Web interface in `src/web/`: `app.py`, `job_manager.py`, `pipeline_runner.py`, templates, static assets.

## Hardware

Tested on:

| Component | Spec |
|-----------|------|
| GPU | 2x NVIDIA RTX 6000 Ada (48 GB VRAM each, 96 GB total) |
| CPU | AMD Threadripper PRO 48-core |
| RAM | 251 GB |
| Storage | NVMe SSD |
<p>
  <a href="https://www.core11.eu/">
    <img src="docs/media/core11_multi.svg" alt="Core 11" height="60">
  </a>
</p>

<br>

<p>
  <a href="https://web.volinga.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/media/volinga-dark.svg">
      <img src="docs/media/volinga.svg" alt="Volinga" height="108">
    </picture>
  </a>
</p>

Minimum: single GPU with 12 GB VRAM (TSDF only; MILo, CoMe, and GaussianWrapping sidecars require a second GPU or sequential scheduling).

## Project Boundaries

This is a fork of LichtFeld Studio. Upstream code (`src/core/`, `src/app/`, `src/mcp/`, `src/rendering/`, `src/training/`) is not modified. All pipeline additions live in `src/pipeline/`, `src/web/`, `docker/`, and `scripts/`. See [BOUNDARIES.md](BOUNDARIES.md) for the full separation policy.

## License

Upstream LichtFeld Studio: GPL-3.0. Pipeline additions: GPL-3.0 (derivative work).
