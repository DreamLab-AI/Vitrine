# Code Boundaries

This document defines what is upstream (LichtFeld Studio), what is our addition (Gaussian Toolkit), and what is experimental. It exists to prevent identity drift between the two projects and to make merge decisions obvious.

## Upstream: LichtFeld Studio (MrNeRF)

LichtFeld Studio is the upstream product. It is a native C++23/CUDA workstation for 3D Gaussian Splatting developed by MrNeRF. We fork it; we do not modify it.

**Upstream directories -- do not modify on the gaussian-toolkit branch:**

| Directory | Contents |
|-----------|----------|
| `src/core/` | Core data structures, Gaussian representation, scene graph |
| `src/app/` | Application entry point, GUI tools, main loop |
| `src/mcp/` | Built-in MCP HTTP server (JSON-RPC, tool/resource registries) |
| `src/rendering/` | Rasterization, camera, viewport |
| `src/training/` | Training loop, optimizers, schedulers |
| `src/geometry/` | Spatial data structures |
| `src/io/` | Import/export (PLY, SOG, SPZ, HTML) |
| `src/sequencer/` | Timeline/animation |
| `src/visualizer/` | GUI framework, panels, assets |
| `src/python/` | Embedded Python plugin runtime |
| `cmake/` | Build system configuration |
| `external/` | Git submodules (OpenMesh, nvImageCodec, libvterm, etc.) |
| `eval/` | Evaluation scripts |
| `tools/` | CLI wrappers shipped by upstream |
| `tests/` | Upstream test suite |
| `CMakeLists.txt` | Root build file |
| `vcpkg.json` | C++ dependency manifest |
| `CONTRIBUTING.md` | Upstream contributing guide |
| `LICENSE` | GPL-3.0 |
| `THIRD_PARTY_LICENSES.md` | Upstream third-party notices |

**Merge policy:** Periodically rebase or merge from upstream `main`. Conflicts in upstream directories are resolved in favour of upstream.

## Our Addition: Gaussian Toolkit

Everything below is written and maintained by us on the `gaussian-toolkit` branch. Upstream does not contain these directories.

### Pipeline (`src/pipeline/`) -- 32 modules

The video-to-structured-3D pipeline. Takes a video file and produces a USD scene with per-object Gaussian and mesh representations, plus a compressed `.ksplat` for web delivery.

| Category | Modules |
|----------|---------|
| Core | `stages.py`, `orchestrator.py`, `cli.py`, `__main__.py`, `config.py`, `preflight.py`, `__init__.py` |
| Reconstruction | `colmap_parser.py`, `coordinate_transform.py`, `frame_selector.py`, `frame_quality.py` |
| Ingestion | `fibonacci_sampler.py` (Fibonacci-sphere viewpoint coverage scoring — added v2) |
| Segmentation | `sam2_segmentor.py`, `sam3_segmentor.py`, `sam3d_client.py`, `mask_projector.py` |
| Mesh extraction | `mesh_extractor.py` (TSDF), `milo_extractor.py` (MILo sidecar), `come_extractor.py` (CoMe sidecar — added v2), `gaussianwrapping_extractor.py` (GaussianWrapping in milo sidecar — added v2), `mesh_cleaner.py` |
| Delivery | `splat_optimizer.py` (splat-transform CLI wrapper — added v2) |
| Texturing | `texture_baker.py`, `material_assigner.py` |
| Scene assembly | `blender_assembler.py` (Blender + Cycles), `usd_assembler.py` (OpenUSD) |
| Rendering | `multiview_renderer.py`, `gsplat_trainer.py`, `hunyuan3d_client.py`, `comfyui_inpainter.py` |
| Utilities | `mcp_client.py`, `quality_gates.py`, `person_remover.py` |

### Web Interface (`src/web/`)

Flask application (port 7860) for video upload, job tracking, log streaming (SSE), 3D preview (model-viewer), and result download. Files: `app.py`, `job_manager.py`, `pipeline_runner.py`, `static/`, `templates/`.

### Deployment

| File | Purpose |
|------|---------|
| `Dockerfile.consolidated` | Main container (Ubuntu 24.04, CUDA 12.8, Python 3.12) |
| `docker/Dockerfile.milo` | MILo + optional GaussianWrapping sidecar (Ubuntu 22.04, CUDA 11.8, Python 3.10) |
| `docker/Dockerfile.come` | CoMe sidecar (Ubuntu 22.04, CUDA 12.1, Python 3.10) — added v2 |
| `docker-compose.consolidated.yml` | Three-container compose: main + milo sidecar + come sidecar |
| `docker/Dockerfile` | Base container (older, superseded) |
| `docker/docker-compose.yml` | Base compose (older, superseded) |
| `docker/entrypoint.sh` | Container entry script |
| `docker/supervisord.conf` | Process manager configuration |
| `docker/install_milo.sh` | MILo dependency installer |
| `docker/install_gaussianwrapping.sh` | GaussianWrapping installer (gated: `INSTALL_GAUSSIANWRAPPING=1`) |
| `docker/install_come.sh` | CoMe installer (gated: `INSTALL_COME=1`) |
| `docker/run_docker.sh` | Launch helper |

**The default deployment story is:** `docker compose -f docker-compose.consolidated.yml up -d`. Three containers (main + milo sidecar + come sidecar), one command. The milo sidecar is optional -- if not present, mesh extraction falls back to TSDF. The come sidecar starts but CoMe is not installed unless `INSTALL_COME=1` is passed at build time (ADR-004; licence pending). GaussianWrapping in the milo sidecar similarly requires `INSTALL_GAUSSIANWRAPPING=1` (ADR-005; licence pending).

### Scripts (`scripts/`)

Pipeline runners, test harnesses, and utilities:
- `run_gallery_pipeline.py` -- Full gallery pipeline
- `run_object_separation.py` -- Object extraction
- `run_tsdf_mesh.py` -- TSDF mesh extraction
- `assemble_gallery_usd.py` -- USD scene assembly
- `lichtfeld_mcp_bridge.py` -- stdio MCP bridge for Claude Desktop/Codex
- `hardware_trace.py` -- GPU/RAM/CPU logging
- `test_*.py` -- Test harnesses for individual pipeline stages

### Research (`research/`)

Research documents covering landscape analysis, pipeline design, component integration, and architecture decisions. This is research context supporting development decisions. It is not product documentation and should not be treated as user-facing.

### Documentation

| Location | Contents |
|----------|----------|
| `docs/architecture.md` | Two-container architecture overview |
| `docs/engineering-log.md` | Development history and key decisions |
| `docs/architecture/` | Detailed architecture docs (overview, cluster setup, performance) |
| `docs/integration/` | Docker, MCP, SplatReady integration guides |
| `docs/workflows/` | Video capture, video-to-splat, scene cleanup workflows |
| `docs/troubleshooting/` | Build issues, headless MCP debugging |
| `docs/renders/` | Pipeline output renders and screenshots |

### Other Our Files

| File | Purpose |
|------|---------|
| `README.md` | Project README (rewritten for the fork) |
| `GAUSSIAN_TOOLKIT_README.md` | Extended README with module status table |
| `BOUNDARIES.md` | This file |
| `AGENTS.md` | Agent operating guide for MCP-driven workflows |
| `CLAUDE_CONTAINER.md` | Claude Code instructions inside the container |

## Experimental

These components are built but not yet validated end-to-end. They may change significantly or be removed.

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| SAM3 concept segmentation | `sam3_segmentor.py`, `sam3d_client.py` | Client built, needs HF_TOKEN | Text+visual concept prompts (4M concepts) |
| Texture baking | `texture_baker.py` | Skeleton written | Depends on clean mesh + xatlas |
| Material assignment | `material_assigner.py` | Skeleton written | Depends on texture baking |
| FLUX background inpainting | `comfyui_inpainter.py` | Client built | ComfyUI workflow dependency |
| Audio-to-scene-graph naming | Planned (not started) | Planned | Whisper transcription to name USD prims |

## Decision Framework

When deciding where new code goes:

1. **Does it modify how LichtFeld trains, renders, or exports Gaussians?** -- Propose it upstream. Do not put it on `gaussian-toolkit`.
2. **Does it extend the video-to-scene pipeline?** -- Put it in `src/pipeline/`.
3. **Does it add a web endpoint or UI page?** -- Put it in `src/web/`.
4. **Does it change container configuration?** -- Put it in `docker/` or update `Dockerfile.consolidated`.
5. **Is it a research exploration or literature review?** -- Put it in `research/`.
6. **Is it a one-off script or test harness?** -- Put it in `scripts/`.
7. **Does it require CUDA 11.8 or older Python (3.9)?** -- Put it in the MILo sidecar (`docker/Dockerfile.milo`).
8. **Does it require Python 3.10 and CUDA 12.1?** -- Put it in the CoMe sidecar (`docker/Dockerfile.come`).
9. **Is it a new mesh extraction backend?** -- Follow the ADR-003 uniform interface: `XConfig` dataclass, `is_X_available() -> bool`, `run_X(colmap_dir, output_dir, config) -> dict`. Name the module `{name}_extractor.py` in `src/pipeline/`. Add dispatch in `stages._select_mesh_backend()`.
