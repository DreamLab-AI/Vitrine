# Architecture

## Three-Container Deployment (v2)

Gaussian Toolkit v2 runs as three Docker containers sharing volumes for output data.

### gaussian-toolkit (main container)

| Property | Value |
|----------|-------|
| Base image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Python | 3.12 |
| GPU assignment | Device 0 (RTX 6000 Ada, 48 GB) |
| Ports | 7860 (web UI), 7681 (ttyd terminal), 8188 (ComfyUI), 45677 (LichtFeld MCP), 5901 (VNC) |
| Process manager | supervisord |
| Memory limit | 200 GB |
| Shared memory | 64 GB |

Runs: COLMAP SfM, LichtFeld Studio 3DGS training (MRNF densification), SAM3 segmentation, Blender scene assembly, Flask web UI, ComfyUI workflows, Claude Code (agentic orchestrator), splat-transform (Node.js, post-training splat optimisation).

### milo (sidecar container)

| Property | Value |
|----------|-------|
| Base image | `nvidia/cuda:11.8.0-devel-ubuntu22.04` |
| Python | 3.10 |
| GPU assignment | Device 1 (RTX 6000 Ada, 48 GB) |
| Entrypoint | `sleep infinity` (called via `docker exec`) |
| Tools | MILo (SIGGRAPH Asia 2025) + optional GaussianWrapping (build-arg gated) |
| CUDA extensions | diff-gaussian-rasterization (3 variants), simple-knn, fused-ssim, nvdiffrast, tetra-triangulation |

MILo requires CUDA 11.8 + GCC <= 11 for its CUDA extension compilation, incompatible with the main container (CUDA 12.8 + GCC 14). GaussianWrapping shares this environment exactly (CUDA 11.8, Python 3.10) and is installed at `/opt/gaussianwrapping` when enabled via `--build-arg INSTALL_GAUSSIANWRAPPING=1` (ADR-005; licence pending).

The main container invokes MILo and GaussianWrapping via:
```bash
docker exec milo python3 train.py --source_path /data/output/JOB/colmap ...
docker exec milo python3 /opt/gaussianwrapping/train.py -s /data/output/JOB/colmap ...
```

### come (sidecar container)

| Property | Value |
|----------|-------|
| Base image | `nvidia/cuda:12.1.1-devel-ubuntu22.04` |
| Python | 3.10 |
| GPU assignment | Device 1 |
| Entrypoint | `sleep infinity` (called via `docker exec`) |
| Tools | CoMe (confidence-based marching tetrahedra; code released 2026-04-22) |
| Build gate | `--build-arg INSTALL_COME=1` (off by default; licence pending — ADR-004) |

CoMe requires Python 3.10 and CUDA 12.1, incompatible with both the main container (CUDA 12.8, Python 3.12) and the MILo sidecar (CUDA 11.8). It therefore occupies a dedicated sidecar. The sidecar definition in `docker-compose.consolidated.yml` is present and the container starts, but CoMe itself is only installed when `INSTALL_COME=1` is passed at build time.

The main container invokes CoMe via:
```bash
docker exec come python3 /opt/come/train.py --splatting_config configs/come_unbounded.json -s /data/... -m /data/...
docker exec come python3 /opt/come/extract_mesh_tets.py -m /data/...
```

### Shared Resources

```
Volumes:
  ./output:/data/output         # All containers read/write pipeline outputs
  hf-cache:/opt/hf-cache        # HuggingFace model cache (shared)
  models-data:/opt/models        # Persistent model storage
  claude-session:/home/ubuntu/.claude  # Claude Code OAuth (main only)
```

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│  gaussian-toolkit container (GPU 0)                      │
│                                                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐             │
│  │ Flask UI │ │ LichtFeld │ │  COLMAP    │             │
│  │  :7860   │ │ MCP :45677│ │  SfM       │             │
│  └────┬─────┘ └─────┬─────┘ └────────────┘             │
│       │              │                                   │
│  ┌────▼──────────────▼─────────────────────┐            │
│  │        Pipeline (32 Python modules)      │            │
│  │  stages → colmap → train →               │            │
│  │  splat_optimize → segment →              │            │
│  │  _select_mesh_backend → blender assemble │            │
│  └────┬─────────────────────────────────────┘            │
│       │                                                  │
│  ┌────▼─────┐ ┌──────────┐ ┌───────────┐               │
│  │ Blender  │ │ ComfyUI  │ │ SAM3      │               │
│  │ (Cycles) │ │  :8188   │ │ segment   │               │
│  └──────────┘ └──────────┘ └───────────┘               │
│                                                          │
│  Claude Code (ttyd :7681) — orchestrates entire pipeline │
└────────────────┬────────────────────┬────────────────────┘
                 │ docker exec / shared /data/output volume
     ┌───────────▼──────────┐   ┌────▼──────────────────┐
     │  milo container       │   │  come container        │
     │  (GPU 1)              │   │  (GPU 1)               │
     │                       │   │                        │
     │  MILo (SIGGRAPH 2025) │   │  CoMe (2026-04-22)    │
     │  CUDA 11.8, Python 3.10│  │  CUDA 12.1, Python 3.10│
     │                       │   │  Gated: INSTALL_COME=1 │
     │  + GaussianWrapping   │   │  (dev only; no licence)│
     │  (opt-in build arg;   │   │                        │
     │   dev only; no licence│   │                        │
     │   CUDA 11.8 shared)   │   │                        │
     └───────────────────────┘   └────────────────────────┘
```

## Mesh Extraction Multi-Backend Strategy

The MeshExtraction bounded context (see `research/ddd/bounded-contexts.md`, section 2.5) spans all four containers. Backend selection is a domain policy evaluated in `stages._select_mesh_backend()` at runtime.

| Backend | Module | Container | CUDA | Speed | Thin Structures |
|---------|--------|-----------|------|-------|-----------------|
| TSDF | `mesh_extractor.py` | main | 12.8 | ~5 min | Poor |
| MILo | `milo_extractor.py` | milo sidecar | 11.8 | ~69 min | Moderate |
| CoMe | `come_extractor.py` | come sidecar | 12.1 | ~25 min | Moderate |
| GaussianWrapping | `gaussianwrapping_extractor.py` | milo sidecar | 11.8 | ~30-50 min | Excellent |

Each backend exposes the same three public symbols: `XConfig` dataclass, `is_X_available() -> bool`, and `run_X(colmap_dir, output_dir, config) -> dict`. The `is_X_available()` guard queries the relevant container before committing to a selection, preventing hangs on unavailable sidecars (ADR-003).

**Auto-selection order** when `mesh_method = "auto"`:
1. Preview mode or speed priority → TSDF
2. Thin-structure scene hint and GaussianWrapping available → GaussianWrapping
3. CoMe sidecar available → CoMe (3x faster than MILo at comparable F1)
4. MILo sidecar available → MILo
5. Fallback → TSDF

## Bounded Context Summary

The pipeline is modelled as seven bounded contexts (see `research/ddd/bounded-contexts.md`):

| Context | Key Modules | Produces |
|---------|-------------|----------|
| Ingestion | `frame_selector.py`, `fibonacci_sampler.py` | `FrameSet` |
| Reconstruction | `colmap_parser.py`, `coordinate_transform.py` | `ColmapDataset` |
| Training | `gsplat_trainer.py`, `mcp_client.py` | `GaussianModel` (PLY) |
| Segmentation | `sam2_segmentor.py`, `sam3_segmentor.py`, `mask_projector.py` | `ObjectMask` arrays |
| MeshExtraction | `mesh_extractor.py`, `milo_extractor.py`, `come_extractor.py`, `gaussianwrapping_extractor.py` | `MeshAsset` (GLB) |
| SceneAssembly | `blender_assembler.py`, `usd_assembler.py` | `UsdScene` |
| Delivery | `splat_optimizer.py`, `src/web/` | `.ksplat` + download ZIP |

Orchestration (`stages.py`, `orchestrator.py`) is a published language that crosses all contexts via `StageResult`.

## Claude Code as Orchestrator

Claude Code runs inside the main container (accessible via ttyd on port 7681). It drives the pipeline by:

1. Receiving a job from the Flask web UI
2. Calling pipeline stages in sequence via Python imports
3. Invoking LichtFeld MCP tools for 3DGS training control (70+ tools on :45677)
4. Running Blender headless for scene assembly and Cycles GPU texture baking
5. Optionally calling the MILo sidecar for high-quality mesh extraction
6. Writing results back to `/data/output/JOB_ID/`

The pipeline modules (`src/pipeline/stages.py`) are designed as independent, stateless functions. Claude Code decides what to run next based on each stage's output. There is no hidden state machine.

## Data Flow

```
/data/output/JOB_ID/
├── input.mp4                    # Uploaded video
├── frames/                      # Extracted JPEG frames
├── colmap/                      # COLMAP sparse model + undistorted images
│   ├── images/
│   └── sparse/0/
├── training/                    # 3DGS training output
│   └── point_cloud.ply
├── segmentation/                # SAM3 per-object masks
├── objects/
│   ├── gaussians/               # Per-object PLY splats
│   └── meshes/                  # Per-object GLB meshes (TSDF or MILo)
├── blender/                     # Blender scene + baked textures
├── usd/                         # USD scene hierarchy
├── previews/                    # Blender-rendered preview images
└── download/                    # ZIP bundle for web download
```

## Key Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| LichtFeld Studio | v0.5.2 (synced 2026-05-26) | 3DGS training + MRNF densification, MCP server (70+ tools), native USD I/O |
| COLMAP | 4.1.0 | Structure-from-Motion |
| Open3D | 0.18+ | TSDF fusion, mesh processing |
| MILo | latest (SIGGRAPH Asia 2025) | High-quality mesh extraction (milo sidecar) |
| CoMe | initial release 2026-04-22 | Confidence-based mesh extraction (come sidecar; dev/opt-in) |
| GaussianWrapping | latest (pushed 2026-05-19) | Thin-structure mesh extraction (milo sidecar; dev/opt-in) |
| splat-transform | `@playcanvas/splat-transform` (npm) | Splat compression + format conversion |
| SAM3 | latest | Concept segmentation (4M concepts) |
| Blender | 5.0.1 | Scene assembly, Cycles GPU texture bake |
| Flask | 3.x | Web interface |
| PyAV | latest | Video frame extraction |
| gsplat | latest | Depth rendering for TSDF |
| OpenUSD | 25.02+ | USD scene export |
| Node.js | system | splat-transform (npm/npx) runtime |
| NumPy | pipeline dep | Fibonacci-sphere scoring, mask projection |
