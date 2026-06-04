# Vitrine / Gaussian Toolkit Architecture

> **Rename note (2026-06-04):** The pipeline product is renamed from **Video-to-Gaussian** to
> **Vitrine**. Code identifiers still say `video2splat` / `gaussian-toolkit`; a full code rename is a
> separate scheduled follow-up (ADR-015).

> **Boundary note:** Vitrine (Gaussian Toolkit) is our fork of [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (MrNeRF). LichtFeld Studio is the upstream product. We add the video-to-scene pipeline, web interface, Docker deployment, and research. See [BOUNDARIES.md](../../BOUNDARIES.md) for the complete separation policy and the decision framework for where new code belongs.

## System Overview (v2 — current)

> See also: [docs/architecture.md](../architecture.md) for the three-container deployment architecture and the **v3 End-to-End Architecture (Proposed)** section.
>
> For the v3 pipeline design reference, see [architecture/v3-pipeline.md](v3-pipeline.md).

Gaussian Toolkit integrates multiple components into a unified 3D Gaussian Splatting pipeline running in two Docker containers on dual RTX 6000 Ada GPUs (96GB total VRAM).

```
┌───────────────────────────────────────────────────────┐
│  gaussian-toolkit container (GPU 0)                    │
│  Ubuntu 24.04 / CUDA 12.8 / Python 3.12               │
├───────────────────────────────────────────────────────┤
│  COLMAP SfM → LichtFeld 3DGS → SAM3 Segmentation     │
│  → TSDF Mesh → Blender Assembly → USD Export          │
│                                                        │
│  Web UI :7860 | ComfyUI :8188 | MCP :45677            │
│  VNC :5901   | ttyd :7681                             │
│  Claude Code (agentic orchestrator)                    │
└────────────────┬──────────────────────────────────────┘
                 │ docker exec / shared /data/output
┌────────────────▼──────────────────────────────────────┐
│  milo container (GPU 1)                                │
│  Ubuntu 22.04 / CUDA 11.8 / Python 3.10               │
│  MILo mesh extraction (SIGGRAPH Asia 2025)             │
└───────────────────────────────────────────────────────┘
                 ▲
                 │ lfs-mcp CLI / MCP bridge / Web UI
                 ▼
         Claude Code / Agents
```

## Component Stack

| Component | Version | License | Purpose |
|-----------|---------|---------|---------|
| LichtFeld Studio | 0.4.2+ | GPL-3.0 | 3DGS training, visualisation, editing, export |
| COLMAP | 4.1.0 | BSD | Structure-from-Motion reconstruction |
| SplatReady | 1.0.0 | Plugin | Video-to-COLMAP pipeline automation |
| SAM3 | latest | Apache-2.0 | Concept segmentation (4M concepts, text+visual prompts) |
| SAM2 | hiera-large | Apache-2.0 | Video segmentation (fallback, validated) |
| Hunyuan3D 2.0 | latest | Tencent | Multi-view to textured mesh creation |
| ComfyUI | latest | GPL-3.0 | Node-based workflow engine (SAM3D/TRELLIS nodes) |
| FLUX | dev/schnell | Apache-2.0 | Inpainting via ComfyUI workflows |
| Open3D | 0.18+ | MIT | TSDF fusion, mesh processing |
| OpenUSD | 25.02+ | Modified Apache 2.0 | USD scene composition |
| METIS | 5.2.1 | Apache-2.0 | Graph partitioning (COLMAP dependency) |
| Flask | 3.x | BSD | Web upload interface on :7860 |
| vcpkg | latest | MIT | C++ dependency management (91 packages) |

## Data Flow

### Full Pipeline (video2splat -> scene assembly)

```
Video File (.mp4/.mov) or Web Upload (:7860)
    │
    ▼ [Stage 1: SplatReady - PyAV frame extraction]
JPEG Frames + GPS EXIF
    │
    ▼ [Stage 2: COLMAP - 6-step SfM pipeline]
    │   feature_extractor → exhaustive_matcher → mapper
    │   → model_aligner → image_undistorter → model_converter
    │
    ▼
COLMAP Undistorted Dataset
    │   images/ + sparse/0/{cameras,images,points3D}.txt
    │
    ▼ [Stage 3: LichtFeld - CUDA-accelerated training]
Trained 3D Gaussian Splat Model (1M gaussians)
    │
    ▼ [Stage 4: SAM3 Concept Segmentation]
    │   text+visual prompts, 4M concepts
    │   (fallback: SAM2 grid-point prompts)
    │
    ▼ [Stage 5: Mask Projection to 3D]
    │   2D masks → 3D Gaussian labels (98.3% coverage)
    │   33 per-object PLY files extracted
    │
    ▼ [Stage 6: Per-Object Mesh Creation]
    │   Hunyuan3D 2.0: multi-view renders → textured mesh
    │   TSDF fallback: Open3D fusion (22K verts, 49K faces)
    │
    ▼ [Stage 7: Background Recovery]
    │   FLUX inpainting via ComfyUI (:8188)
    │
    ▼ [Stage 8: USD Scene Assembly]
USD Scene (59 prims, variant sets: Gaussian + Mesh per object)
    + PLY / SOG / SPZ / HTML exports
```

### MCP-Controlled Pipeline

```
Claude Agent
    │
    ├─► lfs-mcp call scene.load_dataset {"path": "..."}
    ├─► lfs-mcp call training.start
    ├─► lfs-mcp call training.get_state  (poll loop)
    ├─► lfs-mcp call render.capture {"width": 1920}
    ├─► lfs-mcp call selection.by_description {"description": "floaters"}
    ├─► lfs-mcp call gaussians.write {"delete_selected": true}
    └─► lfs-mcp call scene.export_spz {"path": "output.spz"}
```

## Two-Container Docker Architecture

The pipeline runs in two Docker containers: a main container for the full pipeline and a MILo sidecar for high-quality mesh extraction.

```
┌─────────────────────────────────────────────────────────┐
│  gaussian-toolkit (GPU 0)                                │
│  Ubuntu 24.04 / CUDA 12.8 / Python 3.12                 │
├─────────────────────────────────────────────────────────┤
│  Services (supervisord):                                  │
│    Web UI (:7860)        - Flask upload + job manager     │
│    ComfyUI (:8188)       - SAM3D/TRELLIS nodes           │
│    LichtFeld MCP (:45677) - 70+ tools                    │
│    ttyd (:7681)          - Web terminal (Claude Code)    │
│    VNC (:5901)           - Remote desktop (Blender)      │
├─────────────────────────────────────────────────────────┤
│  Pipeline (28 modules):                                   │
│    stages, orchestrator, cli, config, preflight,          │
│    sam2_segmentor, sam3_segmentor, sam3d_client,          │
│    mask_projector, mesh_extractor, milo_extractor,       │
│    mesh_cleaner, blender_assembler, usd_assembler,       │
│    texture_baker, material_assigner, mcp_client,         │
│    multiview_renderer, hunyuan3d_client,                 │
│    comfyui_inpainter, quality_gates, frame_quality,      │
│    frame_selector, colmap_parser, coordinate_transform,  │
│    person_remover, __init__, __main__                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  milo (GPU 1) — sidecar, called via docker exec          │
│  Ubuntu 22.04 / CUDA 11.8 / Python 3.10                 │
├─────────────────────────────────────────────────────────┤
│  MILo (SIGGRAPH Asia 2025)                               │
│  Differentiable mesh-in-the-loop gaussian splatting      │
│  4 rasterizer variants + nvdiffrast + Delaunay           │
└─────────────────────────────────────────────────────────┘
```

## GPU Architecture

- **CUDA Toolkit**: 13.1
- **Target architectures**: sm_89 (RTX 6000 Ada), sm_86 (RTX A6000/3090), sm_75 (RTX 6000/2080)
- **C++ Standard**: C++23
- **CUDA Standard**: C++20
- **Build system**: CMake + Ninja + vcpkg

## MCP Server Architecture

```
scripts/lichtfeld_mcp_bridge.py    <-- stdio MCP client (Claude Desktop/Codex)
        │
        │ HTTP POST to http://127.0.0.1:45677/mcp
        ▼
src/mcp/mcp_http_server.cpp        <-- cpp-httplib HTTP listener
        │
        ▼
src/mcp/mcp_server.cpp             <-- JSON-RPC 2.0 dispatcher
        │
        ├──► ToolRegistry (singleton)       70+ tools
        └──► ResourceRegistry (singleton)   8+ resources
```

### Tool Runtime Model

Tools are registered in two backends depending on the application mode:

- **Headless**: `TrainingContext` singleton manages scene/trainer directly
- **GUI**: `Visualizer` provides the backend with live viewport interaction

Each tool carries metadata:
- `category` (training, scene, render, selection, etc.)
- `kind` (command vs query)
- `runtime` (shared, headless, gui)
- `thread_affinity` (any, training_context, main_thread)
- `destructive`, `long_running`, `user_visible` flags

## Directory Layout

See [BOUNDARIES.md](../../BOUNDARIES.md) for the authoritative ownership map. Summary:

```
LichtFeld-Studio/
├── src/
│   ├── core/              # UPSTREAM (LichtFeld) — do not modify
│   ├── app/               # UPSTREAM
│   ├── mcp/               # UPSTREAM
│   ├── rendering/         # UPSTREAM
│   ├── training/          # UPSTREAM
│   ├── pipeline/          # OURS — 24 Python pipeline modules
│   └── web/               # OURS — Flask web interface (:7860)
├── research/              # OURS — 15 research documents (not product)
├── docker/                # OURS — Docker configuration
├── scripts/               # OURS — Utility scripts and test harnesses
├── Dockerfile.consolidated         # OURS — Consolidated container
├── docker-compose.consolidated.yml # OURS — Single-command deployment
├── GAUSSIAN_TOOLKIT_README.md      # OURS — Authoritative fork README
├── BOUNDARIES.md                   # OURS — Ownership and separation policy
├── docs/                  # Mixed (architecture/* is ours, upstream docs exist)
├── external/              # UPSTREAM — Git submodules
├── README.md              # UPSTREAM — Do not overwrite
└── build/                 # Build output (not committed)
```
