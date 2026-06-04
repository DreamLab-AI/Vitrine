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

---

## v3 End-to-End Architecture (Proposed)

> Everything in this section is **design, not shipped code**. The v2 deployment above is current.
> The v3 design is governed by ADR-012 through ADR-015 in `research/decisions/`.

### Single-manifest input: `exhibit.toml`

The pipeline's one human-authored input is a TOML manifest. It carries:

- `[exhibit]` — project-level identity (id, name, venue, date, curator, description) that flows into USD metadata.
- `[[objects]]` — the list of objects the agent will decompose and recover; each has a stable `id`, a `sam3_concept` string, and a `priority` (`key` | `standard`). Key-priority objects trigger the full hull-reconstruction + FLUX.2 recovery path.
- `[drive]` — Google Drive source folder URL, rclone remote name, and (v3) a `writeback = true` flag so finished artifacts are uploaded back to the same Drive folder.
- `[secrets]` — `env:NAME` references only. Credentials are never inlined, never written to the JSON run snapshot.
- `[pipeline]` — optional overrides onto `PipelineConfig` SOTA defaults (mesh backend, matcher, etc.).
- `[oversight]` — selects the pipeline overseer (see below).
- `[models]` — hardware-selected model/quant choices written by Vitrine Onboarding.

A `manifest.py` loader parses this, resolves `env:` references, and materialises the existing `PipelineConfig` as the runtime artifact. The manifest is the source; the JSON snapshot is the run record. ADR-013.

### Orchestrator and tool: Claude Code + gemma-4

The in-container **Claude Code** agent (accessible via ttyd on port 7681) remains the pipeline overseer. It drives the stateless `stages.py` functions, calls LichtFeld MCP tools, and works around failures end-to-end. There is no hidden state machine.

**gemma-4-26B-A4B** is a unified multimodal vision tool the orchestrator calls — not a second orchestrator. It serves both per-frame artifact triage (FR-27: detecting motion ghosting, specular blowout, rolling-shutter skew, transient occluders) and metadata-fused reasoning (FR-28). The model is vision-capable: architecture `Gemma4ForConditionalGeneration` with a SigLIP vision encoder and `mmproj` projector. It is deployed as a containerised `agent-vlm` service on `v2g-net`. Qwen2.5-VL is retained as an optional fallback behind a capability probe.

The `[oversight].backend` field in the manifest selects the overseer:

- `claude_code` **(default)** — the in-container Claude Code agent, no local GPU cost; requires the user to log in to Claude Code inside the container once (session persists in the `claude-session` volume).
- `gemma_local` — the local gemma-4 model also acts as overseer; fully on-host, no API key, but creates GPU-contention with heavy generative stages.

The `[oversight].artifact_vlm` field (independent of `backend`) selects what performs bulk per-frame triage: `gemma_local` (default; transient, loaded for the artifact stage then unloaded) or `claude_code` (higher API cost on large frame sets).

ADR-013, sections D-013.5 and D-013.6.

### Serial model lifecycle and VRAM bounding

A `ModelLifecycleManager` wraps each pipeline stage as a context manager. Each stage declares a `ModelSpec` with a VRAM estimate and an unload tier:

- **Soft unload** (default) — in-process free via ComfyUI `POST /free`, llama.cpp/vLLM model unload, `torch.cuda.empty_cache()`. Fast, container stays warm.
- **Hard unload** — `docker stop` / `docker start` on the service container. Full driver-level VRAM reclamation; used selectively for back-to-back heavy stages (FLUX.2 → Hunyuan3D) where soft-free leaves fragmentation.

Peak VRAM = `max(stage VRAM)`, not `sum(stages)`. This is the only way the full SOTA model set (FLUX.2-dev ~32 GB fp8 + Hunyuan3D-2.1 ~16 GB + gemma-4 ~20 GB Q5_K_M + SAM3 ~8 GB) fits a single-host budget. ADR-013, section D-013.2.

| Stage | Model | Engine | ~VRAM | Unload tier |
|-------|-------|--------|-------|-------------|
| Frame artifact triage + reasoning | gemma-4-26B-A4B (unified) | `agent-vlm:8080` | ~20 GB (Q5_K_M) | soft |
| SfM matching | ALIKED + LightGlue | in-proc torch | < 4 GB | soft |
| 3DGS training | LichtFeld / gsplat | native | scales with scene | n/a |
| Decomposition | SAM3 | in-proc torch | ~8 GB | soft |
| Inpaint / occluded-face recovery | FLUX.2-dev fp8mixed | `comfyui:8188` | ~32 GB | **hard** |
| Hull reconstruction | Hunyuan3D-2.1 | `comfyui:8188` | ~16 GB | **hard** |
| Mesh extraction | MILo (default) / CoMe | sidecar | sidecar GPU | n/a |

### `v2g-net` Docker mesh

Hardcoded `192.168.2.48:port` endpoints are replaced by a user-defined Docker bridge network on the GPU host. Services resolve by DNS name:

```
v2g-net (bridge)
├── comfyui        :8188   FLUX.2-dev (fp8mixed) + Hunyuan3D-2.1 + Mistral-3 enc + FLUX.2 VAE
│                  :3001   Salad add-on control-plane API (model probe / lifecycle)
├── agent-vlm      :8080   gemma-4-26B-A4B (multimodal) — artifact VLM + reasoner
├── gaussian-toolkit       orchestrator (addresses peers as http://comfyui:8188 etc.)
├── milo  (sidecar)        device_ids ['1'] — docker exec
└── come  (sidecar)        device_ids ['1'] — docker exec (gated, non-commercial)
```

The `ModelLifecycleManager` hard-tier uses `docker stop` / `docker start` on `comfyui` and `agent-vlm` on this network to guarantee VRAM reclamation. ADR-013, section D-013.3.

### Agent-controlled ComfyUI recovery

The existing `.48` ComfyUI is updated to a pinned state via the **Salad add-on control API** (`comfyui:3001`) — an in-container control plane for model probe, download, and lifecycle, not a cloud service. The `RecoveryController` (a stateless helper the orchestrator invokes) runs a per-object loop:

1. **Plan** — compose the FLUX.2-dev inpaint or Hunyuan3D-2.1 hull graph from a template, parameterised by object identity and the artifact report.
2. **Submit** — `POST /prompt` (graph API); poll `/history/{id}`; fetch outputs.
3. **Evaluate** — gemma-4 (vision tool) scores the generated image or mesh render against object identity and artifact criteria.
4. **Decide** — `accept` | `re-prompt` (adjust denoise/guidance/seed/mask, bounded retry budget) | `veto` (unrecoverable). Every attempt is annotated in the per-video ledger; nothing is silently dropped.
5. **Release** — Salad control free/unload the stage model before the next stage.

ADR-014. See also [v3 Pipeline Design](architecture/v3-pipeline.md) for the full `v2g-net` diagram.

### SOTA tooling defaults (ADR-012)

| Axis | v2 default | v3 default (proposed) |
|------|-----------|----------------------|
| Mesh backend | TSDF | **MILo** (SIGGRAPH Asia 2025) |
| SfM matching | SIFT exhaustive | **ALIKED + LightGlue** |
| Hull reconstruction | Hunyuan3D-2.0 | **Hunyuan3D-2.1** |
| Inpainting | FLUX.1-Fill-dev | **FLUX.2-dev** (fp8mixed) |

Fallbacks remain available (TSDF, SIFT, Hunyuan3D-2.0, FLUX.1-Fill) behind capability probes so a missing checkpoint degrades gracefully rather than halting. ADR-012.
