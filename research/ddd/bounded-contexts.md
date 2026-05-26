# Bounded Contexts — Gaussian Toolkit v2

**Extends**: research/decisions/ddd-domain-model.md (v1 model)
**Alignment**: research/decisions/prd-v2-upgrade.md (v2 north star)
**Date**: 2026-05-26

---

## 1. Context Map

The following diagram shows all bounded contexts in the v2 pipeline and their
relationships. Arrows indicate information flow direction (upstream -> downstream).
Relationship type is annotated on each edge.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  EXTERNAL SYSTEMS (outside the fork boundary)                           │
 │                                                                         │
 │   [LichtFeld Core]        [COLMAP]         [ComfyUI]       [Blender]   │
 │   C++ MCP server          SfM binary        HTTP API        subprocess  │
 └──────┬──────────────────────┬──────────────────┬───────────────┬───────┘
        │ Conformist           │ ACL              │ ACL           │ ACL
        │                      │                  │               │
 ┌──────▼──────────────────────▼──────────────────▼───────────────▼───────┐
 │  CORE DOMAIN                                                            │
 │                                                                         │
 │  ┌──────────────┐  Shared   ┌──────────────────┐                        │
 │  │  Ingestion   │  Kernel   │  Reconstruction  │                        │
 │  │              ├──────────►│  (COLMAP SfM)    │                        │
 │  │  frame_      │  FrameSet │                  │                        │
 │  │  selector.py │           │  colmap_parser   │                        │
 │  │  frame_      │           │  coordinate_     │                        │
 │  │  quality.py  │           │  transform.py    │                        │
 │  └──────────────┘           └────────┬─────────┘                        │
 │                                      │ Customer (ColmapDataset)          │
 │                                      │                                  │
 │                             ┌────────▼─────────┐                        │
 │                             │    Training      │                        │
 │                             │   (3DGS / 4D)    │                        │
 │                             │                  │                        │
 │                             │  gsplat_trainer  │                        │
 │                             │  mcp_client.py   │                        │
 │                             │  (MRNF/MCMC/IGS+)│                        │
 │                             └────────┬─────────┘                        │
 │                                      │ Customer (GaussianPLY)            │
 │                      ┌───────────────┴──────────────────┐               │
 │                      │                                  │               │
 │             ┌────────▼─────────┐            ┌───────────▼──────────┐   │
 │             │  Segmentation    │            │  MeshExtraction      │   │
 │             │  (SAM2/3)        │            │  (multi-backend)     │   │
 │             │                  │ Shared     │                      │   │
 │             │  sam2_segmentor  │ Kernel     │  mesh_extractor.py   │   │
 │             │  sam3_segmentor  ├───────────►│  milo_extractor.py   │   │
 │             │  sam3d_client    │  ObjectMask│  [come_extractor.py] │   │
 │             │  mask_projector  │            │  [gaussianwrapping_  │   │
 │             └──────────────────┘            │   extractor.py]      │   │
 │                                             └──────────┬───────────┘   │
 │                                                        │ ACL (sidecar)  │
 │                                                        │               │
 │  ┌─────────────────────────────────────────────────────▼─────────────┐ │
 │  │  SceneAssembly                                                     │ │
 │  │                                                                    │ │
 │  │  blender_assembler.py   usd_assembler.py   texture_baker.py        │ │
 │  │  material_assigner.py   multiview_renderer.py                      │ │
 │  └─────────────────────────────────────────────────────┬─────────────┘ │
 │                                                         │ OHS (USD PL)   │
 │                                             ┌───────────▼──────────┐   │
 │                                             │  Delivery            │   │
 │                                             │  (splat-transform /  │   │
 │                                             │   web)               │   │
 │                                             │                      │   │
 │                                             │  [splat_optimizer.py]│   │
 │                                             │  src/web/app.py      │   │
 │                                             └──────────────────────┘   │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘

 Orchestration crosses all contexts via stages.py / orchestrator.py (published language)
```

---

## 2. Context Definitions

### 2.1 Ingestion Context

**Responsibility**: Accept a raw video file and produce a curated, quality-filtered
frame set ready for SfM.

**Key modules**:
- `stages.py::ingest()` — ffmpeg frame extraction
- `frame_selector.py` — blur/exposure/viewpoint scoring, target-count selection
- `frame_quality.py` — per-frame sharpness and exposure metrics
- `stages.py::remove_people()` — person occlusion removal via `person_remover.py`

**v2 additions**:
- `fibonacci_sampler.py` (planned) — Fibonacci-sphere viewpoint scoring replaces
  sequential stride sampling inside `frame_selector.py`. Selected via
  `config.ingest.frame_selection_strategy = "fibonacci" | "sequential"`.

**Produced artifact**: `FrameSet` — directory of curated JPEG frames with
quality scores; written to `{job_dir}/frames_selected/`.

**Ubiquitous language**: frame, blur score, exposure gate, viewpoint coverage,
Fibonacci sampling, person mask.

---

### 2.2 Reconstruction Context (COLMAP SfM)

**Responsibility**: Recover camera poses and a sparse 3D point cloud from the
frame set. Produce a COLMAP dataset (images + sparse/) consumed by every
downstream context.

**Key modules**:
- `stages.py::reconstruct()` — orchestrates SplatReady or direct COLMAP
- `colmap_parser.py` — reads COLMAP binary format (cameras.bin, images.bin,
  points3D.bin)
- `coordinate_transform.py` — applies Y-up / meters coordinate normalisation
  **CAUTION**: upstream PR #1066 (master only, not v0.5.2) may shift conventions

**External dependency**: COLMAP 4.1.0 binary (CUDA sm_89, headless).
Wrapped behind an ACL; see `anti-corruption-layers.md` for the COLMAP adapter.

**Quality gate**: registration rate >= 30% of frames or job fails.

**Produced artifact**: `ColmapDataset` — path to `{job_dir}/colmap/undistorted/`
containing `images/`, `sparse/0/{cameras,images,points3D}.bin`.

**Ubiquitous language**: sparse model, camera pose, intrinsics, extrinsics,
registration rate, reprojection error, undistorted images.

---

### 2.3 Training Context (3DGS)

**Responsibility**: Train a Gaussian Splatting model from a ColmapDataset.
Produce a trained Gaussian PLY and optional preview renders.

**Key modules**:
- `stages.py::train()` — dispatches to LichtFeld or MILo backend
- `gsplat_trainer.py` — direct gsplat Python API training (alternative path)
- `mcp_client.py` — JSON-RPC calls to LichtFeld MCP server on port 45677
- `multiview_renderer.py` — multi-view depth/RGB rendering for TSDF input
- `stages.py::render_previews()` — gsplat-rendered previews for web carousel

**Training strategies** (domain policy, configured via `config.training.strategy`):
- `mrnf` — MRNF densification (upstream v0.5.2+, new default)
- `mcmc` — MCMC stochastic densification (good for indoor reflective scenes)
- `igs_plus` — IGS+ (legacy)
- `default` — gsplat DefaultStrategy (v1 legacy)

**v2 additions**:
- MRNF as new default strategy (replaces LFS/DefaultStrategy after upstream v0.5.2 sync)
- Scene preset `indoor_reflective` forces MCMC + opacity/scale regularisation
- `dynamic_trainer.py` (Phase 3): 4C4D training path for scenes with motion

**Relationship to LichtFeld Core**: Conformist. The Training context adapts
to LichtFeld's MCP API as-is. Changes to LichtFeld's training API (e.g. post
#984 enhanced MCP) flow into `mcp_client.py` without structural changes to
this context.

**Produced artifact**: `GaussianModel` — trained `.ply` at
`{job_dir}/model/point_cloud/iteration_NNNNN/point_cloud.ply`
(or `{job_dir}/model_milo/` for MILo path).

**Ubiquitous language**: splat, Gaussian, PLY, SH degree, densification,
opacity, scale, training iteration, PSNR, SSIM, scene preset.

---

### 2.4 Segmentation Context (SAM2/3)

**Responsibility**: Label Gaussians by semantic object class. Produce per-object
masks that downstream MeshExtraction uses to carve the scene.

**Key modules**:
- `sam2_segmentor.py` — SAM2 video tracking segmentation
- `sam3_segmentor.py` — SAM3 concept-prompted segmentation (4M concepts)
- `sam3d_client.py` — HTTP client to SAM3 server
- `mask_projector.py` — projects 2D frame masks onto 3D Gaussian positions

**Policy**: SAM3 is preferred; falls back to SAM2 (`config.decompose.sam3_fallback_to_sam2`);
falls back to full-scene (single-object) if both fail.

**Shared Kernel with MeshExtraction**: `ObjectMask` — a per-object numpy mask
array saved as `{job_dir}/sam3_masks/mask_{id:04d}.npy`. This is the only data
structure shared across context boundaries as a kernel (not through events).

**Produced artifact**: ordered list of `{label, object_id, mask_pixels}` dicts
(JSON in StageResult artifacts) + mask files on disk.

**Ubiquitous language**: concept prompt, mask, label, object_id, segmentation
confidence, 3D label projection.

---

### 2.5 MeshExtraction Context (multi-backend)

**Responsibility**: Convert per-object Gaussian PLYs (or full-scene ColmapDatasets)
into polygonal meshes with textures. This is the most structurally complex
context in v2 because it now spans four backends with different infrastructure
requirements.

**Key modules**:
- `mesh_extractor.py` — TSDF primary path (gsplat depth rendering + Open3D TSDF);
  also contains Poisson and marching-cubes fallbacks
- `milo_extractor.py` — MILo sidecar client; calls `docker exec milo` or
  conda fallback
- `come_extractor.py` (planned v2) — CoMe sidecar client; marching tetrahedra
- `gaussianwrapping_extractor.py` (planned v2) — GaussianWrapping sidecar client;
  stochastic surface elements for thin structures
- `mesh_cleaner.py` — decimation, hole filling, degenerate face removal
- `texture_baker.py` — xatlas UV unwrap + vertex-colour-to-texture baking
- `material_assigner.py` — USD PreviewSurface material assignment

#### Backend Registry (domain policy: BackendSelector)

The choice of mesh extraction backend is a domain policy, not an implementation
detail. The policy is evaluated at runtime from scene metadata and configuration.

```
BackendSelector policy:
  IF config.training.mesh_method == "milo":
      → MILo (joint train+mesh, skips standalone mesh_objects stage)
  ELIF scene_type == "preview":
      → TSDF (fastest, gsplat depth render path in mesh_extractor.py)
  ELIF scene_contains_thin_structures:         # future: heuristic or flag
      → GaussianWrapping (stochastic surface elements)
  ELIF config.mesh.backend == "come":
      → CoMe (marching tetrahedra, 3x faster than MILo for quality meshes)
  ELIF config.mesh.backend == "milo":
      → MILo standalone (post-training mesh call)
  ELSE:
      → TSDF (default)
```

**Backend comparison** (v2):

| Backend | Module | Container | CUDA | Speed | Thin Structures | Quality |
|---------|--------|-----------|------|-------|-----------------|---------|
| TSDF | `mesh_extractor.py` | main (CUDA 12.8) | yes | Fast (~5 min) | Poor | Adequate |
| MILo | `milo_extractor.py` | milo sidecar (CUDA 11.8) | yes | Slow (~69 min) | Moderate | High |
| CoMe | `come_extractor.py` | come sidecar (CUDA 12.1) | yes | Medium (~25 min) | Moderate | High |
| GaussianWrapping | `gaussianwrapping_extractor.py` | milo sidecar (CUDA 11.8, shared) | yes | Medium | Excellent | High |

**v2 structural change**: In v1 the MeshExtraction context had one primary
path (TSDF) and one alternative (MILo). In v2 it becomes a pluggable
multi-backend strategy with a policy-driven selector. Each backend is an
interchangeable implementation of the same `MeshBackend` interface
(see `aggregates.md` for the `MeshAsset` aggregate and its invariants).

**Sidecar ACL**: Each non-TSDF backend communicates with an isolated sidecar
container over `docker exec`. The sidecar boundary is an ACL translating
domain commands (`extract(colmap_dir, output_dir, config)`) into
container-specific CLI invocations. See `anti-corruption-layers.md` for
the full sidecar adapter design.

**Produced artifact**: per-object `MeshAsset` — GLB + OBJ + optional diffuse
texture at `{job_dir}/objects/meshes/{label}/`.

**Ubiquitous language**: mesh backend, marching tetrahedra, TSDF, confidence
field, thin structure, watertight mesh, vertex-coloured mesh, UV atlas, texture
bake, decimation.

---

### 2.6 SceneAssembly Context

**Responsibility**: Compose per-object meshes and Gaussian PLYs into a single
coherent USD scene with correct coordinate frame, materials, and variant sets.

**Key modules**:
- `usd_assembler.py` — Python OpenUSD scene composition (v1 primary path)
- `blender_assembler.py` — Blender subprocess for Cycles GPU bake and
  high-quality material assignment (v1 alternative path)
- `texture_baker.py` — xatlas UV + colour bake (shared with MeshExtraction)
- `material_assigner.py` — UsdPreviewSurface material population
- `multiview_renderer.py` — multi-view renders for texture reprojection

**v2 additions**:
- After upstream v0.5.2 sync: LichtFeld native USD I/O (`mcp_client.py` calls
  to LichtFeld's USD export tool) may replace or supplement `usd_assembler.py`.
  Decision pending test of hierarchical-prim support (PRD open question 3).
- Phase 3: `temporal_exporter.py` — Alembic point cache for dynamic prims,
  merged into the USD scene via USD Alembic plugin.

**Relationship to LichtFeld Core**: For USD export, this context calls
LichtFeld's MCP tools via `mcp_client.py`. This makes the context a Conformist
to LichtFeld's USD schema. If LichtFeld native USD proves insufficient, the
context falls back to `usd_assembler.py` (our own OpenUSD code) — the fallback
is an ACL isolating us from LichtFeld's schema choices.

**Produced artifact**: `UsdScene` — `.usda`/`.usdc` file hierarchy at
`{job_dir}/scene/`, plus per-object variant sets (gaussian | mesh).

**Ubiquitous language**: USD prim, prim path, variant set, up-axis, meters per
unit, Xform, UsdGeomMesh, UsdPreviewSurface, Alembic cache.

---

### 2.7 Delivery Context (splat-transform / web)

**Responsibility**: Package pipeline outputs for human consumption via a web
browser: compress Gaussian splats for download, serve the web UI, and present
3D previews.

**Key modules**:
- `splat_optimizer.py` (planned v2) — CLI wrapper around `splat-transform`
  (PlayCanvas npm package): crop, filter, sort, compress `.ply` → `.ksplat`
- `src/web/app.py` — Flask application on port 7860
- `src/web/job_manager.py` — SSE job progress streaming
- `src/web/pipeline_runner.py` — async pipeline invocation from web UI
- `src/web/static/` — 3D viewer assets (model-viewer, planned 2Xplat/SuperSplat)

**v2 additions**:
- `splat_optimizer.py` — first new Delivery module. Runs after Training, before
  SceneAssembly. Produces compressed `.ksplat` alongside full PLY.
- Splat viewer upgrade: replace `model-viewer` with 2Xplat or SuperSplat for
  native Gaussian splat preview (PLY/ksplat rendering in browser).
- KHR_gaussian_splatting glTF extension export for interoperability.
- LOD delivery: progressive loading of splat tiles via splat-transform.

**Relationship**: Open Host Service (OHS). The Delivery context publishes the
pipeline's products using open interchange formats (USD, GLB, PLY, ksplat,
glTF) and standard protocols (HTTP, SSE). It does not own any domain logic;
it is a pure adapter from domain artifacts to user-facing resources.

**Ubiquitous language**: ksplat, splat compression, LOD, web viewer, download
bundle, KHR_gaussian_splatting, model-viewer, progressive loading.

---

## 3. Cross-Cutting Concerns

### 3.1 Orchestration (Published Language)

`stages.py` and `orchestrator.py` are not a separate bounded context but a
published language that all contexts speak. `StageResult` is the canonical
message type crossing context boundaries: `{success, stage, metrics, artifacts, error}`.

The stage sequence in `STAGE_NAMES` defines the pipeline's published language
ordering:
```
ingest → remove_people → select_frames → reconstruct → train →
render_previews → segment → extract_objects → mesh_objects →
texture_bake → assemble_usd → validate
```

v2 adds:
```
... → train → splat_optimize → render_previews → segment → ...
```

`quality_gates.py` implements cross-context quality decisions (PSNR thresholds,
registration rate gates, mesh vertex count minimums).

### 3.2 Sidecar Container Boundary (Shared Infrastructure)

The MILo and GaussianWrapping sidecars share a single Docker container
(`docker/Dockerfile.milo`, CUDA 11.8, Python 3.9). The CoMe sidecar is
separate (`docker/Dockerfile.come`, CUDA 12.1, Python 3.10). The 4C4D sidecar
(Phase 3) will be a fourth container. The main container runs CUDA 12.8,
Python 3.12.

This physical boundary (container) is not a bounded context boundary; it is
infrastructure. The MeshExtraction context spans all four containers, with the
sidecar ACL translating between them.

### 3.3 Fork Boundary (Upstream vs. Ours)

`BOUNDARIES.md` is the physical manifestation of the ACL between our fork and
upstream LichtFeld Studio. See `anti-corruption-layers.md` for the formal ACL
model of this boundary.

---

## 4. Relationship Types Summary

| Pair | Type | Notes |
|------|------|-------|
| Ingestion → Reconstruction | Shared Kernel | `FrameSet` (frames directory) |
| Reconstruction → Training | Customer-Supplier | Training consumes ColmapDataset format |
| Training → Segmentation | Customer-Supplier | Segmentation consumes Gaussian PLY |
| Training → MeshExtraction | Customer-Supplier | MILo backend uses ColmapDataset directly |
| Segmentation → MeshExtraction | Shared Kernel | `ObjectMask` numpy arrays on disk |
| MeshExtraction → SceneAssembly | Customer-Supplier | SceneAssembly assembles MeshAssets |
| SceneAssembly → Delivery | OHS / Published Language | USD + PLY + GLB over file system |
| Delivery → (end user) | OHS | HTTP / SSE / download |
| Training → LichtFeld Core | Conformist | Adapts to MCP API without question |
| Reconstruction → COLMAP | ACL | `colmap_parser.py` + subprocess adapter |
| MeshExtraction → sidecar containers | ACL | `milo_extractor.py` / planned come/gw adapters |
| SceneAssembly → Blender | ACL | `blender_assembler.py` subprocess adapter |
| SceneAssembly → ComfyUI | ACL | `comfyui_inpainter.py` HTTP adapter |
| Our fork → LichtFeld upstream | ACL | `BOUNDARIES.md` + merge policy |
