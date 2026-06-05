# Engineering Log

Development history and key decisions for the Gaussian Toolkit fork of LichtFeld Studio.

---

## Phase 1: Foundation

### LichtFeld Studio Fork

Forked [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio), a native C++23/CUDA workstation for 3D Gaussian Splatting. LichtFeld provides training, visualization, editing, and export with an MCP server exposing 70+ tools. We chose it over alternatives (gsplat standalone, nerfstudio) because:

- MCP integration allows agentic control from Claude Code
- Native C++ performance for training (not Python-bound)
- Built-in scene graph, selection system, and multi-format export (PLY, SOG, SPZ, HTML, USD)

Established [BOUNDARIES.md](../BOUNDARIES.md) to enforce clean separation: upstream code is never modified on our branch.

### Docker Consolidation

Built a single consolidated Dockerfile (`Dockerfile.consolidated`) on `nvidia/cuda:12.8.1-devel-ubuntu24.04` containing:

- COLMAP 4.1.0 (built from source with METIS/GKlib)
- LichtFeld Studio (host-compiled binary, bind-mounted)
- Python 3.12 pipeline modules
- ComfyUI with SAM3D and FLUX nodes
- Claude Code (Node.js 23)
- Blender (headless)
- ttyd web terminal, VNC, supervisord

Single-command deployment: `docker compose -f docker-compose.consolidated.yml up -d`.

### SplatReady Integration

Integrated the SplatReady plugin for automated video-to-COLMAP pipeline: PyAV frame extraction at configurable FPS, automatic COLMAP feature extraction, exhaustive matching, sparse reconstruction, and undistortion.

---

## Phase 2: TSDF Mesh Extraction

### Initial Approach

After 3DGS training produces a gaussian splat model, we need polygonal meshes for downstream use (game engines, USD scenes, web viewers). First approach: render depth maps from the trained gaussians using gsplat, then fuse them into a mesh via Open3D TSDF.

### Implementation

Built `mesh_extractor.py` using:

1. gsplat to render depth + RGB from training viewpoints
2. Open3D `ScalableTSDFVolume` to fuse depth frames
3. Marching cubes to extract a triangle mesh
4. Vertex colour transfer from the gaussian splat

Results: 22K vertices, 49K faces. Geometric accuracy was acceptable for large structures but poor for fine details.

### Vertex Colours vs Texture Baking

TSDF meshes come with vertex colours, not UV-mapped textures. For web delivery this is sufficient (model-viewer handles vertex colours). For production USD scenes, texture baking is needed. Built `texture_baker.py` skeleton using xatlas for UV unwrapping, but deferred full implementation after discovering the quality ceiling.

### Discovery: TSDF Quality Ceiling

TSDF fusion from expected (rendered) depth has a hard quality ceiling. The depth maps from gaussian splatting are noisy at object boundaries and in regions with sparse training views. No amount of TSDF parameter tuning (voxel size, truncation distance, depth scale) fixes this because the problem is in the input signal, not the fusion algorithm.

---

## Phase 3: Mesh Extraction Research

### Methods Evaluated

| Method | Source | Approach | Finding |
|--------|--------|----------|---------|
| SuGaR | Guédon & Lepetit 2024 | Regularise gaussians to lie on surfaces, then Poisson mesh | Good surface alignment but slow (hours). Requires modified training. |
| GOF (Gaussian Opacity Fields) | Yu et al. 2024 | Learn opacity fields, extract level set | Better than TSDF but still limited by training quality |
| MILo | Wewer et al. SIGGRAPH Asia 2025 | Differentiable mesh-in-the-loop: Delaunay triangulation + learned SDF, mesh participates in the gaussian loss | Best quality. Mesh quality is bounded by gaussian quality. |
| CoMe (Compact Mesh) | various | Mesh compression of gaussians | Targets compression, not reconstruction quality |

### Key Insight: Training Quality is the Bottleneck

MILo produces the best meshes among evaluated methods, but all methods share a common ceiling: **the mesh can only be as good as the trained gaussians**. If the gaussians are noisy (floaters, stretched ellipsoids, missing regions), no mesh extraction method recovers lost geometry.

Root causes of poor gaussian quality in our test scenes:

1. **YouTube-compressed video**: H.264 compression artifacts reduce feature matching quality in COLMAP, producing fewer and less accurate camera poses
2. **Featureless walls**: Large uniform surfaces have no visual features for COLMAP to match, creating holes in the sparse reconstruction
3. **Reflective surfaces**: Glass cases, polished floors, and metallic frames violate the Lambertian assumption in both COLMAP and 3DGS
4. **Insufficient view coverage**: Walk-through videos miss ceiling details and behind-object views

The correct fix is better input capture, not better mesh extraction.

---

## Phase 4: MILo Integration

### CUDA Version Conflict

MILo requires:
- CUDA 11.8 (its CUDA extensions fail to compile with 12.x)
- GCC <= 11 (CUDA 11.8 does not support GCC 12+)
- PyTorch 2.3.1 with cu118

Our main container runs CUDA 12.8 + GCC 14 + Python 3.12. These are fundamentally incompatible. Conda environments were attempted but the CUDA toolkit version is a system-level constraint, not a Python-level one.

### Sidecar Container Solution

Built `docker/Dockerfile.milo` on `nvidia/cuda:11.8.0-devel-ubuntu22.04` with:
- Python 3.10 (Ubuntu 22.04 default)
- PyTorch 2.3.1 + cu118
- All 4 MILo rasterizer variants compiled from source
- nvdiffrast, simple-knn, fused-ssim
- tetra-triangulation (Delaunay, CGAL + pybind11)

The sidecar runs on GPU 1, sleeps until called. The main container invokes it via:
```bash
docker exec milo python3 train.py --source_path /data/output/JOB/colmap ...
```

Shared `/data/output` volume allows both containers to read COLMAP data and write mesh results without network transfer.

### MILo Extractor Module

Built `src/pipeline/milo_extractor.py` to:
1. Check if the `milo` container is running
2. Convert pipeline paths to container-relative paths
3. Call MILo training via `docker exec` with appropriate arguments
4. Monitor progress via log file polling
5. Convert MILo's PLY output to GLB for the web viewer
6. Fall back to TSDF if the sidecar is unavailable

---

## Phase 5: Blender Scene Assembly

### Motivation

The pipeline needs a final assembly step that:
- Imports TSDF or MILo meshes
- Cleans debris (small disconnected components)
- Creates proper materials from vertex colours
- Sets up lighting
- Renders preview images
- Exports a USD scene with proper hierarchy

### Implementation

Built `src/pipeline/blender_assembler.py` to run headless:
```bash
blender --background --python blender_assembler.py -- --input mesh.glb --output-usd scene.usda
```

Uses Blender's Cycles renderer with GPU compute for texture baking. Creates a 3-point lighting setup, imports COLMAP camera poses for aligned preview renders.

---

## Phase 6: Web Interface

### Flask App

Built `src/web/` with:
- Video upload (drag-and-drop, file size validation)
- Job management (create, track, cancel, delete)
- SSE log streaming for real-time pipeline progress
- 3D model preview via Google's `<model-viewer>` web component
- Preview image carousel from Blender renders
- ZIP download of all job outputs
- Anthropic API key management (stored on persistent volume, not in container image)

### SAM3 Object Segmentation

SAM3 (Segment Anything Model 3) provides concept-based segmentation with 4M concepts using text + visual prompts. Requires `HF_TOKEN` environment variable for model downloads from HuggingFace. Falls back to SAM2 grid-point prompts if SAM3 is unavailable.

---

## Current State

The end-to-end pipeline works: video upload through web UI, frame extraction, COLMAP SfM, 3DGS training via LichtFeld MCP, object segmentation, TSDF or MILo mesh extraction, Blender assembly, USD export, and web preview/download.

Primary quality limiter remains the input video. High-quality results require:
- 4K or higher resolution source video
- Slow, deliberate camera motion with overlap
- Multiple passes from different heights/angles
- Avoiding reflective and transparent surfaces
- Good, even lighting

The mesh extraction backend (TSDF vs MILo) matters less than the quality of the trained gaussians, which in turn depends almost entirely on the quality of the input video and COLMAP reconstruction.

---

## v2 Upgrade — 2026-05-26

### Overview

This entry records the v2 pipeline upgrade, which was designed by a managed mesh swarm (multi-agent parallel development) and covers the decisions, new modules, and architecture changes introduced on branch `feat/v2-upgrade-swarm`.

### Upstream Sync to v0.5.2 (ADR-002)

The fork had diverged from upstream LichtFeld Studio by approximately 410 commits spanning two stable releases (v0.5.1 and v0.5.2). The high-value features in v0.5.2 were:

- **Native USD import/export** (#1032) — eliminates the need for our custom `usd_assembler.py` as the sole USD path
- **Native mesh support** (#876, #879, #889) — mesh loading, mesh-to-splat conversion, mesh picking inside LichtFeld
- **MRNF densification** (#1031) — new default densification strategy; renamed from LFS
- **Enhanced MCP server** (#984) — additional tools for agentic pipeline control
- **VRAM optimisations** — reduced peak VRAM during evaluation and image loading

We chose to sync to the **v0.5.2 stable tag** (released 2026-04-21) and explicitly deferred the v0.5.3 Vulkan-only rendering migration to ADR-008. The rationale: v0.5.3 is unreleased; the Vulkan migration removed the CUDA and OpenGL renderers entirely (#1170, #1234); and a known coordinate-system regression (issue #1104, from PR #1066) could break `coordinate_transform.py`. The v0.5.2 baseline delivers all capability we need without any of those risks.

**Isolation policy confirmed**: this fork is one-way pull only. We never push to or open PRs against the upstream repository (origin/MrNeRF/LichtFeld-Studio).

### Four Mesh Extraction Backends (ADR-003, ADR-004, ADR-005)

The v1 pipeline supported two mesh extraction backends: TSDF (fast, lower quality) and MILo (high quality, ~69 min). v2 adds two more:

**CoMe** (`come_extractor.py`, ADR-004): Confidence-based Mesh Extraction from github.com/r4dl/CoMe. CoMe trains 3DGS with per-Gaussian confidence values, then extracts a mesh via marching tetrahedra. Benchmarks: ~25 min total on RTX 4090 vs. MILo's ~69 min, at comparable F1 scores (0.521 Tanks & Temples, 0.662 ScanNet++). CoMe requires Python 3.10 and CUDA 12.1, which is incompatible with both the main container (CUDA 12.8, Python 3.12) and the MILo sidecar (CUDA 11.8). It therefore runs in a new dedicated `come` sidecar (`docker/Dockerfile.come`). The sidecar is present in `docker-compose.consolidated.yml` but gated behind `--build-arg INSTALL_COME=1` because CoMe carries no LICENSE file as of 2026-05-26 (SPDX: NOASSERTION). It must not be used in commercial distribution until a permissive licence is published and reviewed.

**GaussianWrapping** (`gaussianwrapping_extractor.py`, ADR-005): From github.com/diego1401/GaussianWrapping. Reinterprets 3D Gaussians as stochastic oriented surface elements and extracts watertight, textured meshes that capture thin structures (bicycle spokes, wires, fences, railings) where TSDF and marching cubes fail. GaussianWrapping requires exactly CUDA 11.8 and Python 3.9 -- matching the MILo sidecar environment -- so it is installed into the existing `milo` container at `/opt/gaussianwrapping` rather than requiring a new container. It is gated behind `--build-arg INSTALL_GAUSSIANWRAPPING=1` for the same licensing reason (no formal LICENSE file).

**Pluggable backend architecture** (ADR-003): All four backends expose the same three-symbol interface (`XConfig` dataclass, `is_X_available() -> bool`, `run_X(colmap_dir, output_dir, config) -> dict`). Backend selection is centralised in `stages._select_mesh_backend()`. When `config.training.mesh_method = "auto"`, the function applies the heuristic: thin-structure hint → GaussianWrapping; CoMe available → CoMe; MILo available → MILo; fallback → TSDF. Explicit values (`"tsdf"`, `"milo"`, `"come"`, `"gaussianwrapping"`) bypass auto-selection for reproducible runs.

**CLI flag notice**: The CoMe and GaussianWrapping CLI flags are inferred from their upstream repositories (the SOF codebase for CoMe; the GaussianWrapping repository structure for GW). They have not been verified against the actual released source. All script names and flag constants are defined as module-level constants in `come_extractor.py` and `gaussianwrapping_extractor.py` so that corrections can be made in one place once the code is reviewed.

### Splat-Transform Delivery Stage (ADR-006)

Added `splat_optimizer.py`, which wraps the PlayCanvas `@playcanvas/splat-transform` npm CLI. The module is invoked as a `SPLAT_OPTIMIZE` stage after 3DGS training and before web delivery. It applies crop, filter, sort, and compress operations to produce a `.ksplat` file targeting under 20 MB from a raw PLY of 100+ MB. Node.js and npx must be present in the main container. The original `.ply` is always kept alongside the compressed form for downstream mesh extraction backends. The stage is opt-in via `config.delivery.enable_splat_optimize = True`.

### Fibonacci-Sphere Frame Selection (ADR-007)

Added `fibonacci_sampler.py`, which provides `fibonacci_sphere()`, `fibonacci_coverage_score()`, and `select_frames_by_coverage()`. The module is imported by `frame_selector.py` when `config.ingest.use_fibonacci_coverage = True`. After COLMAP SfM, camera positions are scored by their coverage of a Fibonacci-sphere distribution (a near-optimal low-discrepancy point set on the unit sphere). The combined frame score is:

```
score = 0.6 * quality_score + 0.4 * fibonacci_coverage_score
```

The weights are configurable via `config.ingest.coverage_weight`. The Fibonacci scoring falls back silently to the v1 quality-only path if COLMAP positions are unavailable (pre-SfM pass or degenerate reconstruction). No new runtime dependencies beyond NumPy.

### Architecture and DDD Model

The v2 upgrade produced eight Architecture Decision Records (`research/decisions/adr-001` through `adr-008`) and a full DDD domain model (`research/ddd/bounded-contexts.md`, `research/ddd/aggregates.md`). The domain model identifies seven bounded contexts (Ingestion, Reconstruction, Training, Segmentation, MeshExtraction, SceneAssembly, Delivery) and four aggregate roots (`ReconstructionJob`, `GaussianModel`, `MeshAsset`, `SceneGraph`, `DeliveryArtifact`).

The MeshExtraction context is architecturally the most complex in v2 because it spans all three containers and four backends. The physical container boundary is treated as infrastructure, not a bounded-context boundary; the sidecar ACL (`milo_extractor.py`, `come_extractor.py`, `gaussianwrapping_extractor.py`) translates domain commands into container-specific CLI invocations.

### What Was Deferred (ADR-008)

The v0.5.3 Vulkan migration is explicitly deferred. Trigger conditions for revisiting:
1. v0.5.3 released as a stable tagged version
2. Coordinate-system regression (issue #1104) resolved upstream
3. Headless Vulkan validated in a Docker container on our GPU model
4. MCP API compatibility verified for `mcp_client.py`
5. Python API audit complete

No v0.5.3-dev commits are merged until all five conditions are met. The `upstream/master-watch` branch tracks upstream progress monthly.

---

## v3 Upgrade — 2026-06-04

### Overview

The v3 increment converts the pipeline from a single-host, hardcoded-IP research
script into a manifest-driven, service-meshed, agent-overseen system targeting
2026 SOTA models. The work was built as a six-agent mesh swarm with disjoint file
ownership and reconciled against ADR-011 through ADR-015 and the v3 work-order
(items 0–9). FR-40 (the `video2gaussian` / `gaussian-toolkit` → `Vitrine` codebase
rename) remains explicitly deferred for blast-radius reasons; GPU-host validation,
live weight staging, and live pin resolution are out of band on the .48 host.

### Single Pre-Run Manifest (ADR-013 / D-013.1)

Replaced ad-hoc CLI flags with one human-authored `exhibit.toml`. `pipeline/manifest.py`
parses it, resolves `env:NAME` secret indirection at load time (a missing referenced
env var is a hard, named failure — exit 2), and materialises a runtime `PipelineConfig`.
Secrets (`hf_token`, `gcloud_credentials`) reject inline literals and are stripped before
the redacted JSON run-record is written. The loader maps objects → `decompose.sam3_concepts`,
`mesh_backend` → `training.mesh_method`, `matcher` → `reconstruct.matcher`, and the
endpoint/oversight overlays onto their config sub-objects. `exhibit.example.toml` documents
the schema. CLI: `python -m pipeline.manifest exhibit.toml [-o run.json]`.

### SOTA Idiot-Check Wired Into Preflight

`pipeline/sota_registry.check_environment()` is now invoked from `preflight.check_all()`
and `print_report()`. It is advisory by default — it logs a registry report (checkpoints
staged, VRAM fit, licence posture, pinning, caveats) and never raises — but escalates a
`FAIL` overall to a hard `RuntimeError` when `SOTA_STRICT` is set. Default posture remains
RESEARCH / non-commercial.

### Serial VRAM Lifecycle + Service-DNS Endpoints (D-013.2, D-013.3)

`pipeline/model_lifecycle.py` introduces `ModelLifecycleManager.stage()`, a context manager
that asserts VRAM headroom before a stage and unloads serially afterwards (soft = POST /free
+ `torch.cuda.empty_cache()`; hard = container stop), so peak VRAM is `max(stage)` rather than
the sum. `pipeline/endpoints.py` replaces hardcoded `192.168.2.48` IPs with an `Endpoints`
dataclass reading `V2G_*` env vars over a docker service-DNS mesh (`comfyui:8188`,
control-plane `:3001`, `agent-vlm:8080`, `milo:8090`, `come:8091`); the legacy single-host IPs
are retained only as named fallback constants.

### Agent-Controlled ComfyUI (ADR-014)

`pipeline/comfyui_control.py` gives the oversight agent direct probe/download/run/free control
over the .48 ComfyUI instance and Salad control-plane (health, `probe_models`, `ensure_model`,
`submit_workflow`, `wait`, `download_outputs`, `free_vram`), with a `requests`→`urllib` fallback
so it runs in a dependency-thin container.

### Web Onboarding (ADR-015)

`onboarding/` is a Rust/Axum service (`:8088`) serving a six-step vanilla-JS wizard that
round-trips `exhibit.toml`. `POST /api/manifest` writes the manifest with `env:` references only;
raw tokens are diverted to a `chmod 0600` `.secrets.env` and never echoed back. `cargo check`
clean.

### SOTA Model Modernisation (work-order items 0, 2, 4, 8)

- **Inpainting**: `comfyui_inpainter.py` adds a FLUX.2 path (`flux2_inpaint.json`, 15-node API
  graph) selected when FLUX.2 weights are present, otherwise falling through unchanged to the
  proven FLUX.1-Fill path.
- **3D recovery**: `hunyuan3d_client.py` adds Hunyuan3D-2.1 textured-PBR multiview
  (`hunyuan3d21_multiview.json`, 16-node graph) with graceful degradation 2.1 → 2.0-mv →
  single-view, and a SAM3D fallback when multi/single-view both fail.
- **Defaults**: `config.py` moves the training strategy default to `igs+`, mesh backend to
  `come`, the inpaint model to `flux2`, and Hunyuan to `2.1`; adds `EndpointsConfig` and
  `OversightConfig` with `validate()` coverage. Matching is ready for ALIKED + LightGlue via
  `reconstruct.matcher`.

### Version Pinning (work-order item 7)

`pins.lock.toml` records 12 upstream components (11 git, 1 pip) with repo / kind / ref /
host-path / clone-site, leaving `resolved_commit` empty rather than fabricating SHAs.
`scripts/resolve_pins.sh` performs a read-only `git rev-parse HEAD` per component on the host
and writes `pins.resolved.toml`. Resolution itself is a host-side step.

### What Was Deferred

FR-40 codebase rename (high blast radius, mechanical, scheduled separately); GPU smoke and
weight staging (host-only); live pin resolution (requires the .48 checkouts); pytest execution
(no pytest in the build container — the two new suites, `test_model_lifecycle.py` and
`test_comfyui_control.py`, are AST-clean and run in CI / on the host).

---

## v4 End-to-End Validation — 2026-06-05

### Overview

The pipeline was run **end to end on a real scene** for the first time, taking it from
*designed* to *demonstrated*. A reused 80-frame indoor capture (`output/milo_run`) was trained
to a 4M-gaussian field (LichtFeld `igs+`, `splat_30000.ply`, SH degree 3) and driven through
segmentation → object isolation → meshing → dual-USD assembly. The run produced five isolated
object PLYs, five per-object meshes, a 901 MB native splat USD, and a composed textured
`scene.usda` with four preview renders. Every stage surfaced a real defect; eight were fixed.

### Object resolution now works (SAM3 + ADR-010 D10)

- **SAM3 #507** — the SAM 3.1 fused `addmm_act` casts operands to bfloat16 and never restores,
  crashing segmentation (`mat1 and mat2 must have the same dtype`). Patched at the source binding
  in `sam3_segmentor.py` (monkey-patch, survives container rebuilds). SAM3 now resolves 5 objects
  (sculptures, furniture, walls, floor, ceiling). We segment **stills** per-frame then union by
  concept — no video tracker — so SAM3 (kept after a web-verified SOTA check) remains the right model.
- **Depth-aware multi-view projection (D10)** — replaced the broken world-XY heuristic in
  `extract_objects`. `segment()` now persists per-frame masks; `_extract_with_mask_mv` projects every
  gaussian centre through each registered COLMAP camera that has a per-frame mask, votes inside/outside
  the mask across views (skipping empty-mask frames), and keeps gaussians inside the object in a
  majority of detected views. Isolated: sculptures 1,080,171; furniture 452,401; floor 964,074;
  walls 357,276; ceiling 11,012 gaussians — correctly keyness-ranked (sculptures/furniture above
  structural surfaces). Three sub-defects fixed en route (empty-mask vote inflation, absent SH
  vertex colours, alignment).

### Meshing, native USD, composed USD

- **gsplat SH-degree loader** — `load_3dgs_ply` hardcoded 45 f_rest coefficients but the trained PLY
  is SH degree 1 (9); now reads the actual degree and zero-pads. gsplat-TSDF then produces real
  meshes (was a degenerate fallback).
- **Hunyuan3D kwargs** — `turbo` keyword crashed the client; kwargs filtered to the constructor signature.
- **LichtFeld runtime** — the prebuilt binary needs the host CUDA-13 runtime, vcpkg OpenUSD libs and
  `libz-ng`; resolved via `LD_LIBRARY_PATH` (CUDA + vcpkg dirs) plus staging `libz-ng.so.2` into the
  bind-mounted `build/`. This unblocked **training** in-container.
- **Native USD** — rewired `_export_native_usd` from the never-running MCP server to the headless
  LichtFeld CLI `convert` subcommand (`LichtFeld-Studio convert <ply> <out.usda>`).
- **Blender** — fixed the bake selecting the glTF `world` root (`Object 'world' is not a mesh`) and the
  Blender 5.0 `wm.usd_export` keyword change (`overwrite_existing_textures` removed); both now pass, so
  `blender_assembled=True` with textured `scene.usda` + 4 renders.

### Honest boundary

Validation was on the **reused** `milo_run` scene, not a fresh Drive→ingest→COLMAP capture. Meshes use
the **gsplat-TSDF fallback**, not the SOTA single-image hulls (TRELLIS.2 / Hunyuan node deps unbuilt).
The FLUX.2 recovery loop and local gemma-4 VLM are staged but not wired. Isolation quality is first-pass
(sparse SAM3 detection, no per-view depth occlusion → coherent but over-inclusive). These are tracked as
in-progress/pending in `report/main_v4.tex` (the consolidated current-state report) and the README.

### Docs

ADR catalogue reconciled to current design (ADR-001 rewritten as the live architecture; evolved ADRs
amended in place). The original bid + pitch brief extracted to `docs/brief/`. New consolidated
current-state report `report/main_v4.tex` (v1/v2/v3 left as historical snapshots).

