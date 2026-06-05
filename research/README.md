# Research: Video to Structured 3D USD Scene

## Objective

Reconstruct 3D polygonal scenes in USD format from video, via Gaussian Splatting. Identify all objects in the scene, isolate them, reconstruct each as individual textured 3D meshes with metadata, and assemble into a hierarchical scene graph.

## Project Scale

- **21 pipeline modules** in `src/pipeline/`
- **5 web interface files** in `src/web/` (upload UI on port 7860)
- **15 research docs** across `research/`
- **25+ git commits** on the `gaussian-toolkit` branch
- **Consolidated Docker** running on remote (192.168.2.48:7860 web, :8188 ComfyUI)
- **33 per-object PLY files** extracted from gallery tour
- **USD scene with 59 prims** assembled
- **TSDF mesh: 22K vertices, 49K faces**
- **SAM3 upgrade in progress** (from SAM2)
- **Hunyuan3D 2.0** multi-view integration built
- **FLUX inpainting client** built
- **Web upload interface** running on :7860

## Pipeline Status

### Complete

| Component | Module | Status |
|-----------|--------|--------|
| Video frame extraction | `orchestrator.py` (PyAV) | Tested, working |
| COLMAP SfM (feature extract, match, sparse, undistort) | `orchestrator.py` + COLMAP 4.1.0 | Tested, working |
| 3DGS Training via LichtFeld MCP | `mcp_client.py` | Tested, 7k iter in 2m15s |
| SAM2 2D segmentation | `sam2_segmentor.py` | Tested, 13 frames in 46s |
| Mask projection (2D masks to 3D Gaussians) | `mask_projector.py` | Tested, 98.3% coverage |
| Mesh extraction (Marching Cubes + TSDF) | `mesh_extractor.py` | Tested, TSDF: 22K verts / 49K faces |
| Mesh cleaning (decimation, hole fill) | `mesh_cleaner.py` | Tested |
| USD scene assembly | `usd_assembler.py` | Tested, 59 prims, variant sets |
| Quality gates (per-stage pass/fail) | `quality_gates.py` | Tested |
| CLI entry point | `cli.py` + `__main__.py` | Working |
| Pipeline configuration | `config.py` | YAML/dict based |
| Coordinate transforms | `coordinate_transform.py` | COLMAP <-> 3DGS <-> USD |
| Frame quality scoring | `frame_quality.py` | Blur/exposure filtering |
| MCP bridge script | `scripts/lichtfeld_mcp_bridge.py` | Working |
| Hardware tracing | `scripts/hardware_trace.py` | GPU/RAM/CPU logging |
| Object separation | `scripts/run_object_separation.py` | 33 objects, 98.3% coverage |
| TSDF mesh extraction | `scripts/run_tsdf_mesh.py` | 22K verts, 49K faces |
| USD gallery assembly | `scripts/assemble_gallery_usd.py` | 59 prims |
| Multi-view renderer | `multiview_renderer.py` | Camera orbit renders |
| Hunyuan3D 2.0 client | `hunyuan3d_client.py` | Multi-view to textured mesh |
| FLUX inpainting client | `comfyui_inpainter.py` | Background recovery via ComfyUI |
| Web upload interface | `src/web/app.py` | Flask on :7860 |
| Consolidated Docker | `Dockerfile.consolidated` | Dual RTX 6000 Ada, all services |

### In Progress

| Component | Module | Status | Blocker |
|-----------|--------|--------|---------|
| SAM3 upgrade | `sam3d_client.py` | Client built, upgrading from SAM2 | SAM3 model integration in consolidated Docker |
| Texture baking | `texture_baker.py` | Skeleton written | Depends on clean mesh extraction |
| Material assignment | `material_assigner.py` | Skeleton written | Depends on texture baking |
| COLMAP output parsing | `colmap_parser.py` | Basic binary reader | Needs robust error handling for malformed models |

### Planned

| Component | Module | Description |
|-----------|--------|-------------|
| Audio-to-scene-graph naming | TBD (`src/pipeline/audio_namer.py`) | Extract audio track from input video using FFmpeg, transcribe with OpenAI Whisper (or whisper.cpp for local inference), run NER/keyword extraction on the transcript, and use the extracted terms to automatically name objects in the USD scene graph. For example, if the narrator says "the red chair by the window", the pipeline would match segmented objects by visual description and assign `red_chair` as the prim name instead of `object_017`. This closes the gap between human-legible scene descriptions and the anonymous object IDs that segmentation produces. Depends on: working SAM3 segmentation (for object descriptions) and USD assembly (for prim naming). |

### Museum Tour Run (2026-03-29)

End-to-end test with a 90-second museum tour video on the consolidated Docker (dual RTX 6000 Ada).

| Metric | Value | Notes |
|--------|-------|-------|
| Input video | 90s museum tour MP4 | Handheld walkthrough |
| Extraction FPS | 4.0 | Oversampled |
| Frames extracted | 180 | At 4fps |
| Frames selected | 80 (target) | Frame selector with quality + diversity |
| COLMAP matcher | sequential | Changed from exhaustive for video |
| COLMAP registration | 21/180 (11.7%) | **FAILED** -- all 180 sent without frame selection |
| SAM3 status | Fallback to SAM2 | BPE vocab file not found in container |
| Pipeline completion | Stopped after training | Claude Code did not continue to mesh/USD |

**Root causes identified and fixed:**
1. Frame selection was not invoked -- all 180 frames sent to COLMAP, causing low registration rate (21/180). Fix: default to 4fps extraction + select best 80 frames.
2. SAM3 BPE vocab file at `/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz` was not on the Python path. Fix: added `SAM3_BPE_PATH` env var and config field.
3. COLMAP used exhaustive matcher, too slow for 180 frames. Fix: default to sequential matcher for video input.
4. Claude Code auto-launch binary path was `claude` (not found in PATH). Fix: use `/usr/local/bin/claude`.
5. Pipeline prompt did not instruct continuation past training. Fix: explicit all-stages prompt.

### Known Issues

1. **COLMAP sparse reconstruction is the bottleneck** -- ~20 minutes on 32 cores for 15 frames. No GPU acceleration available for the sparse BA solver. Workaround: use fewer frames or switch to incremental mapper. For video input, use the sequential matcher instead of exhaustive.

2. **Frame selection is critical** -- Sending all extracted frames to COLMAP causes very low registration rates (11.7% with 180 frames). Always run frame selection to curate 60-80 diverse, high-quality frames first.

3. **SAM3 BPE vocab dependency** -- SAM3 text-prompted segmentation requires `bpe_simple_vocab_16e6.txt.gz` from the sam3 repo assets. The `SAM3_BPE_PATH` environment variable must point to it. Falls back to SAM2 if missing.

4. **Mask projection noise on thin geometry** -- The Gaussian-space voting from 2D masks produces noisy labels on thin structures (branches, wires). Depth-weighted voting and multi-view consistency checks are needed.

5. **Mesh extraction produces non-manifold geometry** -- Marching Cubes on the Gaussian density field can produce self-intersecting faces. TSDF fusion now available as alternative (22K verts, 49K faces).

6. **No texture UV unwrapping** -- Meshes are vertex-coloured only. xatlas is installed but not integrated for proper UV unwrapping and texture baking.

7. **USD variant sets are placeholder** -- The assembler creates Gaussian and Mesh variant sets but the Gaussian variant currently stores only a path reference, not embedded splat data.

## Target Pipeline

```
Video -> Frames (+ per-image metadata sidecar) -> COLMAP SfM (ALIKED+LightGlue)
    -> 3DGS Training (ImprovedGS+ / MRNF) -> .ksplat -> SAM3 Concept Segmentation
    -> Key-item ranking -> Per-object hull (FLUX.2 recovery -> TRELLIS.2 / Hunyuan3D-2.1)
    -> Environment mesh (CoMe default) -> Native USD scene graph (v2g:* metadata)
```

## Research Structure

```
research/
├── README.md                          # This file
├── landscape/
│   ├── tool-catalogue.md              # 31 tools assessed with viability scores
│   ├── segmentation-methods.md        # 3D Gaussian segmentation SOTA
│   ├── mesh-extraction-methods.md     # Gaussian-to-mesh conversion SOTA
│   └── field-overview.md              # Landscape synthesis and gap analysis
├── pipelines/
│   ├── proposed-pipeline.md           # Recommended end-to-end architecture
│   └── alternative-pipelines.md       # Alternative approaches considered
├── components/
│   ├── hunyuan3d-integration.md       # Hunyuan3D 2.0 multi-view mesh creation
│   ├── inpainting-recovery.md         # Background recovery via diffusion
│   └── quality-control.md            # Agent quality decision trees
├── references/
│   ├── existing-capabilities.md       # What LichtFeld/COLMAP already provide
│   └── (papers.md, repos.md)         # Academic references
├── decisions/
│   ├── prd*.md                        # PRDs: v1, consolidated-docker, v2-upgrade, v3-e2e-closure
│   ├── adr-001..015-*.md             # Architecture Decision Records (amended to reflect live choices)
│   ├── work-order-sota-modernisation.md  # Live SOTA stack work plan
│   ├── upstream-sync-runbook.md, video-ingestion-plan.md, ddd-domain-model.md (v1)
│   └── audit-findings.md, gap-analysis-*.md  # point-in-time analyses (historical)
├── ddd/                               # Domain model (current): bounded-contexts, aggregates,
│   └── ...                            #   anti-corruption-layers, ubiquitous-language, v3-e2e-extensions
└── qe/                                # Implementation-status / QE analyses (qe-00..04)
```

> ADRs are maintained as a living catalogue: ADR-001 records the current pipeline
> architecture (the original v1 Gaussian-Grouping/SuGaR design is retired), and the
> evolved records (004, 005, 010, 011, 012, 013, 014) carry dated **Amendment** blocks
> reflecting the current choices (CoMe default mesh, TRELLIS.2 primary hull, FLUX.2-dev,
> gemma-4 VLM, native USD export, the SOTA idiot-check, and the canonical ComfyUI).

## Key Findings

### Critical Path

The current pipeline uses these core components:

1. **SplatReady** (installed) -- Video to COLMAP dataset
2. **SAM2/SAM3** (SAM2 validated, SAM3 upgrading) -- 2D segmentation with concept prompts
3. **Mask Projection** (validated) -- 2D masks to 3D Gaussian labels, 98.3% coverage
4. **TSDF Mesh Extraction** (validated) -- Open3D TSDF fusion, 22K verts / 49K faces
5. **Hunyuan3D 2.0** (client built) -- Per-object multi-view to textured mesh
6. **FLUX Inpainting** (client built) -- Background recovery via ComfyUI
7. **USD Assembly** (validated) -- 59-prim hierarchical scene with variant sets

### Reconstruct-Then-Segment Validated

Evidence strongly favours **reconstruct-then-segment** for our use case:
- SAM2 mask projection achieved 98.3% Gaussian coverage with 33 objects
- Post-hoc segmentation allows quality gating before decomposition
- SAM3 will improve this further with text+visual concept prompts (4M concepts)

### Hybrid Approach (Current)

1. Train full scene 3DGS (7k iter, 2m15s, 1M gaussians)
2. SAM2/SAM3 segmentation on training views
3. Project 2D masks onto 3D Gaussians (98.3% coverage)
4. Extract per-object PLY files (33 objects)
5. Per-object Hunyuan3D 2.0 mesh creation (multi-view to textured mesh)
6. Inpaint removed objects from training views via FLUX/ComfyUI
7. Assemble multi-object USD scene with variant sets (59 prims)

### Gap Analysis

| Capability | Status | Primary Tool |
|-----------|--------|--------------|
| Video to Frames | Complete | SplatReady / PyAV |
| COLMAP SfM | Complete | COLMAP 4.1.0 |
| 3DGS Training | Complete | LichtFeld Studio MCP |
| Object Segmentation | Complete (SAM2), Upgrading (SAM3) | SAM2 + mask projection / SAM3 |
| TSDF Mesh Extraction | Complete | Open3D TSDF fusion |
| Per-Object Mesh | **In Progress** | Hunyuan3D 2.0 |
| Background Inpainting | **Built** | ComfyUI + FLUX |
| Texture Baking | **In Progress** | xatlas + custom baker |
| USD Assembly | Complete | OpenUSD Python (59 prims) |
| Agentic Orchestration | Complete | LichtFeld MCP (70+ tools) |
| Web Interface | Complete | Flask on :7860 |
| Consolidated Docker | Complete | Dual RTX 6000 Ada |
