# Ubiquitous Language — Gaussian Toolkit v2

**Extends**: research/decisions/ddd-domain-model.md (v1 glossary)
**Alignment**: research/decisions/prd-v2-upgrade.md
**Date**: 2026-05-26

This glossary is the authoritative mapping between domain terms used in
conversations, code, and research documents. When a term appears in this file,
code must use that exact spelling. When a term in code differs from a term here,
the code is the thing that needs renaming.

Entries are grouped by bounded context, then sorted alphabetically within
each group. Cross-context terms appear in the Shared section.

---

## Shared / Cross-Context Terms

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **artifact** | `StageResult.artifacts` (stages.py) | A file or directory produced by a pipeline stage and consumed by downstream stages. Artifacts are keyed by semantic name (e.g. `"ply_path"`, `"colmap_dir"`) inside `StageResult.artifacts`. |
| **job** | `job_id`, `job_dir` (stages.py, web/job_manager.py) | A single end-to-end pipeline execution keyed by UUID. All intermediate and final artifacts live under `{output_dir}/{job_id}/`. |
| **job directory** | `job_dir: Path` | The root directory for all artifacts of a single job. Set at `PipelineStages.__init__`. |
| **pipeline** | `src/pipeline/` (28 modules) | The end-to-end video-to-scene processing system. Not the same as LichtFeld Studio's internal pipeline. |
| **quality gate** | `quality_gates.py`, `StageResult.metrics` | A checkpoint where domain metrics are evaluated. If the gate fails, the stage returns `success=False` and may trigger a retry. |
| **stage** | `STAGE_NAMES` (stages.py) | A named, self-contained unit of pipeline work. Each stage takes explicit inputs and returns a `StageResult`. |
| **StageResult** | `dataclass StageResult` (stages.py) | The published message type crossing context boundaries. Contains `{success, stage, metrics, artifacts, error}`. |
| **sidecar** | `docker exec milo ...` (milo_extractor.py) | A companion Docker container running a different CUDA/Python version, accessed via `docker exec` from the main container. MeshExtraction uses sidecars for MILo, CoMe, and GaussianWrapping. |

---

## Current Model Selections (2026-06-05)

The live instances of the domain backends/tools, kept here so the glossary matches the
code (the authoritative, executable registry is `src/pipeline/sota_registry.py`):

| Domain term | Current selection |
|---|---|
| Mesh backend (default) | **CoMe** (MILo / GaussianWrapping / TSDF fallbacks) |
| Object hull generator | **TRELLIS.2-4B** primary, **Hunyuan3D-2.1** fallback, SAM3D last resort |
| Generative recovery (inpaint) | **FLUX.2-dev** |
| Artifact-analysis VLM | **gemma-4-26B-A4B** (Q8_0) |
| SfM matcher | **ALIKED+LightGlue** (SIFT fallback) |
| Training strategy | **ImprovedGS+** (MRNF / MCMC also native) |
| USD assembler | **native `scene.export_usd`** (custom assembler fallback for `v2g:*`) |

Licence posture is **research / non-commercial** by default (CoMe, FLUX.2-dev and SAM3D
are non-commercial); a commercial build swaps to PGSR / Qwen-Image-Edit.

---

## Ingestion Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **blur score** | `frame_quality.py::blur_score()` | Laplacian variance of a frame. Low blur score = blurry frame = excluded from COLMAP input. |
| **exposure gate** | `frame_quality.py::exposure_value()` | Check that a frame's mean luminance falls within acceptable range. Overexposed or underexposed frames are excluded. |
| **Fibonacci sampling** | `fibonacci_sampler.py` (planned) | Frame selection strategy scoring frames by proximity to a Fibonacci-sphere distribution of viewpoints. Produces near-optimal angular coverage compared to sequential stride sampling. |
| **frame** | `Frame` entity (ddd-domain-model.md), `frames/*.jpg` | A single extracted image from the source video, saved as JPEG. |
| **frame extraction** | `stages.py::ingest()` | Running ffmpeg to extract individual JPEG frames from the input video at a target frame rate. |
| **frame set** | `FrameSet` entity, `frames_selected/` directory | The curated set of frames passed to COLMAP — after blur/exposure filtering and viewpoint-coverage selection. |
| **fps** | `config.ingest.fps` | Extraction frame rate (frames per second) from the source video. Default 2.0 fps. |
| **person mask** | `person_remover.py::PersonRemover` | A binary mask identifying person-occupied pixels in a frame. Used to inpaint or drop frames before COLMAP. |
| **viewpoint coverage** | `frame_selector.py::SelectionConfig` | The distribution of camera orientations across the selected frame set. Good coverage = angles spread around all sides of the scene. |

---

## Reconstruction Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **camera intrinsics** | `CameraIntrinsics`, `cameras.bin` | Focal length, principal point, and distortion parameters for a camera model. Stored in COLMAP binary format. |
| **camera extrinsics** | `CameraExtrinsics`, `images.bin` | World-to-camera rotation (quaternion) and translation for each registered image. |
| **COLMAP dataset** | `colmap_dir`, `colmap/undistorted/` | The output of a COLMAP reconstruction: `images/` (undistorted) + `sparse/0/{cameras,images,points3D}.bin`. This is the canonical input format for Training and sidecar mesh backends. |
| **COLMAP sparse model** | `sparse/0/` | The COLMAP output containing registered cameras and a sparse 3D point cloud. Not to be confused with a dense reconstruction. |
| **feature matching** | COLMAP sequential/exhaustive matcher | Finding correspondences between overlapping frames. Sequential matcher used for video; exhaustive for small frame sets. |
| **registration rate** | `n_registered / n_total` (stages.py::reconstruct) | Fraction of input frames successfully registered into the sparse model. Gate: >= 30%. |
| **reprojection error** | `ColmapReconstruction.reprojection_error` | Mean pixel error when projecting 3D points back onto registered images. Lower is better. |
| **SfM** | Structure from Motion | Recovering camera poses and a sparse 3D point cloud from a set of 2D images by triangulating feature correspondences. |
| **SplatReady** | `~/.lichtfeld/plugins/splat_ready/` | A LichtFeld plugin that automates COLMAP video-to-COLMAP-dataset conversion. Used when available; direct COLMAP fallback otherwise. |
| **undistorted images** | `colmap/undistorted/images/` | Frames with lens distortion removed, in pinhole-camera geometry. These are the images used for 3DGS training. |
| **Y-up coordinate frame** | `coordinate_transform.py` | Our canonical world coordinate system: Y is up, Z is into the scene, 1 unit = 1 metre, right-handed. Applied to COLMAP output before any downstream stage consumes it. |

---

## Training Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **3DGS** | 3D Gaussian Splatting | The reconstruction technique that represents a scene as a cloud of 3D Gaussians (position, covariance, opacity, spherical harmonics coefficients). |
| **4C4D** | `dynamic_trainer.py` (Phase 3) | 4D (spatial + temporal) Gaussian Splatting for scenes with motion. Each Gaussian has a temporal trajectory. |
| **densification** | training loop in LichtFeld / gsplat | The process of splitting or cloning Gaussians during training to increase scene detail. Strategy choices: MRNF, MCMC, IGS+, Default. |
| **gsplat** | `gsplat_trainer.py`, pip package | The Python/CUDA 3DGS training library from UC Berkeley. Used as the direct training backend when `backend == "gsplat"` in config. |
| **IGS+** | `strategy = "igs_plus"` (config.py) | IGS+ densification strategy. Legacy option in our config. |
| **LichtFeld** | `mcp_client.py`, LichtFeld-Studio binary | The upstream C++ application providing GUI, training, and USD export. Our pipeline calls it as a headless process or via MCP. |
| **MCMC densification** | `strategy = "mcmc"` (config.py) | Stochastic Control Monte Carlo densification. Default for `indoor_reflective` preset. Upstream fixed defaults in PR #1046. |
| **MCP** | `mcp_client.py`, port 45677 | Model Context Protocol — JSON-RPC HTTP server embedded in LichtFeld. The pipeline's programmatic control interface to LichtFeld. |
| **MRNF densification** | `strategy = "mrnf"` (config.py) | Multi-Resolution Normalised Flow densification. New default strategy from upstream v0.5.2 (was called LFS before rename, PR #1031). |
| **NanoGS** | upstream PR #1014 | Compact Gaussian representation. May be relevant for Delivery (web-optimised splat). Not yet integrated into our pipeline. |
| **PLY** | `.ply` file, `ply_path` artifact | Stanford Polygon Library format. The output format for trained Gaussian splat models (positions, covariances, SH coefficients, opacity per Gaussian). Also used as per-object Gaussian export format. |
| **scene preset** | `config.training.scene_preset` | A named training configuration for a scene type. `"default"` for outdoor/general; `"indoor_reflective"` for indoor scenes with specular surfaces. |
| **SH degree** | `sh_degree: int` (TrainingConfig) | Spherical harmonics degree (0–3). Controls view-dependent colour complexity. Higher = more detail but more VRAM. |
| **splat** | interchangeable with "Gaussian" in code | A single 3D Gaussian in the splat cloud. Colloquial term; used in UI copy and delivery format names (`.ksplat`, `splat-transform`). |
| **training iteration** | `iteration: int` (TrainingProgress event) | One optimisation step in the 3DGS training loop. Typical: 7,000–30,000 iterations. |
| **VkSplat** | upstream PR #1162 (v0.5.3 only) | Vulkan-based VRAM-efficient Gaussian splatting renderer. Not in v0.5.2; requires Vulkan migration. |

---

## Segmentation Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **concept prompt** | `config.decompose.sam3_concepts` | A text string describing an object class used by SAM3 (e.g. `"paintings"`, `"furniture"`). SAM3 segments all instances matching the concept. |
| **label** | `GaussianObject.label`, `obj["label"]` | The human-readable semantic class name for a segmented object (e.g. `"chair_001"`). Set by SAM2/SAM3; may be auto-named or LLM-named. |
| **mask** | `mask_*.npy`, `ObjectMask` | A 2D binary numpy array (H × W) indicating which pixels belong to a segmented object. Saved to `{job_dir}/sam3_masks/mask_{id:04d}.npy`. |
| **mask projection** | `mask_projector.py` | Projecting 2D frame masks onto 3D Gaussian positions using the COLMAP camera models to identify which Gaussians belong to each object. |
| **object_id** | `obj["object_id"]`, integer | Integer identifier for a segmented object within a job. Keys the mask file on disk. |
| **SAM2** | `sam2_segmentor.py` | Segment Anything Model v2 (Meta). Video-capable segmentation model. Used as segmentation fallback when SAM3 is unavailable. |
| **SAM3** | `sam3_segmentor.py`, `sam3d_client.py` | SAM3 (4M concept vocabulary). Text+visual concept prompting for semantic segmentation. Primary segmentation model in v2. |

---

## MeshExtraction Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **backend** | `MeshBackend` value object, `config.training.mesh_method` | The specific algorithm and container used for mesh extraction. One of: TSDF, MILo, CoMe, GaussianWrapping. Chosen by the BackendSelector policy. |
| **CoMe** | `come_extractor.py` (planned) | Confidence-based Mesh Extraction (github.com/r4dl/CoMe, code released 2026-04-22). Trains 3DGS with per-Gaussian confidence, extracts via marching tetrahedra. ~25 min on RTX 4090. Runs in `come` sidecar (CUDA 12.1, Python 3.10). |
| **confidence field** | CoMe training objective | Per-Gaussian scalar confidence value trained alongside the Gaussian attributes. High-confidence Gaussians correspond to actual surfaces; low-confidence ones are floaters. Used by CoMe's marching-tetrahedra extraction. |
| **decimation** | `mesh_cleaner.py::decimate()` | Reducing polygon count of a mesh while preserving shape. Applied after extraction to hit `config.mesh.max_vertices` target. |
| **GaussianWrapping** | `gaussianwrapping_extractor.py` (planned) | Stochastic Oriented Surface Elements mesh extraction (github.com/diego1401/GaussianWrapping). Specialised for thin structures (wires, spokes, railings). Runs in the MILo sidecar (CUDA 11.8, Python 3.9 — shared). |
| **GLB** | `.glb`, `mesh_glb_path` | Binary glTF format. Primary mesh output format for web viewer and Blender import. |
| **marching cubes** | `mesh_extractor.py::extract_from_pointcloud()` | Isosurface extraction from a volumetric occupancy field. Used in the TSDF path and as a fallback. |
| **marching tetrahedra** | CoMe extraction step | Finer-grained isosurface extraction using tetrahedral cells. Produces fewer artefacts on smooth surfaces than marching cubes. Used by CoMe. |
| **mesh backend** | `MeshBackend` value object | See "backend" above. |
| **MILo** | `milo_extractor.py` | Mesh-in-the-Loop (github.com/Anttwo/MILo, SIGGRAPH Asia 2025). Joint 3DGS training + differentiable mesh extraction. ~69 min on RTX 4090. Runs in `milo` sidecar (CUDA 11.8, Python 3.9). |
| **OBJ** | `.obj` + `.mtl`, `mesh_obj_path` | Wavefront OBJ format. Secondary mesh output, used for USD import and Blender assembly. |
| **Poisson reconstruction** | `mesh_extractor.py::_mesh_with_open3d()` | Surface reconstruction from oriented point cloud using Poisson equation (Open3D). Last-resort fallback. |
| **sidecar** | `docker/Dockerfile.milo`, `docker/Dockerfile.come` | See Shared terms above. Each non-TSDF backend runs in a sidecar container with a different CUDA/Python environment. |
| **thin structure** | scene type triggering GaussianWrapping | Scene geometry with very small cross-section relative to length: bicycle spokes, wires, fences, railings, thin columns. Standard TSDF and marching cubes fail on thin structures. |
| **TSDF** | `mesh_extractor.py::MeshExtractor`, `TSDFConfig` | Truncated Signed Distance Field. Fuses depth maps rendered from the trained Gaussian model into a volumetric SDF and extracts a surface via marching cubes. The default fast-path backend. |
| **UV atlas** | `texture_baker.py` (xatlas) | A mapping from 3D mesh surface to a 2D texture image. Generated by xatlas. Required for UV-mapped textures (as opposed to vertex-coloured meshes). |
| **vertex-coloured mesh** | `mesh_glb_path` without `texture_path` | A mesh where colour is stored per vertex rather than in a UV-mapped texture. Exported as GLB. Produced when texture baking is skipped (face count > 30,000). |
| **watertight mesh** | GaussianWrapping objective | A mesh with no holes or open edges. Required for reliable physics simulation and 3D printing. GaussianWrapping and MILo target watertight output; TSDF does not guarantee it. |

---

## SceneAssembly Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **Alembic** | `temporal_exporter.py` (Phase 3) | A CG interchange format for animated geometry. Used to export dynamic Gaussian point caches (4C4D output) as animated prims in USD. |
| **assembler backend** | `assembler_backend` field on `SceneGraph` | Which code path assembled the USD scene: `"usd_assembler"` (our Python code), `"lichtfeld_native"` (LichtFeld MCP USD tools post v0.5.2 sync), or `"blender"` (Blender subprocess). |
| **Cycles bake** | `blender_assembler.py` | Blender's path-traced renderer used to bake high-quality textures from Gaussian renders onto mesh UVs. Slower than xatlas vertex-colour baking but produces photorealistic results. |
| **LichtFeld native USD** | upstream PR #1032, `mcp_client.py` | USD import/export built into LichtFeld Studio as of v0.5.2. May replace `usd_assembler.py` if it supports hierarchical per-object prims. PRD open question 3. |
| **prim** | `ScenePrim.prim_path`, `/World/Objects/Chair_001` | A USD primitive — the fundamental addressable node in a USD scene graph. |
| **prim path** | `ScenePrim.prim_path` | The absolute path to a prim in the USD hierarchy, e.g. `/World/Objects/Chair_001`. Our convention: `/World/Objects/{Label}` for objects, `/World/Background` for background, `/World/Cameras/{Name}` for cameras. |
| **UsdGeomMesh** | used by `usd_assembler.py` | The USD schema type for polygonal mesh geometry. |
| **UsdPreviewSurface** | used by `material_assigner.py` | The standard USD material model supported by all major DCC tools. |
| **variant set** | `UsdObject.variant_set`, `{gaussian: bool, mesh: bool}` | A USD mechanism for switching between multiple representations of the same object. Our scenes publish a `representation` variant set with `gaussian` and `mesh` variants. |
| **Xform** | USD prim type | A USD transform node. Our scene hierarchy is: `/World` (Xform) → `/World/Objects` (Xform) → `/World/Objects/{Label}` (Xform holding variants). |

---

## Delivery Context

| Domain Term | Code Term / Location | Definition |
|-------------|---------------------|------------|
| **compressed splat** | `.ksplat`, `SplatOptResult` | A Gaussian PLY file compressed by `splat-transform` (PlayCanvas). Typical reduction: 100+ MB PLY → <20 MB ksplat. |
| **download bundle** | ZIP, `DeliveryArtifact.download_bundle_path` | A ZIP archive containing all output formats (USD, PLY, GLB, ksplat) for a single job, offered as a single download link from the web UI. |
| **KHR_gaussian_splatting** | planned glTF extension output | The Khronos glTF extension for embedding Gaussian splat data in a glTF/GLB file, enabling standard 3D toolchain interoperability. |
| **ksplat** | `.ksplat` file | PlayCanvas compressed splat format produced by `splat-transform`. Binary format with quantised SH coefficients and half-precision positions. |
| **LOD** | Level of Detail, `splat-transform crop/filter` | Multiple quality levels of a splat: full PLY (training quality) and compressed ksplat (web delivery). splat-transform can also crop to different bounding boxes for spatial LOD. |
| **model-viewer** | `src/web/static/` | Google's `<model-viewer>` web component. Currently used for mesh (GLB) preview. To be supplemented or replaced by a splat-native viewer (2Xplat / SuperSplat) in v2. |
| **splat-transform** | `splat_optimizer.py` (planned), npm package | PlayCanvas JavaScript/CLI library for PLY splat manipulation: crop, filter, sort, compress. Wraps `@nicedoc/splat-transform`. |
| **SuperSplat** | planned web viewer | Web-based Gaussian splat viewer from PlayCanvas. Alternative to 2Xplat. Renders `.ply` and `.ksplat` in-browser. |
| **web UI** | `src/web/` | The Flask application (port 7860) providing video upload, job progress (SSE), 3D preview, and result download. |
| **2Xplat** | planned web viewer component | Cross-platform Gaussian splatting renderer. Candidate to replace `model-viewer` for splat preview in the web UI. |

---

## v1 Terms Superseded or Renamed in v2

| v1 Term | v2 Term | Notes |
|---------|---------|-------|
| `LFS densification` | `MRNF densification` | Upstream renamed in PR #1031. Our config already uses `"mrnf"` string. |
| `SuGaR` | removed | SuGaR mesh extraction not in our modules. Was referenced in ADR-001 but never implemented. TSDF + MILo replaced it. |
| `SOF / GOF` | `MILo` | SOF/GOF referenced in v1 ddd-domain-model.md as `extraction_method` enum values. MILo is the current sidecar. |
| `SUGAR / SOF / GOF / TSDF` enum | `MeshBackend` value object | v2 replaces the enum string with a structured value object carrying container and CUDA metadata. |
| `DefaultStrategy` | `mrnf` (new default) | After upstream v0.5.2 sync, MRNF replaces gsplat DefaultStrategy as the default training strategy. |
| `usd_assembler.py` (primary) | `lichtfeld_native` (post-sync candidate) | Post v0.5.2 sync, LichtFeld native USD export may become the primary SceneAssembly path. `usd_assembler.py` becomes the fallback. |
