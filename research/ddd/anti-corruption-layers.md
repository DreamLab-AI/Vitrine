# Anti-Corruption Layers — Gaussian Toolkit v2

**Extends**: research/decisions/ddd-domain-model.md (v1 context map)
**Alignment**: research/decisions/prd-v2-upgrade.md, BOUNDARIES.md
**Date**: 2026-05-26

An Anti-Corruption Layer (ACL) is an adapter that translates between two
different ubiquitous languages, preventing the semantics of an external system
from leaking into our core domain model. Each ACL section below identifies:
- What external system it isolates
- What domain concepts it protects
- Which module(s) implement it
- The translation contract (our domain type ↔ external representation)

---

## 1. COLMAP ACL

**External system**: COLMAP 4.1.0 binary (SfM tool with its own binary file
format and CLI conventions)

**Isolates**: Reconstruction Context from COLMAP's binary formats, coordinate
conventions, and CLI argument schema.

**Modules**: `colmap_parser.py`, `coordinate_transform.py`,
`stages.py::reconstruct()` (subprocess invocation layer)

### What the ACL translates

| COLMAP concept | Our domain concept | Translation |
|----------------|-------------------|-------------|
| `cameras.bin` (binary) | `CameraIntrinsics` entity | `colmap_parser.py` reads binary structs, emits Python dataclasses |
| `images.bin` (binary) | `CameraExtrinsics` entity | Binary quaternion + translation → `Transform3D` value object |
| `points3D.bin` | sparse point cloud (position + colour arrays) | Binary read → numpy arrays |
| `sparse/0/` path convention | `ColmapDataset.sparse_dir` | Path normalisation; handles missing `0/` subdirectory |
| COLMAP coordinate frame (Z-forward, right-hand) | Y-up, right-hand, 1 m = 1 unit | `coordinate_transform.py::apply_yup_transform()` |
| COLMAP camera models (OPENCV, PINHOLE, FISHEYE) | uniform `CameraIntrinsics` | Model-aware parameter extraction; fisheye supported via `config.reconstruct.use_fisheye` |
| CLI: `feature_extractor`, `sequential_matcher`, `mapper`, `image_undistorter` | `stages.py::_run_colmap_direct()` | Subprocess call with typed parameters from `ReconstructConfig` |
| SplatReady plugin API | `stages.py::reconstruct()` | JSON config file + `runner.py` subprocess; falls back to direct COLMAP if plugin absent |

### CAUTION: v2 coordinate risk

Upstream PR #1066 (in master, not v0.5.2) modifies the upstream coordinate
conventions and has caused issue #1104 ("ERP + GUT training produces degenerate
flat-plane output"). Our `coordinate_transform.py` ACL must be re-validated
against the upstream convention before any sync to master. This is the highest-
risk ACL change in v2.

---

## 2. LichtFeld MCP ACL (Conformist + Thin Adapter)

**External system**: LichtFeld Studio C++ application with embedded MCP
JSON-RPC server on port 45677.

**Isolates**: Training Context (and partly SceneAssembly) from LichtFeld's
MCP tool schema, JSON-RPC wire format, and async training progress model.

**Modules**: `mcp_client.py`

**Relationship type**: Primarily Conformist (we adapt to LichtFeld's API
as-is) with a thin adapter layer for type safety and error handling.

### Why Conformist, not pure ACL?

We deliberately conform to LichtFeld's MCP API because:
1. LichtFeld provides 70+ pre-built tools; wrapping them all would be
   prohibitive.
2. Our pipeline does not modify LichtFeld's training semantics — we pass
   parameters through.
3. When LichtFeld's API changes (e.g. post PR #984 enhanced MCP), we update
   `mcp_client.py` to follow, not to shield the domain.

The thin adapter layer in `mcp_client.py` provides:
- Python type wrappers around JSON-RPC responses
- Timeout handling and retry logic
- Training progress polling (MCP does not push; we poll)
- Connection-failure fallback (direct binary invocation)

### Translation contract

| Our domain command | MCP call | Notes |
|-------------------|----------|-------|
| `train(colmap_dir, iterations, strategy)` | `POST /tools/train` with JSON body | Strategy names must match LichtFeld's internal names; `mrnf`/`mcmc` verified post v0.5.2 sync |
| `export_usd(scene, output_path)` | `POST /tools/export_usd` | Post v0.5.2 sync; replaces `usd_assembler.py` for native export path |
| `get_training_progress()` | `GET /resources/training/status` | Polling loop; returned as `TrainingProgress` domain event |
| `load_ply(ply_path)` | `POST /tools/load_scene` | Loads trained PLY into LichtFeld for rendering/export |

### v2 MCP risk (post-sync)

PR #984 (v0.5.2) hardened the MCP server and added new capabilities. PR #1231
(master only) adds a TCP event server as an alternative to MCP polling. After
the v0.5.2 sync, `mcp_client.py` must be tested against the updated MCP tool
schema. If the TCP event server is adopted in a future sync, this ACL will need
a new transport adapter.

---

## 3. Sidecar Container ACL (MeshExtraction)

**External systems**: Three isolated Docker containers running different
CUDA/Python environments: `milo` sidecar (CUDA 11.8, Python 3.9),
`come` sidecar (CUDA 12.1, Python 3.10), and the main container (CUDA 12.8,
Python 3.12).

**Isolates**: MeshExtraction Context from the container-specific CLI
invocations, file path conventions, environment setup, and error output
format of each sidecar backend.

**Modules**: `milo_extractor.py`, `come_extractor.py` (planned),
`gaussianwrapping_extractor.py` (planned)

### Design pattern

Each sidecar ACL implements a common adapter interface:

```python
class MeshSidecarAdapter:
    """Abstract sidecar adapter interface (not yet formalised as ABC in code)."""

    def is_available(self) -> bool:
        """Check if the sidecar container/conda env is reachable."""
        ...

    def extract(
        self,
        colmap_dir: Path,
        output_dir: Path,
        config: dict,       # backend-specific config dict
    ) -> MeshSidecarResult:
        """Run extraction; return domain result or raise SidecarError."""
        ...

class MeshSidecarResult:
    success: bool
    ply_path: str | None
    mesh_path: str | None
    glb_path: str | None
    duration: float
    error: str | None
```

The calling code in `stages.py::mesh_objects()` and the `BackendSelector`
policy see only `MeshSidecarResult` — never the raw sidecar output.

### MILo sidecar ACL (`milo_extractor.py`)

| Domain concept | MILo external concept | Translation |
|----------------|----------------------|-------------|
| `extract(colmap_dir, output_dir, config)` | `docker exec milo python train.py --source_path ... --output_path ...` | `_milo_exec_prefix()` builds the docker exec prefix; `run_milo()` assembles the full command |
| `MiloConfig.rasterizer` | `--rasterizer radegs\|gof\|ms` CLI flag | Enum-to-flag mapping |
| `MiloConfig.imp_metric` | `--imp_metric indoor\|outdoor` | String passthrough |
| MILo `mesh.ply` output | `MeshAsset.mesh_obj_path` | `load_milo_mesh()` finds the output file by glob pattern |
| MILo `point_cloud.ply` output | `GaussianModel.ply_path` | Copied to main container's `{job_dir}/model_milo/` |
| Docker container name `milo` | `_milo_exec_prefix()` | `docker exec milo` prefix; falls back to `conda run -n milo` |
| Sidecar unavailable | `is_milo_available() == False` | Falls back to TSDF backend transparently |

### CoMe sidecar ACL (`come_extractor.py` — planned)

| Domain concept | CoMe external concept | Translation |
|----------------|----------------------|-------------|
| `extract(colmap_dir, output_dir, config)` | `docker exec come python train.py --source_path ...` then `python extract_mesh.py ...` | Two-phase CLI: train (~18 min) then extract (~7 min) |
| COLMAP dataset path | CoMe `--source_path` | Direct passthrough; CoMe accepts standard COLMAP format (images/ + sparse/) |
| CoMe mesh output (geometry-only PLY) | `MeshAsset.mesh_obj_path` | CoMe does not produce UV textures; texturing pass (xatlas + reprojection) runs in main container after extraction |
| CoMe confidence threshold | `--conf_threshold float` | From `config.mesh.come_confidence_threshold` |
| Container `come` | `docker exec come` | Same pattern as MILo |

**License note** (ADR candidate): CoMe uses SOF custom license (NOASSERTION).
The ACL layer cleanly separates CoMe from our codebase — if license review
blocks usage, only `come_extractor.py` needs to be removed.

### GaussianWrapping sidecar ACL (`gaussianwrapping_extractor.py` — planned)

GaussianWrapping shares the MILo sidecar container (both run CUDA 11.8,
Python 3.9), but runs a different set of CLI commands.

| Domain concept | GaussianWrapping external concept | Translation |
|----------------|----------------------------------|-------------|
| `extract(colmap_dir, output_dir, config)` | `docker exec milo python /opt/gaussianwrapping/train.py ...` | Different Python path inside the same container |
| `GWConfig.rasterizer` | `--rasterizer radegs\|median_depth` CLI flag | `radegs` = higher quality; `median_depth` = faster (preview) |
| `GWConfig.primal_adaptive` | `--primal_adaptive_meshing` flag | Enables high-resolution targeted extraction for specific regions |
| Output mesh | `MeshAsset.mesh_glb_path` | GLB export via GaussianWrapping's built-in exporter |
| Thin-structure scene type | `BackendSelector` policy → `GaussianWrapping` | Policy decision happens in main container before dispatching to sidecar |
| Container `milo` | `docker exec milo python /opt/gaussianwrapping/...` | Re-uses MILo's container; path prefix distinguishes the two backends |

**License note**: GaussianWrapping has no formal license file. The ACL layer
isolates the risk — if license review is negative, only
`gaussianwrapping_extractor.py` is affected.

---

## 4. BOUNDARIES.md ACL (Upstream Fork Boundary)

**External system**: MrNeRF/LichtFeld-Studio upstream repository.

**Isolates**: Our Gaussian Toolkit domain from upstream LichtFeld Studio's
internal architecture, build system, and C++ API changes.

**Physical artifact**: `BOUNDARIES.md` — the written merge policy.
**Enforcement**: Git merge workflow (conflict resolution rules), directory
ownership conventions, and the Docker architecture (our code is never in
upstream directories).

### How BOUNDARIES.md functions as an ACL

The upstream/our boundary is not a software adapter but an architectural
convention backed by directory ownership:

| Upstream territory | Our territory | ACL rule |
|-------------------|---------------|----------|
| `src/core/`, `src/app/`, `src/mcp/`, etc. | `src/pipeline/`, `src/web/`, `docker/` | Conflicts in upstream dirs resolve in favour of upstream; our dirs are never modified upstream |
| `CMakeLists.txt`, `vcpkg.json` | none (we don't build our own C++) | Accept upstream; re-add our build-arg additions post-merge |
| `LICENSE` (GPL-3.0) | our modules (separate license status) | Our `src/pipeline/` modules inherit GPL-3.0 from the repo; must not add code with incompatible licenses |
| Upstream coordinate conventions | `coordinate_transform.py` | Any upstream coordinate change (e.g. PR #1066) requires updating our ACL (coordinate_transform.py), not the rest of the pipeline |
| LichtFeld MCP tool schema | `mcp_client.py` | Breaking MCP changes (e.g. post PR #984) require updating mcp_client.py only |

### v2 sync trigger points

Each upstream sync event has an ACL checkpoint:

```
Sync to v0.5.2 tag
    → CMakeLists.txt: accept upstream, preserve our CUDA 12.8 build arg
    → mcp_client.py: test all 70+ MCP tools against updated server (#984)
    → coordinate_transform.py: v0.5.2 does NOT include PR #1066 (safe)
    → usd_assembler.py: test LichtFeld native USD export as potential replacement

Future sync to master (v0.5.3 Vulkan)
    → coordinate_transform.py: REQUIRED update for PR #1066 convention change
    → mcp_client.py: TCP event server (#1231) may replace polling adapter
    → Dockerfile.consolidated: add vulkan-tools, mesa-vulkan-drivers
    → Any call to CUDA renderer in mcp_client.py: remove (renderer removed #1234)
```

---

## 5. Blender ACL

**External system**: Blender 5.0.1 subprocess (Python bpy API inside
a Blender-embedded Python runtime).

**Isolates**: SceneAssembly Context from Blender's bpy API, Python version
(Blender embeds its own Python 3.11), and subprocess communication model.

**Module**: `blender_assembler.py`

### Translation contract

| Domain concept | Blender external concept | Translation |
|----------------|--------------------------|-------------|
| `assemble_scene(objects, background, output_path)` | `blender --background --python script.py` subprocess | `blender_assembler.py` generates a temporary bpy script, launches Blender headless, parses stdout |
| `MeshAsset.mesh_obj_path` | `bpy.ops.import_scene.obj(filepath=...)` | OBJ import via Blender's built-in importer |
| `Transform3D` | `obj.location`, `obj.rotation_euler`, `obj.scale` | Matrix decomposition; Y-up convention passed as Blender axis setting |
| `UsdScene.usd_path` | `bpy.ops.wm.usd_export(filepath=...)` | Blender's native USD exporter (Blender 5.0 has good USD support) |
| Blender process crash | `StageResult.success = False` | stderr parsing; non-zero returncode → stage failure |

---

## 6. ComfyUI ACL

**External system**: ComfyUI HTTP API (diffusion workflow server,
running locally on a configurable port).

**Isolates**: Ingestion Context (person removal) and SceneAssembly Context
(background inpainting) from ComfyUI's workflow graph JSON format and
websocket-based progress model.

**Modules**: `comfyui_inpainter.py`, `person_remover.py` (FLUX endpoint)

### Translation contract

| Domain concept | ComfyUI external concept | Translation |
|----------------|--------------------------|-------------|
| `inpaint(image, mask)` → inpainted image | ComfyUI workflow graph JSON | `comfyui_inpainter.py` holds a pre-built workflow template; substitutes image/mask paths |
| FLUX inpainting result | `ComfyUI /history/{prompt_id}` → image URL | HTTP polling until `status == "complete"`; downloads output image |
| ComfyUI unavailable | `config.inpaint.enabled = False` | ACL returns no-op; stage skips gracefully |

---

## 7. ACL Inventory Summary

| ACL | Module(s) | Protects | Risk |
|-----|-----------|----------|------|
| COLMAP binary format + CLI | `colmap_parser.py`, `coordinate_transform.py`, `stages.py::reconstruct()` | Reconstruction Context | MEDIUM — PR #1066 coordinate risk on master sync |
| LichtFeld MCP (Conformist) | `mcp_client.py` | Training, SceneAssembly | MEDIUM — MCP schema changed in v0.5.2 (#984) |
| MILo sidecar | `milo_extractor.py` | MeshExtraction | LOW — stable; container boundary is well-defined |
| CoMe sidecar | `come_extractor.py` (planned) | MeshExtraction | MEDIUM — new; code released 2026-04-22, not yet validated |
| GaussianWrapping sidecar | `gaussianwrapping_extractor.py` (planned) | MeshExtraction | MEDIUM — new; no formal license; shares MILo container |
| BOUNDARIES.md fork policy | merge workflow + directory convention | Entire codebase | LOW — convention is working; enforced by code review |
| Blender subprocess | `blender_assembler.py` | SceneAssembly | LOW — stable; Blender 5.0 USD export is mature |
| ComfyUI HTTP | `comfyui_inpainter.py`, `person_remover.py` | Ingestion, SceneAssembly | LOW — optional; graceful no-op if unavailable |

---

## 8. ACL Design Principles (for v2 additions)

When adding a new external integration (e.g. RT-Splatting, 2Xplat, 4C4D):

1. **Never import an external system's types into a domain module.** Domain
   modules (`stages.py`, `config.py`, etc.) should only reference our own
   `MeshBackend`, `TrainingStrategy`, etc. The ACL module does the translation.

2. **One ACL module per external system.** `milo_extractor.py` handles MILo.
   `come_extractor.py` will handle CoMe. They do not cross-contaminate.

3. **Availability checks are part of the ACL.** `is_milo_available()` lives in
   `milo_extractor.py`, not in `stages.py`. The domain code just calls
   `BackendSelector.select(...)` and gets a `MeshBackend` — it never checks
   container availability directly.

4. **Error translation.** Sidecar errors (stderr, non-zero exit codes) are
   translated into `StageResult.error` strings before leaving the ACL. The
   domain never sees raw subprocess output.

5. **Fallback is a domain policy, not an ACL responsibility.** The ACL
   (`milo_extractor.py`) reports availability and results. The `BackendSelector`
   policy in the domain decides which backend to try next. This keeps fallback
   logic in one place (the policy) rather than scattered across ACL adapters.
