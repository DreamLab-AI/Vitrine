# Aggregates — Gaussian Toolkit v2

**Extends**: research/decisions/ddd-domain-model.md (v1 model)
**Alignment**: research/decisions/prd-v2-upgrade.md
**Date**: 2026-05-26

Each section follows the pattern:
  Aggregate Root / Entities / Value Objects / Invariants / Domain Events / Repository interface

---

## 1. ReconstructionJob (Root Aggregate)

**Bounded context**: Orchestration (spans all contexts)

**Purpose**: The top-level consistency boundary for a single video-to-scene
processing run. Owns the lifecycle and state of every subordinate aggregate.
Enforces pipeline-level invariants (e.g. quality gates) before allowing
downstream stages to proceed.

### Entities (owned by ReconstructionJob)

```
ReconstructionJob {
    id: UUID                         # job identifier; also {job_dir} basename
    job_dir: Path                    # {output_dir}/{id}/
    video_path: Path                 # original video input (immutable after creation)
    config: PipelineConfig           # full config snapshot at job creation time
    state: JobState                  # current lifecycle state (value object)
    frame_set_id: UUID | None        # populated after Ingestion completes
    colmap_dataset_id: UUID | None   # populated after Reconstruction completes
    gaussian_model_id: UUID | None   # populated after Training completes
    scene_graph_id: UUID | None      # populated after SceneAssembly completes
    delivery_artifact_id: UUID | None
    created_at: datetime
    completed_at: datetime | None
    error: str | None
}
```

### Value Objects

```
JobState {
    stage: StageEnum   # PENDING | INGESTING | RECONSTRUCTING | TRAINING |
                       # SEGMENTING | MESHING | ASSEMBLING | DELIVERING |
                       # DONE | FAILED
    progress: float    # 0.0–1.0 within current stage
    retry_count: int
    started_at: datetime
    stage_started_at: datetime
}

PipelineConfig {
    # Snapshot of config.PipelineConfig at job creation.
    # Immutable after recording; stored as JSON in {job_dir}/config.json.
    ingest: IngestConfig
    reconstruct: ReconstructConfig
    training: TrainingConfig
    decompose: DecomposeConfig
    mesh: MeshConfig
    ...
}
```

### Invariants

1. `job_dir` must exist and be writable before any stage executes.
2. Stage ordering is strictly `STAGE_NAMES` (defined in `stages.py`). A job
   cannot transition to RECONSTRUCTING before Ingestion succeeds.
3. PSNR >= `config.quality.min_psnr` (default 25.0) before mesh extraction.
4. COLMAP registration rate >= 30% before training.
5. If `state.stage == FAILED`, `error` must be non-null.
6. A completed job (`state.stage == DONE`) must have at least one delivery
   artifact (USD or compressed splat).

### Domain Events

```
JobCreated {
    job_id: UUID
    video_path: str
    config_snapshot: dict
    timestamp: datetime
}

StageStarted {
    job_id: UUID
    stage: str
    timestamp: datetime
}

StageCompleted {
    job_id: UUID
    stage: str
    metrics: dict          # stage-specific metrics from StageResult
    artifacts: dict        # artifact paths
    duration_seconds: float
    timestamp: datetime
}

QualityGatePassed {
    job_id: UUID
    gate_name: str         # e.g. "colmap_registration", "training_psnr"
    metrics: dict
    timestamp: datetime
}

QualityGateFailed {
    job_id: UUID
    gate_name: str
    metrics: dict
    retry_count: int
    timestamp: datetime
}

JobCompleted {
    job_id: UUID
    total_duration_seconds: float
    final_metrics: dict
    artifact_paths: dict
    timestamp: datetime
}

JobFailed {
    job_id: UUID
    stage: str
    error: str
    partial_output_path: str | None
    timestamp: datetime
}
```

### Repository Interface

```python
class ReconstructionJobRepository:
    def save(self, job: ReconstructionJob) -> None: ...
    def find_by_id(self, job_id: UUID) -> ReconstructionJob | None: ...
    def find_all_active(self) -> list[ReconstructionJob]: ...
    def find_by_state(self, stage: StageEnum) -> list[ReconstructionJob]: ...
    def delete(self, job_id: UUID) -> None: ...
```

*Implementation*: JSON file at `{job_dir}/job_state.json` (current impl in
`src/web/job_manager.py`). Redis-backed implementation planned for
multi-worker deployments.

---

## 2. GaussianModel (Aggregate Root)

**Bounded context**: Training

**Purpose**: Encapsulates a trained 3D Gaussian Splatting model. Owns training
metadata, the physical PLY artifact, and per-Gaussian segmentation labels
once applied.

### Entities

```
GaussianModel {
    id: UUID
    job_id: UUID
    ply_path: Path                 # {job_dir}/model/.../point_cloud.ply
    num_gaussians: int
    training_iterations: int
    strategy: TrainingStrategy     # value object
    scene_preset: str              # "default" | "indoor_reflective"
    final_loss: float | None
    psnr: float | None
    ssim: float | None
    has_segmentation_labels: bool
    objects: list[GaussianObject]  # populated after Segmentation
    backend: str                   # "lichtfeld" | "milo" | "gsplat"
    created_at: datetime
}

GaussianObject {
    id: UUID
    model_id: UUID
    label: str                   # semantic name, e.g. "chair_001"
    object_id: int               # integer id from segmentor
    num_gaussians: int
    bounding_box: AABB           # value object
    gaussian_ply_path: Path      # per-object extracted PLY
    confidence: float            # segmentation confidence
    mesh_asset_id: UUID | None   # set after MeshExtraction
}
```

### Value Objects

```
TrainingStrategy {
    name: str          # "mrnf" | "mcmc" | "igs_plus" | "default"
    sh_degree: int     # 0–3
    iterations: int
    densification_interval: int | None
}

AABB {
    min: Vec3
    max: Vec3
}

Vec3 {
    x: float
    y: float
    z: float
}

QualityMetrics {
    psnr: float
    ssim: float
    lpips: float | None
    mesh_roundtrip_ratio: float | None
}
```

### Invariants

1. `ply_path` must exist before any Segmentation or MeshExtraction event fires.
2. `num_gaussians` > 10,000 for a valid non-trivial scene.
3. If `strategy.name == "mrnf"`, `training_iterations` >= 7,000
   (MRNF requires warmup period).
4. If `has_segmentation_labels == True`, every `GaussianObject` must have
   a non-empty `gaussian_ply_path`.

### Domain Events

```
TrainingStarted {
    model_id: UUID
    job_id: UUID
    strategy: str
    iterations: int
    backend: str
    timestamp: datetime
}

TrainingProgress {
    model_id: UUID
    iteration: int
    loss: float
    num_gaussians: int
    timestamp: datetime
}

ModelTrained {
    model_id: UUID
    job_id: UUID
    ply_path: str
    num_gaussians: int
    psnr: float | None
    duration_seconds: float
    backend: str
    timestamp: datetime
}

FramesExtracted {
    job_id: UUID
    frame_count: int
    selected_count: int
    fps: float
    selection_strategy: str    # "fibonacci" | "sequential"
    timestamp: datetime
}

PosesEstimated {
    job_id: UUID
    colmap_dir: str
    registered_cameras: int
    total_frames: int
    registration_rate: float
    reprojection_error: float | None
    timestamp: datetime
}

GaussianObjectLabelled {
    model_id: UUID
    object_id: int
    label: str
    mask_pixels: int
    segmentation_method: str   # "sam3" | "sam2" | "full_scene"
    timestamp: datetime
}
```

### Repository Interface

```python
class GaussianModelRepository:
    def save(self, model: GaussianModel) -> None: ...
    def find_by_id(self, model_id: UUID) -> GaussianModel | None: ...
    def find_by_job_id(self, job_id: UUID) -> GaussianModel | None: ...
    def find_by_strategy(self, strategy: str) -> list[GaussianModel]: ...
```

---

## 3. MeshAsset (Aggregate Root)

**Bounded context**: MeshExtraction

**Purpose**: Encapsulates a polygonal mesh extracted from a GaussianObject (or
full scene). Owns the mesh geometry, UV atlas, texture, and extraction metadata.
Enforces geometry quality invariants before passing to SceneAssembly.

This aggregate changed structurally in v2 because the extraction_backend field
now carries meaning that triggers different ACLs and container paths.

### Entities

```
MeshAsset {
    id: UUID
    gaussian_object_id: UUID | None   # None for full-scene meshes
    job_id: UUID
    label: str
    mesh_glb_path: Path               # primary format for web viewer
    mesh_obj_path: Path | None        # OBJ + MTL for USD / Blender
    texture_path: Path | None         # diffuse PNG (None = vertex-colored)
    material_path: Path | None        # .mtl
    vertex_count: int
    face_count: int
    has_uv: bool
    extraction_backend: MeshBackend   # value object (see below)
    quality_score: float | None       # round-trip PSNR ratio
    created_at: datetime
}
```

### Value Objects

```
MeshBackend {
    name: str       # "tsdf" | "milo" | "come" | "gaussianwrapping" |
                    # "hunyuan3d" | "open3d" | "pointcloud"
    container: str  # "main" | "milo_sidecar" | "come_sidecar"
    cuda_version: str   # "12.8" | "11.8" | "12.1"
    is_sidecar: bool
}

# Canonical instances:
TSDF_BACKEND       = MeshBackend("tsdf", "main", "12.8", False)
MILO_BACKEND       = MeshBackend("milo", "milo_sidecar", "11.8", True)
COME_BACKEND       = MeshBackend("come", "come_sidecar", "12.1", True)
GAUSSIANWRAPPING_BACKEND = MeshBackend("gaussianwrapping", "milo_sidecar", "11.8", True)
```

**Backend selection domain policy** (`BackendSelector` service):

```python
class BackendSelector:
    """Domain policy: choose mesh extraction backend from scene context."""

    def select(
        self,
        scene_type: str,          # "preview" | "quality" | "thin_structure"
        config_backend: str,       # explicit override from PipelineConfig
        milo_available: bool,
        come_available: bool,
        gaussianwrapping_available: bool,
    ) -> MeshBackend:
        if config_backend == "milo" and milo_available:
            return MILO_BACKEND
        if config_backend == "come" and come_available:
            return COME_BACKEND
        if config_backend == "gaussianwrapping" and gaussianwrapping_available:
            return GAUSSIANWRAPPING_BACKEND
        if scene_type == "preview":
            return TSDF_BACKEND
        if scene_type == "thin_structure" and gaussianwrapping_available:
            return GAUSSIANWRAPPING_BACKEND
        if come_available:
            return COME_BACKEND   # prefer CoMe (3x faster than MILo, comparable quality)
        if milo_available:
            return MILO_BACKEND
        return TSDF_BACKEND       # always-available fallback
```

### Invariants

1. `vertex_count` >= 100 for a valid mesh (below 100 indicates degenerate
   extraction — all backends tried, convex hull used).
2. `face_count` >= 50.
3. If `has_uv == True`, `texture_path` must exist on disk.
4. `extraction_backend.name` must be one of the canonical names above.
5. `mesh_glb_path` must exist before `MeshExtracted` event fires.

### Domain Events

```
MeshExtractionStarted {
    asset_id: UUID
    job_id: UUID
    label: str
    backend: str
    container: str
    timestamp: datetime
}

MeshExtracted {
    asset_id: UUID
    job_id: UUID
    label: str
    vertex_count: int
    face_count: int
    has_uv: bool
    backend: str
    duration_seconds: float
    glb_path: str
    timestamp: datetime
}

MeshExtractionFailed {
    job_id: UUID
    label: str
    backend: str
    error: str
    fallback_backend: str | None
    timestamp: datetime
}

TextureBaked {
    asset_id: UUID
    texture_path: str
    face_count: int
    timestamp: datetime
}
```

### Repository Interface

```python
class MeshAssetRepository:
    def save(self, asset: MeshAsset) -> None: ...
    def find_by_id(self, asset_id: UUID) -> MeshAsset | None: ...
    def find_by_job_id(self, job_id: UUID) -> list[MeshAsset]: ...
    def find_by_gaussian_object_id(
        self, gaussian_object_id: UUID
    ) -> MeshAsset | None: ...
    def find_by_backend(self, backend_name: str) -> list[MeshAsset]: ...
```

---

## 4. SceneGraph (Aggregate Root)

**Bounded context**: SceneAssembly

**Purpose**: Represents the assembled 3D scene as a USD scene graph. Owns the
mapping from GaussianObjects and MeshAssets to USD prims. Enforces coordinate
system and schema invariants.

### Entities

```
SceneGraph {
    id: UUID
    job_id: UUID
    usd_path: Path                # main .usda / .usdc
    up_axis: str                  # "Y" (always, our convention)
    meters_per_unit: float        # 1.0 (always)
    scene_prims: list[ScenePrim]
    cameras: list[UsdCamera]
    assembler_backend: str        # "usd_assembler" | "lichtfeld_native" | "blender"
    created_at: datetime
}

ScenePrim {
    id: UUID
    scene_graph_id: UUID
    prim_path: str               # e.g. "/World/Objects/Chair_001"
    gaussian_object_id: UUID | None
    mesh_asset_id: UUID | None
    transform: Transform3D       # value object
    has_gaussian_variant: bool
    has_mesh_variant: bool
    material_id: UUID | None
}

UsdCamera {
    id: UUID
    scene_graph_id: UUID
    prim_path: str               # e.g. "/World/Cameras/Camera_001"
    focal_length_mm: float
    sensor_width_mm: float
    transform: Transform3D
}
```

### Value Objects

```
Transform3D {
    translation: Vec3
    rotation: Quaternion         # XYZW
    scale: Vec3
}

Quaternion {
    x: float
    y: float
    z: float
    w: float
}
```

### Invariants

1. `up_axis` must be "Y" (our published coordinate convention).
2. `meters_per_unit` must be 1.0.
3. Every `ScenePrim.prim_path` must be unique within the graph.
4. Every `ScenePrim` must have at least one of `has_gaussian_variant` or
   `has_mesh_variant` == True.
5. `usd_path` must point to a valid USD file before `SceneAssembled` fires.

### Domain Events

```
SceneAssembled {
    scene_graph_id: UUID
    job_id: UUID
    prim_count: int
    object_count: int
    usd_path: str
    assembler_backend: str
    timestamp: datetime
}

PrimAdded {
    scene_graph_id: UUID
    prim_path: str
    gaussian_object_id: UUID | None
    mesh_asset_id: UUID | None
    has_variants: bool
    timestamp: datetime
}
```

### Repository Interface

```python
class SceneGraphRepository:
    def save(self, graph: SceneGraph) -> None: ...
    def find_by_id(self, graph_id: UUID) -> SceneGraph | None: ...
    def find_by_job_id(self, job_id: UUID) -> SceneGraph | None: ...
```

---

## 5. DeliveryArtifact (Aggregate Root)

**Bounded context**: Delivery

**Purpose**: Represents the final packaged output of a pipeline run — the
products a user can download or preview in a browser. Owns compressed splat
variants alongside USD and GLB outputs.

### Entities

```
DeliveryArtifact {
    id: UUID
    job_id: UUID
    scene_graph_id: UUID | None
    outputs: list[ArtifactOutput]  # one per format
    download_bundle_path: Path | None   # ZIP of all outputs
    web_preview_url: str | None
    created_at: datetime
}

ArtifactOutput {
    id: UUID
    artifact_id: UUID
    format: ArtifactFormat         # value object
    path: Path
    size_bytes: int
    checksum_sha256: str
}
```

### Value Objects

```
ArtifactFormat {
    name: str     # "usd" | "ply" | "ksplat" | "glb" | "glb_khr_gs" | "alembic"
    mime_type: str
    is_compressed: bool
    compression_method: str | None  # "splat_transform" | "gzip" | None
}

# Canonical instances (v2):
USD_FORMAT         = ArtifactFormat("usd", "model/vnd.usdz+zip", False, None)
PLY_FORMAT         = ArtifactFormat("ply", "application/octet-stream", False, None)
KSPLAT_FORMAT      = ArtifactFormat("ksplat", "application/octet-stream", True, "splat_transform")
GLB_FORMAT         = ArtifactFormat("glb", "model/gltf-binary", False, None)
GLB_KHR_GS_FORMAT  = ArtifactFormat("glb_khr_gs", "model/gltf-binary", False, None)
```

### Invariants

1. At least one `ArtifactOutput` must exist before `ArtifactPublished` fires.
2. If `KSPLAT_FORMAT` output exists, a corresponding `PLY_FORMAT` output must
   also exist (the PLY is always kept alongside its compressed form).
3. `size_bytes` must match the actual file size on disk.

### Domain Events

```
SplatOptimized {
    job_id: UUID
    input_ply_path: str
    output_ksplat_path: str
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    operations: list[str]      # e.g. ["crop", "filter", "sort", "compress"]
    timestamp: datetime
}

ArtifactPublished {
    artifact_id: UUID
    job_id: UUID
    formats: list[str]
    download_url: str | None
    total_size_bytes: int
    timestamp: datetime
}
```

### Repository Interface

```python
class DeliveryArtifactRepository:
    def save(self, artifact: DeliveryArtifact) -> None: ...
    def find_by_id(self, artifact_id: UUID) -> DeliveryArtifact | None: ...
    def find_by_job_id(self, job_id: UUID) -> DeliveryArtifact | None: ...
```

---

## 6. Domain Services

### BackendSelector (MeshExtraction context)

Stateless domain service implementing the backend selection policy
(see `MeshAsset` aggregate above). Lives in `mesh_extractor.py` or a new
`mesh_backend_policy.py`.

### QualityGateService (Orchestration cross-cutting)

Stateless. Takes `StageResult` metrics + `PipelineConfig.quality` thresholds
and returns a `GateDecision {passed, gate_name, reason, retry_advised}`.
Implemented in `quality_gates.py`.

### CoordinateNormaliser (Reconstruction context)

Stateless service that applies the Y-up / meters / right-hand coordinate
transform to COLMAP outputs. Implemented in `coordinate_transform.py`.
Enforces invariant: any artifact leaving the Reconstruction context is in
Y-up, 1 metre = 1 unit, right-hand coordinate frame.

**CAUTION (v2)**: Upstream PR #1066 (in master, not v0.5.2) changes the
upstream coordinate convention. This service must be validated against the
upstream convention after any sync to master.

### SplatOptimizerService (Delivery context)

Wraps the `splat-transform` npm CLI. Accepts a PLY path and a
`SplatOptConfig` value object, returns a `SplatOptResult` with paths and
compression metrics. Implemented in `splat_optimizer.py`.

```
SplatOptConfig {
    crop_bbox: AABB | None
    opacity_threshold: float    # default 0.01
    max_scale: float | None
    sort: bool                  # default True
    compress: bool              # default True
    output_format: str          # "ksplat" | "splat"
}

SplatOptResult {
    ksplat_path: Path
    original_size_bytes: int
    compressed_size_bytes: int
    gaussians_before: int
    gaussians_after: int
}
```

---

## 7. Aggregate Relationships (Event Flow)

```
JobCreated
    → ReconstructionJob created

FramesExtracted
    → ReconstructionJob.frame_set_id set
    → QualityGatePassed/Failed (frame count gate)

PosesEstimated
    → ReconstructionJob.colmap_dataset_id set
    → QualityGatePassed/Failed (registration rate gate)

ModelTrained
    → GaussianModel created
    → ReconstructionJob.gaussian_model_id set
    → QualityGatePassed/Failed (PSNR gate)

GaussianObjectLabelled (x N per object)
    → GaussianModel.objects populated

MeshExtracted (x N per object)
    → MeshAsset created per object
    → GaussianObject.mesh_asset_id set

SceneAssembled
    → SceneGraph created
    → ReconstructionJob.scene_graph_id set

SplatOptimized
    → DeliveryArtifact.outputs updated with ksplat entry

ArtifactPublished
    → DeliveryArtifact created
    → ReconstructionJob.delivery_artifact_id set
    → JobCompleted fires
```
