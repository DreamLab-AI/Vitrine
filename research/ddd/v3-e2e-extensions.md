# Domain Model Extensions — v3 End-to-End Closure

**Date**: 2026-06-04
**Status**: Draft

## What this document is

This document **EXTENDS** the existing v2 domain model. It does **not** restate or
rewrite it. It adds new aggregates, entities, value objects, and ubiquitous-language
terms required to close the aspirational end-to-end workflow, traced through the
fourteen deltas (D1–D14) of
[`../decisions/gap-analysis-e2e-aspiration.md`](../decisions/gap-analysis-e2e-aspiration.md)
and the forty functional requirements of
[`../decisions/prd-v3-e2e-closure.md`](../decisions/prd-v3-e2e-closure.md).

It builds on, and must be read alongside, the four existing model files:

- [`bounded-contexts.md`](bounded-contexts.md)
- [`aggregates.md`](aggregates.md)
- [`ubiquitous-language.md`](ubiquitous-language.md)
- [`anti-corruption-layers.md`](anti-corruption-layers.md)

It aligns by exact filename with the commissioned decision records (created by a
peer agent; cross-referenced here whether or not they are yet present):
[`adr-009-per-video-ingest-and-metadata.md`](../decisions/adr-009-per-video-ingest-and-metadata.md),
[`adr-010-key-item-hull-recon.md`](../decisions/adr-010-key-item-hull-recon.md),
[`adr-011-usd-metadata-enrichment.md`](../decisions/adr-011-usd-metadata-enrichment.md),
[`adr-012-sota-tooling-modernisation.md`](../decisions/adr-012-sota-tooling-modernisation.md),
[`adr-013-ingest-manifest-serial-model-lifecycle.md`](../decisions/adr-013-ingest-manifest-serial-model-lifecycle.md),
[`adr-014-agent-controlled-comfyui-integration.md`](../decisions/adr-014-agent-controlled-comfyui-integration.md),
[`adr-015-vitrine-web-onboarding.md`](../decisions/adr-015-vitrine-web-onboarding.md). The §2.6 agent
loop is ADR-014; the §7 Onboarding/Setup context is ADR-015.

### Exactly what this extension builds on (so nothing is duplicated)

**Existing bounded contexts reused as-is** (from `bounded-contexts.md` §2):
Ingestion, Reconstruction, Training, Segmentation, MeshExtraction, SceneAssembly,
Delivery, and the cross-cutting Orchestration published language. This document
**refines** Ingestion, **splits** a new Per-Object Reconstruction context out of
Segmentation + MeshExtraction, and **extends** SceneAssembly. The other contexts are
untouched.

**Existing root aggregates reused as-is** (from `aggregates.md`):
`ReconstructionJob`, `GaussianModel` (and its `GaussianObject` entity, `AABB`/`Vec3`
value objects), `MeshAsset` (and its `MeshBackend` value object), `SceneGraph` (and
its `ScenePrim`/`UsdCamera` entities, `Transform3D`/`Quaternion` value objects),
`DeliveryArtifact`. The new aggregates here are **owned by, or referenced from**,
these — they do not replace them. In particular:

- `Video` and `CaptureSession` sit **upstream** of `ReconstructionJob`; a
  `CaptureSession` yields exactly one `ReconstructionJob`.
- `KeyItem` is a **refinement of selection over** existing `GaussianObject`s — it does
  not replace `GaussianObject`; it ranks and gates a subset of them.
- `Hull` is a **specialisation of** `MeshAsset` for per-object watertight recon; it
  reuses `MeshBackend` (the `hunyuan3d` instance already enumerated in
  `aggregates.md` §3).
- `MetadataRecord`, `EnvironmentMesh`, and `Lineage` are **added inside** `SceneGraph`.

**Existing ubiquitous-language terms reused, never redefined**: frame (v1
`Frame` entity is promoted here — see note in §1), frame set, blur score, exposure
gate, Fibonacci sampling, viewpoint coverage, COLMAP dataset, registration rate,
splat, Gaussian, PLY, GaussianObject, label, object_id, mask, mask projection,
concept prompt, MeshBackend, watertight mesh, UV atlas, texture bake, prim, prim path,
variant set, UsdPreviewSurface, Xform, sidecar (the **Docker** sense), StageResult,
quality gate, artifact, job, job directory. Where this document needs one of these, it
**uses the existing spelling**. The §4 glossary lists **only genuinely new** terms.

**Existing ACLs reused, never redefined** (from `anti-corruption-layers.md`):
COLMAP ACL, LichtFeld MCP ACL, the Sidecar Container ACL family (MILo/CoMe/
GaussianWrapping), BOUNDARIES.md fork ACL, Blender ACL, ComfyUI ACL. The §5 notes
**extend** the ComfyUI ACL (new per-object inpaint usage), and add adapter notes for
two boundaries the new aggregates newly cross: the **rclone/Drive** boundary and the
**Hunyuan3D** boundary. The OpenUSD boundary is already owned by SceneAssembly; §5
records only the new `v2g:*`/lineage translation responsibility.

> Terminology note — `sidecar` is overloaded by design. In `ubiquitous-language.md`
> it means a companion Docker container (MILo/CoMe). In this document, the
> per-frame metadata file is consistently called a **sidecar tag** (never bare
> "sidecar") to avoid collision. Both spellings are glossed in §4.

---

## 1. Ingestion context — refinement (D1, D2, D3, D14-root)

**Refines** `bounded-contexts.md` §2.1. The Ingestion context today produces a single
pooled `FrameSet` from a session folder (`drive_ingestor.py` `_extract_pooled_frames`).
v3 introduces a **per-video unit of work** and a **per-image provenance record**, while
preserving the existing pooled `FrameSet` as the output handed to Reconstruction.

The crux (from ADR-009): **ingest is per-video, reconstruction is combined.** A room is
captured as several videos; each video is copied, extracted, tagged, and purged
*individually* (so local NVMe never holds more than one raw video — `prd-v3` NFR-3),
but all surviving frames are *pooled* into one `FrameSet` that drives **one** combined
reconstruction. This per-video-ingest / combined-reconstruct tension is modelled
explicitly by making `CaptureSession` the pooling aggregate that owns many `Video`s and
yields one `ReconstructionJob`.

### 1.1 New aggregate root: `Video`

The per-video unit of work. Its lifecycle ledger row (extending the session ledger at
`drive_ingestor.py:137-200` to video granularity) is what makes an overnight batch
resumable and retention-bounded.

```
Video {                              # Aggregate Root (Ingestion)
    id: UUID
    session_id: UUID                 # owning CaptureSession
    source_drive_path: str           # rclone remote path — source of truth, never deleted
    local_scratch_path: Path | None  # NVMe copy; None once purged
    checksum_sha256: str             # verifies the Drive copy and the local copy match
    duration_seconds: float | None
    expected_frame_count: int | None # from duration × fps — used by the verified-extraction rule
    frames: list[Frame]              # extracted Frame entities (this Video only)
    state: VideoState                # value object (lifecycle below)
    created_at: datetime
    error: str | None
}

VideoState {                         # Value Object
    phase: VideoPhase                # pending → copied → extracted → tagged → purged → done | failed
    retained_frame_count: int        # frames surviving the per-video quality gate
    started_at: datetime
    phase_started_at: datetime
}
```

`VideoPhase` lifecycle (the per-video state machine FR-1/FR-7 resume against):

```
pending  → copied   : Video.copy_to_scratch() succeeds, checksum verified
copied   → extracted: frames extracted; retained_frame_count recorded
extracted→ tagged   : an ImageMetadataTag written for every retained Frame
tagged   → purged   : local_scratch_path deleted (retention rule, below)
purged   → done      : Frame contributions merged into the CaptureSession FrameSet
any      → failed    : error set; Drive copy untouched (source of truth survives)
```

### 1.2 Entity: `Frame` (promoted, belongs to a `Video`)

`Frame` already exists in the v1 glossary (`ddd-domain-model.md`) as a loose entity and
appears in `ubiquitous-language.md` Ingestion. v3 **promotes** it to a first-class
entity owned by `Video` (identity matters: a frame must trace to exactly one source
video). It carries no new geometry — only identity and its tag.

```
Frame {                              # Entity (owned by Video)
    id: UUID
    video_id: UUID                   # the owning Video — root of the lineage chain
    frame_index: int                 # index within this Video's extraction
    image_path: Path
    tag: ImageMetadataTag            # value object, written at the `extracted → tagged` transition
    kept: bool                       # survived the per-video quality gate
}
```

### 1.3 Value object: `ImageMetadataTag` (D3, the sidecar tag)

The per-frame provenance record. Persisted as a JSON **sidecar tag** at
`<frame>.json` (FR-3). Immutable once written (NFR-2 idempotency). This value object
**is** the per-image metadata schema; `schema_version` makes it byte-stable (NFR-5).

```
ImageMetadataTag {                   # Value Object — written to <frame>.json
    source_video: str                # Video.id — root of the lineage chain (FR-5)
    capture_session: str             # CaptureSession.id
    frame_index: int
    source_timestamp: datetime | None# capture time if recoverable from container metadata
    blur_score: float                # reuses Ingestion "blur score"
    exposure_score: float            # reuses Ingestion "exposure gate"
    sharpness_score: float
    phash: str                       # perceptual hash — near-duplicate detection
    kept: bool                       # gate decision recorded with the frame
    selection_reason: str            # why kept/dropped: "blur", "exposure", "duplicate", "selected"
    pose_hint: PoseHint | None       # optional slot; backfilled post-COLMAP (see "pose backfill", §4)
    schema_version: str              # e.g. "v2g.frame.1" — fixes the field set (NFR-5)
}

PoseHint {                           # Value Object — optional, backfilled
    camera_position: Vec3 | None     # reuses existing Vec3
    camera_rotation_quat: Quaternion | None  # reuses existing Quaternion
    registered: bool                 # did COLMAP register the source frame
}
```

### 1.4 New aggregate root: `CaptureSession` (the pooling boundary)

Models the per-video-ingest / combined-reconstruct tension directly. Owns many
`Video`s; yields exactly **one** `ReconstructionJob`. The combined `FrameSet` is built
by pooling the retained, tagged `Frame`s of all member `Video`s in the `done` phase.

```
CaptureSession {                     # Aggregate Root (Ingestion)
    id: UUID
    drive_session_path: str          # the remote session folder
    videos: list[Video]              # many — ingested one at a time
    pooled_frame_set_id: UUID | None # the single combined FrameSet (existing artifact)
    reconstruction_job_id: UUID | None # the ONE job this session yields
    state: SessionState
}

SessionState {                       # Value Object
    phase: str                       # ingesting → pooled → reconstructing → done | failed
    videos_total: int
    videos_done: int
}
```

**Invariants**

1. A `CaptureSession` pools frames into its `FrameSet` only from `Video`s whose
   `state.phase == done`. (Per-video completion gates pooling.)
2. At most **one** `Video` per `CaptureSession` may hold a non-null `local_scratch_path`
   at any instant (retention ceiling — `prd-v3` G2 / NFR-3).
3. A `CaptureSession` yields exactly one `ReconstructionJob`
   (`reconstruction_job_id`). The combined reconstruction never runs per-video.
4. `pooled_frame_set_id` is set only after every member `Video` is `done` or `failed`,
   and at least one is `done`.

**Retention as a domain rule** (D2, FR-2). Deletion is not an implementation detail; it
is an invariant of `Video`:

> A `Video`'s `local_scratch_path` is deleted at the `tagged → purged` transition, on
> **verified extraction** only. *Verified* = `retained_frame_count >= 1` **and**
> extracted frame count `>= expected_frame_count` (tolerance per ADR-009). The
> `source_drive_path` is **never** deleted — Drive is the source of truth. A `failed`
> `Video` retains its scratch copy for diagnosis; a `purged` `Video` cannot be
> re-extracted without re-copying from Drive (idempotent: checksum match ⇒ no-op).

**New domain events** (Ingestion):

```
VideoCopied      { video_id, session_id, checksum_sha256, timestamp }
FramesExtractedPerVideo { video_id, extracted_count, retained_count, timestamp }
FramesTagged     { video_id, tag_count, schema_version, timestamp }
VideoPurged      { video_id, freed_bytes, timestamp }       # retention rule fired
SessionPooled    { session_id, frame_set_id, total_retained_frames, timestamp }
```

> Note: the existing `FramesExtracted` event (`aggregates.md`, fired per *job*) is
> retained. `FramesExtractedPerVideo` is the finer-grained per-`Video` event; the
> session-level `FramesExtracted` still fires once at pooling.

---

## 2. Per-Object Reconstruction context — new (D6, D7, D8, D9, D10, D11)

**Split** from the existing Segmentation (`bounded-contexts.md` §2.4) and MeshExtraction
(§2.5) contexts. Segmentation still labels Gaussians and produces `GaussianObject`s;
MeshExtraction still produces full-scene/environment `MeshAsset`s. What is *new* is the
per-object loop that **ranks** segmented objects into key items, **recovers** their
unseen faces, and **reconstructs** a textured watertight hull at its preserved pose.
This is the home of the Hunyuan3D hull path (`stages.py:1778-1806`, "Strategy 1").

**Relationship to existing contexts**: Customer-Supplier downstream of Segmentation
(consumes `GaussianObject`s + masks) and downstream of Reconstruction (consumes the
`ColmapDataset` for depth-aware projection); Shared Kernel with SceneAssembly (supplies
`Hull` + `ObjectPose` for placement). It uses the existing Hunyuan3D and ComfyUI ACLs.

### 2.1 Entity: `KeyItem` (D7 — ranked, selected object)

A `KeyItem` is **not** a new object representation; it is a **ranked selection over** an
existing `GaussianObject`. Today every mask with `mask_pixels>0` is kept
(`stages.py:1410-1419`) and `min_object_gaussians` (`config.py:109`) is never enforced.
`KeyItem` makes keyness a first-class, gated decision.

```
KeyItem {                            # Entity (Per-Object Reconstruction)
    id: UUID
    gaussian_object_id: UUID         # the existing GaussianObject this ranks (never replaces it)
    concept: str                     # the SAM3 concept prompt that detected it (existing term)
    rank_score: float                # composite: size × gaussian_count × confidence × concept_priority
    gaussian_count: int              # per-object Gaussian count (drives the keyness threshold)
    confidence: float                # segmentation confidence (existing GaussianObject.confidence)
    is_key: bool                     # passed the keyness threshold
    hull_id: UUID | None             # set after hull recon
}
```

**Domain rule — keyness threshold**: a `GaussianObject` becomes a `KeyItem` with
`is_key == True` only if `gaussian_count >= config.decompose.min_object_gaussians`
(**enforcing** the previously-dead config knob) **and** `rank_score` is above the
profile-configured keyness cut. Only `is_key` items proceed to hull recon (FR-9, G6/G7).
**keyness** is glossed in §4.

### 2.2 Aggregate root: `Hull` (the reconstructed textured 3D object)

`Hull` is a **specialisation of `MeshAsset`** (`aggregates.md` §3) for per-object
watertight recon. It reuses `MeshBackend` (the `hunyuan3d` canonical instance already
enumerated there) and the existing `watertight mesh` / `texture bake` vocabulary.

```
Hull {                               # Aggregate Root (Per-Object Reconstruction)
    id: UUID
    key_item_id: UUID
    mesh_asset_id: UUID              # the underlying MeshAsset (reuses its GLB/OBJ/texture fields)
    backend: MeshBackend             # reuses existing value object — typically the hunyuan3d instance
    is_watertight: bool              # reuses existing "watertight mesh" invariant
    has_texture: bool                # D11: guaranteed textured (decimate-then-bake or native PBR)
    pose: ObjectPose                 # value object — the data that MUST survive to USD (D9)
    inpainted_views: list[InpaintedView]  # recovered unseen views fed to hull recon (D8)
    created_at: datetime
}
```

**Invariants**

1. A `Hull` exists only for a `KeyItem` with `is_key == True`.
2. `has_texture` must be `True` before SceneAssembly consumes it (D11, FR-13, G11).
   No untextured grey hull is delivered.
3. `pose` is non-identity wherever a real pose was recovered (D9, FR-12, G10); a hull
   never silently lands at origin/identity when pose data exists.
4. Every `InpaintedView` covers a **genuinely unobserved** region (anti-hallucination
   invariant, §2.5).

### 2.3 Value object: `ObjectPose` (D9 — the data that must survive to USD)

The pose persisted from segmentation through to the `ObjectDescriptor` and into the USD
xform. Today it is normalised and **discarded** (`stages.py:1683-1693`) while the
placement machinery (`usd_assembler.py:169-175`) waits for data that never arrives. This
value object is the contract that plumbs it end-to-end. Field names match the existing
consumer (`obj.centroid/rotation_quat/scale`).

```
ObjectPose {                         # Value Object
    centroid: Vec3                   # reuses existing Vec3 — world position
    rotation_quat: Quaternion        # reuses existing Quaternion (XYZW)
    scale: Vec3                      # reuses existing Vec3
    frame: str                       # the coordinate frame: "intra_scene" (v3 contract, FR-8)
}
```

> "Correctly placed" contract (D6, FR-8): `frame == "intra_scene"` — COLMAP-relative,
> Y-up, `SCENE_SCALE=0.5`. Survey/georeferenced placement is explicitly deferred to
> ADR-010 and is **not** a v3 field value.

### 2.4 Value object: `InpaintedView` (D8 — FLUX-recovered unseen view)

A single view recovered by the local FLUX inpainter (`comfyui_inpainter.py:86-107`,
local ComfyUI :3001) for a key item whose orbit render
(`multiview_renderer.py:148-240`) leaves faces unseen, **before** Hunyuan3D hull recon.

```
InpaintedView {                      # Value Object
    view_index: int                  # which orbit view
    recovered_image_path: Path       # the FLUX-inpainted RGB
    coverage_mask_path: Path         # which pixels were genuinely unobserved (the inpaint region)
    coverage_fraction: float         # fraction of the view that was unobserved and recovered
    confidence: float                # inpaint confidence — feeds the anti-hallucination gate
}
```

### 2.5 Domain rule — anti-hallucination invariant (D8)

> **Only genuinely-unobserved regions are inpainted.** A view is eligible for FLUX
> recovery only where its `coverage_mask` marks pixels that no observed frame covers.
> Observed regions are **never** overwritten by the generator. Inpaint is gated behind a
> visible-coverage threshold (FR-11): a key item is recovered only if its observed
> coverage is below threshold, and only the missing region is generated. This is the
> proactive guard (`prd-v3` Risk Register) against plausible-but-wrong hull geometry,
> verified per item by gate G8.

**Depth-aware projection rule** (D10, FR-10): mask→3D assignment is depth-gated, not an
XY-plane majority vote (replacing `mask_projector.py:153-214`), so co-located distinct
objects do not merge into one `KeyItem`'s `GaussianObject` subset (G9).

**New domain events** (Per-Object Reconstruction):

```
KeyItemsRanked   { job_id, ranked_count, key_count, dropped_below_threshold, timestamp }
ViewInpainted    { key_item_id, view_index, coverage_fraction, confidence, timestamp }
HullReconstructed{ hull_id, key_item_id, backend, is_watertight, has_texture, timestamp }
ObjectPosePersisted { key_item_id, centroid, rotation_quat, scale, timestamp }
```

### 2.6 Agent-controlled recovery loop (ADR-014)

The FLUX.2/Hunyuan3D recovery is no longer one-shot. The **orchestrator** (the in-container Claude Code
agent; `docs/architecture.md` §"Claude Code as Orchestrator") runs a verify-or-retry loop, calling a
**local vision tool** (`agent-vlm` = gemma-4-26B-A4B, ADR-013 D-013.5) to *see* the generated result
and retaining the judgment itself. gemma-4 is a **tool, not a second orchestrator** — this preserves
the "no hidden state machine; Claude decides what to run next" architecture.

New **value objects** (on the existing per-object recovery flow — *no new aggregate*):

```
RecoveryRequest  { key_item_id, kind: inpaint|hull, target_region, object_identity, artifact_report_ref }
RecoveryAttempt  { request_ref, attempt_index, params: {denoise, guidance, seed, mask}, output_ref }
RecoveryVerdict  { attempt_ref, source: gemma-4|claude, score, decision: accept|re-prompt|veto, reason }
```

**Domain policy** `RecoveryController` (stateless helper the orchestrator invokes): emits
`RecoveryAttempt`s and consumes `RecoveryVerdict`s under a config-bounded retry budget until
`accept`/`veto`. Every attempt and verdict is appended to the per-video ledger and `v2g:*` `Lineage`
(§3.3) — **annotate, never silently drop** (D8 anti-hallucination invariant §2.5 still gates which
regions may be generated; this loop gates whether a *generated* result is *accepted*).

**New domain events**:

```
RecoveryAttempted { key_item_id, kind, attempt_index, timestamp }
RecoveryJudged    { key_item_id, attempt_index, source, score, decision, reason, timestamp }
```

---

## 3. SceneAssembly context — extension (D12, D13, D14)

**Extends** the existing `SceneGraph` aggregate (`aggregates.md` §4) and the
SceneAssembly context (`bounded-contexts.md` §2.6). No existing field is removed; three
additions make the USD self-describing.

### 3.1 Value object: `MetadataRecord` (D12 — the `v2g:*` schema per node)

The namespaced metadata block populated on every node via the (currently unused)
`ObjectDescriptor.metadata` hook (`usd_assembler.py:221-222`). Attribute names are the
`v2g:*` namespace already specified by FR-15. This value object **is** the v3 USD
metadata schema.

```
MetadataRecord {                     # Value Object — populated on a ScenePrim
    semantic_label: str              # v2g:semantic_label   (reuses existing "label")
    quality_score: float             # v2g:quality_score
    bbox_extent: AABB                # v2g:bbox_extent      (reuses existing AABB)
    gaussian_count: int              # v2g:gaussian_count
    recon_method: str                # v2g:recon_method     (e.g. the MeshBackend name)
    confidence: float                # v2g:confidence
    capture_timestamp: datetime      # v2g:capture_timestamp
    processing_timestamp: datetime   # v2g:processing_timestamp
    lineage: Lineage                 # v2g:source_video + v2g:source_frames (§3.3)
    schema_version: str              # v2g schema version (NFR-5 byte-stable)
}
```

> Namespace discipline: existing prims carry `lichtfeld:mesh_path` / `:diffuse_path`
> (`assemble_usd_scene.py:723-725`) — those are **kept**. `MetadataRecord` adds the
> `v2g:*` namespace alongside; it never renames or removes the `lichtfeld:*` attributes.

A `ScenePrim` (existing entity) gains one optional reference:
`metadata: MetadataRecord | None`. **Invariant** (FR-18, G13): every object `ScenePrim`
must carry a fully-populated `MetadataRecord` (all ≥10 required fields) before
`SceneAssembled` fires.

### 3.2 Entity: `EnvironmentMesh` (D13 — textured environment surface)

Today `/World/Environment/Background` holds only a `DomeLight`
(`usd_assembler.py:147-154`) and the full-scene mesh is mis-homed under
`/World/Objects/full_scene` (`assemble_usd_scene.py:619-726`). `EnvironmentMesh`
re-homes the textured polygonal scene surface to `/World/Environment` as a first-class
entity owned by `SceneGraph`.

```
EnvironmentMesh {                    # Entity (owned by SceneGraph)
    id: UUID
    scene_graph_id: UUID
    prim_path: str                   # "/World/Environment/EnvironmentMesh" (re-homed)
    mesh_asset_id: UUID              # the full-scene MeshAsset produced by the mesh_method branch
    has_texture: bool                # textured surface, not a backdrop
    metadata: MetadataRecord         # the environment is annotated too (scene-scope v2g:*)
}
```

**Invariant** (FR-18, G12): a `SceneGraph` for a non-trivial scene must own an
`EnvironmentMesh` with `has_texture == True`; a bare `DomeLight` no longer satisfies the
environment requirement.

### 3.3 Value object: `Lineage` (D14 — Video → Frame → KeyItem → Hull → USD node)

Threads provenance through the whole pipeline so the USD is self-describing without the
pipeline present. Carried inside `MetadataRecord`; resolvable by the FR-19 lineage query.

```
Lineage {                            # Value Object
    source_video: str                # Video.id      → v2g:source_video
    source_frames: list[str]         # Frame.ids     → v2g:source_frames
    key_item_id: str | None          # KeyItem.id (None for the environment)
    hull_id: str | None              # Hull.id    (None for the environment)
}
```

The chain — **Video → Frame → KeyItem → Hull → USD node** — is exactly the lineage
closure measured by `prd-v3` O4/G14. Its root is the `ImageMetadataTag.source_video`
written in §1.3; its leaf is the `v2g:source_video`/`v2g:source_frames` on the
`ScenePrim` here.

**New domain events** (SceneAssembly):

```
MetadataPopulated     { scene_graph_id, prim_path, field_count, schema_version, timestamp }
EnvironmentMeshHomed  { scene_graph_id, prim_path, mesh_asset_id, has_texture, timestamp }
LineageResolved       { prim_path, source_video, source_frame_count, timestamp }  # FR-19 query
```

---

## 4. Ubiquitous Language — NEW terms only

These extend `ubiquitous-language.md`. Existing terms are **not** repeated; where a
definition references an existing term it uses that term's established spelling.

| Domain Term | Definition |
|-------------|------------|
| **Video** | The per-video unit of work in Ingestion: one source video copied from Drive, extracted, tagged, and purged individually, with its own resumable ledger row. Aggregate root. |
| **Frame** | (Promoted) A single extracted image owned by exactly one `Video`; the root identity of the lineage chain. Was a loose v1 entity; now first-class with a `video_id`. |
| **ImageMetadataTag** | The per-frame provenance value object written as a JSON **sidecar tag** at `<frame>.json`: source video, session, frame index, timestamp, quality scores, phash, kept flag, selection reason, pose hint, schema version. |
| **CaptureSession** | The pooling aggregate that owns many `Video`s but yields exactly one combined `ReconstructionJob`; models the per-video-ingest / combined-reconstruct tension. |
| **KeyItem** | A ranked, gated selection over an existing `GaussianObject` that passes the keyness threshold and proceeds to hull recon. Does not replace `GaussianObject`. |
| **keyness** | The composite ranking (size × gaussian count × confidence × concept priority) plus the `min_object_gaussians` floor that decides whether a segmented object is a `KeyItem`; the criterion that drops noise. |
| **Hull** | A textured watertight 3D object reconstructed per `KeyItem` (typically via the Hunyuan3D backend); a specialisation of `MeshAsset` carrying `ObjectPose` and recovered views. |
| **ObjectPose** | The persisted placement value object (centroid + rotation quaternion + scale, in the intra-scene frame) that must survive from segmentation through `ObjectDescriptor` into the USD xform. |
| **InpaintedView** | A single FLUX-recovered view of a `KeyItem`'s genuinely-unobserved faces, with its coverage mask and confidence, fed to hull recon before it runs. |
| **MetadataRecord** | The `v2g:*` namespaced metadata value object populated on every USD node (semantic label, quality, extent, gaussian count, recon method, confidence, timestamps, lineage, schema version). |
| **EnvironmentMesh** | The textured polygonal environment surface re-homed under `/World/Environment`, replacing the bare `DomeLight` backdrop as a first-class scene entity. |
| **Lineage** | The provenance value object threading Video → Frame → KeyItem → Hull → USD node, carried in `MetadataRecord` and resolvable by the lineage query. |
| **sidecar tag** | The per-frame JSON metadata file (`<frame>.json`) holding an `ImageMetadataTag`. Distinct from the Docker-container **sidecar** of `ubiquitous-language.md` — always written as two words. |
| **pose backfill** | The post-COLMAP step that fills a `Frame`'s `pose_hint` (`PoseHint`) once its source frame is registered; turns the optional tag slot into real camera pose. |
| **Vitrine** | The project's name (ADR-015 D-015.1): a museum display case, pairing with upstream LichtFeld Studio. CLI/package id `vitrine`; the web setup tool is **Vitrine Onboarding**. Supersedes "Video-to-Gaussian". |
| **Vitrine Onboarding** | The schema-driven web setup tool (`vitrine-setup`): a re-entrant, no-history wizard that authors the single active `exhibit.toml` from a JSON Schema and provisions the run. Models the agentbox setup-tool pattern. |
| **HardwareProfile** | The typed host probe (GPU count, per-GPU VRAM, RAM, disk) produced by `/api/hardware`; the input to hardware-aware `ModelSelection`. |
| **ModelSelection** | The per-stage model/quant choice recommended to fit the `HardwareProfile` and written to the `[models]` / `[models.vram_plan]` manifest block; user-overridable. |
| **SecretRef** | An `env:`-indirected reference (`env:HF_TOKEN`, `env:GOOGLE_APPLICATION_CREDENTIALS`) that is all the manifest ever holds; the actual secret lives only in the server-side keyring/Docker-secret. |
| **Provisioning** | The deterministic setup phase: download/integrate models, ensure+pin the .48 ComfyUI, bring up `v2g-net`, verify readiness; ends by writing `provision.status = "ready"` and emitting the hand-off event. |
| **setup/agent hand-off** | The boundary (ADR-015 D-015.5) where deterministic `Provisioning` ends and the interpretive internal Claude Code overseer begins: *setup makes the system runnable; the agent decides how to run it.* |
| **OutputWriteBack** | The additive upload of finished artifacts back to the source Drive folder (or its `writeback_subdir`) when `[drive].writeback == true`; requires the Drive write scope from the browser-OAuth grant. |

**New-term count: 22** (14 pipeline + 8 onboarding/setup).

---

## 5. Anti-Corruption notes — new boundaries the new aggregates cross

These **extend** `anti-corruption-layers.md`. Only boundaries newly crossed by the new
aggregates are noted; existing ACLs (COLMAP, LichtFeld MCP, MILo/CoMe/GaussianWrapping
sidecars, BOUNDARIES.md, Blender) are unchanged.

### 5.1 rclone / Google Drive ACL — new, read **and** write (serves `Video`, `CaptureSession`, `Provisioning`)

**External system**: Google Drive accessed via rclone. **Isolates** the Ingestion and
Onboarding/Setup contexts from rclone path conventions, remote listing format, and
credential handling. ADR-015 D-015.6 extends this ACL from **read-only ingest** to
**read+write**: outputs are written back to the *same* source folder, which requires the
Drive **write** scope in the FR-37 OAuth grant.

| Domain concept | External concept | Translation |
|----------------|------------------|-------------|
| `Video.source_drive_path` | `rclone` remote path | `drive_ingestor` builds and verifies the remote path; the domain sees a `Video`, not an rclone URL |
| `Video.copy_to_scratch()` | `rclone copy <remote> <nvme>` | One-video copy; checksum verified before `pending → copied` |
| `CaptureSession` discovery | `rclone lsjson <session>` | Remote listing → list of `Video`s (one ledger row each) |
| `OutputWriteBack` (new, ADR-015) | `rclone copy <nvme> <source-folder>[/vitrine-output]` | When `[drive].writeback == true`, finished artifacts (USD scene, ksplat, per-object hulls, run report) are pushed to the **same** folder the source video came from (or its `writeback_subdir`); Drive remains source of truth — write-back is additive, never overwrites the source video |
| ingest credentials | service-account key | **NFR-4 / FINDING-006**: supplied as a **Docker secret**, never a plaintext env var or image layer; the ACL reads the mounted secret, the domain never sees it |
| write-back credentials | Google OAuth **refresh token** (Drive read+write scope) | ADR-015 D-015.4: obtained by the browser-OAuth flow in Vitrine Onboarding, **stored server-side** (keyring/Docker secret), referenced from the manifest only as `env:GOOGLE_APPLICATION_CREDENTIALS`; the ACL holds the token and proxies the call, the domain and the browser never see it |

> Retention rule (§1.4) is enforced **above** this ACL: the domain decides when a
> `Video` is purged; rclone only ever copies, never the authority on deletion. The
> write scope is **additive** — the ACL may create/upload under the source folder but is
> never granted authority to delete the source video.

### 5.2 ComfyUI FLUX ACL — extended (serves `InpaintedView`)

The ComfyUI ACL already exists (`anti-corruption-layers.md` §6) for **background**
inpaint. v3 **adds a new caller**: the per-object recovery loop.

| Domain concept | ComfyUI concept | Translation |
|----------------|-----------------|-------------|
| `InpaintedView` (per key item) | FLUX.2-dev workflow graph JSON (ADR-012/014) | `comfyui_inpainter.py` substitutes the orbit-view image + coverage mask into the pre-built workflow; result → `InpaintedView` |
| anti-hallucination invariant (§2.5) | mask supplied to the workflow | The ACL passes **only** the unobserved-region mask; observed pixels are never sent for generation — the invariant is enforced at the adapter boundary |
| `RecoveryAttempt` / `RecoveryVerdict` (§2.6) | graph API `/prompt`+`/history`+`/view`; Salad control API model load/free | The ACL exposes **two ComfyUI surfaces** (ADR-014): the graph API for execution and the Salad add-on control API for model lifecycle; the orchestrator's verify-or-retry loop crosses only the adapter, ComfyUI's API shape never enters the domain |
| ComfyUI unavailable | `config.inpaint.enabled = False` | No-op (existing behaviour): key item proceeds to hull recon with whatever views exist; gate G8 records the skip |

### 5.3 Hunyuan3D ACL — new note (serves `Hull`)

**External system**: Hunyuan3D hull recon (`hunyuan3d_client.py:528-685`,
`stages.py:1778-1806` "Strategy 1"). The translation reuses the existing `MeshBackend`
value object (the `hunyuan3d` instance) and the Sidecar Container ACL **family pattern**
of `anti-corruption-layers.md` §3 — `Hull` is the domain result; the raw client output
never leaves the adapter.

| Domain concept | Hunyuan3D concept | Translation |
|----------------|-------------------|-------------|
| `Hull.mesh_asset_id` | Hunyuan3D GLB output | `hunyuan3d_client` returns a GLB; adapter wraps it as a `MeshAsset`, then `Hull` |
| `InpaintedView[]` | 4-view orbit input | Recovered views (§2.4) are passed as the multiview input; the domain supplies completed views, not raw renders |
| `Hull.has_texture` guarantee | vertex-colour GLB / bake | D11: adapter runs decimate-then-bake when faces exceed the bake ceiling so `has_texture` is always `True` before the domain sees the `Hull` |

### 5.4 OpenUSD — extended responsibility (serves `MetadataRecord`, `EnvironmentMesh`, `Lineage`)

The OpenUSD boundary is already owned by SceneAssembly (`usd_assembler.py`, and the
LichtFeld MCP ACL for native USD). v3 adds **one** translation responsibility, no new
ACL module:

| Domain concept | USD concept | Translation |
|----------------|-------------|-------------|
| `MetadataRecord` | `v2g:*` prim attributes | `usd_assembler` writes each field as a namespaced attribute via the `ObjectDescriptor.metadata` hook; `lichtfeld:*` attributes are preserved alongside |
| `EnvironmentMesh` | `UsdGeomMesh` at `/World/Environment` | Re-homed from `/World/Objects/full_scene`; reuses existing `UsdGeomMesh` + `UsdPreviewSurface` translation |
| `Lineage` | `v2g:source_video` / `v2g:source_frames` | Written as prim attributes; the FR-19 lineage query reads them back — round-trip stable (NFR-5) |

### 5.5 Vitrine Onboarding host/secret ACL — new (serves the Onboarding/Setup context, §7)

**External systems**: the host hardware (`nvidia-smi`/`/proc`), the Hugging Face Hub, and
Google's OAuth endpoint, fronted by the `vitrine-setup` Rust/Axum backend. **Isolates** the
Onboarding/Setup context (and the browser) from raw probe output, token handling, and
external HTTP — the agentbox `/api/proxy` containment pattern (ADR-015 D-015.4).

| Domain concept | External concept | Translation |
|----------------|------------------|-------------|
| `HardwareProfile` | `nvidia-smi --query-gpu`, `/proc/meminfo`, `df` | `/api/hardware` parses raw probe output into a typed `HardwareProfile` (GPU count, per-GPU VRAM, RAM, disk); the wizard sees the value object, never the CLI text |
| `ModelSelection` | the FR-29 / D-013.4 per-stage table | The backend maps `HardwareProfile` → recommended model/quant per stage that fits VRAM; writes the `[models]` + `[models.vram_plan]` manifest block |
| `SecretRef` (HF) | HF token string | Pasted token is `POST`ed to the backend, stored as a keyring/Docker-secret entry; the manifest holds only `env:HF_TOKEN`; the token never returns to the browser |
| `SecretRef` (Google) | OAuth refresh token | Browser consent flow → refresh token stored server-side; manifest holds only `env:GOOGLE_APPLICATION_CREDENTIALS`; proxied Drive calls cross §5.1, never the browser |
| `ManifestDraft` ⇆ `exhibit.toml` | `toml_edit::DocumentMut` | The backend round-trips the file (parse→validate→re-serialise) preserving comments/key order; `/api/validate` server-validates against the JSON Schema before save |
| external HTTP (HF / Google / Drive) | `/api/proxy/{path}` | The backend injects `Authorization: Bearer` server-side and forwards; the browser issues no authenticated request itself — secrets stay off the wire to the client |

> The ACL is the **only** holder of credentials. Its output `exhibit.toml` carries
> `env:`-indirected references exclusively — the secret-containment invariant (FR-37, O19,
> G-O3) is enforced at this adapter boundary, not by convention in the domain.

---

## 6. Context map — new and extended contexts

```mermaid
flowchart TB
    subgraph EXT["External systems (ACLs)"]
        DRIVE["Google Drive / rclone"]
        FLUX["ComfyUI FLUX.2-dev"]
        HY3D["Hunyuan3D"]
        USD["OpenUSD"]
    end

    subgraph ING["Ingestion (REFINED)"]
        CS["CaptureSession (new agg)"]
        VID["Video (new agg)"]
        FR["Frame (promoted)"]
        TAG["ImageMetadataTag (sidecar tag)"]
        CS --> VID --> FR --> TAG
    end

    subgraph RECON["Reconstruction + Training (unchanged)"]
        FS["FrameSet (existing)"]
        GM["GaussianModel / GaussianObject (existing)"]
    end

    subgraph POR["Per-Object Reconstruction (NEW — split from Segmentation + MeshExtraction)"]
        KI["KeyItem (rank/gate over GaussianObject)"]
        IV["InpaintedView"]
        HULL["Hull (specialises MeshAsset)"]
        POSE["ObjectPose"]
        KI --> IV --> HULL
        HULL --> POSE
    end

    subgraph SA["SceneAssembly (EXTENDED)"]
        SG["SceneGraph (existing)"]
        MR["MetadataRecord (v2g:*)"]
        ENV["EnvironmentMesh"]
        LIN["Lineage"]
        SG --> MR
        SG --> ENV
        MR --> LIN
    end

    DRIVE -. "rclone ACL (new)" .-> VID
    ING -- "Customer-Supplier: pooled FrameSet" --> RECON
    RECON -- "Customer-Supplier: GaussianObject + ColmapDataset" --> POR
    FLUX -. "ComfyUI ACL (extended)" .-> IV
    HY3D -. "Hunyuan3D ACL (new note)" .-> HULL
    POR -- "Shared Kernel: Hull + ObjectPose" --> SA
    USD -. "OpenUSD (extended: v2g:* + lineage)" .-> SA
    TAG -. "Lineage root: source_video" .-> LIN
    KI -. "Lineage: key_item_id" .-> LIN
    HULL -. "Lineage: hull_id" .-> LIN

    classDef new fill:#fff3cd,stroke:#d39e00;
    classDef ext fill:#f8d7da,stroke:#c82333;
    class CS,VID,KI,IV,HULL,POSE,MR,ENV,LIN,POR new;
    class DRIVE,FLUX,HY3D,USD ext;
```

**Reading the map**: the lineage chain (dotted, bottom) — `Video.source_video` →
`KeyItem.key_item_id` → `Hull.hull_id` → `Lineage` on the USD node — is the spine that
makes every delivered prim resolvable to its source video and frames (D14, O4, G14).

---

## 7. Onboarding/Setup context — new (ADR-015)

**New bounded context**, upstream of every other context. It is the **only** producer of
the `exhibit.toml` manifest (ADR-013 §1) that the whole pipeline consumes, and the
**only** holder of credentials. Two actors live here, explicitly bounded by the
setup/agent hand-off (ADR-015 D-015.5): the deterministic **Setup** (the `vitrine-setup`
tool) and the interpretive **internal overseer** (the in-container Claude Code agent,
ADR-013 D-013.6). It does **no** reconstruction — it makes the system *runnable* and
hands off.

**Relationship to existing contexts**: Customer-Supplier **upstream** of Ingestion,
Reconstruction, and Generative Recovery — it supplies the validated manifest + provisioned
models + a live `v2g-net`, and they consume it. It is **Conformist** toward three external
systems (host probe, Hugging Face, Google OAuth, ComfyUI/Salad) via the §5.5 + §5.1 ACLs.
It is a **config wizard that also provisions** — unlike the agentbox setup tool it emulates,
which only edits config (ADR-015 D-015.2/D-015.5).

### 7.1 Aggregate root: `ExhibitManifest` (the single active configuration)

The one human-authored input, modelled as an aggregate so its **re-entrant, no-history**
rule (FR-35) is a domain invariant rather than a UI behaviour. There is exactly one
active `ExhibitManifest`; editing reloads and overwrites the **same** file.

```
ExhibitManifest {                    # Aggregate Root (Onboarding/Setup)
    exhibit: ExhibitIdentity         # name, description (free text)
    objects: list[ObjectOfInterest]  # the object sub-list (ADR-013 [[objects]])
    drive: DriveBinding              # source URL + write-back target (§7.4)
    secrets: list[SecretRef]         # env:-indirected only — never a literal credential
    models: ModelSelection           # hardware-selected per-stage choice (§7.3)
    oversight: OversightChoice       # backend = claude_code (default) | gemma_local; artifact_vlm
    provision: ProvisionStatus       # value object — the hand-off latch (§7.5)
    schema_version: str              # fixes the manifest field set (NFR-5)
}

ObjectOfInterest {                   # Entity (owned by ExhibitManifest)
    id: str                          # stable id (survives edits)
    description: str                 # free-text object the agent will mediate
    sam3_concept: str | None         # filled by the internal agent at hand-off, not by setup
    priority: str                    # "key" → enters the ADR-010 hull-recon path
}
```

**Invariants**

1. **Single active manifest** (FR-35, D-015.2): exactly one `ExhibitManifest` exists; a
   restart loads it and repopulates editable fields. No past-project list, no archive.
2. **Secret containment** (FR-37, O19, G-O3): `secrets` holds only `SecretRef`s
   (`env:`-indirected). A literal credential in the manifest is an invariant violation.
3. `ObjectOfInterest.sam3_concept` is **not** authored by Setup — it is the interpretive
   output the internal overseer fills after hand-off (the setup/agent boundary, §7.5).
4. `provision.status` advances monotonically `unconfigured → provisioning → ready | failed`;
   the hand-off event fires only on `ready`.

### 7.2 Value object: `HardwareProfile` (D-015.3 — the probe result)

```
HardwareProfile {                    # Value Object — produced by /api/hardware
    gpus: list[GpuInfo]              # per-GPU: name, total_vram_gb, free_vram_gb
    ram_gb: float
    disk_free_gb: float
}
```

### 7.3 Value object: `ModelSelection` (D-015.3 — hardware-fit choices)

The per-stage model/quant recommendation, written to the `[models]` /
`[models.vram_plan]` manifest block. **Domain rule — VRAM fit** (FR-36, O18, G-O2): no
selected stage model's `serial_peak_estimate_gb` may exceed the smallest GPU's
`total_vram_gb` in the `HardwareProfile` (the serial lifecycle, ADR-013 D-013.2, bounds
peak VRAM to the largest single stage). The user may override; an over-budget override is
flagged, not silently accepted.

```
ModelSelection {                     # Value Object
    inpaint: str                     # e.g. "flux2-dev-fp8mixed" (fallback flux1-fill)
    hull: str                        # e.g. "hunyuan3d-2.1"      (fallback 2.0)
    matcher: str                     # e.g. "aliked_lightglue"   (fallback sift)
    mesh: str                        # milo | tsdf | come | gaussianwrap
    artifact_vlm_quant: str          # gemma-4 quant (e.g. Q5_K_M)
    serial_peak_estimate_gb: float   # max single-stage VRAM (informational; from probe)
}
```

### 7.4 Value object: `DriveBinding` (D-015.6 — source + write-back)

```
DriveBinding {                       # Value Object
    url: str                         # source (ingest) Drive folder
    rclone_remote: str
    writeback: bool                  # D-015.6 — upload outputs back to source
    writeback_subdir: str | None     # defaults to the source folder
}
```

`OutputWriteBack` (the act) crosses the §5.1 rclone ACL (now read+write); the Drive write
scope it needs comes from the §5.5 browser-OAuth grant.

### 7.5 Value object: `ProvisionStatus` + the setup/agent hand-off (D-015.5)

The latch that bounds **deterministic Setup** from the **interpretive agent**. Setup
performs the §5.5-mediated downloads, the ADR-014 ComfyUI ensure+pin (FR-32), and
`v2g-net` bring-up, then writes `status = "ready"` and emits `ProvisionReady`. The internal
Claude Code overseer consumes that event and does the judgement work — `ObjectOfInterest`
free text → SAM3 concept candidates + per-object recovery plans (FR-9/FR-11).

```
ProvisionStatus {                    # Value Object — the hand-off latch
    status: str                      # unconfigured → provisioning → ready | failed
    models_integrated: list[str]     # which model checkpoints are present + pinned
    comfyui_pinned: bool             # ADR-014 FR-32 ensure+pin succeeded
    network_up: bool                 # v2g-net reachable
    ready_at: datetime | None
}
```

> **The boundary, stated once**: *Setup makes the system runnable; the agent decides how
> to run it.* Everything in §7.1–§7.4 is deterministic, idempotent, and scriptable.
> Everything downstream of `ProvisionReady` is the overseer's interpretation — consistent
> with `docs/architecture.md` "no hidden state machine; Claude decides what to run next".

**New domain events** (Onboarding/Setup):

```
ManifestSaved     { schema_version, object_count, timestamp }        # re-entrant overwrite
HardwareProbed    { gpu_count, total_vram_gb, timestamp }
ModelsSelected    { selection, serial_peak_estimate_gb, timestamp }
SecretStored      { ref_name, backend: keyring|docker_secret, timestamp }  # value never logged
ProvisionStarted  { selection_ref, timestamp }
ProvisionReady    { models_integrated, comfyui_pinned, network_up, timestamp }  # hand-off fires
```

### 7.6 Context map — Onboarding/Setup upstream of the pipeline

```mermaid
flowchart LR
    subgraph EXTO["External (ACL §5.5 / §5.1)"]
        HOST["Host GPUs (nvidia-smi)"]
        HF["Hugging Face Hub"]
        GOOG["Google OAuth"]
        CUI["ComfyUI / Salad (.48)"]
    end

    subgraph ONB["Onboarding/Setup (NEW — ADR-015)"]
        WIZ["Vitrine Onboarding (vitrine-setup)"]
        EM["ExhibitManifest (single active)"]
        HW["HardwareProfile"]
        MS["ModelSelection"]
        PS["ProvisionStatus"]
        WIZ --> EM
        HW --> MS --> EM
        WIZ --> PS
    end

    AGENT["Internal overseer (Claude Code, ADR-013)"]

    HOST -. "§5.5 hardware probe" .-> HW
    HF -. "§5.5 model pulls" .-> PS
    GOOG -. "§5.5 OAuth (Drive r+w)" .-> EM
    CUI -. "ADR-014 ensure+pin" .-> PS
    EM -- "the one manifest" --> AGENT
    PS -- "ProvisionReady (hand-off)" --> AGENT
    AGENT -- "consumes" --> PIPE["Ingestion → … → Delivery"]

    classDef new fill:#fff3cd,stroke:#d39e00;
    classDef ext fill:#f8d7da,stroke:#c82333;
    class WIZ,EM,HW,MS,PS,ONB new;
    class HOST,HF,GOOG,CUI ext;
```

**Reading the map**: the Onboarding/Setup context is the single source of the
`ExhibitManifest` and the single holder of secrets; it provisions deterministically and
hands off to the internal overseer on `ProvisionReady` — the overseer, not setup, drives
the pipeline from there.
