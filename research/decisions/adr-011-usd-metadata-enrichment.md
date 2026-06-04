# ADR-011: USD Scene-Graph Metadata Enrichment and Textured Environment Mesh

## Status

Proposed

## Context

The USD assembly stage is the deliverable's self-description layer, and it is the weakest in
annotation. The hierarchy and the real `{Gaussian|Mesh}` variant sets exist and work — but the
nodes carry almost no metadata, the environment surface is a light rather than a mesh, and the
population hook that should fix this is unused. These are registered as D12, D13, D14 in
`research/decisions/gap-analysis-e2e-aspiration.md` (§4) and commissioned as FR-14..FR-19 in
`research/decisions/prd-v3-e2e-closure.md`.

- **Hierarchy and variant sets exist (good).** `usd_assembler.py:144-151` defines
  `/World/{Environment/Background, Objects, Cameras, Materials}`, and `usd_assembler.py:182-203`
  authors a real `representation` variantSet with `gaussian` and `mesh` variants. The poster claim
  of `{Gaussian|Mesh}` variant sets is true in code. This is not a gap.

- **Environment is only a DomeLight (D13).** `/World/Environment/Background` carries **only** a
  `DomeLight` (`usd_assembler.py:147-154`). The actual full-scene polygonal mesh is mis-homed: the
  assembler script places every mesh — including the one labelled `full_scene` — under
  `/World/Objects/{safe_name}` in its object placement loop
  (`scripts/assemble_usd_scene.py:619-726`), so the environment surface lands under `/World/Objects`
  instead of `/World/Environment`. There is no dedicated textured environment surface a consumer
  can occlude, light, or walk.

- **Node metadata is thin (D12).** Object prims carry only `lichtfeld:mesh_path` and
  `lichtfeld:diffuse_path` (`scripts/assemble_usd_scene.py:723-725`); mesh prims add
  `mesh:vertex_count` / `mesh:face_count` (`scripts/assemble_usd_scene.py:536-538`). None of the
  semantic, quality, provenance or recon-method fields the archival consumer needs (U6) are
  present. Critically, the `ObjectDescriptor.metadata` dict hook
  (`usd_assembler.py:221-222`, which copies every key/value into prim customData) is **unused** —
  the data carrier exists, but nobody fills the dict.

- **No lineage in the USD (D14).** Because there is no per-image metadata upstream (ADR-009) and
  no per-object provenance threading, the USD cannot answer "which video and which frames produced
  this object". The video→image→object→USD chain has no termination in the scene graph.

## Decision

### (a) Namespaced `v2g:*` USD metadata schema (D12, FR-15/FR-16)

Define a single namespaced metadata schema, authored as **prim customData** under the `v2g:`
namespace, with REQUIRED and OPTIONAL fields per prim type. Required fields fail the validation
gate (FR-18) if absent.

**Object prims** (`/World/Objects/obj_NN`):

| Field | Req? | Source |
|-------|------|--------|
| `v2g:semantic_label` | required | SAM3 concept (`config.py:113-116`) via ObjectDescriptor |
| `v2g:concept` | required | SAM3 prompt concept that matched |
| `v2g:gaussian_count` | required | per-object subset count (ADR-010 ranking) |
| `v2g:bbox_extent` | required | `extent` computed at `stages.py:1684` |
| `v2g:recon_method` | required | `hunyuan3d` \| `tsdf` \| ... (hull backend taken) |
| `v2g:confidence` | required | SAM3 detection score (ADR-010) |
| `v2g:quality_score` | required | aggregate of contributing-frame quality (ADR-009 sidecars) |
| `v2g:capture_timestamp` | required | earliest source-frame timestamp (ADR-009) |
| `v2g:processing_timestamp` | required | assembly time |
| `v2g:source_video` | required | lineage from ADR-009 sidecar (D14) |
| `v2g:source_frames` | required | contributing frame indices (D14) |
| `v2g:hull_watertight` | optional | hull QC flag |
| `v2g:texture_resolution` | optional | baked/native texture dimensions |

**Scene prim** (`/World`): `v2g:pipeline_profile` (the declarative profile, PRD FR-6),
`v2g:branch_choices` (matcher × strategy × preset × hull × mesh actually taken),
`v2g:frame_counts` (extracted / kept / selected), `v2g:registration_rate` (COLMAP %),
`v2g:object_count`, `v2g:aggregate_quality`, `v2g:schema_version`.

**Camera prims** (`/World/Cameras`): `v2g:visibility_set` (which object prims this camera observed),
alongside the existing `colmap:*` intrinsics/extrinsics customData.

### (b) Populate from data already computed upstream (D12/D14, FR-17)

No new computation. Wire the **unused** `ObjectDescriptor.metadata` hook
(`usd_assembler.py:221-222`) by filling the `metadata` dict before assembly from three sources that
already produce the data:

- **ADR-009 sidecars** → `v2g:source_video`, `v2g:source_frames`, `v2g:capture_timestamp`,
  `v2g:quality_score` (joined via the per-frame `source_video`/`frame_index` lineage root).
- **ADR-010 pose/ranking** → `v2g:gaussian_count`, `v2g:confidence`, `v2g:semantic_label`,
  `v2g:concept`, `v2g:recon_method`, `v2g:bbox_extent`.
- **Assembly time** → `v2g:processing_timestamp`, `v2g:pipeline_profile`, `v2g:branch_choices`.

Because the carrier (`prim.SetCustomDataByKey` loop, `usd_assembler.py:221-222`) already exists,
this is wiring, not new schema machinery.

### (c) Real textured environment mesh prim (D13, FR-14)

Stop mis-homing the full-scene mesh under `/World/Objects/full_scene`
(`scripts/assemble_usd_scene.py:619-726`). Author the polygonal scene mesh — produced by the
`mesh_method` branch (ADR-003) — as a dedicated textured mesh prim under `/World/Environment`,
bound to a `UsdPreviewSurface` material exactly as object meshes are
(`scripts/assemble_usd_scene.py:536-538` extent/counts + material bind at `:718-725`). The
`DomeLight` (`usd_assembler.py:154`) remains as fill lighting; the environment is now a surface,
not only a light. The environment prim carries the object-prim metadata schema plus
`v2g:recon_method` = the `mesh_method` backend used.

### (d) Lineage threading (D14, FR-17/FR-19)

Thread the ADR-009 sidecar `source_video`/`frame_index` through selection, segmentation and
assembly so `v2g:source_video` and `v2g:source_frames` on each object resolve to real provenance.
Provide a lineage-resolution query (FR-19): given an object prim, read its `v2g:*` lineage
attributes and return the source video and contributing frames, closing the
video→image→object→USD chain.

## Rationale

- **customData over USD schema classes over primvars.** Prim **customData** is the correct carrier
  for this metadata: it is arbitrary, namespaced key/value data that travels with the prim, is
  authored by the already-present `SetCustomDataByKey` hook (`usd_assembler.py:221-222`,
  `scripts/assemble_usd_scene.py:536-538,723-725` already use exactly this), and requires no schema
  registration or plugin. A formal **USD schema class** (`IsA`/applied API schema) would be more
  rigorous but demands a registered schema plugin and a typed codegen step — disproportionate for
  descriptive annotation that no renderer consumes geometrically, and a churn risk against archival
  consumers (NFR-5). **Primvars** are for per-vertex/per-face interpolated data bound for shading
  (the pipeline already uses the `st` primvar for UVs); object-level scalar metadata like
  `semantic_label` or `source_video` is not interpolated and does not belong in a primvar. Hence:
  customData for the `v2g:*` namespace, primvars reserved for shading data, schema classes deferred.
- **Populate from existing data only.** Every required field is already computed somewhere
  upstream (sidecars, ranking, pose, extent, backend choice). The gap is purely that the carrier
  dict is empty; filling it from existing sources is the cheapest close of D12 and avoids any new
  reconstruction work.
- **Environment as a mesh, not a label fix only.** Re-homing `full_scene` to `/World/Environment`
  and binding a material turns the mis-homed object into the usable textured surface the consumer
  asked for (U8), reusing the exact material-bind path object meshes already use.
- **Lineage is the payoff of ADR-009.** The per-image sidecar root only becomes self-describing
  when it terminates in the USD; threading `source_video`/`source_frames` to object metadata is
  what makes the deliverable answerable months later without the pipeline (U7).

## Consequences

### Positive

- The USD is self-describing: every object prim carries ≥10 required `v2g:*` fields (O8, G13),
  satisfying the archival consumer (U6).
- The environment is a real textured polygonal surface under `/World/Environment` (O9, G12),
  occludable and lightable, no longer mis-homed under `/World/Objects` (U8).
- Full video→image→object→USD lineage resolves via FR-19 (O4, G14), so provenance survives the
  handoff (U7).
- No new computation: the unused `metadata` hook (`usd_assembler.py:221-222`) is finally wired;
  the blast radius is `usd_assembler.py`, `scripts/assemble_usd_scene.py`, and the validator at
  `stages.py:2134`.

### Negative

- The validator must be extended (FR-18, `stages.py:2134`) to assert metadata richness, environment
  mesh presence, and hull texturing — a stricter gate that will fail-closed on incomplete scenes
  that previously passed the existence-only check.
- The `v2g:*` schema is a contract: once consumers depend on it, fields cannot be removed without
  a schema-version bump.
- Re-homing the environment mesh changes the prim path of the full-scene surface; any external
  reference to `/World/Objects/full_scene` must be updated to `/World/Environment`.

### Risks

- **`v2g:*` schema churn breaks archival consumers.** Mitigation: NFR-5 byte-stable schema,
  `v2g:schema_version` recorded on the scene prim (FR-16), additive-only changes.
- **Population depends on upstream lineage (ADR-009) and pose/ranking (ADR-010).** If those are
  absent, required fields are unpopulated and the FR-18 gate fails the scene (correctly), but the
  failure surfaces late at assembly. Mitigation: G13 fails closed at the validation gate, naming
  the missing field and prim.
- **customData is not type-checked by USD.** A wrongly-typed value (e.g. string where float
  expected) is authored silently. Mitigation: the FR-18 validator schema-checks each `v2g:*` field
  type, not just presence.

## Alternatives Considered

- **Formal USD applied API schema classes for `v2g:*`.** Rejected for v3: requires a registered
  schema plugin and typed codegen, disproportionate for descriptive metadata no renderer consumes
  geometrically, and higher churn risk for archival consumers. May be revisited if the schema
  stabilises.
- **Encode metadata in primvars.** Rejected: primvars are for interpolated per-vertex/per-face
  shading data (the `st` UV primvar is the legitimate use); object-level scalar provenance is not
  interpolated and would abuse the mechanism.
- **Keep environment as a DomeLight, add the mesh under `/World/Objects`.** Rejected: leaves the
  environment mis-homed (the D13 gap) and gives the consumer no dedicated, addressable environment
  surface to occlude/light.
- **Side-car the metadata in an external JSON beside the USD.** Rejected: the deliverable must be
  self-describing *inside* the scene graph (U6/U7); an external file does not survive a single-file
  USD handoff. (The ADR-009 per-frame sidecar is upstream provenance, not the delivered scene's
  self-description.)
- **Store only file paths and counts (status quo).** Rejected: it is the gap — the scene cannot
  answer what/how-good/where-from without external context.

## Related Decisions

- `research/decisions/prd-v3-e2e-closure.md` — commissions this ADR; realises FR-14..FR-19, closes
  D11 (texturing gate), D12, D13, D14.
- `research/decisions/gap-analysis-e2e-aspiration.md` — delta register D12, D13, D14 (§4, §5).
- `research/pipelines/aspirational-e2e-flowchart.md` — Phase 5 environment mesh and Phase 6
  richly-annotated USD (§1).
- `adr-009-per-video-ingest-and-metadata.md` — supplies the per-image sidecar lineage root
  (`source_video`/`frame_index`, quality, capture timestamp) that this ADR threads into `v2g:*`
  object/scene metadata (D14).
- `adr-010-key-item-hull-recon.md` — supplies persisted pose (xform), ranking (`gaussian_count`,
  `confidence`), semantic label and `recon_method` consumed by the `v2g:*` schema.
- `adr-003-pluggable-mesh-extraction-backends.md` — the `mesh_method` backend that produces the
  environment mesh; recorded as `v2g:recon_method` on the environment prim, not redesigned here.
- `adr-006-splat-transform-web-delivery.md` — the annotated USD plus `.ksplat` and GLB hulls form
  the deliverable bundle this ADR completes.
