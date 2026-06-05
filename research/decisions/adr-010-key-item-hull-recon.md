# ADR-010: Key-Item Selection/Ranking, FLUX-Inpainted Per-Object Hull Recovery, and Pose Preservation

## Status

Proposed Amended 2026-06-05.

## Amendment (2026-06-05) — current hull stack + build status

- **Hull generator:** TRELLIS.2-4B (MIT, single-image, PBR) is now the **primary**;
  Hunyuan3D-2.1 (multiview, matches the orbit renderer) is the fallback; SAM3D is a
  last-resort, non-commercial fallback. This supersedes the ADR-012 "Hunyuan3D-2.1"
  primary.
- **Recovery model:** FLUX.2-dev (not FLUX.1-Fill).
- **Build status:** key-item ranking + the (previously dead) `min_object_gaussians`
  threshold are enforced in `extract_objects` (FR-9, built). Wiring the FLUX.2 recovery
  loop into the per-object hull path and persisting per-object pose to USD are the active
  integration (FR-11 / FR-12, in progress).

## Context

The per-object decomposition path is structurally complete but has five wiring/scope gaps,
registered as D6–D10 in `research/decisions/gap-analysis-e2e-aspiration.md` (§3) and commissioned
as FR-8..FR-13 in `research/decisions/prd-v3-e2e-closure.md`. Every heavy component already exists;
the failure is that they are not connected into the per-key-item recovery loop the aspiration
demands (`research/pipelines/aspirational-e2e-flowchart.md` Phase 3/4).

- **No key-item ranking (D7).** SAM3 segmentation is concept/text-prompted
  (`sam3_segmentor.py:192-248`) over a fixed concept list (`config.py:113-116`). Every detection
  with any mask area is kept: `stages.py:1410-1419` appends an object whenever `mask_pixels > 0`.
  The threshold that should drop noise — `min_object_gaussians`, defined `config.py:109` with a
  default of 100 — is **never enforced** at the selection site. There is no notion of "key item":
  a one-pixel speck and a sofa are treated identically.

- **FLUX inpainter built but unwired (D8).** The local FLUX.1-Fill inpainter runs against a local
  ComfyUI instance on :3001 and is implemented (`comfyui_inpainter.py:86-107`), but it was built
  for **background** inpaint and is **not called** anywhere in the per-object flow. Occluded or
  unseen faces of an object are therefore presented to hull reconstruction as empty views, which
  is the principal cause of hallucinated geometry on the back of objects.

- **Mask→3D is depth-lossy (D10).** Per-object Gaussian subsets are assigned by an XY-plane
  majority vote (`mask_projector.py:153-214`). Two objects that overlap in the image plane but sit
  at different depths can be voted into a single subset, merging them into one hull.

- **Hunyuan3D hull recon is wired (good).** Multiview orbit render → Hunyuan3D 2.0 → textured GLB
  is the wired Strategy 1 (`stages.py:1778-1806`; `multiview_renderer.py:148-240`). This is not a
  gap and must not be rebuilt.

- **Per-object pose computed then discarded (D9).** At `stages.py:1683-1693` the pipeline computes
  `centroid`, `extent` and `scale` for each object's point cloud, normalises the points by them,
  and then **drops** the pose: only the normalised mesh is exported (re-multiplied by scale +
  centroid inside `_export_mesh`, but the transform itself is never persisted to an
  `ObjectDescriptor`). Meanwhile the USD placement machinery at `usd_assembler.py:169-175` is
  *waiting* for exactly this data — it reads `obj.centroid`, `obj.rotation_quat`, `obj.scale` to
  author the prim xform. The capability is present; the data plumbing is severed.

- **"Correctly placed" is undefined (D6).** Placement is COLMAP-relative; `coordinate_transform.py`
  fixes `SCENE_SCALE = 0.5` (`coordinate_transform.py:34`). Whether the aspiration's "correctly
  placed" requires survey/georeferenced placement or is satisfied by intra-scene consistency has
  never been decided, leaving the requirement open-ended.

## Decision

### (a) Key-item ranking + threshold policy (D7, FR-9)

At the selection site (`stages.py:1410-1419`) compute a **keyness score** per detection and rank,
rather than keeping everything with `mask_pixels > 0`:

```
keyness = gaussian_count(obj) * concept_priority(label) * detection_confidence(obj)
```

- `gaussian_count` is the number of Gaussians projected to the object subset (after the
  depth-aware projection in (d)), replacing the raw `mask_pixels` proxy.
- `concept_priority` is a per-concept weight on the SAM3 concept list (`config.py:113-116`);
  structural background concepts (`walls`, `floor`, `ceiling`) weight low, foreground items
  (`paintings`, `sculptures`, `furniture`) weight high.
- `detection_confidence` is the SAM3 score, already thresholded by
  `sam3_confidence_threshold` (`config.py:112`).

**Enforce `min_object_gaussians`** (`config.py:109`): any object whose `gaussian_count` falls below
the threshold is dropped before hull recon (closes G7). The ranked survivors are kept down to a
configurable `top_k` cap (default: keep all above threshold). Only ranked key items proceed to
Phase 4.

### (b) Wire local FLUX inpaint into the per-object multiview loop (D8, FR-11)

Insert the existing local FLUX inpainter (`comfyui_inpainter.py:86-107`) into the per-object loop
**between** the multiview orbit render (`multiview_renderer.py:148-240`) and Hunyuan3D
(`stages.py:1778-1806`). For each key item, after the orbit render, compute a per-view
**coverage mask** = the fraction of the object silhouette in that view that received splat
coverage. A view (or region within a view) is eligible for inpaint **only** where coverage falls
below a `visible_coverage_threshold`; genuinely observed regions are never inpainted.

The guardrail against hallucination (PRD §7, top risk) is structural: the inpaint mask is the
*complement* of the coverage mask, so FLUX only paints pixels the cameras never saw. The
threshold is profile-configurable, every inpaint call is traced for G8, and the inpainted views
feed Hunyuan3D *as additional conditioning*, never replacing observed views.

### (c) Persist per-object world pose through ObjectDescriptor to USD (D9, FR-12)

Stop discarding the pose at `stages.py:1683-1693`. Capture the computed transform as a value
object and carry it end-to-end:

- `centroid` → object world position (already computed at `:1683`).
- `scale` → object scale (already computed at `:1685`).
- `rotation_quat` → object orientation. Where SAM3/COLMAP yield no oriented frame, default to
  identity, but compute a PCA-aligned principal-axis quaternion from the object point cloud when
  the cloud is non-degenerate, so the hull inherits the object's natural axes.

Populate `ObjectDescriptor.centroid/rotation_quat/scale` so the existing placement code at
`usd_assembler.py:169-175` authors a real xform. This is pure plumbing of data already computed —
no new pose estimation science.

### (d) Depth-aware mask projection (D10, FR-10)

Replace the XY-plane majority vote (`mask_projector.py:153-214`) with a depth-gated assignment:
project each Gaussian into the image, and assign it to a mask only if its rendered depth agrees
with the local depth of that mask's surface within a tolerance. Co-located objects at different
depths are kept as separate subsets (G9). Where depth is ambiguous, fall back to per-mask seeds
(the v1 vote) rather than merging — fail-separate, not fail-merge.

### (e) Resolve "correctly placed" (D6, FR-8)

**The v3 contract for "correctly placed" is intra-scene consistency**: every hull sits at its
persisted COLMAP-relative pose (from (c)), so spatial relationships within the room are preserved
(`coordinate_transform.py:34` `SCENE_SCALE = 0.5` is the fixed relative scale). Survey /
georeferenced world placement is a **separate concern, explicitly deferred** as a future option
under this ADR — it is not a v3 requirement. This bounds the requirement and removes the
open-ended survey-pose ambiguity from scope.

## Rationale

- **Ranking by `gaussian_count × concept_priority × confidence`** combines the three signals that
  actually distinguish a key item from noise: 3D mass (not 2D pixels, which over-weight large flat
  surfaces), semantic importance, and detector certainty. Using `gaussian_count` rather than
  `mask_pixels` ties keyness to reconstructable 3D substance, and reuses the same per-object subset
  the hull recon needs anyway.
- **Enforcing `min_object_gaussians`** finally uses a config field that has been dead since it was
  defined (`config.py:109`); the default of 100 is a sensible floor below which a hull cannot be
  meaningfully reconstructed.
- **Coverage-masked inpaint** is the only safe way to wire FLUX: painting *only* unobserved pixels
  means the generative model cannot overwrite real geometry, which directly mitigates the
  highest-impact risk in the PRD register. Inpaint-everything would trade hallucinated-empty for
  hallucinated-wrong.
- **Pose persistence is plumbing, not science.** The transform is already computed
  (`stages.py:1683-1685`) and already consumed (`usd_assembler.py:169-175`); the only change is to
  not throw it away in between. This is the cheapest possible close of D9.
- **Fail-separate depth fallback** is the conservative choice: a wrongly-split object yields two
  recoverable hulls, whereas a wrongly-merged object is unrecoverable downstream.
- **Intra-scene contract for D6** matches what the data can actually support today (COLMAP gives
  relative pose, not georeference) and aligns with FR-8; promising survey placement would commit to
  capability the pipeline does not have.

## Consequences

### Positive

- Key items are clean objects, not noise blobs: ranking + threshold enforcement raises key-item
  precision (O5, G6) and rejects sub-threshold noise (G7).
- Occluded faces are recovered before hull recon, reducing hallucinated back-geometry (O6, G8)
  without risking overwrite of observed surfaces.
- Hulls land at their real pose instead of stacked at origin/identity (O9, G10), preserving the
  room's spatial relationships for the archival consumer (U9).
- Co-located objects no longer merge (G9), so adjacent items reconstruct as distinct hulls.
- The "correctly placed" contract is bounded and testable; survey placement is cleanly deferred.

### Negative

- The coverage-mask computation adds a per-view pass before Hunyuan3D and a FLUX inference call
  per low-coverage item, increasing per-object Phase 4 latency.
- Depth-aware projection is more expensive than the XY vote and needs per-object rendered depth,
  which the projector must now request.
- PCA-based default orientation requires a non-degenerate point cloud; thin or sparse objects fall
  back to identity orientation, which may look less aligned than a hand-tuned pose.

### Risks

- **FLUX hallucinates plausible-but-wrong geometry on occluded faces.** Mitigation: inpaint gated
  behind `visible_coverage_threshold`, mask = complement of coverage, every call traced (G8),
  reviewer spot-check on the test scene.
- **Depth-lossy merge of co-located objects.** Mitigation: depth-gated assignment with
  fail-separate fallback; G9 two-object overlap fixture asserts two subsets.
- **Key-item ranking drops a genuine item (false negative).** Mitigation: `min_object_gaussians`
  and `top_k` are profile-configurable; G6 measures precision, complemented by a recall
  spot-check; thresholds tuned on the curated test scene.
- **Pose plumbing regression.** If `ObjectDescriptor` is populated but the xform author path is
  bypassed, hulls silently land at identity. Mitigation: G10 asserts a non-identity prim xform for
  every object that has a persisted pose.

## Alternatives Considered

- **Keep-all detections (status quo).** Rejected: it is the gap — `stages.py:1410-1419` ships
  noise into hull recon, and `min_object_gaussians` stays dead.
- **Rank by `mask_pixels` (2D area) only.** Rejected: over-weights large flat surfaces (walls,
  floor) and under-weights small high-value items; 3D `gaussian_count` is the better mass proxy.
- **Inpaint all views unconditionally.** Rejected: lets the generative model overwrite observed
  geometry, converting the occlusion problem into a hallucination problem (the PRD's top risk).
- **Skip FLUX, rely on Hunyuan3D priors for unseen faces.** Rejected: defers the same
  hallucination to a model with no scene-specific signal; the local FLUX inpainter is already
  built (`comfyui_inpainter.py`) and gives a controllable, coverage-masked recovery.
- **Survey/georeferenced placement as the v3 contract.** Rejected/deferred: the pipeline produces
  COLMAP-relative pose only (`coordinate_transform.py:34`); survey registration is a separate
  concern, deferred per FR-8.
- **Re-implement pose estimation.** Rejected: pose is already computed (`stages.py:1683-1685`) and
  already consumed (`usd_assembler.py:169-175`); only the plumbing is missing.

## Related Decisions

- `research/decisions/prd-v3-e2e-closure.md` — commissions this ADR; realises FR-8..FR-13, closes
  D6, D7, D8, D9, D10.
- `research/decisions/gap-analysis-e2e-aspiration.md` — delta register D6–D10 (§2, §3, §5).
- `research/pipelines/aspirational-e2e-flowchart.md` — Phase 3 key-item ID and Phase 4 hull recon
  (§1).
- `adr-009-per-video-ingest-and-metadata.md` — supplies the `pose_hint`/source-video lineage that
  this ADR's pose persistence carries into `ObjectDescriptor`.
- `adr-011-usd-metadata-enrichment.md` — consumes the persisted pose (xform), ranking
  (`gaussian_count`, `confidence`) and `recon_method` into the USD `v2g:*` metadata.
- `adr-003-pluggable-mesh-extraction-backends.md` — the hull-backend branch (Hunyuan3D | TSDF
  fallback) is governed by the existing backend-selection policy; this ADR does not change it.
