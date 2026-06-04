# PRD v3: End-to-End Workflow Closure

**Date**: 2026-06-04
**Status**: Draft
**Supersedes scope**: extends [prd-v2-upgrade.md](prd-v2-upgrade.md) (technology/sync) and the
original [prd.md](prd.md) (object-decomposition vision). v2 brought capabilities *into* the repo;
v3 **wires the workflow shape** the user actually asked for.
**Drives**: ADR-009, ADR-010, ADR-011, and the DDD extension
[../ddd/v3-e2e-extensions.md](../ddd/v3-e2e-extensions.md).
**Traces to**: every delta D1–D14 in the gap register
[gap-analysis-e2e-aspiration.md](gap-analysis-e2e-aspiration.md), against the target picture
[../pipelines/aspirational-e2e-flowchart.md](../pipelines/aspirational-e2e-flowchart.md).

This PRD is written with **PACT** discipline (Proactive — defects designed out, not caught late;
Autonomous — gates run unattended on the GPU host; Collaborative — schemas shared across phases;
Targeted — effort weighted to the four shape gaps, not to rebuilding working reconstruction
science) and **risk-based prioritisation** (highest-severity × highest-likelihood deltas first).

---

## 1. Executive Summary

The codebase is structurally mature: every heavy stage — COLMAP, 3DGS, SAM3, Hunyuan3D hull
recon, USD assembly with real `{Gaussian|Mesh}` variant sets — exists and works. The product is
**not** missing reconstruction capability. It is missing the *shape* of the end-to-end workflow
the operator asked for. The gap register identifies exactly **four shape gaps**, and this PRD
closes all fourteen deltas under them:

1. **Granularity** — ingestion is per-session-folder *pooled*, not per-video *looped*. A single
   pooled extraction (`drive_ingestor.py:655-674`, `_extract_pooled_frames`) means there is no
   per-video unit of work, no per-video deletion, and no per-video resume. Closes **D1, D2**.
2. **Provenance** — there is **no per-image metadata at all**. Only a session `manifest.json`
   (`drive_ingestor.py:434-441`) and a session-level ledger (`:137-200`) exist; `FrameQuality`
   is computed in-memory (`frame_quality.py:79-92`) and **never persisted per frame**. The
   image→source-video→object lineage chain does not exist. Closes **D3, D14**.
3. **Selection + recovery** — SAM3 detects but **never ranks** "key items" (`min_object_gaussians`
   is defined `config.py:109` but never enforced; `stages.py:1410-1419` keeps everything with
   `mask_pixels>0`); the **local FLUX inpainter is built but unwired** from per-object recovery
   (`comfyui_inpainter.py:86-107`); and per-object **pose is not persisted** into
   `ObjectDescriptor` (`stages.py:1683-1693` normalises then discards, while the placement
   machinery in `usd_assembler.py:169-175` waits for data that never arrives). Closes
   **D6, D7, D8, D9, D10, D11**.
4. **Annotation** — USD nodes carry only file paths and counts; the `ObjectDescriptor.metadata`
   hook (`usd_assembler.py:221-222`) is unused, the environment surface is only a `DomeLight`
   (`usd_assembler.py:147-154`), and the textured environment mesh prim is missing. Closes
   **D4, D5, D12, D13**.

Branch model-choice (matcher / strategy / preset / mesh backend / hull backend) is already real
and config-driven (`config.py:40,50,73,60,111,151,191`); v3 **surfaces** it as a declarative
profile — it does **not** redesign the selection policy (that is ADR-003's remit). The work is
**wiring, granularity, and annotation**, not new reconstruction.

---

## 2. Objectives & Success Metrics

| # | Objective | Metric | Target |
|---|-----------|--------|--------|
| O1 | Per-video granularity | Videos processed as discrete units with own ledger row | 100% of remote videos have an individual ledger entry |
| O2 | Bounded local retention | Wall-clock a processed video occupies local NVMe past its extraction window | 0 s — deleted on verified extraction, never held through reconstruction |
| O3 | Total provenance | Retained frames carrying a per-image metadata sidecar | 100% |
| O4 | Lineage closure | USD object/scene prims resolvable back to source video & frame | 100% (every object → ≥1 source frame → source video) |
| O5 | Key-item precision | Ranked key items that are genuine objects (not noise) vs. analyst ground truth | ≥ 90% precision on the curated test scene |
| O6 | Occlusion recovery | Key items whose unseen/occluded faces are FLUX-inpainted before hull recon | 100% of items below the visible-coverage threshold |
| O7 | Hull texturing | Reconstructed hulls that are textured (baked or native PBR), no untextured grey | 100% |
| O8 | Rich USD annotation | Required `v2g:*` metadata fields present on every object prim | ≥ 10 required fields, 100% populated |
| O9 | Environment surface | Scenes with a real textured polygonal environment mesh prim under `/World/Environment` | 100% |
| O10 | Resumability | Pipeline restart after kill resumes at the failed *video* / *stage*, no redo of completed work | 100% of restart cases |
| O11 | Reproducibility | Same input + profile → byte-identical metadata schema, deterministic stage order | 100% |

Success is the conjunction O1–O11 on the curated test scene, verified by the §6 quality gates.

---

## 3. Personas & User Stories

### Primary — the Batch-Ingest Operator
Runs the one-time Google Drive batch on the GPU host, unattended, overnight. Cares about: not
babysitting; not running out of NVMe; being able to kill and resume; trusting that "done" means
done. Technically fluent, but will not hand-tune per-video.

- **U1** — *As the operator*, I point the worker at a Drive remote dir and walk away; it ingests
  **one video at a time**, extracts, quality-checks, deletes the local copy, tags the images,
  and loops until the remote dir is exhausted — so my NVMe never fills with raw video. (D1, D2)
- **U2** — *As the operator*, when the host is killed mid-run, I restart and it resumes at the
  exact video/stage it failed on — so an 8-hour batch never restarts from zero. (D1, D5, D10-NFR)
- **U3** — *As the operator*, I pass a single declarative **pipeline profile** (matcher ×
  strategy × preset × hull backend × mesh backend) — so branch choices are explicit and
  reproducible, not buried defaults. (D4)
- **U4** — *As the operator*, the worker auto-selects and ranks the **key items** in each room
  and recovers their occluded faces before hull recon — so I get clean object hulls, not noise
  blobs or hallucinated geometry. (D7, D8)
- **U5** — *As the operator*, service-account credentials are mounted as a **Docker secret**, not
  baked into an env var — so a leaked image layer or `docker inspect` does not expose Drive
  access. (NFR security, FINDING-006)

### Secondary — the Heritage / Archival Consumer of the USD
Opens the delivered USD months later in a DCC tool, with none of the original operators present.
Cares about: self-description. The scene must answer "what is this, how good is it, where did it
come from" without external context.

- **U6** — *As the archival consumer*, every object prim carries a semantic label, quality score,
  bbox/extent, gaussian count, recon method, confidence, and capture/processing timestamps — so
  the USD is self-describing. (D12)
- **U7** — *As the archival consumer*, every object and the scene carry **source-video lineage**
  (which video, which frames) — so provenance survives the handoff. (D14)
- **U8** — *As the archival consumer*, the environment is a real textured polygonal surface I can
  occlude/light/walk, not a dome light — so the scene is usable, not a backdrop. (D13)
- **U9** — *As the archival consumer*, hulls sit at their **correct world pose**, not stacked at
  the origin — so spatial relationships are preserved. (D6, D9)

---

## 4. Functional Requirements

Each FR is tagged with the delta IDs it closes and the file it touches. Grouped by phase.
`*` marks the highest-risk (severity × likelihood) requirements — these get gate priority in §6.

### Phase 1 — Per-video ingest loop (`drive_ingestor.py`, `frame_quality.py`)

- **FR-1\*** — Introduce a per-**video** unit of work. Replace folder-pooled iteration
  (`drive_ingestor.py:655-674`) with a loop that picks the next single video `V`, copies only
  `V` to NVMe scratch, and processes it to completion before the next. Each video gets its own
  resumable ledger row (extends `:137-200` from session- to video-granularity). *Closes D1.*
- **FR-2\*** — Delete the local video immediately after a **verified** extraction. Move the purge
  (`drive_ingestor.py:583-587`, currently after reconstruct+train+mesh) to fire on the post-
  extraction verification of `V` only. "Verified" = frame count ≥ expected-from-duration AND
  ≥1 frame passed the quality gate. Raw remains on Drive as source of truth. *Closes D2.*
- **FR-3\*** — Author and write a **per-image metadata sidecar** for every retained frame.
  Persist `FrameQuality` (`frame_quality.py:38-64,79-92`), which is currently in-memory only, to
  a per-frame sidecar (`<frame>.json`) carrying: `source_video`, `session`, `frame_idx`,
  `timestamp`, `blur`, `exposure`, `sharpness`, `phash`, and a `pose_hint` slot. *Closes D3.*
- **FR-4** — Run the quality gate **per video** (blur/exposure), not on the pooled aggregate
  distribution (`quality_gates.py` `FrameStats`), so a single bad video cannot skew another's
  thresholds; drop bad frames before the metadata write. *Closes D3 (scope).*
- **FR-5** — Carry the per-image sidecar forward so the image→source-video mapping is queryable
  at every later phase; this is the root of the lineage chain consumed by FR-19. *Closes D14 (root).*

### Phase 2 — Reconstruction core & branching (`config.py`, `stages.py`, `cli.py`)

- **FR-6\*** — Surface branch choices as one explicit, declarative **pipeline profile**: a single
  validated object selecting `matcher` (`config.py:40`), `strategy` (`:50`), `scene_preset`
  (`:73`), `mesh_method` (`:60`), hull backend, `sam3` concepts (`:111`), `hunyuan3d` (`:151`),
  `delivery` (`:191`). No new selection *policy* (ADR-003 owns that) — just one declarative
  surface over the existing knobs. *Closes D4.*
- **FR-7** — Add an **optional** resumable DAG / state-machine layer over the linear
  caller-driven sequence (`stages.py:43-56` `STAGE_NAMES`, `cli.py:119-214`) providing
  stage-level resume keyed on the per-video ledger from FR-1. Default execution order is
  unchanged; the DAG only governs resume/routing. *Closes D5.*
- **FR-8** — Define and document what "correctly placed" means: **intra-scene consistency**
  (COLMAP-relative, `coordinate_transform.py:34` `SCENE_SCALE=0.5`) is the v3 contract; survey/
  georeferenced placement is explicitly deferred to ADR-010 as a future option, not a v3
  requirement. *Closes D6.*

### Phase 3 — SAM3 key-item identification (`sam3_segmentor.py`, `stages.py`, `mask_projector.py`)

- **FR-9\*** — Implement **key-item ranking + threshold**. Rank SAM3 detections
  (`sam3_segmentor.py:192-248`) by a composite of size, per-object gaussian count, detection
  confidence, and concept priority; **enforce** `min_object_gaussians` (`config.py:109`, defined
  but never applied at `stages.py:1410-1419`) to drop noise. Only ranked key items proceed to
  hull recon. *Closes D7.*
- **FR-10** — Make mask→3D projection **depth-aware**. Replace the XY-plane majority vote
  (`mask_projector.py:153-214`) with a depth-gated assignment so co-located objects do not merge
  into one subset. *Closes D10.*

### Phase 4 — Per-key-item hull reconstruction (`comfyui_inpainter.py`, `stages.py`, `multiview_renderer.py`)

- **FR-11\*** — **Wire the local FLUX inpainter into the per-object multiview loop.** Before
  Hunyuan3D hull recon (`stages.py:1778-1806`), for each key item whose orbit render
  (`multiview_renderer.py:148-240`) has unseen/occluded views, call the already-built local FLUX
  inpainter (`comfyui_inpainter.py:86-107`, local ComfyUI :3001) to recover those views. This is
  the proactive guard against hallucinated hull geometry. *Closes D8.*
- **FR-12\*** — **Persist per-object world pose** (centroid + rotation quaternion + scale) from
  segmentation through to `ObjectDescriptor`. The placement machinery already consumes
  `obj.centroid/rotation_quat/scale` (`usd_assembler.py:169-175`); stop discarding it at
  `stages.py:1683-1693` (which normalises then drops it). Plumb the real pose end-to-end so hulls
  land at their pose, not at origin/identity. *Closes D9.*
- **FR-13** — Guarantee hull texturing. Today only `<30k`-face hulls are baked
  (`stages.py:1741-1771`) and large hulls ship untextured. Add a decimate-then-bake path (or
  accept native PBR from the backend) so **every** hull is textured before assembly. *Closes D11.*

### Phase 5 — Environment mesh (`usd_assembler.py`, `assemble_usd_scene.py`)

- **FR-14\*** — Create a real **textured polygonal environment mesh prim** under
  `/World/Environment`. Today `/World/Environment/Background` holds only a `DomeLight`
  (`usd_assembler.py:147-154`) and the full-scene mesh is mis-homed under
  `/World/Objects/full_scene` (`assemble_usd_scene.py:619-726`). Re-home the scene mesh to
  `/World/Environment` as the textured environment surface, produced by the `mesh_method` branch
  (`config.py:60`). *Closes D13.*

### Phase 6 — Richly-annotated USD scene graph (`usd_assembler.py`, `assemble_usd_scene.py`)

- **FR-15\*** — Author and populate a namespaced **`v2g:*` USD metadata schema** on every node,
  using the `ObjectDescriptor.metadata` hook (`usd_assembler.py:221-222`) that is currently
  unused. Object prims today carry only `lichtfeld:mesh_path`/`:diffuse_path`
  (`assemble_usd_scene.py:723-725`) and `vertex_count`/`face_count` (`:536-538`). Required fields
  per object prim: `v2g:semantic_label`, `v2g:quality_score`, `v2g:bbox_extent`,
  `v2g:gaussian_count`, `v2g:recon_method`, `v2g:confidence`, `v2g:capture_timestamp`,
  `v2g:processing_timestamp`, `v2g:source_video`, `v2g:source_frames`. *Closes D12.*
- **FR-16** — Populate `v2g:*` scene-level metadata on the stage root (profile used, stage
  versions, total object count, aggregate quality) so the scene is self-describing without the
  pipeline. *Closes D12 (scene scope).*
- **FR-17** — Carry **video→image→object lineage** into object and scene metadata. Thread the
  FR-3 sidecar `source_video`/`frame_idx` through selection, segmentation, and assembly so
  `v2g:source_video` and `v2g:source_frames` (FR-15) resolve to real provenance. *Closes D14.*
- **FR-18** — Extend USD validation (`stages.py:2134`, currently checks USD-present / mesh>0) to
  assert **metadata richness**: fail if any object prim is missing a required `v2g:*` field, if
  `/World/Environment` has no mesh, or if any hull is untextured. *Closes D11, D12, D13 (gate).*
- **FR-19** — Provide a lineage-resolution query: given a USD object prim, return its source
  video and contributing frames via the `v2g:*` lineage attributes. *Closes D14.*

**FR count: 19.**

---

## 5. Non-Functional Requirements

- **NFR-1 Resumability** — Any kill is recoverable. The per-video ledger (FR-1) plus optional DAG
  (FR-7) guarantee restart resumes at the failed video/stage; completed videos are never
  re-extracted or re-trained.
- **NFR-2 Idempotency** — Re-running a `done` video is a no-op (ledger checksum match); re-writing
  a sidecar or `v2g:*` attribute yields identical output. No stage produces drift on replay.
- **NFR-3 Retention / storage ceiling** — Local NVMe holds at most **one** raw video at a time
  (FR-2). A processed video is deleted within its extraction window; the working-set ceiling is
  one video + its frame set, not the whole remote dir.
- **NFR-4 Security** — Drive service-account credentials are supplied via **Docker secret**, not a
  plain environment variable. This directly remediates **FINDING-006**
  ([v2-security-audit.md](v2-security-audit.md), §FINDING-006: secrets exposed as plain env vars,
  mitigation = Docker Swarm secrets / BuildKit `--secret` mounts) for the credential path in
  `drive_ingestor.py`. No credential is written to a logged env or an image layer.
- **NFR-5 Reproducibility** — Given the same input + pipeline profile (FR-6), the metadata schema
  is byte-stable, the stage order is deterministic, and the `v2g:*` field set is fixed. Schema
  versions are recorded in scene metadata (FR-16).
- **NFR-6 Observability** — Each per-video iteration emits a structured ledger transition
  (start/extracted/deleted/tagged/done) so an unattended overnight batch is auditable after the
  fact.

---

## 6. Acceptance Criteria — Quality Gates (PACT / risk-based)

Gates are ordered by risk weight. Each is autonomous (runs on the GPU host, no human in the loop),
proactive (designed to catch the defect at its phase, not at delivery), and targeted (one metric,
one threshold). The pipeline **fails closed** on any gate marked blocking.

| Gate | Closes | Metric | Threshold | How verified | Blocking |
|------|--------|--------|-----------|--------------|----------|
| G1 Per-video unit | D1 | Videos with own ledger row | 100% | Count ledger rows vs `rclone lsjson` video count | Yes |
| G2 Retention ceiling | D2 | Max raw videos on NVMe at any instant | ≤ 1 | NVMe scratch poll during run; assert ≤1 `*.mp4` | Yes |
| G3 Video deleted | D2 | Processed videos still local after their extraction window | 0 | Post-extraction scratch scan per video | Yes |
| G4 Sidecar coverage | D3 | Retained frames with a metadata sidecar | 100% | `count(sidecars) == count(retained frames)` | Yes |
| G5 Sidecar completeness | D3 | Sidecars with all required fields populated | 100% | Schema-validate each `<frame>.json` | Yes |
| G6 Key-item precision | D7 | Ranked key items that are real objects | ≥ 90% | Compare ranked set vs analyst ground truth on test scene | Yes |
| G7 Noise rejection | D7 | `min_object_gaussians` enforced (sub-threshold items dropped) | 100% | Assert no proceeding item below threshold | Yes |
| G8 Occlusion recovery | D8 | Low-coverage key items FLUX-inpainted pre-hull | 100% | Trace inpaint call per item below coverage threshold | Yes |
| G9 Depth separation | D10 | Co-located distinct objects kept as separate subsets | 100% on test pair | Two-object overlap fixture; assert 2 subsets | No |
| G10 Pose preserved | D9 | Object hulls placed at persisted pose (not origin/identity) | 100% non-identity where pose exists | Assert prim xform ≠ identity for posed objects | Yes |
| G11 Hull texturing | D11 | Hulls shipping textured | 100% | Per-hull material/diffuse-path presence check | Yes |
| G12 Environment mesh | D13 | Scenes with textured mesh prim under `/World/Environment` | 100% | USD traversal asserts mesh + material, not just DomeLight | Yes |
| G13 USD metadata richness | D12 | Object prims with all ≥10 required `v2g:*` fields | 100% | FR-18 validator over every object prim | Yes |
| G14 Lineage closure | D14 | Object prims resolvable to source video + frames | 100% | FR-19 query returns non-empty video + frames per object | Yes |
| G15 Profile reproducibility | D4 | Repeat run, same profile → identical schema + stage order | 100% | Re-run; diff metadata schema + stage log | No |
| G16 Resume correctness | D5 | Killed run resumes at failed video/stage, no redo | 100% | Inject kill; assert no completed video re-processed | No |
| G17 Secret handling | NFR-4 | Credentials absent from env/image; present as Docker secret | Pass | `docker inspect` + image-layer scan find no plaintext cred | Yes |

**Gate count: 17.**

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FLUX hallucinates plausible-but-wrong geometry on occluded faces → bad hulls | Medium | High | Gate inpaint behind a visible-coverage threshold (FR-11); only inpaint genuinely unseen views; G8 traces every call; reviewer spot-check on test scene |
| Depth-lossy mask merge collapses co-located objects into one hull | Medium | High | Depth-aware projection (FR-10); G9 two-object overlap fixture; fall back to per-mask seeds if depth ambiguous |
| Survey-pose ambiguity ("correctly placed" over-scoped) | Medium | Medium | FR-8 fixes the v3 contract to intra-scene consistency; survey/georef explicitly deferred to ADR-010 — out of v3 scope |
| Per-video deletion races extraction verification → data loss | Low | High | FR-2 deletes only on *verified* extraction (frame-count + ≥1 passing frame); raw always retained on Drive as source of truth |
| Key-item ranking drops a genuine item (false negative) | Medium | Medium | Tune `min_object_gaussians` on test scene; G6 measures precision, complement with recall spot-check; threshold is profile-configurable |
| Large hulls untextured after decimation degrades quality | Low | Medium | FR-13 decimate-then-bake guarantees texture; G11 blocks delivery of any untextured hull |
| `v2g:*` schema churn breaks archival consumers | Low | Medium | NFR-5 byte-stable schema + schema version in scene metadata (FR-16); additive-only changes |
| DAG layer destabilises the proven linear path | Low | Medium | FR-7 is optional and resume-only; default order unchanged; G16 is non-blocking |
| Docker-secret migration breaks the unattended batch launch | Low | Medium | NFR-4 follows the v2-security-audit FINDING-006 remediation pattern; validate launch on test host before batch night |

---

## 8. Out of Scope

This PRD **does not** redesign, and explicitly defers to, the existing decisions:

- **ADR-003** — pluggable mesh-extraction backend **selection policy** (`auto|tsdf|milo|come|
  gaussianwrapping`). v3 only *surfaces* the choice (FR-6); it does not change how a backend is
  chosen.
- **ADR-002** — upstream-sync strategy.
- **ADR-006** — splat-transform / web-splat **delivery**.
- **ADR-007** — Fibonacci-sphere frame **selection** algorithm.

Also out of scope: new reconstruction science (COLMAP→3DGS→`.ksplat` core, SAM3 concept
segmentation, Hunyuan3D multiview hull recon, the `{Gaussian|Mesh}` variant set, camera prims and
UsdPreviewSurface materials all work and must not be rebuilt — see gap analysis §6); dynamic/4D
scenes (ADR/PRD-v2 Phase 3); survey/georeferenced world placement (deferred per FR-8).

---

## 9. Decision Dependencies

This PRD's requirements are realised by three new ADRs and one DDD extension, which it commissions
by exact filename:

| Artefact (filename) | Decides | Realises FR / closes deltas |
|---------------------|---------|------------------------------|
| **ADR-009** (`adr-009-*.md`) | Per-video ingest unit + per-image metadata sidecar schema + delete-on-verified-extraction | FR-1..FR-5 — D1, D2, D3, D14(root) |
| **ADR-010** (`adr-010-*.md`) | Key-item ranking + threshold; FLUX-inpainted hull recovery; per-object world-pose persistence; depth-aware projection; the "correctly placed" contract | FR-8..FR-13 — D6, D7, D8, D9, D10 |
| **ADR-011** (`adr-011-*.md`) | USD `v2g:*` metadata schema enrichment + textured environment mesh prim + lineage into USD | FR-14..FR-19 — D11, D12, D13, D14 |
| **DDD extension** (`research/ddd/v3-e2e-extensions.md`) | New aggregates/value objects: per-video work unit, per-image metadata, key-item ranking, object pose, `v2g:*` annotation — extending the existing DDD model (`research/ddd/{aggregates,bounded-contexts,ubiquitous-language,anti-corruption-layers}.md`) | All phases — vocabulary + aggregate boundaries for D1–D14 |

D4 (declarative profile) and D5 (resumable DAG) are realised directly in this PRD's FR/NFR
(FR-6, FR-7, NFR-1) per the gap register's "closes via" column, and do not require a separate ADR.

---

## 10. Traceability — Delta → FR → Gate

| Delta | Shape gap | FR(s) | Gate(s) | ADR |
|-------|-----------|-------|---------|-----|
| D1 | Granularity | FR-1 | G1 | 009 |
| D2 | Granularity | FR-2 | G2, G3 | 009 |
| D3 | Provenance | FR-3, FR-4, FR-5 | G4, G5 | 009 |
| D4 | Annotation | FR-6 | G15 | PRD §4 |
| D5 | Annotation | FR-7 | G16 | PRD §5 |
| D6 | Selection | FR-8 | — (contract) | 010 |
| D7 | Selection | FR-9 | G6, G7 | 010 |
| D8 | Recovery | FR-11 | G8 | 010 |
| D9 | Recovery | FR-12 | G10 | 010 |
| D10 | Selection | FR-10 | G9 | 010 |
| D11 | Recovery | FR-13 | G11 | PRD §4 / 011 |
| D12 | Annotation | FR-15, FR-16, FR-18 | G13 | 011 |
| D13 | Annotation | FR-14 | G12 | 011 |
| D14 | Provenance | FR-5, FR-17, FR-19 | G14 | 011 |

Every delta D1–D14 traces to at least one FR; every blocking gate traces to a delta. Coverage is
complete.
