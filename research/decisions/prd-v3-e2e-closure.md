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
| O12 | SOTA mesh + matching | Default mesh = MILo; indoor SfM = ALIKED+LightGlue (not TSDF/SIFT) | 100% of default runs |
| O13 | SOTA generative models | Hull = Hunyuan3D-2.1; inpaint = FLUX.1 Kontext (with declared fallbacks) | 100% where checkpoints present |
| O14 | Build reproducibility | Pinned version/commit for every model + tool (no HEAD clones) | 100% of build inputs pinned |
| O15 | Semantic artifact rejection | VLM-flagged artifact frames vetoed/repaired before pooling, vs analyst ground truth | ≥ 90% precision on test scene |
| O16 | Metadata-aware scaffolding | Reconstruction candidates selected with VLM + capture metadata in the loop | 100% of candidates carry a fused score |
| O17 | Non-technical onboarding | A user authors a valid `exhibit.toml` via the web wizard with zero file editing | 100% of required manifest fields settable from the UI |
| O18 | Hardware-correct model selection | Wizard-recommended model/quant per stage fits the probed per-GPU VRAM | 100% — no recommendation exceeds `/api/hardware` VRAM |
| O19 | Secret containment | Credentials (HF token, Google refresh token) absent from browser JS, TOML, and git | 100% — secrets only as server-side keyring/Docker-secret, `env:`-referenced |
| O20 | Output write-back | Finished artifacts uploaded back to the source Drive folder the video came from | 100% of completed runs with `[drive].writeback = true` |

Success is the conjunction O1–O16 on the curated test scene (verified by the §6 quality gates);
the onboarding objectives O17–O20 (ADR-015) are verified by the §6 onboarding gates G-O1..G-O4.

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

### Cross-cutting — SOTA tooling modernisation (`config.py`, Dockerfiles, `hunyuan3d_client.py`, `comfyui_inpainter.py`, new VLM stage) — ADR-012

These FRs are infrastructure, applied across phases; they realise the **T-series** gaps (gap
analysis §6) and are decided by **ADR-012**. `†` marks items requiring a model pull / HF token.

- **FR-20\*** — Flip the default mesh backend from `tsdf` to **MILo** (`config.py:60`); keep TSDF as
  the no-sidecar fallback in `_select_mesh_backend()` (`stages.py:895-941`). No change to the ADR-003
  selection *policy*. *Closes T1.*
- **FR-21** — Add **learned features + matching** to COLMAP: `feature ∈ {sift, aliked}` and
  `matcher += lightglue` (`config.py:40`); route `_run_colmap_direct()` (`stages.py:624`) through
  COLMAP 4.1 ALIKED+LightGlue, default for indoor presets, `sift+exhaustive` fallback. *Closes T2.*
- **FR-22†** — Upgrade the image-to-3D hull backend to **Hunyuan3D-2.1** (`hunyuan3d_client.py:61,77`)
  behind a capability probe that degrades to 2.0. *Closes T3.*
- **FR-23†** — Upgrade the inpainter to **FLUX.1 Kontext** (`comfyui_inpainter.py:88`); ADR-010 FR-11
  targets Kontext from the outset; degrade to FLUX.1-Fill if absent. *Closes T4.*
- **FR-24†** — Add an optional **neural feed-forward SfM** branch `sfm ∈ {colmap, vggt, mast3r}`
  feeding the same posed-image contract; COLMAP stays the accuracy default. *Closes T5.*
- **FR-25\*** — **Pin every model and tool** to a tag/commit (COLMAP, gsplat, ComfyUI, SAM3, PyTorch,
  usd-core) across all Dockerfiles; no unpinned HEAD clones. Build-layer realisation of NFR-5.
  *Closes T6.*
- **FR-26** — **Adopt native v0.5.x + plugin ecosystem** where it replaces custom code: install/pin
  `splat_ready` (or remove the dead reference `stages.py:574`); enable PPISP / bilateral-grid /
  3DGUT / ImprovedGS+ training flags per-preset (`stages.py:748-756`); evaluate native USD export
  vs `scripts/assemble_usd_scene.py` for the `v2g:*` schema. *Closes T7.*
- **FR-27\*†** — Add a **VLM artifact-analysis** stage the containerised agent calls on photometric
  survivors: the **unified multimodal gemma-4-26B-A4B agent** (ADR-013 D-013.5; Qwen2.5-VL / InternVL3
  fallback) emits a typed `artifact_report` (motion ghosting, rolling-shutter, specular blowout, flare,
  transient occluders, compression blocking; label + confidence + bbox) written as a `vlm` block in the
  ADR-009 per-frame sidecar. *Closes T8.*
- **FR-28\*** — **Metadata-aware candidate scaffolding.** Ingest per-capture/project metadata
  (camera/lens, capture session, operator notes, EXIF/SRT GPS) as an additive `capture` sidecar
  block, and fuse it with the FR-27 `artifact_report` and the FR-3 photometric tags to rank/scaffold
  reconstruction candidates: Fibonacci coverage (ADR-007) as geometric prior, VLM artifact score as
  semantic veto, occluder/blowout regions flagged for FLUX inpaint (FR-11). The agent **annotates,
  never silently drops**; every veto/repair reason is recorded in the ledger and `v2g:*` lineage.
  *Closes T8.*

### Cross-cutting — Ingest manifest, serial model lifecycle, docker mesh (`manifest.py`, `model_lifecycle.py`, `config.py`, compose) — ADR-013

These requirements give the pipeline a single human-authored input and a VRAM-bounded model
schedule. They are decided by **ADR-013** and are gated on its Open Questions Q1–Q7. `†` marks
items requiring a model pull / HF token.

- **FR-29** — **Pre-run TOML ingest manifest** (`exhibit.toml`). A loader parses exhibit identity,
  an object sub-list (array of tables, each with stable `id`/`sam3_concept`/`priority`), the Drive
  URL, and `env:`-indirected secrets (HF, Google Cloud), and **materialises** the existing
  `PipelineConfig`. Secrets are resolved at load and stripped before the JSON run snapshot; a missing
  referenced env-var fails by name. `priority="key"` objects enter the ADR-010 hull-recon path. The
  manifest also carries `[oversight]`: `backend` (**default `claude_code`** — the in-container Claude
  Code overseer, which the user must log into inside the container once; `gemma_local` optional with a
  GPU-contention tradeoff per ADR-013 D-013.6) and `artifact_vlm` (transient bulk-triage tool, default
  `gemma_local`).
- **FR-30†** — **Serial model load/unload lifecycle.** Stages run serially behind a
  `ModelLifecycleManager`; each declares a `ModelSpec(engine, checkpoint, vram_estimate, gpu_affinity,
  isolation)`. The manager asserts VRAM headroom, loads, yields, then unloads — `soft` (ComfyUI
  `/free`, `cuda.empty_cache`) by default, `hard` (container stop/start) for FLUX.2↔Hunyuan3D. Peak
  VRAM is bounded to the largest single stage, enabling the most-performant model per stage.
- **FR-31** — **Docker-network model mesh (`v2g-net`).** Replace hardcoded `192.168.2.48:port` /
  `localhost:port` endpoints (`config.py:135-136,152-153,171`) with service-DNS names (`comfyui:8188`,
  `vlm:8081`, `reasoner:8080`), overridable from the manifest `[pipeline]` block. The lifecycle
  manager owns container start/stop on this network. *gemma-4-26B-A4B is multimodal
  (`Gemma4ForConditionalGeneration`, SigLIP vision, `mmproj`), so one `agent-vlm` model serves both the
  FR-27 artifact VLM and the FR-28 reasoner — no separate Qwen2.5-VL required (fallback only).*

### Cross-cutting — Agent-controlled ComfyUI on .48 (`comfyui_inpainter.py`, `workflows/`) — ADR-014

- **FR-32** — **Update & pin the existing .48 ComfyUI.** Idempotently ensure the FLUX.2-dev /
  Hunyuan3D-2.1 checkpoints and pinned custom nodes via the Salad add-on control API
  (`probe_models`/`download_model`); missing checkpoints degrade to declared fallbacks (FLUX.1-Fill /
  Hunyuan3D-2.0) behind a capability probe. Pins recorded in the ADR-012 T6 version lock.
- **FR-33** — **Connect over `v2g-net`.** Replace hardcoded `192.168.2.48:{3001,8189,8188}` defaults
  (`comfyui_inpainter.py:282-284`) with service-DNS endpoints (`comfyui:8188` graph, `comfyui:3001`
  Salad control); the `ImageServer` binds the orchestrator's network address so ComfyUI fetches inputs
  by DNS, not `192.168.2.1`. Literal-IP form retained only as a manifest override.
- **FR-34** — **Agent control loop (VLM-in-the-loop).** Promote the one-shot `inpaint()` into a
  `RecoveryController`: the gemma-4 agent plans the FLUX.2/Hunyuan3D graph, submits, **looks at** the
  output, and decides `accept` | `re-prompt` (bounded retries, adjust denoise/guidance/seed/mask) |
  `veto`. Annotates never drops — every attempt + VLM verdict + reason to the ledger + `v2g:*` lineage.
  Operationalises FR-11/FR-27/FR-28.

### Cross-cutting — Web onboarding & setup tool ("Vitrine Onboarding") (`vitrine-setup/`, `schema/exhibit.toml.schema.json`, `frontend/`) — ADR-015

These requirements give the human-authored `exhibit.toml` (FR-29) an ergonomic, non-technical
authoring surface modelled on the proven agentbox setup tool, and bound the setup-vs-agent hand-off.
They are decided by **ADR-015** and gated on its Open Questions Q-015.1–Q-015.3. `†` marks items
requiring a model pull / external credential.

- **FR-35\*** — **Schema-driven onboarding wizard.** A frameworkless vanilla-JS frontend + a single
  static Rust/Axum binary (`vitrine-setup`, ephemeral `127.0.0.1:0`) render a stepper (Exhibit →
  Objects → Hardware/Models → Secrets/Login → Provision & Hand-off) from a **JSON Schema**
  (`schema/exhibit.toml.schema.json`) that is the single source of truth for the ADR-013 manifest.
  `toml_edit::DocumentMut` round-trips the file so comments and key order survive. The wizard's output
  **is** the FR-29 `exhibit.toml` — zero translation. *Re-entrant, no history*: on start the backend
  loads the existing manifest (if present) and populates editable fields; the user re-saves the **same**
  single active file (no past-project list). *Realises ADR-015 D-015.2.*
- **FR-36** — **Hardware-aware model selection.** A `/api/hardware` endpoint probes the host
  (GPU count, per-GPU VRAM via `nvidia-smi`, RAM, disk) and maps it to the FR-29/D-013.4 per-stage
  table, **recommending** the most-performant model/quant that fits (e.g. 48 GB → FLUX.2-dev fp8mixed +
  Hunyuan3D-2.1 + gemma-4 Q5_K_M; 24 GB → smaller quants, `hard` unload everywhere). The user accepts
  or overrides; choices write a new `[models]` + `[models.vram_plan]` manifest block. No recommendation
  may exceed probed VRAM (O18). *Realises ADR-015 D-015.3.*
- **FR-37\*†** — **Secret entry & browser-based login (server-side containment).** No credential ever
  enters the browser JS or the TOML (extends NFR-4 / FINDING-006 to the onboarding surface). The HF
  token is pasted → `POST` to the backend → stored as a host-keyring / Docker-secret entry, referenced
  from the manifest only as `hf_token = "env:HF_TOKEN"` (FR-29 secret indirection), shown masked. Google
  **browser OAuth** (Drive read + write scope) runs the consent flow in the browser; the **refresh token
  is stored server-side**, never in the browser or TOML, referenced as
  `gcloud_credentials = "env:GOOGLE_APPLICATION_CREDENTIALS"` (or an rclone remote). The Rust backend
  holds the token and proxies Drive calls (`/api/proxy/*`, server-side `Authorization: Bearer`).
  *Realises ADR-015 D-015.4.*
- **FR-38†** — **Deterministic provisioning + setup/agent hand-off boundary.** Unlike the agentbox
  config wizard, `vitrine-setup` **provisions**: it downloads & integrates the FR-36 models (HF pulls
  via the stored token), ensures+pins the .48 ComfyUI checkpoints/nodes via the ADR-014 Salad control
  API (FR-32), brings up `v2g-net`, and verifies readiness — idempotent, scriptable, no interpretation,
  progress streamed to the UI. It ends by writing `provision.status = "ready"` and emitting a hand-off
  event. The **internal Claude Code overseer** (FR-29 `[oversight]`, ADR-013 D-013.6) then does the
  *interpretive* scaffolding: turning the free-text objects-of-interest into SAM3 concept candidates +
  per-object recovery plans (FR-9/FR-11). Boundary: **setup makes the system runnable; the agent decides
  how to run it.** *Realises ADR-015 D-015.5.*
- **FR-39** — **Output write-back to the source Google Drive.** The manifest `[drive]` block gains
  `writeback` / `writeback_subdir`; finished artifacts (USD scene, ksplat, per-object meshes, run report)
  are uploaded back to the **same Drive folder** the source video came from (or a `vitrine-output/`
  subfolder). Requires the Drive **write** scope from the FR-37 OAuth grant — the rclone/Drive ACL
  extends from read-only ingest to read+write (DDD §5.1). *Realises ADR-015 D-015.6.*
- **FR-40** — **Project rename to Vitrine.** New docs, the CLI/package id (`vitrine`), and the onboarding
  tool adopt the name **Vitrine** now; a full code/repo/remote rename is an explicitly-scheduled,
  separate follow-up (high blast radius — not done silently). *Realises ADR-015 D-015.1.*

**FR count: 40** (19 workflow-shape + 9 tooling + 3 ingest/lifecycle + 3 ComfyUI agent-control +
6 onboarding/setup).

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
| G-T1 Mesh default | T1 | Default `mesh_method` resolves to MILo (TSDF only on sidecar-down) | Pass | Assert config default == milo; fallback path covered | Yes |
| G-T2 Learned matching | T2 | Indoor preset SfM uses ALIKED+LightGlue | Pass | Inspect COLMAP feature/matcher in run log | No |
| G-T3 Model versions | T3, T4 | Hull == Hunyuan3D-2.1, inpaint == FLUX Kontext (or declared fallback logged) | Pass | Assert requested model IDs; fallback emits explicit log | Yes |
| G-T4 Pins | T6 | Every model/tool pinned to tag/commit (no HEAD) | 100% | Scan Dockerfiles for unpinned clones / bare pip | Yes |
| G-T5 VLM artifact precision | T8 | VLM-vetoed frames that are genuine artifacts vs analyst GT | ≥ 90% | Compare `artifact_report` vetoes vs ground truth on test scene | Yes |
| G-T6 Fused scaffolding | T8 | Pooled candidates carry a fused (photometric + VLM + capture) score, no silent drops | 100% | Assert every pooled frame has a recorded score + every veto a reason | Yes |
| G-O1 Schema round-trip | O17 | Wizard save → `exhibit.toml` re-loads and re-renders identical field values (comments/order preserved) | 100% | Save, reload, assert `toml_edit` diff is comment/order-stable and values equal | Yes |
| G-O2 VRAM fit | O18 | Wizard model recommendation fits probed per-GPU VRAM | 100% | Assert each recommended `serial_peak_estimate_gb` ≤ `/api/hardware` GPU VRAM | Yes |
| G-O3 Secret containment | O19 | No credential present in browser payloads, `exhibit.toml`, or git; only `env:`-references | Pass | Scan served JS + manifest + tree for plaintext token; assert only `env:` indirection | Yes |
| G-O4 Write-back | O20 | Completed run with `writeback = true` uploads artifacts to the source Drive folder | 100% | Post-run `rclone lsjson` of the source/`vitrine-output` folder finds the artifact set | No |

**Gate count: 27** (17 workflow-shape + 6 tooling + 4 onboarding).

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
| MILo-as-default raises no-sidecar failure surface | Low | Medium | FR-20 keeps TSDF fallback in `_select_mesh_backend()`; G-T1 covers the fallback path |
| Gated FLUX Kontext / Hunyuan3D-2.1 weights unavailable on the host | Medium | Medium | FR-22/23 capability probe degrades to 2.0 / FLUX-Fill; model-pull step requires a HF token with licences accepted; G-T3 logs which model actually ran |
| VLM hallucinates artifacts or slows large batches | Medium | Medium | FR-27 runs VLM only on photometric survivors, dedups by `phash`, requires confidence threshold; annotates not drops (G-T6); A/B precision on test scene (G-T5) |
| Native v0.5.x flags (PPISP/3DGUT) alter output character | Low | Medium | FR-26 enables per-preset, A/B against curated test scene before making preset-default |

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

This PRD's requirements are realised by seven new ADRs and one DDD extension, which it commissions
by exact filename:

| Artefact (filename) | Decides | Realises FR / closes deltas |
|---------------------|---------|------------------------------|
| **ADR-009** (`adr-009-*.md`) | Per-video ingest unit + per-image metadata sidecar schema + delete-on-verified-extraction | FR-1..FR-5 — D1, D2, D3, D14(root) |
| **ADR-010** (`adr-010-*.md`) | Key-item ranking + threshold; FLUX-inpainted hull recovery; per-object world-pose persistence; depth-aware projection; the "correctly placed" contract | FR-8..FR-13 — D6, D7, D8, D9, D10 |
| **ADR-011** (`adr-011-*.md`) | USD `v2g:*` metadata schema enrichment + textured environment mesh prim + lineage into USD | FR-14..FR-19 — D11, D12, D13, D14 |
| **ADR-012** (`adr-012-*.md`) | SOTA tooling modernisation: MILo default, ALIKED+LightGlue, Hunyuan3D-2.1, **FLUX.2-dev** (amended from Kontext), neural-SfM branch, version pins, native v0.5.x + plugin adoption, VLM artifact analysis + metadata-aware scaffolding | FR-20..FR-28 — T1..T8 |
| **ADR-013** (`adr-013-*.md`) | Pre-run TOML ingest manifest (`exhibit.toml`) + serial model load/unload lifecycle (VRAM-bounded, soft/hard unload) + docker-network model mesh (`v2g-net`); unified multimodal gemma-4 `agent-vlm` (artifact VLM + reasoner); target host .48 | FR-29..FR-31 — operationalises FR-27/FR-28; Q1–Q4/Q7 resolved 2026-06-04 |
| **ADR-014** (`adr-014-*.md`) | Agent-controlled ComfyUI on .48: update+pin checkpoints/nodes via Salad control API; connect over `v2g-net` (service DNS); VLM-in-the-loop `RecoveryController` (gemma-4 plans→submits→evaluates→accept/re-prompt/veto); Generative Recovery bounded context + ACL | FR-32..FR-34 — operationalises FR-11/FR-27/FR-28 |
| **ADR-015** (`adr-015-*.md`) | **Vitrine** rename; schema-driven web onboarding/setup tool (`vitrine-setup`, agentbox pattern: vanilla JS + Rust/Axum, JSON-Schema-driven `exhibit.toml` editor, `toml_edit` round-trip, re-entrant/no-history); hardware-aware model selection; browser-OAuth + server-side secret containment; deterministic-provisioning vs internal-agent hand-off; Drive output write-back; new Onboarding/Setup bounded context + Drive write-back ACL | FR-35..FR-40 — authors FR-29 manifest; Q-015.1–Q-015.3 open |
| **DDD extension** (`research/ddd/v3-e2e-extensions.md`) | New aggregates/value objects: per-video work unit, per-image metadata, key-item ranking, object pose, `v2g:*` annotation — extending the existing DDD model (`research/ddd/{aggregates,bounded-contexts,ubiquitous-language,anti-corruption-layers}.md`). Tooling (ADR-012) adds **no** new aggregate — the VLM `artifact_report` and `capture` block are additive fields on the existing Frame / ImageMetadataTag | All phases — vocabulary + aggregate boundaries for D1–D14 |

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
| T1 | Tooling | FR-20 | G-T1 | 012 |
| T2 | Tooling | FR-21 | G-T2 | 012 |
| T3 | Tooling | FR-22 | G-T3 | 012 |
| T4 | Tooling | FR-23 | G-T3 | 012 |
| T5 | Tooling | FR-24 | — (optional branch) | 012 |
| T6 | Tooling | FR-25 | G-T4 | 012 |
| T7 | Tooling | FR-26 | G-T2 | 012 |
| T8 | Tooling | FR-27, FR-28 | G-T5, G-T6 | 012 |
| O1 | Ingest/ops | FR-29 | — (manifest contract) | 013 |
| O2 | Ingest/ops | FR-30 | — (VRAM-bounded run) | 013 |
| O3 | Ingest/ops | FR-31 | — (docker mesh) | 013 |
| O4 | Generative | FR-32 | — (pin/probe contract) | 014 |
| O5 | Generative | FR-33 | — (network connect) | 014 |
| O6 | Generative | FR-34 | G8 (verified recovery) | 014 |
| O17 | Onboarding | FR-35 | G-O1 | 015 |
| O18 | Onboarding | FR-36 | G-O2 | 015 |
| O19 | Onboarding | FR-37 | G-O3 | 015 |
| O20 | Onboarding | FR-38, FR-39 | G-O4 | 015 |
| — | Onboarding (rename) | FR-40 | — (docs contract) | 015 |

Every delta D1–D14 and tooling gap T1–T8 traces to at least one FR; every blocking gate traces to a
delta or tooling gap. FR-29..FR-31 (ADR-013) + FR-32..FR-34 (ADR-014) operationalise FR-11/FR-27/FR-28;
FR-35..FR-40 (ADR-015) author the FR-29 manifest and bound the setup/agent hand-off.
ADR-013 Q1–Q4/Q7 resolved 2026-06-04, Q5/Q6 open (non-blocking); ADR-015 Q-015.1–Q-015.3 open
(non-blocking). Coverage is complete.
