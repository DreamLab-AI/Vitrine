# ADR-012: SOTA Tooling Modernisation — Native v0.5.x, Plugin Ecosystem, and Model Upgrades

## Status

Proposed Amended 2026-06-05.

## Amendment (2026-06-05) — current SOTA stack (folds the live decisions in)

The model selections in this ADR have evolved; the live choices, now staged and verified
on the reference host and enforced by the `sota_registry.py` idiot-check:

| Element | Decision | Note |
|---|---|---|
| Env mesh | **CoMe** default | supersedes MILo-default (ADR-004) |
| 3D hull | **TRELLIS.2-4B** primary, Hunyuan3D-2.1 fallback | supersedes Hunyuan-2.1-only (ADR-010) |
| Inpaint | **FLUX.2-dev** (fp8) | supersedes FLUX.1-Kontext/Fill |
| VLM | **gemma-4-26B-A4B Q8_0** | Q5_K_M not published; ~28 GB, fits 48 GB serial |
| SfM | **ALIKED+LightGlue** via LichtFeld COLMAP plugin | SIFT fallback |
| Training | **ImprovedGS+** (native) | switch default from MRNF (config change pending) |
| Pins | `pins.lock.toml` + `scripts/resolve_pins.sh` -> `pins.resolved.toml` | T6 |

**Licence posture (research / non-commercial) is the governing default**, enforced by the
idiot-check: CoMe (CC BY-NC-ND), FLUX.2-dev and SAM3D (non-commercial) are allowed under
it and FAIL a `--commercial` check. A commercial build swaps CoMe->PGSR and
FLUX.2->Qwen-Image-Edit (both permissive). The detailed, executable registry is
`src/pipeline/sota_registry.py`; the work plan is
`research/decisions/work-order-sota-modernisation.md`.

## Context

ADR-009/010/011 and PRD-v3 close the *workflow-shape* gaps (granularity, provenance,
selection/recovery, annotation) but assume the **current** tool stack. An online + in-repo audit
(2026-06-04) found that the stack defaults to the **weakest available option at every fork** and
has **forked past native capability** that the upstream engine now ships. These are infrastructure
gaps, not domain gaps — they add no new aggregates to the DDD model — but they leave substantial
planned performance and quality on the table.

We forked **LichtFeld Studio v0.5.2** (`CMakeLists.txt:33`, sync commit `9517dd51`, upstream
`MrNeRF/LichtFeld-Studio`). The v0.5.x line and its plugin registry now provide features our custom
Python pipeline re-implements or ignores. Audit findings, with evidence:

- **T1 — Default mesh backend is the weakest baseline.** `config.py:60` sets `mesh_method = "tsdf"`.
  MILo (SIGGRAPH Asia 2025, `milo_extractor.py`) and CoMe (`come_extractor.py`) are **fully wired**
  (`stages.py:701-718`) but never the default. TSDF is gsplat-depth + marching cubes — the
  intentional fallback, not the quality path.

- **T2 — COLMAP uses plain SIFT, not learned features/matching.** `stages.py:639`,
  `config.reconstruct.matcher ∈ {exhaustive, sequential, vocab_tree}` — all SIFT. The
  `lichtfeld-360-plugin` and `Lichtfeld-COLMAP-Plugin` ship **COLMAP 4.1 with ALIKED extraction and
  LightGlue matching**, which dominate SIFT on textureless / low-overlap indoor scenes — exactly the
  heritage-room failure case.

- **T3 — Hunyuan3D pinned at 2.0.** `hunyuan3d_client.py:61,77,93` request `hunyuan3d-dit-v2-0` /
  `-v2-mv`. Hunyuan3D-2.1 improves texture fidelity and adds a paint-only mode. No 2.1 reference
  exists in the repo.

- **T4 — Inpainting pinned at FLUX.1-Fill-dev.** `comfyui_inpainter.py:88` loads
  `flux1-fill-dev.safetensors`. A context-aware edit model is materially better for the per-object
  occluded-face recovery that ADR-010 FR-11 commissions. **Superseded by ADR-013:** the owner has
  designated **FLUX.2-dev** (already staged on this host) as the target inpaint/edit model rather
  than FLUX.1 Kontext — see ADR-013 §model-selection. Earlier "FLUX.1 Kontext" wording below is
  retained for history but the binding target is now FLUX.2-dev.

- **T5 — No neural feed-forward SfM option.** COLMAP is the only SfM path and is cloned at an
  **untagged HEAD** (`Dockerfile.consolidated:104`). VGGT / MASt3R-SfM / DUSt3R reconstruct in
  seconds without iterative bundle adjustment and recover poses on low-overlap captures COLMAP
  fails on. `research/pipelines/proposed-pipeline.md:200` already names DUSt3R as the alternative;
  no code path exists.

- **T6 — Zero version pins → non-reproducible builds.** COLMAP, gsplat (`pip install gsplat`,
  `Dockerfile.consolidated:176`), ComfyUI, SAM3 (`Dockerfile.consolidated:199`), and the main
  PyTorch (`--index-url cu128`, no version) are all unpinned HEAD/latest. A silent upstream change
  breaks the GPU host with no diff. This violates PRD-v3 NFR-5 (reproducibility) at the build layer.

- **T7 — Native v0.5.x capability re-implemented in Python.** The engine ships features we fork
  past and rebuild: **native USD import/export** (v0.5.1) vs. our hand-rolled
  `scripts/assemble_usd_scene.py`; **plugin marketplace + MCP automation** (v0.5.0) vs. our
  subprocess orchestration in `stages.py`; **ImprovedGS+** (v0.5.0), **PPISP** camera-response
  modelling — exposure drift / vignetting / white balance (v0.4.1), **bilateral grid**, **3DGUT**
  distorted-camera support and **pose optimisation** training flags — none enabled. The
  `splat_ready` plugin is *referenced* (`stages.py:574`) but **not installed** (`~/.lichtfeld/
  plugins/splat_ready/` is empty), so we always fall to `_run_colmap_direct()`.

- **T8 — No vision-language artifact analysis; metadata-blind candidate selection.** Frame quality
  today is purely photometric: blur / exposure / sharpness scalars (`frame_quality.py:79-92`) with
  no semantic understanding of *reconstruction* artifacts (motion ghosting, rolling-shutter skew,
  specular blowout, lens flare, transient occluders, compression blocking). There is no VLM in the
  loop, and the per-image / capture-context metadata a VLM would reason over is not ingested
  (ADR-009 introduces the per-frame sidecar; capture/project metadata beyond it is still absent).
  The containerised agent therefore scaffolds reconstruction candidates blind to both semantic
  artifacts and provenance context.

These map to a new **T-series** in `research/decisions/gap-analysis-e2e-aspiration.md` (§7) and are
commissioned as FR-20..FR-28 in `research/decisions/prd-v3-e2e-closure.md`.

## Decision

Adopt SOTA tooling across the stack in four tranches, ordered by quality-gain-per-unit-effort. Each
sub-decision is independently shippable and gated.

### D-012.1 — Flip the default mesh backend to MILo (closes T1)

Change `config.py:60` default from `"tsdf"` to `"milo"`; keep `tsdf` as the no-sidecar fallback in
`_select_mesh_backend()` (`stages.py:895-941`). CoMe remains opt-in (Inria/MPII non-commercial,
`INSTALL_COME=0`). The selection *policy* is unchanged (ADR-003 owns it) — only the default value.

### D-012.2 — Adopt learned features + matching for COLMAP (closes T2)

Add `feature ∈ {sift, aliked}` and extend `matcher` with `lightglue` to `ReconstructConfig`
(`config.py:40`). Build COLMAP 4.1 with the ALIKED + LightGlue path (GPU wheels) and route
`_run_colmap_direct()` (`stages.py:624`) through it. Default the **indoor** presets to
`aliked + lightglue`; keep `sift + exhaustive` as the universal fallback. Where the
`Lichtfeld-COLMAP-Plugin` / `lichtfeld-360-plugin` provides this natively, prefer the plugin over
re-implementation (see D-012.4).

### D-012.3 — Upgrade the generative models (closes T3, T4)

Move the image-to-3D hull backend to **Hunyuan3D-2.1** (`hunyuan3d_client.py` model IDs) and the
inpainter to **FLUX.2-dev** (`comfyui_inpainter.py`) — *amended from FLUX.1 Kontext per owner
preference; see ADR-013.* Both are served through the existing ComfyUI endpoint; this is a
checkpoint + workflow-graph change, not new infrastructure. The local FLUX wiring that ADR-010
FR-11 commissions targets FLUX.2-dev from the outset. Keep Hunyuan3D-2.0 / FLUX.1-Fill as declared
fallbacks behind a capability probe so a missing checkpoint degrades, not breaks.

### D-012.4 — Adopt native v0.5.x + plugin ecosystem; pin everything (closes T5, T6, T7)

- **Pin every model and tool** (T6): COLMAP tag/commit, `gsplat==<ver>`, ComfyUI commit, SAM3
  commit, PyTorch version, USD/usd-core version. No `--depth 1` HEAD clones without a pinned ref.
  This is the build-layer realisation of PRD-v3 NFR-5.
- **Install and pin the plugin ecosystem** (T7) where it replaces custom code: `SplatReady`
  (video→COLMAP) and/or `Lichtfeld-COLMAP-Plugin` for ingest, `lichtfeld-360-plugin` for the
  ALIKED+LightGlue path. Make `splat_ready` (`stages.py:574`) actually present, or delete the dead
  reference.
- **Enable native training-quality flags** (T7): PPISP, bilateral grid, 3DGUT (distorted cameras),
  pose optimisation, and ImprovedGS+ via the LichtFeld binary invocation (`stages.py:748-756`),
  presets-driven. PPISP directly addresses the video exposure-variance failure mode.
- **Add an optional neural-SfM branch** (T5): a `sfm ∈ {colmap, vggt, mast3r}` selector feeding the
  same posed-image contract COLMAP produces, so 3DGS training is unchanged downstream. Feed-forward
  SfM is the fast/low-overlap path; COLMAP remains the accuracy default.
- **Prefer native USD export** (T7): evaluate routing the ADR-011 `v2g:*` schema through the engine's
  native USD export rather than `scripts/assemble_usd_scene.py`; adopt if it carries arbitrary
  customData, otherwise keep the script and record why.

### D-012.5 — VLM artifact analysis + metadata-aware candidate scaffolding (closes T8)

Add a **vision-language model** stage that the containerised agent calls to reason about each
candidate frame *semantically*, alongside the metadata ADR-009 ingests. Components:

- **Local VLM** (e.g. Qwen2.5-VL / Qwen3-VL or InternVL3, served locally or on the ComfyUI host),
  pinned by commit + checkpoint (T6 discipline). The agent prompts it per frame (or per frame
  cluster) to detect **reconstruction-relevant artifacts** the photometric scalars miss: motion
  ghosting, rolling-shutter skew, specular blowout, lens flare, transient occluders (people,
  reflections), and compression blocking. Output is a structured `artifact_report` (typed labels +
  confidence + bbox) written into the ADR-009 per-frame sidecar as a new `vlm` block.
- **Metadata-aware search.** The agent fuses the VLM `artifact_report` with the ADR-009 sidecar
  (`source_video`, `blur`, `exposure`, `phash`, `pose_hint`) **and** newly-ingested **capture/project
  metadata** (camera model, lens, capture session, operator notes, EXIF/SRT GPS where present — the
  "project metadata we don't yet ingest"). This requires a small ingest extension to capture
  per-capture context that the sidecar then carries.
- **Candidate scaffolding.** The fused signal **ranks and scaffolds reconstruction candidates** —
  selecting frames into the COLMAP pool (ADR-007 Fibonacci coverage as the geometric prior, VLM
  artifact score as the semantic veto) and flagging frames/regions for FLUX inpaint recovery
  (ADR-010 FR-11) where a transient occluder or blowout is detected. The agent does not silently
  drop; it annotates, so the per-video ledger and `v2g:*` lineage (ADR-011) record *why* a frame was
  vetoed or repaired.

This is additive: the photometric gate (`frame_quality.py`) remains the cheap first pass; the VLM is
the second, semantic pass invoked by the agent on the surviving candidates.

## Rationale

- **Biggest win, least effort first.** D-012.1 is a one-line default change unlocking the
  already-built MILo path — the highest quality-per-effort item in the audit. D-012.2/3 are
  config + checkpoint changes over existing infrastructure.
- **Stop re-implementing the engine.** Every Python re-implementation of a native v0.5.x feature is
  maintenance we own and the upstream maintains for free; forking past it was an accident of the
  v0.5.2 sync, not a decision.
- **Reproducibility is a correctness property.** Unpinned HEAD clones make the GPU host
  non-deterministic; PRD-v3 NFR-5 cannot hold while the build layer floats.
- **Learned SfM/matching targets our actual failure mode.** ALIKED+LightGlue and PPISP both attack
  textureless/low-overlap/exposure-variant indoor capture — the heritage-room case that motivates
  the whole programme.

## Consequences

### Positive
- Quality jump (MILo mesh, learned matching, Hunyuan3D-2.1, FLUX.2-dev) with mostly config-level
  change; the heavy infrastructure already exists.
- Deterministic, reproducible builds; PRD-v3 NFR-5 holds end-to-end.
- Less bespoke code to maintain as native/plugin paths absorb custom orchestration and USD export.
- A fast SfM option (neural feed-forward) for low-overlap captures COLMAP currently fails.

### Negative
- Larger model/VRAM/disk footprint (Hunyuan3D-2.1, FLUX.2-dev, ALIKED/LightGlue weights, neural
  SfM checkpoints) on the GPU host and the ComfyUI server.
- MILo/learned-matching sidecars add build complexity vs. the always-available in-process TSDF/SIFT.
- Native-flag and plugin adoption couples us to upstream v0.5.x behaviour and its release cadence.

### Risks
- **Gated/licensed weights.** FLUX.2-dev (and the FLUX.1 fallbacks) are gated on Hugging Face (licence
  acceptance + token); Hunyuan3D weights have their own licence. *Mitigation*: model-pull step
  requires a HF token with the licences accepted; capability probe degrades to the declared
  fallback when a checkpoint is absent.
- **MILo as default raises the no-sidecar failure surface.** *Mitigation*: `_select_mesh_backend()`
  already falls back to TSDF when the sidecar is down; keep that fallback and gate on G-T1.
- **Native-flag regressions.** PPISP/3DGUT/bilateral-grid can alter output character. *Mitigation*:
  enable per-preset, A/B against the curated test scene before making them preset defaults.
- **Plugin ecosystem maturity.** SplatReady has no formal release; the COLMAP plugin's version is
  undocumented. *Mitigation*: pin to a specific commit (T6 discipline), or treat the plugin as a
  reference implementation and vendor the ALIKED+LightGlue path directly.
- **VLM latency / hallucinated artifacts (D-012.5).** A per-frame VLM pass is slow on large captures
  and may invent artifacts. *Mitigation*: run the VLM only on photometric survivors (second pass,
  not first); cluster near-duplicate frames by `phash` and analyse one representative; require a
  confidence threshold; the VLM *annotates and ranks*, it never silently drops — the human-auditable
  ledger records every veto/repair reason.
- **Capture/project-metadata ingest scope creep (D-012.5).** *Mitigation*: extend the ADR-009
  sidecar with an additive `capture` + `vlm` block only; no new aggregate (DDD unchanged), reuse the
  existing per-frame writer.

## Alternatives Considered

1. **Status quo (TSDF default, SIFT, Hunyuan3D-2.0, FLUX-Fill, no pins).** Rejected: the audit shows
   we default to the weakest option at every fork while better, already-wired paths sit idle.
2. **Rip out the custom Python pipeline and run entirely on native v0.5.x + plugins.** Rejected for
   now: the custom pipeline encodes the per-video/provenance/USD-annotation logic of ADR-009/010/011
   that no plugin provides; adopt native *components* incrementally, don't replace wholesale.
3. **Upgrade models only, skip native/plugin/pin work.** Rejected: leaves reproducibility (T6) and
   the largest re-implementation cost (T7) unaddressed.
4. **Sync to a newer upstream than v0.5.2.** Deferred to ADR-002 (upstream-sync strategy); this ADR
   adopts capability already present at v0.5.2 plus the external plugin registry, independent of a
   resync.

## Related Decisions

- **ADR-002** — upstream-sync strategy (a future resync may supersede parts of D-012.4).
- **ADR-003** — pluggable mesh-backend *selection policy*; D-012.1 changes only the **default**
  value, not the policy.
- **ADR-006** — `.ksplat` / web-splat delivery (unaffected).
- **ADR-009/010/011** — workflow-shape closure; ADR-010 FR-11 now targets FLUX.2-dev (ADR-013/014), ADR-011's
  `v2g:*` schema is evaluated against native USD export. D-012.5's VLM `artifact_report` and capture
  metadata extend the ADR-009 per-frame sidecar and feed ADR-010's inpaint recovery and ADR-011's
  lineage.
- **PRD-v3** — commissions this ADR as FR-20..FR-28 (§9), gates G-T1..G-T6 (§6).
- **DDD `v3-e2e-extensions.md`** — unchanged: tooling modernisation (incl. the VLM stage) is
  infrastructure over the existing Frame entity / ImageMetadataTag value object; it adds no new
  aggregate, only an additive `vlm` + `capture` block on the per-frame sidecar.
