# QE-02: Implementation Coverage Audit
**Date**: 2026-06-04
**Auditor**: QE automated coverage pass
**Plan reference**: `research/decisions/prd-v3-e2e-closure.md` FR-1..FR-40
**Branch**: feat/v2-upgrade-swarm

---

## Executive Summary

Of 40 functional requirements in the v3 PRD, current status is:

| Status | Count | % |
|--------|-------|---|
| Implemented | 4 | 10% |
| Partial | 10 | 25% |
| Stubbed | 4 | 10% |
| Missing | 22 | 55% |

Technical debt is concentrated in four areas: (1) the per-video ingest loop and provenance chain (FR-1..5) are pooled/session-level only; (2) key-item ranking and the inpaint recovery loop are built but unwired (FR-9, FR-11); (3) the entire USD annotation layer (`v2g:*`) does not exist (FR-14..19); and (4) infrastructure that underlies 12 FRs — `manifest.py` loader, `model_lifecycle.py`, `v2g-net` service mesh, and the `vitrine-setup/` onboarding tool — does not exist at all.

Estimated technical debt to close all gaps: approximately 120–160 engineering hours excluding model-weight acquisition and Docker network provisioning.

---

## Coverage Matrix

| FR | Title (short) | Status | Evidence (file:line) | Gap note |
|----|--------------|--------|----------------------|----------|
| FR-1 | Per-video unit of work | Missing | `drive_ingestor.py:320-354` `_extract_pooled_frames`; `drive_ingestor.py:655-674` loops over sessions, not videos | Ledger at session granularity only; no per-video ledger row; pooled extraction never replaced |
| FR-2 | Delete-on-verified-extraction | Missing | `drive_ingestor.py:582-587` purge fires post-upload, not post-extraction | Purge is at the wrong lifecycle point; no frame-count + quality verification before delete |
| FR-3 | Per-image metadata sidecar | Missing | `frame_quality.py:38-64` FrameQuality defined; never serialised to `<frame>.json`; no `source_video`, `frame_idx`, `timestamp`, `pose_hint` fields | In-memory only; no sidecar written anywhere in the codebase |
| FR-4 | Per-video quality gate | Missing | `quality_gates.py:44` `FrameStats` is a session-level aggregate; no per-video distribution computed | Pooled stats can skew per FR description; no video-scoped threshold path |
| FR-5 | Sidecar carried forward | Missing | No sidecar exists (FR-3 absent), so no lineage root to carry | Blocked on FR-3 |
| FR-6 | Declarative pipeline profile | Partial | `config.py:40,50,60,73,109,151,191` all knobs exist; no single validated profile object or loader surfacing them together; no `exhibit.toml` or profile dataclass wrapping them | Individual config fields present; no profile-level validation, no declarative surface |
| FR-7 | Resumable DAG / state-machine | Partial | `stages.py:43-56` `STAGE_NAMES` list exists; `cli.py:119-214` linear sequential execution; `drive_ingestor.py:656-658` ledger skip-done on session level | Stage names list and session-level ledger exist; no stage-level resume keyed on per-video ledger; no DAG routing layer |
| FR-8 | "Correctly placed" contract | Partial | `coordinate_transform.py:34` `SCENE_SCALE=0.5` exists; comment in `stages.py:1683` notes intra-scene consistency | Contract partially documented in code; no explicit gate or documented contract artefact; survey/georef deferral not formally recorded in code |
| FR-9 | Key-item ranking + threshold | Stubbed | `config.py:109` `min_object_gaussians=100` defined; `stages.py:1410-1419` `if mask_pixels > 0` only — threshold never applied; `sam3_segmentor.py:192-248` returns scores but stages discards them | `min_object_gaussians` is a dead config knob; scores returned by SAM3 are ignored; no composite ranking by size/confidence/concept-priority |
| FR-10 | Depth-aware mask projection | Stubbed | `mask_projector.py:153-214` `assign_labels` projects onto XY plane via majority vote; `mask_projector.py:73-74` depth (`p_cam[:,2]`) used only for front/back culling, not for object separation | No depth-gated multi-object disambiguation; co-located objects merge into one subset |
| FR-11 | Wire FLUX inpainter | Missing | `comfyui_inpainter.py:686` `inpaint()` method exists and is complete; `stages.py` has no call to `ComfyUIInpainter` anywhere in the multiview hull loop (`stages.py:1778-1806`) | Inpainter is built; the call site in the per-key-item orbit render loop does not exist |
| FR-12 | Persist per-object world pose | Missing | `stages.py:1683-1693` computes centroid/scale then normalises and discards; no pose returned in result dict; `usd_assembler.py:169-175` placement machinery waits for `obj.centroid/rotation_quat/scale` on `ObjectDescriptor` (default identity) | `ObjectDescriptor` has pose fields (`usd_assembler.py:55-56`) but they are never populated from real segmentation data |
| FR-13 | Guarantee hull texturing | Partial | `stages.py:1744-1771` bakes only for `fc <= 30_000`; large hulls skip bake and ship vertex-colored GLB (`stages.py:1768-1771`); Hunyuan3D path (`stages.py:1799-1804`) returns no texture field | Decimate-then-bake path absent; native PBR from Hunyuan3D not checked; large hulls remain untextured |
| FR-14 | Textured environment mesh prim | Missing | `usd_assembler.py:147-154` `/World/Environment/Background` holds only a `DomeLight`; `scripts/assemble_usd_scene.py:619-726` places scene mesh under `/World/Objects/full_scene`, not `/World/Environment` | No textured polygonal environment mesh; mis-homed location not corrected |
| FR-15 | `v2g:*` metadata schema on objects | Missing | `usd_assembler.py:220-222` metadata hook exists (`for mk, mv in obj.metadata.items()`); `ObjectDescriptor.metadata` always empty `{}`; `scripts/assemble_usd_scene.py:723-725` writes only `lichtfeld:mesh_path` and `lichtfeld:diffuse_path` | `v2g:semantic_label`, `v2g:quality_score`, `v2g:gaussian_count`, `v2g:recon_method`, `v2g:confidence`, `v2g:capture_timestamp`, `v2g:processing_timestamp`, `v2g:source_video`, `v2g:source_frames`, `v2g:bbox_extent` — all absent |
| FR-16 | `v2g:*` scene-level metadata | Missing | `scripts/assemble_usd_scene.py:774-781` writes `lichtfeld:` namespace keys only; no `v2g:` namespace; no profile-used, stage-versions, or aggregate-quality fields | Scene-level annotation writes `lichtfeld:` not `v2g:`; required fields absent |
| FR-17 | Video→image→object lineage in USD | Missing | No sidecar (FR-3 absent); no `source_video` threading through selection, segmentation, or assembly | Entire lineage chain is absent; blocked on FR-3/FR-5 |
| FR-18 | USD validation metadata richness | Missing | `stages.py:2134-2166` `validate()` checks USD file presence and mesh count only; no `v2g:*` field assertion, no environment mesh check, no untextured hull check | Gate is structural-only; semantic richness check not implemented |
| FR-19 | Lineage-resolution query | Missing | No query utility anywhere in codebase | Blocked on FR-15/FR-17 |
| FR-20 | Flip default mesh to MILo | Partial | `config.py:60` default is `"tsdf"`; `stages.py:895-941` `_select_mesh_backend()` correctly prefers MILo when available under `auto` mode; but `mesh_method` default is `"tsdf"`, not `"auto"` or `"milo"` | PRD requires default `== milo`; actual default is `tsdf`; ADR-003 auto-select only triggers if user sets `auto`; contradicts FR-20 and G-T1 |
| FR-21 | ALIKED+LightGlue COLMAP | Missing | `config.py:40` `matcher: str = "exhaustive"`; `stages.py:624-638` `_run_colmap_direct` passes `self.config.reconstruct.matcher + "_matcher"` to COLMAP but no ALIKED feature extractor is wired | No `aliked` feature extractor path; no LightGlue matcher integration; indoor preset default not changed |
| FR-22 | Hunyuan3D-2.1 upgrade | Partial | `hunyuan3d_client.py:59-100` model specs reference `tencent/Hunyuan3D-2mv` (v2.0 repo); class docstring says "Hunyuan3D 2.0"; no capability probe for 2.1; no 2.1 checkpoint URLs | Still targeting 2.0; 2.1 weights and probe logic absent |
| FR-23 | FLUX.1 Kontext upgrade | Missing | `comfyui_inpainter.py:86-107` references `FLUX.1-Fill-dev` and `FLUX.1-dev`; no Kontext or FLUX.2-dev checkpoint; no capability probe | FLUX.1-Fill is the ceiling; no Kontext/FLUX.2-dev path or fallback logic |
| FR-24 | Neural SfM branch (VGGT/MASt3R) | Missing | `config.py:36` `method: str = "colmap"` only; no `sfm` enum extending beyond colmap | No VGGT or MASt3R integration; not started |
| FR-25 | Pin all models and tools | Missing | `Dockerfile.consolidated:104` COLMAP cloned with no tag (`git clone --depth 1`); `:148` ComfyUI unpinned; `:125` gaussian-toolkit unpinned; `:189` SplatReady unpinned; `:199` SAM3 cloned at HEAD | Multiple unpinned HEAD clones; contradicts NFR-5 and G-T4 |
| FR-26 | Native gsplat v0.5.x plugin adoption | Stubbed | `stages.py:535-564` attempts `SplatReady` plugin from `~/.lichtfeld/plugins/splat_ready`; `stages.py:574` references dead `splat_ready` path; no PPISP/3DGUT/ImprovedGS+ flags | SplatReady path exists but is a runtime-optional plugin, not pinned; training flag extensions absent |
| FR-27 | VLM artifact-analysis stage | Missing | No VLM stage, no `artifact_report` data type, no gemma-4/Qwen2.5-VL integration anywhere | Entire VLM artifact analysis layer absent |
| FR-28 | Metadata-aware candidate scaffolding | Missing | No capture metadata ingestion, no fused score, no VLM veto recording | Blocked on FR-27 and FR-3 |
| FR-29 | Pre-run TOML ingest manifest | Missing | No `manifest.py`, no `exhibit.toml` loader, no `manifest.py` file anywhere in `src/pipeline/` or repo root | `manifest.py` does not exist; confirmed by filesystem search |
| FR-30 | Serial model lifecycle manager | Missing | No `model_lifecycle.py`; no `ModelLifecycleManager`; no `ModelSpec`; no VRAM assertion or soft/hard unload logic | `model_lifecycle.py` does not exist |
| FR-31 | Docker-network model mesh (v2g-net) | Missing | `config.py:135-136` `comfyui_api_url = "http://192.168.2.48:3001"`; `config.py:152-153` `comfyui_url = "http://192.168.2.48:8189"`; `config.py:171` `local_ip = "192.168.2.1"`; `comfyui_inpainter.py:282-284` same literals; `hunyuan3d_client.py:192-193` same literals | Hardcoded IP literals throughout; no service-DNS names; no `v2g-net` Docker network |
| FR-32 | Update + pin .48 ComfyUI | Missing | `comfyui_inpainter.py` has `probe_models` and `download_model` for Salad API; no idempotent ensure-checkpoints logic for Hunyuan3D-2.1 or FLUX.2-dev; no pin records | Probe/download primitives exist; idempotent provisioning script absent |
| FR-33 | Connect ComfyUI over v2g-net | Missing | Blocked on FR-31; same hardcoded IPs noted above | Service-DNS form not present anywhere |
| FR-34 | RecoveryController VLM-in-the-loop | Missing | `comfyui_inpainter.py:686` one-shot `inpaint()` only; no retry loop, no accept/re-prompt/veto, no ledger annotation | No `RecoveryController` class or equivalent; gemma-4 integration absent |
| FR-35 | Schema-driven onboarding wizard | Missing | No `vitrine-setup/` directory; no `schema/exhibit.toml.schema.json`; confirmed absent | Entire onboarding tool absent |
| FR-36 | Hardware-aware model selection | Missing | No `/api/hardware` endpoint; no GPU probe logic; no VRAM-fit recommendation | Absent |
| FR-37 | Secret entry + browser OAuth | Missing | `config.py:139` `hf_token: str = ""` plaintext field; `drive_ingestor.py` uses rclone config path — no Docker secret indirection; no OAuth consent flow | Credentials still in plain config fields; FINDING-006 not remediated |
| FR-38 | Deterministic provisioning + hand-off | Missing | No setup/agent hand-off boundary; no `provision.status`; no `vitrine-setup` binary | Absent |
| FR-39 | Output write-back to Drive | Missing | No `writeback` flag in `DriveIngestConfig`; `drive_ingestor.py` uploads to `remote_path/output_subfolder` but not back to the source Drive folder the video came from | No write-back scope or subfolder routing |
| FR-40 | Project rename to Vitrine | Missing | CLI, package, docs still use `LichtFeld-Studio` / `lichtfeld` naming throughout | Rename not started |

---

## Build-Order Readiness

### Can be done in existing code now (no absent infrastructure required)

These FRs require only modifications to existing Python files and Dockerfile; they depend on no missing module:

| FR | Work required | Estimated effort |
|----|--------------|-----------------|
| FR-1 | Replace `_extract_pooled_frames` loop with per-video loop; add `video_id` column to `Ledger` schema | 4–6 h |
| FR-2 | Move `_purge(local_raw)` to fire after per-video extraction verification; add frame-count + quality check | 2 h |
| FR-3 | After each retained frame write `<frame>.json` from `FrameQuality` fields plus `source_video`/`frame_idx`/`timestamp` | 3–4 h |
| FR-4 | Run `FrameQualityAssessor` per-video before pooling | 2 h |
| FR-9 | Read `mask_pixels` and SAM3 scores; compute composite rank; enforce `min_object_gaussians` threshold | 3–4 h |
| FR-10 | Extend `MaskProjector.assign_labels` with per-Gaussian depth binning before majority vote | 4–6 h |
| FR-11 | Add coverage-check after orbit render; call `ComfyUIInpainter.inpaint()` for low-coverage views before Hunyuan3D call | 4–6 h |
| FR-12 | Return `centroid`/`rotation_quat`/`scale` from `_mesh_one_object`; populate `ObjectDescriptor` fields | 2–3 h |
| FR-13 | Add decimate-then-bake path for hulls exceeding `_TEXTURE_BAKE_FACE_LIMIT`; check Hunyuan3D result for native PBR | 4–6 h |
| FR-14 | Re-home full-scene mesh prim from `/World/Objects/full_scene` to `/World/Environment` in `assemble_usd_scene.py` | 2 h |
| FR-15/16 | Populate `ObjectDescriptor.metadata` with `v2g:*` fields; extend `_write_scene_metadata` in `usd_assembler.py` | 4–6 h |
| FR-18 | Extend `validate()` in `stages.py` to assert `v2g:*` fields, environment mesh, textured hulls | 2–3 h |
| FR-20 | Change `config.py:60` default from `"tsdf"` to `"milo"`; update `validate()` method | 30 min |
| FR-25 | Pin all `git clone` commands in `Dockerfile.consolidated` to specific tags/commits | 2–3 h |

### Blocked on absent infrastructure

| FR(s) | Missing module / system | Blocks |
|-------|------------------------|--------|
| FR-29, FR-6 (profile) | `manifest.py` / `exhibit.toml` loader | Profile surface; FR-29 feeds FR-30, FR-35..38 |
| FR-30 | `model_lifecycle.py` / `ModelLifecycleManager` | VRAM-bounded serial execution across all GPU stages |
| FR-31, FR-33 | `v2g-net` Docker network; service-DNS endpoint config | Endpoint hardcoding in `config.py:135-136,152-153,171` and `comfyui_inpainter.py:282-284`; blocks FR-33, FR-34 |
| FR-32, FR-34 | v2g-net + `RecoveryController` | ComfyUI agent-control loop; VLM-in-the-loop inpaint |
| FR-35..38 | `vitrine-setup/` Rust/Axum binary + `schema/exhibit.toml.schema.json` | Entire onboarding surface |
| FR-27, FR-28 | gemma-4-26B-A4B agent container (`agent-vlm`) | VLM artifact analysis and metadata-aware scaffolding |
| FR-17, FR-19 | FR-3 sidecar (per-image metadata must exist first) | Lineage threading; lineage-resolution query |
| FR-5 | FR-3 | Sidecar carry-forward |
| FR-24 | VGGT/MASt3R container and model weights | Neural SfM branch |

### Infrastructure that blocks the most FRs

1. **`manifest.py` + `exhibit.toml` loader** — blocks FR-29, FR-6 (profile surface), FR-30 (lifecycle), FR-35..39 (onboarding). Removing the hardcoded IPs (FR-31) should be done in parallel.
2. **`v2g-net` Docker network** — blocks FR-31, FR-33, FR-34 (9 sub-requirements across ADR-013/014).
3. **Per-image sidecar (FR-3)** — blocks FR-5, FR-17, FR-19 (lineage chain). This is achievable in existing code now and is the highest-leverage unlock.

---

## Code Contradictions with the Plan

| Location | Contradiction |
|----------|--------------|
| `config.py:60` `mesh_method = "tsdf"` | FR-20 requires default `"milo"`; current default means G-T1 fails on every default run |
| `config.py:109` `min_object_gaussians = 100` | FR-9 requires enforcement; `stages.py:1411` only checks `mask_pixels > 0` — the config knob is dead |
| `comfyui_inpainter.py:282-284`, `config.py:135-136,152-153,171`, `hunyuan3d_client.py:192-193` | FR-31/FR-33 require service-DNS names; hardcoded `192.168.2.48` literals contradict this |
| `Dockerfile.consolidated:104,125,148,161,162,189,199` | FR-25 requires pinned tags; multiple bare `git clone` with no tag/commit |
| `stages.py:1744` `_TEXTURE_BAKE_FACE_LIMIT = 30_000` | FR-13 requires all hulls textured; large hulls above this limit ship untextured — violates G11 |
| `usd_assembler.py:276-281` | Scene metadata uses `lichtfeld:` namespace; FR-15/FR-16 require `v2g:` namespace |
| `drive_ingestor.py:320` `_extract_pooled_frames` | FR-1 explicitly replaces this function; it remains the sole extraction path |
