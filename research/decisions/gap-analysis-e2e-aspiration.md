# Gap Analysis — Aspirational E2E Workflow vs. Current Implementation

**Date**: 2026-06-04
**Method**: 5-agent mesh-swarm recon (ingestion · reconstruction core · segmentation/per-object 3D · mesh/USD/metadata · design-doc catalogue), `swarm_1780586490487_wwk5p6h`.
**Aspiration**: [`../pipelines/aspirational-e2e-flowchart.md`](../pipelines/aspirational-e2e-flowchart.md)
**Companion (older, different scope)**: [`gap-analysis-output.md`](gap-analysis-output.md) compares *local vs Docker output quality* (2026-03-30). This document is orthogonal: it compares the *aspirational E2E shape* against *what the code does today*.

All citations are `file:line` under `src/pipeline/` unless noted.

---

## Executive summary

The codebase is **structurally mature** — every heavy stage exists and works (COLMAP, 3DGS,
SAM3, Hunyuan3D hull recon, USD assembly with real `{Gaussian|Mesh}` variant sets, materials).
The gaps are **not** missing reconstruction capability; they are **four shape/wiring gaps**
between the aspiration and the implementation:

1. **Granularity** — ingestion is **per-session-folder pooled**, not **per-video looped**.
2. **Provenance** — there is **no per-image metadata** at all; only a session-level manifest.
3. **Selection + recovery** — SAM3 detects but **never ranks "key items"**, and the **local
   FLUX inpainter is built but unwired** from per-object 360 recovery; object **pose is not
   persisted** for placement.
4. **Annotation** — USD nodes carry only file paths and counts; the **"richly populated
   metadata"** aspiration is unrealised, despite the population hook already existing.

Branching model-choice (matcher / strategy / preset / mesh backend / hull backend) is **real
and config-driven** — already covered by ADR-003; **do not redesign it**, only surface it.

---

## 1 · Phase 1 — Per-video ingest loop

| Aspiration step | State | Evidence | Gap |
|---|---|---|---|
| Ingest **one video at a time** | 🟥 Missing | `drive_ingestor.py:655-674` loops over **session folders**; `_extract_pooled_frames` pools **all** videos with `set00_/set01_` prefixes | Loop is per-folder, not per-video. No single-video unit of work. |
| Extract frames from that video | 🟩 Implemented | `stages.py` ingest stage (ffmpeg/PyAV); `_extract_pooled_frames` | Works, but on the pooled set. |
| Quality-check images (blur/exposure) | 🟨 Partial | `frame_quality.py:79-92` blur/exposure; `quality_gates.py:53-123` aggregate `FrameStats` | Runs on the **pooled aggregate distribution**, not per-video; scores not retained. |
| **Delete the local video** after extraction | 🟨 Partial | `drive_ingestor.py:583-587` `_purge(local_raw)` runs **after** reconstruct+train+mesh | Deletion is per-session, **after the whole pipeline** — not per-video after extraction. Raw held far longer than needed. |
| **Metadata-tag the images** of that video | 🟥 Missing | Only session `manifest.json` `drive_ingestor.py:434-441`; `FrameQuality` (`frame_quality.py:38-64`) computed **in-memory, never persisted per frame** | No EXIF, no sidecar JSON, no per-frame DB rows. No image→source-video mapping, no per-frame quality/pose/timestamp persisted. |
| Loop until remote dir exhausted | 🟨 Partial | `drive_ingestor.py:633-657` batch over sessions, skip `done` | Resumable, but at **session** granularity, not per-video. |
| Raw stays on Drive as source of truth | 🟩 Implemented | `drive_ingestor.py:19-20`, `:567` outputs → `<remote>/outputs/` | Correct by design (plan §8 decision 5). |

**Deltas**: (D1) introduce a per-**video** unit of work with its own ledger row; (D2) move
deletion to immediately after a **verified** extraction of that video; (D3) author a
**per-image metadata sidecar** schema and write it for every retained frame.

---

## 2 · Phase 2 — Reconstruction core & branching

| Aspiration | State | Evidence | Gap |
|---|---|---|---|
| Pool tagged frames + Fibonacci select | 🟩 Implemented | ADR-007; `fibonacci_sampler.py`, `frame_selector.py`; `stages.py` select_frames | Selection ignores per-image tags (none exist yet). |
| **Branching matcher** | 🟩 Implemented | `config.py:40` `matcher ∈ {exhaustive*, sequential, vocab_tree}` | Real branch. |
| COLMAP SfM + registration gate | 🟨 Partial | `stages.py` reconstruct (~`:520`), 30% registration check | Gate exists; no auto re-capture flag loop back to source. |
| **Branching training** (strategy × preset) | 🟩 Implemented | `config.py:50` `strategy ∈ {mrnf*, mcmc}`, `:73` `scene_preset ∈ {default*, indoor_reflective}` | Real branch. |
| 3DGS train → field, optional `.ksplat` | 🟩 Implemented | `gsplat_trainer.py`; `splat_optimizer.py`; `config.py:190-191` (ADR-006) | — |
| **Orchestration** | 🟨 Partial | `stages.py:43-56` `STAGE_NAMES`; **linear, caller-driven** `cli.py:119-214`; **no DAG, no stage-level resume** | "Branching choices of models *through the system*" implies a routed graph; today it's a fixed linear sequence with config knobs and deep mesh fallbacks. |
| Real-world placement / surveyed pose | 🟥 Missing | `coordinate_transform.py:34` `SCENE_SCALE=0.5` fixed; COLMAP relative only; no georef | Output is scene-relative, never world/surveyed. "Correctly placed" is currently relative-correct only. |

**Deltas**: (D4) surface branch choices as an explicit, declarative pipeline profile (not buried
defaults); (D5) optional state-machine/DAG layer for resumable routing; (D6) decide whether
"correctly placed" requires survey registration or is satisfied by intra-scene consistency.

---

## 3 · Phase 3/4 — SAM3 key-items, FLUX hulls, placement

| Aspiration | State | Evidence | Gap |
|---|---|---|---|
| SAM3 identifies items | 🟨 Partial | `sam3_segmentor.py:192-248` concept/text-prompted; concepts `config.py:113-116` | Detects by fixed concept list; not exemplar-driven. |
| **Select/rank KEY items** | 🟥 Missing | `stages.py:1410-1419` keeps all (mask_pixels>0); `min_object_gaussians` defined `config.py:109` **never enforced** | No keyness ranking by size / gaussian-count / confidence / concept priority; noise kept. |
| Mask → 3D per-object subset | 🟨 Partial | `mask_projector.py:153-214` majority-vote; `stages.py:1539-1595` | XY-plane vote is **depth-lossy**; co-located objects can merge. |
| **LOCAL FLUX** inpaint unseen views | 🟥 Missing (unwired) | `comfyui_inpainter.py:4-14,86-107` FLUX.1-Fill, local ComfyUI :3001 — built for **background** inpaint, **not called** in per-object flow | Occluded object faces are not recovered before hull recon → risk of hallucinated geometry. |
| Hunyuan3D hull recon wired in | 🟩 Implemented | `stages.py:1778-1806` (Strategy 1); `multiview_renderer.py:148-240` 4-view orbit → GLB; `hunyuan3d_client.py:528-685` | Solid. |
| Textured watertight hull | 🟨 Partial | Hunyuan GLB vertex-colour; bake only `<30k` faces `stages.py:1741-1771` | No PBR/UV QC; large hulls untextured. |
| **Correct placement / preserved pose** | 🟨 Partial | Placement machinery exists `usd_assembler.py:169-175` (translate/orient/scale from `obj.centroid/rotation_quat/scale`), but **upstream does not persist** real pose into `ObjectDescriptor` (`stages.py:1683-1693` normalises then discards) | Hulls risk landing at origin/identity; the *capability* is there, the *data plumbing* is not. |

**Deltas**: (D7) implement key-item ranking + threshold; (D8) wire FLUX inpaint into the
per-object multiview loop before Hunyuan3D; (D9) persist per-object world pose
(centroid+rotation+scale) from segmentation through to `ObjectDescriptor`; (D10) depth-aware
mask projection; (D11) guaranteed hull texturing (decimate-then-bake or native PBR).

---

## 4 · Phase 5/6 — Environment mesh & richly-annotated USD

| Aspiration | State | Evidence | Gap |
|---|---|---|---|
| Prim hierarchy | 🟩 Implemented | `usd_assembler.py:144-151` `/World/{Environment/Background, Objects, Cameras, Materials}` | — |
| **`{Gaussian|Mesh}` variant sets** | 🟩 Implemented | `usd_assembler.py:182-203` real `representation` variantSet | Poster claim is **true in code**. |
| **Polygonal textured ENVIRONMENT mesh** | 🟥 Missing | `/World/Environment/Background` holds **only a DomeLight** `usd_assembler.py:147-154`; full-scene mesh dumped under `/World/Objects/full_scene` (`assemble_usd_scene.py:619-726`) | No dedicated textured environment surface; scene mesh mis-homed. |
| Hull placement transform | 🟨 Partial | `usd_assembler.py:169-175,313-318` translate→orient→scale | Machinery present; depends on D9 pose plumbing. |
| **Richly populated node metadata** | 🟥 Missing | Object prim: only `lichtfeld:mesh_path`, `:diffuse_path` (`assemble_usd_scene.py:723-725`); mesh prim: `vertex_count/face_count` (`:536-538`); `ObjectDescriptor.metadata` hook (`usd_assembler.py:221-222`) **unused** | Missing semantic label, quality score, bbox/extent, gaussian_count, recon method, confidence, capture/processing timestamps, **source-video lineage**. |
| Materials/textures bound | 🟩 Implemented | `material_assigner.py:42-94,119-134` UsdPreviewSurface + diffuse + `st` primvar | — |
| Camera prims | 🟩 Implemented | `assemble_usd_scene.py` `colmap:*` customData | Rich intrinsics; no object-visibility maps. |
| Validate + deliver | 🟩 Implemented | `stages.py` validate (`:2134`) USD present / mesh>0 | Validates existence, not metadata richness. |

**Deltas**: (D12) author a USD **metadata schema** (namespaced `v2g:*` attrs) and populate it on
every node from data already computed upstream; (D13) create a real textured environment mesh
prim under `/World/Environment`; (D14) carry per-image/video lineage into object + scene
metadata so the USD is self-describing.

---

## 5 · Consolidated delta register

| # | Delta | Phase | Severity | Closes via |
|---|---|---|---|---|
| D1 | Per-video unit of work + ledger row | 1 | High | ADR-009 |
| D2 | Delete local video right after verified extraction | 1 | High | ADR-009 |
| D3 | Per-image metadata sidecar schema + writer | 1 | High | ADR-009 |
| D4 | Declarative pipeline/branch profile | 2 | Medium | PRD-v3 §FR |
| D5 | Optional resumable DAG/state machine | 2 | Medium | PRD-v3 §NFR |
| D6 | Define "correctly placed" (survey vs intra-scene) | 2/6 | Medium | ADR-010 |
| D7 | Key-item ranking + threshold | 3 | High | ADR-010 |
| D8 | Wire local FLUX inpaint into per-object recovery | 4 | High | ADR-010 |
| D9 | Persist per-object world pose to ObjectDescriptor | 4/6 | High | ADR-010 |
| D10 | Depth-aware mask projection | 3 | Medium | ADR-010 |
| D11 | Guaranteed hull texturing | 4 | Medium | PRD-v3 §FR |
| D12 | USD `v2g:*` metadata schema + population | 6 | High | ADR-011 |
| D13 | Textured environment mesh prim | 5/6 | Medium | ADR-011 |
| D14 | Video→image→object lineage into USD | 1/6 | High | ADR-011 |

**Out of scope to redesign**: branching backend *selection policy* (ADR-003), upstream sync
(ADR-002), web-splat delivery (ADR-006), Fibonacci selection (ADR-007).

---

## 6 · SOTA tooling gaps (T-series)

Distinct from D1–D14 (which are *workflow-shape* gaps): the T-series are *tooling* gaps where the
stack defaults to the weakest available option or has forked past native v0.5.x capability. Source:
online + in-repo audit 2026-06-04 (upstream `MrNeRF/LichtFeld-Studio`, plugin registry
`lichtfeld.io/plugins`). These are **infrastructure**, not domain — no new DDD aggregates. All
closed by **ADR-012**.

| # | Tooling gap | Evidence | Today | SOTA | Severity |
|---|---|---|---|---|---|
| T1 | Default mesh backend is the weakest baseline | `config.py:60` `mesh_method="tsdf"`; MILo/CoMe wired at `stages.py:701-718` | TSDF (gsplat-depth + marching cubes) | MILo (SIGGRAPH Asia 2025), CoMe | High |
| T2 | COLMAP uses plain SIFT, not learned features/matching | `stages.py:639` `matcher ∈ {exhaustive,sequential,vocab_tree}` (all SIFT) | SIFT + brute/sequential | COLMAP 4.1 ALIKED + LightGlue (per 360/COLMAP plugins) | High |
| T3 | Image-to-3D hull pinned at Hunyuan3D 2.0 | `hunyuan3d_client.py:61,77,93` request `dit-v2-0`/`-v2-mv` | Hunyuan3D 2.0 | Hunyuan3D 2.1 (texture fidelity, paint-only) | Medium |
| T4 | Inpainting pinned at FLUX.1-Fill-dev | `comfyui_inpainter.py:88` `flux1-fill-dev.safetensors` | FLUX.1-Fill-dev | FLUX.1 Kontext (context-aware edit) | Medium |
| T5 | No neural feed-forward SfM option | only COLMAP; untagged HEAD `Dockerfile.consolidated:104`; DUSt3R named but unbuilt `proposed-pipeline.md:200` | COLMAP only | VGGT / MASt3R-SfM / DUSt3R branch | Medium |
| T6 | Zero version pins → non-reproducible builds | gsplat `Dockerfile.consolidated:176`, SAM3 `:199`, PyTorch `cu128` no ver, COLMAP/ComfyUI HEAD | unpinned HEAD/latest | pinned tag/commit per tool (NFR-5) | High |
| T7 | Native v0.5.x capability re-implemented in Python | native USD export (v0.5.1) vs `scripts/assemble_usd_scene.py`; MCP+plugins (v0.5.0) vs subprocess `stages.py`; PPISP/bilateral/3DGUT/ImprovedGS+ unused; `splat_ready` referenced `stages.py:574` but uninstalled | bespoke Python | native engine + plugin ecosystem | High |
| T8 | No VLM artifact analysis; metadata-blind candidate selection | photometric scalars only `frame_quality.py:79-92`; no semantic artifact detection; capture/project metadata not ingested | blur/exposure/sharpness | VLM (Qwen2.5-VL/InternVL3) artifact report + metadata-fused candidate scaffolding | High |

**Closes via**: T1–T8 → **ADR-012** (D-012.1..D-012.5) → PRD-v3 **FR-20..FR-28**, gates
**G-T1..G-T6**.

---

## 7 · What is already good (do not rebuild)

- COLMAP→3DGS→`.ksplat` core, Fibonacci selection, indoor presets.
- SAM3 concept segmentation + 2D→3D projection.
- Hunyuan3D multiview hull reconstruction, wired as Strategy 1 with a deep fallback chain.
- USD hierarchy, **real** `{Gaussian|Mesh}` variant sets, UsdPreviewSurface materials, camera prims.
- Per-session resumable Drive ingest (rclone, service-account creds, push-back-to-Drive).

The work is **wiring, granularity, and annotation** — not new reconstruction science.
