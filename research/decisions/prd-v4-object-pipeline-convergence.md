# PRD v4: Object Pipeline Convergence — single-image → textured high/low assets

**Date**: 2026-07-09
**Status**: In implementation — Phases 1–3 landed + runtime-verified 2026-07-09

> **Implementation status (2026-07-09):** R1 ✅ (root cause: HWC/CHW bug in our
> Sam3Processor call, one-line fix — silhouettes verified on rawcapdev);
> R2 ✅ (fallbacks deleted, loud per-object failures); R3 ✅ (`object_crops`
> stage, 3/3 crops on rawcapdev); R4 ✅ (multiview subsystems deleted);
> R5 ◐ (single-image client live on the ComfyUI executor; native service
> scaffolded, env stand-up pending); R6 ✅ (verbatim GLB + lineage,
> texture_bake skips generators); R6a ✅ (Hy3D21 graph in code, broken JSONs
> deleted); R7 ✅ (best-of-N seed re-rolls + front-silhouette scorer +
> 2026-07-10 the image-edit "show the back" rung (b) — `image_edit_view.py`,
> Qwen-Image-Edit, self-gated + eval-scored; a winning edit rewrites `surface`
> to `image-edit-inferred`); R8 ◐ (2026-07-10: `pixal3d_client.py` drop-in
> scaffold + `--generator pixal3d` eval + head-to-head plan; MIT gate passes;
> weights not staged so VERIFY-ON-ENV-BUILD, not live-run); R9 ✅
> (harness + 3-object baseline committed, live run: vessel 492k faces/441s,
> bottle 483k/158s, block 466k/130s, 0 regressions); R10 ✅ (2026-07-10:
> position + uniform scale + **upright yaw orientation** all solved and applied
> at assembly — `object_placement.py` + `object_orientation.py` →
> `usd/placements.json` → assembler `AddOrientOp`, live-verified; roll/pitch
> pinned upright per the single-view constraint). Proof assets:
> `docs/renders/object-pipeline-2026-07-09/`.
**Supersedes scope**: the object-generation half of [prd-v3-e2e-closure.md](prd-v3-e2e-closure.md)
(selection + recovery); does **not** reopen ingest/provenance/annotation (v3 scope) or
reconstruction science.
**Drives**: [ADR-025](adr-025-single-image-object-generation.md) (single-image conditioning; supersedes ADR-015/017).
**Traces to**: defects **F1–F12** in the audit
[docs/audit/object-pipeline-audit-2026-07-09.md](../../docs/audit/object-pipeline-audit-2026-07-09.md)
(numbered as in its Part A table).

Written with PACT discipline: Proactive — the conditioning anti-pattern is designed out rather
than parameter-tuned; Autonomous — quality gates run unattended per object; Collaborative —
crop/lineage schema shared between segment, generate, and assemble phases; Targeted — effort
goes to the one broken arc (object generation), not to re-litigating capture, training, or USD.

---

## 1. Executive Summary

The object arc has never produced a validated per-object asset on real data. The audit shows a
chain of compounding defects: SAM3 emits boxes not silhouettes (F1), the isolation stage falls
back to full-scene PLY copies (F1), the generator is conditioned on six near-black premultiplied
CPU splat renders with lying directional labels (F3, F4), padded with a duplicated front image
(F5) and 512px text-prompted FLUX hallucinations (F6), through an *unofficial* multiview node
whose upstream equivalent measurably underperforms single-image conditioning. The proven path —
one clean crop into a modern generator — exists only as uncommitted scripts.

v4 converges the pipeline on that proven path, per ADR-025:

```
frames ─► SAM3 silhouettes ─► best-frame crop + matte ─► TRELLIS.2 native, single image
       ─► to_glb ×2 (high-poly / low-poly, 4K PBR) ─► pose-solve from splat ─► USD
```

The splat contributes **scale and placement only**. ComfyUI shrinks to the 2D stages. Multiview
conditioning survives only for genuine multi-photo captures, routed to models trained for it
(Hunyuan3D-2mv / Omni), never to panel sets.

## 2. Objectives & Success Metrics

| # | Objective | Metric | Target |
|---|-----------|--------|--------|
| O1 | Real object isolation | SAM3 mask IoU vs box baseline on the 10-crop eval set | silhouettes, not boxes; visual sign-off per object |
| O2 | Conditioning fidelity | Generator input = matted best-frame crop ≥1024² | 100% of objects; 0 splat-render conditioning |
| O3 | Asset completeness | Objects yielding high+low GLB with baked 4K PBR | ≥90% of segmented objects per run |
| O4 | Texture integrity | Generator GLB bytes persisted unmodified (hash-verified) | 100%; `texture_bake` touches 0 generator meshes |
| O5 | Quality floor | Turntable review + geometry metric vs dreamlab manual baseline | ≥ parity with manual Hy3D-2.1 results on 10-crop set |
| O6 | Loud failure | Full-scene-copy and world-XY fallbacks removed | 0 silent fallbacks; failed objects reported, not faked |
| O7 | Placement | Per-object pose/scale in scene frame at USD assembly | 100% of successful objects |

## 3. Requirements

### Phase 1 — Unblock isolation (F1)

- **R1** Fix SAM3 silhouette output. Diagnose boxes-not-masks (suspects: `Sam3Processor`
  mask extraction, the #507 bf16 patch). Acceptance: pixel-accurate silhouettes on the eval set.
- **R2** Delete the full-scene-PLY fallback (stages.py:1893-1917) and the world-XY heuristic
  (`_extract_with_mask`, stages.py:2099). An object that cannot be isolated is a reported
  failure with cause, never a fake success. *(F1, F12; O6)*

### Phase 2 — Commit the crop path (F2, F3, F4, F5, F6)

- **R3** New `object_crops` stage between `segment` and `mesh_objects`: best-frame selection
  (silhouette area × sharpness × frontality), padded crop, matte (rembg/BiRefNet-HR),
  square-pad ≥1024², persisted with provenance (frame id, bbox, camera pose) in the manifest.
- **R4** Retire `multiview_renderer.py` + `view_completer.py` from the object arc; delete
  `trellis2_multiview_pbr.json` and `flux2_turnaround.json` submission paths. Renderer may
  remain for previews only (straight-alpha fix required if kept). *(F3–F6)*

### Phase 3 — Native generator service (F7, F9, F11)

- **R5** Thin HTTP service wrapping the TRELLIS.2 **native** pipeline (own env, pinned):
  `run(image)` at configurable resolution (default 1024³, hero 1536³), `to_glb` twice
  (`decimation_target` high/low, 4K PBR bake), returns both GLBs + lineage. Serial VRAM
  lifecycle as per ADR-013. Config surfaces resolution, seeds, decimation targets, steps —
  no dead knobs (`inspect.signature` filtering removed). *(F9)*
- **R6** `mesh_objects` persists returned GLB bytes verbatim (hash recorded); `texture_bake`
  skips generator meshes entirely (splat-projection baking remains only for TSDF/environment
  meshes). Remove dependence on the out-of-tree cv2 monkey-patches by exiting the ComfyUI
  TRELLIS wrapper. *(F7, F11)*
- **R6a** Hunyuan fallback: either align the client workflows to the proven `Hy3D21*` graph
  from `scripts/hy3d_turnaround.py` or delete them; the current non-validating JSONs
  (`ImageOnlyCheckpointLoader` on an empty dir, fictional node classes) must not remain as a
  silent no-op tier. *(F8)*

### Phase 4 — Quality, escalation, evaluation (F10, F12)

- **R7** Escalation ladder per object: (a) seed re-rolls, best-of-N; (b) image-edit second
  view ("show the back") as an alternative single-image attempt; (c) optional SAM 3D Objects
  track for badly occluded items / scene-space pose priors. Lineage flags
  `surface:observed|inferred`. *(replaces ADR-017's role)*
- **R8** Gated evaluation: Pixal3D vs TRELLIS.2 head-to-head on the 10-crop set (licence
  review before any adoption); Hunyuan3D-2mv/Omni reserved for genuine multi-photo captures.
- **R9** Object-quality eval harness: fixed 10-crop dreamlab set, automated turntable renders,
  ≥1 numeric geometry metric, committed reference renders. CI-runnable; regression-gates
  generator/service changes. *(F12; O5)*
- **R10** Pose-solve: place each generated asset in scene frame from splat + crop camera pose
  (position, orientation, scale) at `assemble_usd`. *(O7)*

## 4. Out of scope

Capture protocol changes; splat training quality; environment/full-scene meshing (gsplat-TSDF
path untouched); UE/Unreal export (ADR-016/018/019); rigging (UniRig noted as future);
quad-retopo automation (manual Instant Meshes/Quad Remesher + `app_texturing.py` re-texture is
the documented manual path, not pipeline work).

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAM3 silhouette fix is deep (model-side, not extraction-side) | M | Blocks Phase 1 | Timebox; fallback = SAM3 boxes → SAM2 point-prompted refinement inside the box |
| TRELLIS.2 backside hallucination unacceptable for hero assets | M | M | R7 ladder; SAM 3D occlusion prior; capture-protocol note for future shoots |
| Native service env drift (flash-attn/nvdiffrast/CuMesh pins) | M | M | Dedicated pinned env per ADR-021 vendoring discipline; smoke test in R9 harness |
| Pose-solve accuracy | M | M | Crop camera pose is known from COLMAP; validate against dreamlab ground truth |
| Pixal3D licence blocks adoption | M | L | Evaluation gated on licence review (R8); TRELLIS.2 remains default |

## 6. Sequencing

Phase 1 (R1–R2) is prerequisite to everything and independently valuable. Phase 2 (R3–R4) and
Phase 3 (R5–R6a) can proceed in parallel once R1 lands (R3 can be developed against manual
dreamlab masks). Phase 4 gates release: no v4 sign-off without R9 parity vs the manual
baseline. Highest severity × likelihood first: R1 → R3 → R5 → R6 → R9; R7/R8/R10 follow.
