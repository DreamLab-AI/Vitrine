# Pixal3D vs TRELLIS.2 — head-to-head evaluation plan (PRD v4 R8)

**Date**: 2026-07-10
**Status**: Plan — client scaffold landed (`src/pipeline/pixal3d_client.py`),
weights NOT staged, eval not yet run
**Traces to**: PRD v4 R8 (gated evaluation), ADR-025 amendment 2026-07-09
(licence correction), audit §B.1 + recommendation 6
(`docs/audit/object-pipeline-audit-2026-07-09.md`)

---

## 1. Why

Pixal3D (TencentARC, May 2026, SIGGRAPH 2026) is pixel-aligned generation **on
the TRELLIS.2 backbone**: it back-projects pixel features from the conditioning
image into 3D, targeting near-reconstruction fidelity for the *observed* side
of the object. That is precisely Vitrine's weak axis — ADR-025 conditions the
generator on one clean source-frame crop, and everything the crop shows should
survive into the asset. If Pixal3D delivers its claim, front-face geometry and
texture fidelity improve with zero change to the pipeline architecture
(single-image in, PBR GLB out).

- Repo/weights: `https://huggingface.co/TencentARC/Pixal3D`
- Paper: arXiv 2605.10922

## 2. Licence gate — PASSES

ADR-025 originally rejected Pixal3D as default partly on a "custom non-MIT
licence". The 2026-07-09 amendment **corrected this: Pixal3D is MIT** (GitHub
LICENSE + HF model card, both SPDX MIT). MIT is compatible with Vitrine's
GPL-3.0 and with the non-commercial deployment context — no Tencent-UK-style
territorial carve-out applies (that was the Hunyuan3D community licence, a
different instrument). The remaining rationale for gating was community soak;
Pixal3D has now had ~2 months (May → July 2026). **The gate is quality
evidence only. Re-verify the licence file at the pinned revision when
staging** — model cards have changed licence between revisions before.

## 3. What is compared

Both generators run the identical contract: one matted RGBA crop →
single-image generation → PBR GLB persisted verbatim. Same seeds, same crops,
same machine, serial (never concurrent — both want the full GPU).

### 3.1 Numeric (from `eval/objects/run_eval.py` `mesh_stats`, no GPU to grade)

| Metric | Source | What it tells us |
|---|---|---|
| faces / vertices | mesh_stats | Topology budget parity (gate: within ±30% of TRELLIS.2 unless quality justifies) |
| watertight | mesh_stats | Downstream Nanite/collision sanity |
| bbox_extents | mesh_stats | Gross proportion sanity |
| has_material / material_type / has_uv | mesh_stats | PBR pipe intact (must be PBRMaterial + UVs — hard gate) |
| GLB bytes + sha256 | mesh_stats | Artifact size; hash pins the exact bytes reviewed |
| duration_s | harness | Wall-time cost per object (gate: ≤1.5× TRELLIS.2 at same resolution) |

### 3.2 Front-fidelity score (the axis Pixal3D claims to win)

Reuse the R7 scorer (`pipeline/object_candidate_score.py`): front-silhouette
proportion match vs the observed crop + mesh sanity. This is exactly the
"pixel-aligned" claim made measurable with code we already trust in the
best-of-N ladder. Run it on both generators' outputs per crop and compare
`total` / `proportion` components.

### 3.3 Human review (necessary — the numeric gate is not sufficient)

8-view turntables per object (harness renders them via headless Blender
automatically). Side-by-side review sheet per crop, three judgments:

1. **Observed-side fidelity** — does the front match the crop (geometry
   *and* texture)? This is Pixal3D's claimed win; judge it first.
2. **Inferred-side plausibility** — is the hallucinated back at least as
   plausible as TRELLIS.2's? (Backbone is shared, so the completion prior
   should be similar; a regression here is disqualifying.)
3. **PBR material quality** — roughness/metallic separation, bake artifacts,
   seam visibility at UV islands.

## 4. Crop set

The R9 dreamlab set. Committed baseline today is **3 objects**
(`eval/objects/references.json`, live-run 2026-07-09):

| Crop | TRELLIS.2 reference |
|---|---|
| `0001_metal_container` | 492k faces, 441 s, PBRMaterial, not watertight |
| `0002_bottle` | 483k faces, 158 s, PBRMaterial |
| `0003_wooden_block` | 466k faces, 130 s, PBRMaterial |

PRD v4 R9 targets a 10-crop set; extend to 10 before the eval verdict (the
audit's "5–10 dreamlab crops"). Selection criteria for the additional 7:
cover the failure axes — one heavily occluded object, one reflective, one
thin-structure, one textureless, one organic shape — so the verdict is not
"wins on three easy solids".

## 5. Staging steps (before any run)

1. **Pin**: resolve the current `TencentARC/Pixal3D` HF revision; record the
   commit hash. Verify the LICENSE file at that revision is still MIT.
2. **Stage weights** into the unified tree `data/comfyui/models/` (directive
   §3: pre-staged, never re-downloaded at runtime). Expect TRELLIS.2-class
   VRAM (≥24 GB official; the backbone is shared) — VERIFY-ON-ENV-BUILD.
3. **Register** in `pipeline/sota_registry.py` (licence=MIT, VRAM, pinned
   revision) so `python -m pipeline.sota_registry check` gates the run.
4. **Executor** — in preference order (audit rec. 5: native pipelines for the
   3D half):
   a. Stand up `scripts/pixal3d_native_service.py` mirroring the
      `trellis2_native_service.py` HTTP contract (`/generate` multipart →
      `{glb_high_b64, glb_low_b64?, lineage}`); set `pixal3d.native_url`.
      The client already speaks this contract.
   b. Only if a runtime-verified ComfyUI node pack materialises: author
      `src/pipeline/workflows/pixal3d_single_image_pbr.json` and re-pin
      `Pixal3DClient._build_prompt` to explicit node ids. As of 2026-07-10
      **no verified node pack exists**; the client fails fast on this path
      by design.
5. **Smoke test**: `Pixal3DClient(native_url=...).health_check()` then one
   crop end-to-end; confirm the GLB opens with PBRMaterial + UVs before
   burning the full sweep.

## 6. Running the eval

With the integration diff applied (adds `--generator pixal3d`):

```bash
# Incumbent (regenerate at eval time — same machine, same day, honest timings)
python3 eval/objects/run_eval.py --crops <dreamlab_crops> \
    --out /data/output/eval_pixal3d/trellis2 --generator trellis2 --seed 42

# Challenger
python3 eval/objects/run_eval.py --crops <dreamlab_crops> \
    --out /data/output/eval_pixal3d/pixal3d --generator pixal3d --seed 42
```

Compare the two `metrics.json` side by side plus turntables. Do **not**
`--write-references` from a Pixal3D run — `references.json` stays the
committed TRELLIS.2 baseline until an adoption decision is made. The
reference "regression" report for the pixal3d run is read as a *diff vs
incumbent*, not a failure (face-count deltas beyond ±30% are expected to
show up there; judge them, don't auto-fail them).

Seeds: 42 for the primary sweep; then a 3-seed re-roll (42/43/44) on the two
hardest crops to sample variance — a generator that wins on mean but has
higher variance is worse under the R7 best-of-N ladder economics only if its
per-run cost is also higher.

## 7. Decision criteria

Adopt **Pixal3D as primary** (TRELLIS.2 demoted to fallback slot ahead of
Hunyuan3D) iff ALL of:

1. **Front fidelity wins**: R7 proportion/sanity score ≥ TRELLIS.2 on ≥7/10
   crops AND human review prefers Pixal3D's observed side on a clear
   majority (no crop where it is *badly* worse).
2. **No PBR regression**: PBRMaterial + UVs on 10/10; material quality judged
   at-least-equal in review.
3. **No back-side regression**: inferred surfaces at least as plausible.
4. **Cost sane**: ≤1.5× TRELLIS.2 wall time per object at equivalent
   resolution; fits the 24–48 GB envelope without new infrastructure.
5. **Ops sane**: native service stands up under ADR-021 pinning discipline
   without dependency conflicts against the TRELLIS.2 env (shared backbone
   should make this cheap; if the envs conflict, that cost enters the
   verdict).

**Keep TRELLIS.2 as primary** if any hard gate (2, 4-VRAM) fails or the
fidelity win is not decisive. Intermediate outcome worth naming: adopt
Pixal3D as an **R7 escalation rung** (hero assets only) if it wins on
fidelity but loses on cost — the ladder already selects best-of-N, and a
slow-but-better generator slots naturally as rung (c).

Whatever the verdict: amend ADR-025 in place (dated block, per the
living-catalogue convention), update `sota_registry`, and record the sweep
under `docs/renders/` like the R9 baseline.

## 8. Honest status

- `pixal3d_client.py` is a **contract-validated scaffold**: 13 hermetic tests
  pin drop-in compatibility with `Trellis2Client` (result fields, signature,
  verbatim-GLB, lineage incl. MIT licence + backbone note, fail-fast when no
  executor exists). It has **never generated a real object** — weights are
  not staged, no service or node pack exists in this environment.
- The `resolution`/steps knob names assume the TRELLIS.2 backbone surfaces
  the same ladder — marked VERIFY-ON-ENV-BUILD in the client; the native
  service is the mapping layer if Pixal3D's actual API differs.
- Integration into `config.py` / `stages.py` / `run_eval.py` is prepared as
  an unapplied diff (see the R8 task notes) — nothing in the live pipeline
  changes until the eval is scheduled.
