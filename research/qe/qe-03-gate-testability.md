# QE-03: Gate Testability Audit

**Date**: 2026-06-04
**Auditor**: QE Agent (read-only)
**Scope**: PRD v3 §6 quality gates G1..G17, G-T1..G-T6, G-O1..G-O4 (27 gates total)
**Sources inspected**:
- `research/decisions/prd-v3-e2e-closure.md` (§2 O1..O20, §6 gates, §7 Risk Register)
- `research/decisions/adr-013-ingest-manifest-serial-model-lifecycle.md`
- `research/decisions/adr-014-agent-controlled-comfyui-integration.md`
- `research/decisions/adr-015-vitrine-web-onboarding.md`
- `src/pipeline/quality_gates.py`
- `src/pipeline/preflight.py`
- `scripts/test_orchestrator.py`
- `scripts/test_usd_pipeline.py`
- All files under `tests/python/`

---

## 1. Executive Verdict

**The gate framework is aspirational, not executable.** The PRD §6 defines 27 quality gates, of which 20 are blocking. None of the 27 gates maps 1:1 to an automated test that could run today in CI. The four existing automation surfaces — `quality_gates.py`, `preflight.py`, `scripts/test_orchestrator.py`, and `tests/python/` — address a completely orthogonal concern: PSNR/SSIM of training output, mesh vertex counts, and the upstream LichtFeld Studio C++ plugin system. Not one line of test code covers the workflow-shape gaps (per-video granularity, sidecar persistence, key-item ranking, lineage, USD metadata richness, Docker secret containment, or the onboarding schema/VRAM/secret gates).

Coverage of PRD §6 gates by any existing automation: **0 / 27 (0%)**.

Blocking gates with zero verification code: **20 / 20 (100%)**.

---

## 2. Per-Gate Classification Table

Verifiability classes:
- **A** — Automatable-now: the verification logic is a pure unit/integration assertion requiring no external fixture, live model, or live infrastructure.
- **F** — Needs-fixture: automatable, but requires a curated test-scene asset (ground-truth labels, a specific video/USD file) that does not yet exist in the repo.
- **I** — Needs-infra: automatable only when specific runtime infrastructure is running (Docker network, .48 GPU host, live ComfyUI, rclone remote, live GPU with VRAM budget).
- **U** — Underspecified: the "how verified" description is vague enough that the test cannot be written from the PRD text alone.

| Gate | Closes | Blocking | Class | Has code today | Notes |
|------|--------|----------|-------|----------------|-------|
| **G1** Per-video unit | D1 | Yes | A/I | No | "Count ledger rows vs `rclone lsjson` video count" — the ledger schema is not yet built (`manifest.py` does not exist; `drive_ingestor.py:320` still uses `_extract_pooled_frames`, the exact anti-pattern the PRD closes). The count assertion is simple once the ledger row exists, but rclone contact makes it I-class for a CI run. Splittable: unit part (row exists per video) = A; remote count part = I. |
| **G2** Retention ceiling | D2 | Yes | I | No | Requires polling NVMe during a live run. Not mockable without a filesystem snapshot harness. No file-count-during-run helper exists. |
| **G3** Video deleted | D2 | Yes | A | No | Post-extraction scratch scan is a pure filesystem assertion. Automatable now as a unit test once `_extract_pooled_frames` is replaced by a per-video loop (FR-1/FR-2). Currently, the deletion logic does not exist at the required granularity (`drive_ingestor.py:583-587` purge fires after reconstruct+train+mesh, not after extraction). |
| **G4** Sidecar coverage | D3 | Yes | A | No | `count(sidecars) == count(retained_frames)` is a pure filesystem assertion. No `<frame>.json` sidecar writer exists anywhere in the pipeline — `frame_quality.py` computes `FrameQuality` in-memory and never persists it (confirmed by inspection: no `json.dump` / `write_text` call in `frame_quality.py`). |
| **G5** Sidecar completeness | D3 | Yes | A | No | Schema-validate each `<frame>.json` against required fields (`source_video`, `session`, `frame_idx`, `timestamp`, `blur`, `exposure`, `sharpness`, `phash`, `pose_hint`). Pure JSON-schema assertion once sidecars exist. The schema is not yet written. |
| **G6** Key-item precision | D7 | Yes | F | No | Requires "analyst ground truth on the curated test scene." No curated scene asset exists in the repo. `min_object_gaussians` is defined in `config.py:109` but the PRD notes it is "never enforced" (`stages.py:1410-1419` keeps everything with `mask_pixels>0`). The ranking logic (FR-9) is unbuilt. Even with a fixture, ≥90% precision requires a ground-truth labels file matched against ranked output. |
| **G7** Noise rejection | D7 | Yes | A | No | "Assert no proceeding item below threshold." This is a pure unit assertion against the ranked list once FR-9 is built. The threshold enforcement path does not exist today. |
| **G8** Occlusion recovery | D8 | Yes | A/I | No | "Trace inpaint call per item below coverage threshold." Verifiable via log/ledger inspection once the recovery controller (ADR-014 D-014.3) and the per-object coverage computation (FR-11) exist. The `comfyui_inpainter.py` client exists but is unwired from the per-object loop. Pure assertion over call-trace records is A-class; actual inpaint execution is I-class (requires .48 ComfyUI). |
| **G9** Depth separation | D10 | No | F | No | "Two-object overlap fixture; assert 2 subsets." The fixture (a scene with two co-located objects at different depths) does not exist. The depth-aware projection (FR-10) is not built — `mask_projector.py:153-214` uses XY-plane majority vote today. |
| **G10** Pose preserved | D9 | Yes | A | No | "Assert prim xform ≠ identity for posed objects." USD traversal assertion is pure code once per-object world pose is plumbed through `ObjectDescriptor` (FR-12). Today `stages.py:1683-1693` normalises and discards pose; `usd_assembler.py:169-175` waits for data that never arrives. |
| **G11** Hull texturing | D11 | Yes | A | No | "Per-hull material/diffuse-path presence check." USD traversal for material binding is straightforward once FR-13 (decimate-then-bake) is implemented. Currently hulls >30k faces ship untextured. The check itself is automatable. |
| **G12** Environment mesh | D13 | Yes | A | No | "USD traversal asserts mesh + material, not just DomeLight." Pure USD traversal. Today `/World/Environment/Background` holds only a `DomeLight` (`usd_assembler.py:147-154`); FR-14 is not implemented. The assertion code would be 10 lines. |
| **G13** USD metadata richness | D12 | Yes | A | No | FR-18 validator is specified but not built. `usd_assembler.py:221-222` `ObjectDescriptor.metadata` hook exists and iterates `obj.metadata.items()`, but no `v2g:*` fields are ever populated — `config.py` has no `v2g` schema, and `assemble_usd_scene.py` only writes `lichtfeld:mesh_path`/`:diffuse_path` and `vertex_count`/`face_count`. The validator itself would be an automatable unit test once FR-15 populates the fields. |
| **G14** Lineage closure | D14 | Yes | A | No | "FR-19 query returns non-empty video + frames per object." Pure USD attribute read once `v2g:source_video` and `v2g:source_frames` are populated (FR-17). Neither the lineage threading nor the query function (FR-19) exists today. |
| **G15** Profile reproducibility | D4 | No | A | No | "Re-run; diff metadata schema + stage log." Automatable as a determinism test: run twice, diff outputs. Requires the declarative profile (FR-6) to exist. The config system has the knobs but no single validated profile object. |
| **G16** Resume correctness | D5 | No | I | No | "Inject kill; assert no completed video re-processed." Requires a live run that can be killed mid-stage. Automatable as an integration test using `subprocess` and SIGTERM injection, but needs the per-video DAG (FR-7) to exist. |
| **G17** Secret handling | NFR-4 | Yes | A/I | No | "`docker inspect` + image-layer scan find no plaintext cred." The scan part (`docker inspect` / `dive`-style layer scan) requires Docker access (I-class). The manifest secret-indirection check (assert `env:` prefix only, no inline token) is A-class and testable against the manifest loader once `manifest.py` exists. Currently `InpaintConfig.hf_token = ""` in `config.py:138` is a plain string field — the anti-pattern the PRD closes. |
| **G-T1** Mesh default | T1 | Yes | A | No | "Assert config default == milo; fallback path covered." Pure config assertion (`config.py:60`). Currently `config.py:60` uses `tsdf` as default (the PRD says flip to `milo`). The fallback path test requires the `_select_mesh_backend()` logic change. Once FR-20 is done, this is a 5-line unit test. |
| **G-T2** Learned matching | T2 | No | A | No | "Inspect COLMAP feature/matcher in run log." Automatable as a unit test asserting that indoor preset config maps to `aliked_lightglue`. Requires FR-21 config addition. |
| **G-T3** Model versions | T3, T4 | Yes | I | No | "Assert requested model IDs; fallback emits explicit log." The model ID assertion is A-class (inspect `hunyuan3d_client.py` and `comfyui_inpainter.py` config). The fallback path assertion requires a live capability probe against ComfyUI. Mixed A/I. |
| **G-T4** Pins | T6 | Yes | A | No | "Scan Dockerfiles for unpinned clones / bare pip." This is a static analysis test — grep Dockerfiles for `git clone` without `--branch`/`--depth` + pinned tag, and for bare `pip install` without pinned versions. Fully automatable today as a pre-commit or CI check. No implementation exists. |
| **G-T5** VLM artifact precision | T8 | Yes | F | No | "Compare `artifact_report` vetoes vs ground truth on test scene." Requires the same curated test scene as G6 plus ground-truth artifact annotations (which frames are genuine artifacts). The VLM artifact stage (FR-27) does not exist. |
| **G-T6** Fused scaffolding | T8 | Yes | A | No | "Assert every pooled frame has a recorded score + every veto a reason." Pure ledger/sidecar assertion once FR-28 (metadata-aware scaffolding) is built. The `vlm` sidecar block in the per-frame JSON (ADR-013 D-013.1) is not yet written. |
| **G-O1** Schema round-trip | O17 | Yes | A | No | "`toml_edit` diff is comment/order-stable and values equal." Pure unit test against `vitrine-setup` Rust binary — save, reload, compare. `vitrine-setup` does not exist yet (ADR-015 D-015.2). Once built, this is a standard round-trip test. |
| **G-O2** VRAM fit | O18 | Yes | A/I | No | "Assert each recommended `serial_peak_estimate_gb` ≤ `/api/hardware` GPU VRAM." The assertion logic is A-class; `/api/hardware` endpoint requires the `vitrine-setup` binary (I-class if probing real GPU, mockable for unit test). `vitrine-setup` does not exist. |
| **G-O3** Secret containment | O19 | Yes | A | No | "Scan served JS + manifest + tree for plaintext token; assert only `env:` indirection." Static file scan — fully automatable as a git pre-commit hook or CI scan once `vitrine-setup` and the manifest loader exist. |
| **G-O4** Write-back | O20 | No | I | No | "Post-run `rclone lsjson` of the source/`vitrine-output` folder finds the artifact set." Requires live Google Drive access and a completed run. Not CI-automatable without a Drive test account. |

---

## 3. Summary by Verifiability Class

| Class | Count | Gate IDs |
|-------|-------|----------|
| Automatable-now (A, or A with minor split) | 13 | G3, G4, G5, G7, G10, G11, G12, G13, G14, G15, G-T1, G-T2, G-T4, G-T6, G-O1, G-O3 (16 total, several overlap A/I) |
| Needs-fixture (F) | 3 | G6, G9, G-T5 |
| Needs-infra (I, or A/I) | 5 | G1 (partial), G2, G8 (partial), G16, G17 (partial), G-T3 (partial), G-O2 (partial), G-O4 |
| Underspecified (U) | 1 | G2 (boundary: "NVMe scratch poll" has no defined harness or polling interval) |

Pure-A count (no infrastructure dependency in the assertion path): **13 out of 27** gates can be fully automated as unit/integration tests once the corresponding source code exists.

**None** are automatable today because the source code the assertions verify does not exist.

---

## 4. Coverage Gaps

### 4.1 Code that exists but does not map to any PRD §6 gate

`quality_gates.py` implements:
- `assess_input_quality` (blur, exposure, frame count) — partially maps to G4/G5 intent (frame quality) but does not verify sidecar existence or schema completeness.
- `assess_training_quality` (PSNR, SSIM, convergence) — no corresponding PRD §6 gate. These are pre-v3 metrics.
- `assess_mesh_quality` (vertex count, watertightness) — no corresponding PRD §6 gate.
- `assess_roundtrip_quality` (round-trip PSNR) — no corresponding PRD §6 gate.
- `assess_final_quality` (final PSNR, object count, materials) — the `has_materials` flag is a weak proxy for G11 but does not assert per-hull diffuse-path presence or the `/World/Environment` mesh.

`preflight.py` checks dependency availability (torch, gsplat, COLMAP) — no PRD §6 gate.

`scripts/test_orchestrator.py` tests the above `quality_gates.py` functions with synthetic data, and tests config load/save and MCP client payload structure. No PRD §6 gate covered.

`tests/python/` (79 files) tests the LichtFeld Studio C++ plugin system (operators, panels, hooks, rendering, tensor ops, scene validity). These are upstream application tests with zero coverage of the pipeline's own acceptance criteria.

`scripts/test_usd_pipeline.py` exercises USD assembly with two synthetic objects — checks up-axis and meters-per-unit. Does not assert `v2g:*` fields, environment mesh presence, lineage attributes, or any PRD §6 gate.

### 4.2 Blocking gates with zero verification code: 20 / 20

All 20 blocking gates (G1, G2, G3, G4, G5, G6, G7, G8, G10, G11, G12, G13, G14, G17, G-T1, G-T3, G-T4, G-T5, G-T6, G-O1, G-O2, G-O3) have zero corresponding test code.

### 4.3 The curated test scene fixture does not exist

Gates G6, G9, and G-T5 require "analyst ground truth on the curated test scene." The repo contains no fixture directory, no ground-truth label file, and no fixture video. The PRD refers to "the curated test scene" as a known entity but never specifies its content, format, or location. This is a blocking dependency for two of the highest-risk gates (key-item precision and VLM artifact precision, both ≥90% precision thresholds).

---

## 5. Proposed Test Pyramid

### Layer 0 — Static analysis (runnable today, no source changes required)

Implement immediately as a CI step:

- **G-T4 pin scanner**: `grep -rE 'git clone(?!.*--branch)' Dockerfiles/` and `grep -E 'pip install [^=]' Dockerfiles/` to catch unpinned clones and unversioned pip installs. Zero dependencies. Covers one blocking gate today.
- **G17 (partial) secret scan**: `grep -rE '(hf_token|HF_TOKEN|GOOGLE_APPLICATION_CREDENTIALS)\s*=' src/` to detect inline credential assignments vs `env:` indirection. Runnable once `manifest.py` is written.
- **G-O3 secret containment scan**: `grep -rE '[A-Za-z0-9_]{40,}' src/pipeline/manifest.py` (crude token pattern) plus a TOML parse asserting all credential fields match `^env:`.

### Layer 1 — Unit tests (automatable once corresponding source code exists)

Each of the following maps to exactly one or two PRD §6 gates and requires no live infrastructure:

| Test module (proposed) | Gate(s) covered | Prerequisite |
|------------------------|-----------------|--------------|
| `tests/pipeline/test_ledger.py` | G1 (unit half), G3 | FR-1, FR-2: per-video ledger row + deletion |
| `tests/pipeline/test_sidecar.py` | G4, G5 | FR-3: `<frame>.json` writer in `frame_quality.py` |
| `tests/pipeline/test_key_item_ranking.py` | G7 | FR-9: `min_object_gaussians` enforcement |
| `tests/pipeline/test_inpaint_trace.py` | G8 (unit half) | FR-11: `RecoveryController` call-trace logging |
| `tests/pipeline/test_pose_plumbing.py` | G10 | FR-12: pose in `ObjectDescriptor` |
| `tests/pipeline/test_hull_texture.py` | G11 | FR-13: decimate-then-bake path |
| `tests/pipeline/test_usd_env_mesh.py` | G12 | FR-14: `/World/Environment` mesh prim |
| `tests/pipeline/test_usd_metadata.py` | G13 | FR-15/FR-18: `v2g:*` validator |
| `tests/pipeline/test_lineage.py` | G14 | FR-17/FR-19: lineage query |
| `tests/pipeline/test_profile_determinism.py` | G15 | FR-6: declarative profile |
| `tests/pipeline/test_manifest_secrets.py` | G17 (A half), G-O3 | FR-29/ADR-013: `manifest.py` |
| `tests/pipeline/test_config_defaults.py` | G-T1, G-T2 | FR-20, FR-21: config default changes |
| `tests/pipeline/test_dockerfile_pins.py` | G-T4 | Static analysis (no source changes needed) |
| `tests/pipeline/test_fused_score.py` | G-T6 | FR-28: fused score in per-frame sidecar |
| `tests/onboarding/test_schema_roundtrip.py` | G-O1 | ADR-015: `vitrine-setup` binary |
| `tests/onboarding/test_vram_recommendation.py` | G-O2 (mockable) | ADR-015: `/api/hardware` with mocked `nvidia-smi` |

### Layer 2 — Integration tests (requires in-process pipeline execution, no live external services)

- **Per-video loop end-to-end** (G1, G3, G4, G5): feed a synthetic 5-second H.264 video through a mocked `drive_ingestor` run; assert one ledger row, deletion, and sidecar per retained frame.
- **USD gate suite** (G10, G11, G12, G13, G14): build a synthetic USD from `UsdSceneAssembler` with known `ObjectDescriptor` data; run the FR-18 validator; assert all gate conditions pass/fail as expected.
- **Resume test** (G16): run a 3-video batch, inject `SIGTERM` after video 1, restart, assert video 1 is skipped.

### Layer 3 — E2E / fixture-dependent tests (require the curated test scene)

The "curated test scene" fixture must contain:
1. A short (30–60 s), controlled walkthrough video of a room with at least 3 identifiable objects and 2 intentionally co-located objects at different depths (for G9).
2. A **ground-truth object manifest**: for each visible object, a stable label, bounding box in at least one key frame, and whether it is genuine (for G6 precision) or a noise/background region.
3. A **ground-truth artifact annotation file**: for each frame, whether it contains genuine artifacts (motion blur, rolling shutter, specular blowout, etc.) and the artifact type (for G-T5 precision).
4. The video must be stored as a Git LFS asset (or an external fixture URL) so CI can access it without a live Drive connection.

These fixture-dependent tests cover gates G6, G9, G-T5 and serve as the sole precision benchmarks for the ≥90% thresholds specified in the PRD.

### CI integration

The ubuntu.yml workflow (`github/workflows/ubuntu.yml`) currently builds the C++ binary with no Python pipeline test step. Add:

```yaml
- name: Pipeline unit tests
  run: |
    pip install -e src/
    pytest tests/pipeline/ -m "not slow and not integration" -x
```

Layer 2 integration tests should run nightly (`.github/workflows/nightly.yml` already exists) with the fixture video available.

---

## 6. Risk Register Coverage Check

Cross-checking PRD §7 Risk Register against §6 gates:

| Risk | Likelihood | Impact | Mitigating gate(s) | Gap |
|------|-----------|--------|-------------------|-----|
| FLUX hallucinates on occluded faces → bad hulls | Medium | High | G8 (inpaint traced) | Gate exists; RecoveryController not built. G8 verifies call-trace but not output quality — no perceptual quality gate on inpaint output. |
| Depth-lossy mask merge collapses co-located objects | Medium | High | G9 (two-object fixture) | G9 is non-blocking. A medium-high risk has only a non-blocking gate. **Recommend upgrading G9 to blocking.** |
| Per-video deletion races extraction verification → data loss | Low | High | G3 (deletion scan), G4 (sidecar coverage) | Both gates are blocking and cover the risk. G3/G4 have zero code. |
| Key-item ranking drops genuine item (false negative) | Medium | Medium | G6 (precision only) | G6 measures precision; the PRD itself notes "complement with recall spot-check" — no recall gate exists. A false-negative scenario (genuine item ranked below threshold) is unmitigated by any gate. **Recommend adding a recall gate.** |
| Large hulls untextured after decimation | Low | Medium | G11 (hull texturing) | Adequately covered once G11 has code. |
| `v2g:*` schema churn breaks archival consumers | Low | Medium | G15 (profile reproducibility) | G15 is non-blocking and tests schema stability per-run, not across schema versions. No gate asserts additive-only changes or schema version recording. **Recommend adding a schema-versioning gate.** |
| DAG layer destabilises proven linear path | Low | Medium | G16 (resume correctness) | G16 is non-blocking; the risk is low so this is proportionate. |
| Docker-secret migration breaks unattended launch | Low | Medium | G17 (secret handling) | G17 is blocking. The `rclone_config` path (`drive_ingestor.py:105`, `--config /run/secrets/rclone.conf`) already uses the secrets path but the HF token (`config.py:138 hf_token=""`) is a plain string. Mixed state. |
| Gated FLUX Kontext / Hunyuan3D-2.1 weights unavailable | Medium | Medium | G-T3 (model versions + fallback logged) | G-T3 is blocking and covers the fallback path. Adequately specified once built. |
| VLM hallucinates artifacts or slows large batches | Medium | Medium | G-T5 (precision), G-T6 (fused score/no silent drops) | Both gates exist. G-T5 requires the fixture. G-T6 is a coverage gate, not a quality gate. No gate bounds VLM latency. |
| Native v0.5.x flags alter output character | Low | Medium | None explicit | PRD §7 mentions "A/B against curated test scene before making preset-default" but no gate in §6 covers this. **Unmitigated by any gate.** |
| MILo-as-default raises no-sidecar failure surface | Low | Medium | G-T1 (fallback path) | G-T1 is blocking and covers both default and fallback. Adequately specified. |

**Unmitigated high risks:**
1. The FLUX inpaint output quality has no perceptual gate — G8 only checks that the call happened, not that the result is usable.
2. Key-item false negatives (recall) have no gate.
3. The `v0.5.x` native-flags output change has no gate.
4. G9 (co-located object collapse) is non-blocking despite medium/high risk profile.

---

## 7. Single Most Important Recommendation

**Build the curated test scene fixture first.**

Gates G6 (key-item precision, blocking, medium/high risk), G-T5 (VLM artifact precision, blocking, medium/medium risk), and G9 (co-located object separation, non-blocking but should be blocking) all require it. Without this fixture, the two ≥90% precision thresholds in the PRD are unverifiable by any means — not by unit tests, not by integration tests, and not by CI. The fixture is also the only way to tune `min_object_gaussians` (G7) and the VLM confidence threshold (G-T5) to realistic values before shipping. Every other gate in the pyramid can be built incrementally in parallel; these three cannot proceed at all without the fixture.

The fixture specification (a controlled 30–60 s walkthrough video + ground-truth object manifest + ground-truth artifact annotation file, stored in Git LFS or an external fixture URL) should be the first deliverable of the QE workstream, before any test code is written for G6/G9/G-T5.
