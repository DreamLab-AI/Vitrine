# QE Fleet Synthesis — Vitrine (v3 E2E Closure) Plan & Partial Implementation

**Date**: 2026-06-04
**Scope**: ADR-009..015, PRD-v3 (FR-1..FR-40, 27 quality gates), DDD v3 extension, the
aspirational flowchart, and the partial Python implementation in `src/pipeline/`.
**Method**: four parallel QE specialists (plan-coherence, implementation-coverage,
gate-testability, security), each producing a standalone report (`qe-01`..`qe-04`); this
document synthesises them and gives one prioritised remediation backlog.

---

## 1. Executive verdict

**Overall RAG: AMBER.** The *design* is sound and internally near-coherent; the *plan-to-code
gap is the entire story*. The seven-ADR closure set traces cleanly (every D1–D14, T1–T8,
FR-1..FR-40 closes with no orphan requirements), but the implementation realises only a
fraction of it, the verification framework has **zero** executing coverage, and the
single security regression the plan was written to fix (FINDING-006) is **still live in
the running code**.

| Lens | Verdict | Headline metric |
|------|---------|-----------------|
| Plan coherence (`qe-01`) | AMBER | 1 Critical, 3 High doc-drift findings; traceability spine intact |
| Implementation coverage (`qe-02`) | RED | 40 FRs → **4 Implemented / 10 Partial / 4 Stubbed / 22 Missing** |
| Gate testability (`qe-03`) | RED | **0 / 27** gates automated; **20 / 20** blocking gates have zero code |
| Security (`qe-04`) | Plan: CONDITIONAL PASS · Code: **FAIL** | FINDING-006 remediated in plan, **not** in code |

The AMBER (not RED) overall reflects that this is a *planning milestone* — the artefacts are
proposed/draft, infrastructure is deliberately not yet built — and the design is implementable
as written. It is RRED on execution readiness, AMBER on design quality.

---

## 2. The single root cause

Three v3 infrastructure components are specified but **do not exist yet**, and their absence
is the dominant cause across three of the four reports:

| Absent component | Specified by | FRs blocked | Security findings caused |
|------------------|--------------|-------------|--------------------------|
| `manifest.py` / `exhibit.toml` loader (`env:` resolution + secret stripping) | ADR-013, ADR-015 | **12** (FR-6, FR-29..31, FR-35..39) | SEC-03 (snapshot leak), root of SEC-01/02 remediation |
| `model_lifecycle.py` (`ModelLifecycleManager`, soft/hard unload) | ADR-013 | FR-30 + gates the serial-VRAM design | — |
| `vitrine-setup/` Rust/Axum binary + `schema/exhibit.toml.schema.json` | ADR-015 | FR-35..38 (+ gates G-O1..G-O4) | SEC-06/07 (OAuth, proxy allowlist) |
| `v2g-net` Docker mesh (replacing hardcoded `192.168.2.48`) | ADR-013, ADR-014 | FR-31, FR-33, FR-34 | SEC-04 (11 hardcoded endpoints) |

**Highest-leverage single action** (named independently by both `qe-02` and `qe-04`): build
`manifest.py` with proper `env:`-indirection resolution and pre-snapshot secret stripping. It
unblocks the most FRs and closes the FINDING-006 code regression at its source.

---

## 3. Consolidated findings by theme

### 3.1 Documentation drift (from `qe-01`) — cheap, high-coherence-value fixes

These are stale strings left over from earlier decisions; each is a trap for an implementer
reading the wrong document. **All are low-effort and should be fixed in the doc-update pass (task #30).**

| Ref | Severity | Location | Drift |
|-----|----------|----------|-------|
| F-01 | Critical | `aspirational-e2e-flowchart.md` Phase 4 + §5 REPAIR node; `adr-012` §Consequences | Still says "FLUX.1 Kontext"; canonical decision is **FLUX.2-dev** |
| F-02 | High | `adr-013` Context section | States gemma-4 is "text-only, no vision tower" then reverses in D-013.5 — the wrong line is not struck |
| F-03 | High | `aspirational-e2e-flowchart.md` entry-point para; DDD §6 context map | Pre-unification two-service topology (`Qwen2.5-VL` + text reasoner); DDD map labels ComfyUI `FLUX.1-Fill` |

### 3.2 Implementation gaps — the 5 highest-leverage (from `qe-02`)

Ranked by (impact on blocking gates) × (1/effort). The first three are *near-trivial* and live
entirely in existing code:

1. **FR-3 — persist the per-image metadata sidecar.** `FrameQuality` is fully computed
   (`frame_quality.py:38-64`) but never written to disk. This one write unlocks the entire USD
   lineage chain (FR-5, FR-17, FR-19). ~half a day, no new infra.
2. **FR-9 — enforce `min_object_gaussians`.** Defined (`config.py:109`), ignored
   (`stages.py:1411` keeps everything with `mask_pixels>0`). Dead config knob; directly gates G6/G7.
3. **FR-20 — flip default mesh `tsdf → milo`** (`config.py:60`). One line; every default run
   currently contradicts the plan. Unblocks G-T1.
4. **FR-11 — wire the (already-working) FLUX inpainter** `comfyui_inpainter.py:686` into the hull
   orbit loop `stages.py:1778-1806`. Wiring gap, not a build gap. Blocking gate G8.
5. **FR-12 — stop discarding per-object pose** (`stages.py:1683-1693`); `ObjectDescriptor` already
   has the fields (`usd_assembler.py:55-56`). Every hull lands at origin today. Blocking gate G10.

### 3.3 Verification void (from `qe-03`)

- **0 / 27** PRD gates have any executing automation; **20 / 20 blocking gates** have zero code.
- The four existing test/validation surfaces (`quality_gates.py`, `preflight.py`,
  `scripts/test_orchestrator.py`, `tests/python/`) verify PSNR/SSIM/mesh-counts/the C++ plugin —
  all **orthogonal** to the PRD §6 acceptance criteria.
- Verifiability classes: 13 Automatable-now · 3 Needs-fixture (G6, G9, G-T5) · 8 Needs-infra ·
  1 Underspecified (G2 harness).
- **Blocking dependency**: G6 and G-T5 (both ≥90 % precision) are *unverifiable* without a
  **curated golden test scene** (30–60 s walkthrough + ground-truth object manifest + artifact
  annotations, Git-LFS) — which does not exist. It is also the only way to tune G7's threshold
  non-arbitrarily. **This fixture is the critical path for the whole gate framework.**

### 3.4 Security (from `qe-04`)

- **FINDING-006 (plaintext-env-var secrets): remediated in plan, NOT in code.** `HF_TOKEN` and
  `ANTHROPIC_API_KEY` are still plain Docker env vars loaded from an on-disk `.env`; visible to
  `docker inspect`. `PipelineConfig.save()` would serialise a resolved token into the JSON run
  snapshot (SEC-03). The Drive service-account Docker-secret mount *is* already correct — partial credit.
- Counts: **1 Critical (SEC-01 HF_TOKEN), 3 High (SEC-02, SEC-03, SEC-04), 5 Medium, 2 Low, 2 Info.**
- The onboarding threat surface (FR-37/D-015.4) is sound in principle but the spec omits **OAuth
  PKCE + CSRF `state`** (SEC-06) and the `/api/proxy/*` **URL allowlist** (SEC-07) — both must be
  added to ADR-015 before the tool is built.

---

## 4. Prioritised remediation backlog

Ordered for maximum unblock-per-unit-effort. P0 = do first.

| # | Action | Closes | Effort | Reports |
|---|--------|--------|--------|---------|
| P0-1 | Fix the 3 doc-drift items (FLUX.2, gemma multimodal strike, unified-agent topology) | F-01/02/03 | Low | qe-01 |
| P0-2 | Build `manifest.py`: parse `exhibit.toml`, resolve `env:` refs, **strip secrets before JSON snapshot** | FR-29, SEC-01/02/03; unblocks 12 FRs | Med | qe-02, qe-04 |
| P0-3 | Add OAuth PKCE+state and `/api/proxy` allowlist to ADR-015 spec | SEC-06/07 | Low | qe-04 |
| P1-1 | Quick-win code fixes: FR-3 sidecar write, FR-9 threshold enforce, FR-20 MILo default | G4/G5/G6/G7/G-T1 | Low–Med | qe-02 |
| P1-2 | Wire FR-11 inpaint loop + FR-12 pose persistence | G8, G10 | Med | qe-02 |
| P1-3 | Author the **curated golden test scene** fixture (Git-LFS) | unblocks G6, G9, G-T5, G7 tuning | Med | qe-03 |
| P2-1 | Replace hardcoded `192.168.2.48` with service-DNS / manifest endpoints | FR-31/33, SEC-04 | Med | qe-02, qe-04 |
| P2-2 | `model_lifecycle.py` + `v2g-net` compose | FR-30/31 | High | qe-02 |
| P2-3 | Stand up `vitrine-setup` skeleton + JSON Schema | FR-35..38 | High | qe-02 |
| P3 | Build the gate test pyramid against the golden scene | 20 blocking gates | High | qe-03 |

---

## 5. What the fleet confirms about the design

Despite the execution gap, the audits validate the architecture: the ADR dependency graph is a
clean DAG with no cycles; the DDD extension is purely additive (no existing aggregate is
mutated); the setup/agent hand-off boundary (ADR-015 D-015.5) is clean; and the
serial-lifecycle ↔ hardware-aware-selection coupling (ADR-013 D-013.2 ↔ ADR-015 D-015.3) is
coherent. The plan is **buildable as written** once the doc-drift is fixed and the open
questions (Q-014.1 Salad endpoints, Q-015.1 OAuth client owner) are closed — both currently
unclassified-but-blocking and should be promoted in their ADRs.

**Bottom line**: the project is at a healthy *planning-complete, build-pending* milestone.
Priorities P0-1..P1-3 convert the strongest paper design in the repo's history into a testable
spine, and close the one live security regression, for modest effort.
