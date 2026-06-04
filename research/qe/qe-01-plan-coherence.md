# QE-01: Plan Coherence Audit — Vitrine v3 ADR/PRD/DDD Set

**Date**: 2026-06-04
**Auditor**: Code Analyzer Agent (Claude Sonnet 4.6)
**Scope**: ADR-009..015, PRD-v3 (`prd-v3-e2e-closure.md`), gap analysis (`gap-analysis-e2e-aspiration.md`), DDD (`v3-e2e-extensions.md`), flowchart (`aspirational-e2e-flowchart.md`). ADR-001..008 skimmed for baseline context.
**Method**: Cross-document traceability analysis, consistency checking, gap identification.

---

## Executive Verdict

**RAG Status: AMBER**

The document set is structurally sound and internally consistent at high fidelity. The traceability spine (Delta → FR → Gate → ADR) is complete and closes every D1–D14 and T1–T8 gap. The DDD model is coherent and additive. The ADR chain from ADR-009 through ADR-014 is logically sequenced with clean dependencies.

However, three issues prevent a GREEN rating: (1) a confirmed model-identity error that persists across multiple documents after a partial correction — the FLUX version conflict is only half-resolved; (2) the flowchart retains stale pre-correction wording that will mislead implementers; and (3) two open questions (Q-014.1 on the Salad API and Q-015.1 on GCP OAuth credentials) are operationally blocking but labelled or treated as non-blocking. Three medium-severity gaps exist in traceability or specification completeness that a downstream implementer would notice.

**Finding count: 13** — 1 Critical, 3 High, 5 Medium, 4 Low.

---

## Findings Table

| ID | Severity | Location | Finding | Recommendation |
|----|----------|----------|---------|----------------|
| F-01 | Critical | `adr-012-sota-tooling-modernisation.md` §Consequences + `aspirational-e2e-flowchart.md` §4 + §5 | **FLUX version conflict — partially resolved, still stale in two locations.** ADR-013 D-013.1 note, ADR-012 D-012.3, and ADR-013 §D-013.4 table all correctly state the binding target is **FLUX.2-dev** (amended from FLUX.1 Kontext). However: (a) `adr-012-sota-tooling-modernisation.md` §Consequences bullet 1 reads "Quality jump (MILo mesh, learned matching, Hunyuan3D-2.1, **FLUX Kontext**)" — stale FLUX.1 Kontext wording persists in the summary that reviewers read first; (b) `aspirational-e2e-flowchart.md` Phase 4 node `FLUX` is labelled `"LOCAL FLUX.1 Kontext inpaint unseen/occluded views (ComfyUI, ADR-012)"` — still says Kontext; (c) the same flowchart §5 VLM subgraph says `"FLUX Kontext"` in the `REPAIR` node label. A downstream implementer reading the flowchart (the canonical target picture) is told FLUX.1 Kontext; the ADRs say FLUX.2-dev. | Update `adr-012` §Consequences bullet 1 to read "FLUX.2-dev". Update `aspirational-e2e-flowchart.md` Phase 4 FLUX node label and §5 REPAIR node to say "FLUX.2-dev". These are the three remaining stale references. |
| F-02 | High | `adr-013-ingest-manifest-serial-model-lifecycle.md` §Context "What the audit found", paragraph 2 | **gemma-4-26B-A4B multimodal claim contradicted within the same document.** The Context section §"What the audit found" states: *"The linked gemma-4-26B-A4B-it-ara-abliterated is a text-only MoE (26 B total / ~4 B active, Arabic-tuned, abliterated). It has no vision tower — it cannot detect frame artifacts."* Then §D-013.5 "Corrected 2026-06-04" immediately below reverses this, asserting the model IS multimodal with a SigLIP vision encoder. Both are present in the same file. The Context section therefore contains a statement the Decision section explicitly retracts, but without deleting or striking it. A reader who stops reading at the Context section (the normal review pattern for ADRs) receives incorrect information that the model is text-only. | Either delete the "text-only MoE" paragraph from the Context section, or annotate it inline as "~~text-only — corrected, see D-013.5~~". The retraction in D-013.5 is correct and should be the only statement of the model's capability. |
| F-03 | High | `adr-013-ingest-manifest-serial-model-lifecycle.md` §D-013.3 topology diagram + `v3-e2e-extensions.md` §6 context map external box | **Topology diagram references `vlm:8081` + `reasoner:8080` as separate services, contradicting D-013.5 and D-013.3's own text.** The topology in D-013.3 shows `agent-vlm :8080` as a single unified service — correct per D-013.5. However, the same ADR-013 §D-013.3 diagram line reads: `├── agent-vlm :8080 gemma-4-26B-A4B (multimodal) — unified artifact VLM (FR-27) + reasoner (FR-28)` — that is consistent. BUT `v3-e2e-extensions.md` §6 context map `EXT` box still lists `FLUX["ComfyUI FLUX.1-Fill"]` as the external ComfyUI entry name — not FLUX.2-dev — perpetuating the FLUX version drift in the DDD context map. Additionally, `aspirational-e2e-flowchart.md` §1 entry point note (line 8) says `"vlm Qwen2.5-VL, reasoner gemma/qwen-text"` — a pre-D-013.5 two-service model that contradicts the unified agent decision. | Update `aspirational-e2e-flowchart.md` entry-point paragraph: replace `"vlm Qwen2.5-VL, reasoner gemma/qwen-text"` with `"agent-vlm gemma-4-26B-A4B (unified VLM + reasoner; Qwen2.5-VL optional fallback)"`. Update `v3-e2e-extensions.md` §6 EXT box `FLUX` label from "FLUX.1-Fill" to "ComfyUI FLUX.2-dev". |
| F-04 | High | `adr-014-agent-controlled-comfyui-integration.md` §Open Questions Q-014.1 | **Q-014.1 (Salad API model-lifecycle endpoints) is operationally blocking but marked with no blocking designation.** Q-014.1 asks: *"Confirm the model-lifecycle endpoints (load/free/unload) [the Salad control API] exposes, so the ADR-013 hard tier drives it rather than `docker stop`."* The `hard`-tier VRAM lifecycle (D-013.2) and the `RecoveryController` (D-014.3, step 5 "Release") both depend on this API being present. If the Salad add-on does not expose a `free`/`unload` endpoint, the hard-tier lifecycle must fall back to `docker stop/start`, which materially changes the architecture (cold-start impact, network disruption, container restart overhead for every FLUX.2↔Hunyuan3D transition). FR-30 (`ModelLifecycleManager`) and FR-38 (provisioning) cannot be fully implemented without resolving this. PRD §10 marks FR-30 as having no gate and ADR-013 Q5/Q6 as non-blocking, but Q-014.1 is silently unclassified while being operationally critical. | Classify Q-014.1 as **blocking** in ADR-014. Add a fallback decision: "If Salad does not expose load/free/unload: the `hard` tier uses `docker stop`/`docker start <svc>` — accepting the cold-start cost — and the lifecycle contract remains unchanged." Document this decision in ADR-013 D-013.2 as a resolution path. |
| F-05 | Medium | `prd-v3-e2e-closure.md` §9 commission table + §10 traceability table | **Traceability table §10 introduces O1–O6 rows with no corresponding Objectives in §2.** PRD §2 defines Objectives O1–O20. PRD §10 traceability table adds rows labelled `O1`, `O2`, `O3`, `O4`, `O5`, `O6` in the "Delta" column — but these collide with the O1–O6 objective IDs. These rows map to ADR-013/014 `[oversight]`/`[pipeline]`/`[drive]` manifest feature groups (`O1=FR-29`, `O2=FR-30`, `O3=FR-31`, `O4=FR-32`, `O5=FR-33`, `O6=FR-34`). This creates an **ID collision**: the same symbols O1..O6 refer to two different things (Objectives in §2, and manifest-group rows in §10). The gap register D1–D14 and T1–T8 are cleanly namespaced; O1–O6 in §10 are not gap IDs and are not labelled as such. | Rename the §10 manifest-group rows to `M1–M6` (or `ADR-013-1..3`, `ADR-014-1..3`) to avoid collision with the §2 Objective O-numbers. Update the notes below the table accordingly. |
| F-06 | Medium | `adr-015-vitrine-web-onboarding.md` §Open Questions Q-015.1 | **Q-015.1 (Google OAuth client ownership) is de-facto blocking for FR-37 and G-O3 but not classified as blocking.** Q-015.1 asks who owns the GCP project / client credentials for the OAuth consent screen. Without a registered OAuth client, the browser-based Google login flow (FR-37 D-015.4) cannot be implemented — there is no redirect URI, no `client_id`, no consent screen. FR-37 is marked `*` (high-risk) and G-O3 is a blocking gate. A downstream implementer cannot wire the OAuth flow without this answer. | Classify Q-015.1 as **blocking for FR-37 / G-O3**. Add a provisional fallback: "If GCP client registration is delayed, the ADR-009 service-account key path (NFR-4 Docker secret, no browser OAuth) suffices for Drive read — write-back (FR-39) requires the OAuth scope and remains blocked until the client is registered." |
| F-07 | Medium | `adr-011-usd-metadata-enrichment.md` §Decision (a) `v2g:*` schema table + `prd-v3-e2e-closure.md` FR-15 | **`v2g:concept` field present in ADR-011 schema but absent from PRD FR-15 required-fields list.** ADR-011 §(a) object-prim table lists both `v2g:semantic_label` and `v2g:concept` as required fields (10 rows total + concept = 11 distinct fields). PRD FR-15 lists the required fields and names 10, omitting `v2g:concept`. The PRD says "≥ 10 required fields"; technically this is met, but the field enumeration mismatch means a validator built strictly from the PRD list would not check `v2g:concept`. Gate G13 requires "all ≥10 required `v2g:*` fields" — ambiguous as to whether `v2g:concept` is in scope. | Reconcile: either add `v2g:concept` to the PRD FR-15 enumeration (making it 11 fields) and update the ≥10 threshold accordingly, or demote it to OPTIONAL in ADR-011 §(a). The field is useful (the SAM3 concept prompt that matched) and should remain required; update the PRD list. |
| F-08 | Medium | `v3-e2e-extensions.md` §2.6 `RecoveryRequest` value object; `adr-014-agent-controlled-comfyui-integration.md` §D-014.3 | **`RecoveryRequest.artifact_report_ref` has no upstream type definition.** The DDD §2.6 defines `RecoveryRequest { ..., artifact_report_ref }`. The `artifact_report` is described in ADR-012 D-012.5 and PRD FR-27 as a structured typed output (motion ghosting, rolling-shutter, etc., with label + confidence + bbox) written as a `vlm` block in the per-frame sidecar. However, no value object for `ArtifactReport` is defined anywhere in the DDD extension or the ADR set. `ImageMetadataTag` in DDD §1.3 mentions a `vlm` block only by implication (ADR-012 says the report is "written as a `vlm` block in the per-frame sidecar"). The type is used in the `RecoveryController` loop (ADR-014 §D-014.3 step 1) but never declared as a domain value object. | Add an `ArtifactReport` value object to DDD §1.3 (as a nested block of `ImageMetadataTag`) or §2.6. Minimum fields: `artifact_type: str`, `confidence: float`, `bbox: AABB | None`, `reason: str`. Reference it from `RecoveryRequest.artifact_report_ref`. |
| F-09 | Medium | `prd-v3-e2e-closure.md` §10 traceability table + gap register | **D11 is assigned to both "PRD §4 / 011" in §10 and listed under ADR-011 in the §9 commission table, with different FR assignments.** The §9 commission table assigns D11 to ADR-011 ("FR-14..FR-19 — D11, D12, D13, D14"). The §10 traceability table shows D11 → FR-13 → G11 → "PRD §4 / 011". FR-13 is defined in PRD §4 under "Phase 4 — Per-key-item hull reconstruction" (closes D11), and ADR-010 is where hull texturing logic lives (guaranteed texturing before hull is accepted). ADR-011 only adds the validation gate (FR-18 checks hull texturing). The split attribution means D11 is owned by neither ADR solely and the primary implementation location (ADR-010 or ADR-011?) is ambiguous. | In §10, clarify D11 → FR-13 → G11 → **ADR-010** (the implementation: decimate-then-bake guarantee); ADR-011 FR-18 is the *validation gate* downstream. Update the §9 table to reflect that ADR-010 owns D11 implementation and ADR-011 owns the gate. |
| F-10 | Low | `adr-012-sota-tooling-modernisation.md` §Risks bullet 1 | **ADR-012 §Risks still references "FLUX.1 Kontext" in the gated/licensed weights risk item.** The bullet reads: *"FLUX.1 Kontext and FLUX.1-dev are gated on Hugging Face"*. After the amendment to FLUX.2-dev, the risk item should name FLUX.2-dev. FLUX.1-dev is the fallback, not the primary. | Replace "FLUX.1 Kontext" with "FLUX.2-dev" in the §Risks entry. Add FLUX.2-dev licence acceptance as a prerequisite note (the fp8mixed checkpoint staged on .48 may already be past the licence gate, but the ADR should be accurate). |
| F-11 | Low | `adr-015-vitrine-web-onboarding.md` §D-015.5; `v3-e2e-extensions.md` §7.1 `ObjectOfInterest` | **`ObjectOfInterest.sam3_concept` attribution boundary is stated in DDD §7.1 but not cross-referenced in ADR-015 D-015.5.** DDD §7.1 invariant 3 states: *"`ObjectOfInterest.sam3_concept` is NOT authored by Setup — it is the interpretive output the internal overseer fills after hand-off."* ADR-015 D-015.5 describes the hand-off boundary but does not explicitly state that `sam3_concept` is left `None` in the manifest until the agent fills it. A setup-tool implementer reading only ADR-015 may add a `sam3_concept` field to the wizard UI. | Add a note to ADR-015 D-015.5 (and the wizard step description): "`sam3_concept` is NOT a wizard field — it is populated by the internal agent on hand-off from the free-text `ObjectOfInterest.description`. The wizard only captures `description` and `priority`." |
| F-12 | Low | `prd-v3-e2e-closure.md` §9 commission table + `adr-009` related decisions | **ADR-009 "Drives" line in PRD §9 header and in the ADR itself omits ADR-012/013/014 as dependents that consume ADR-009 sidecar data.** The PRD §1 says "Drives: ADR-009, ADR-010, ADR-011, and the DDD extension" — a literal statement from the header that was written before ADR-012/013/014 were commissioned. The PRD body (§4 FR-27, FR-28, §9 table) correctly adds ADR-012/013/014. The ADR-009 "Related Decisions" section lists only ADR-007/010/011/001 as consumers; it does not mention ADR-012 D-012.5 (VLM artifact stage writes into the ADR-009 sidecar `vlm` block) or ADR-013 (manifest feeds the ingest, D-013.1). | Update ADR-009 "Related Decisions" to add ADR-012 (VLM stage writes `vlm` block into the sidecar) and ADR-013 (the manifest's `exhibit.toml` is the upstream of the ingest loop). Update the PRD §1 "Drives" header to note it was extended by ADR-012/013/014/015. |
| F-13 | Low | `v3-e2e-extensions.md` §6 context map (Mermaid diagram) | **Context map §6 labels the ComfyUI external node as `FLUX["ComfyUI FLUX.1-Fill"]` — a stale pre-amendment name.** This also means the context map does not show `agent-vlm` as an external system or the Salad control API surface, despite both being new boundaries that the new `Per-Object Reconstruction` context crosses via the ComfyUI ACL (§5.2). The Onboarding/Setup context §7.6 has its own correct diagram (shows `CUI["ComfyUI / Salad (.48)"]`), but the main §6 context map is inconsistent. | In §6 context map, rename `FLUX` to `COMFYUI["ComfyUI (.48): FLUX.2-dev / Hunyuan3D-2.1 + Salad control"]` and add `AGENTLM["agent-vlm: gemma-4-26B-A4B"]` to the EXT box. |

---

## Collated Open Questions

All open questions from ADR-013/014/015, with blocking classification:

| ID | Source | Question | Blocking? | Status |
|----|--------|----------|-----------|--------|
| Q1 | ADR-013 | Host & VRAM — which host runs the pipeline? | Non-blocking | RESOLVED: .48 |
| Q2 | ADR-013 | VLM vs reasoner — is gemma-4 multimodal? | Non-blocking | RESOLVED: multimodal (D-013.5) |
| Q3 | ADR-013 | Hunyuan3D 2.1 — present on .48? | Non-blocking | RESOLVED: yes |
| Q4 | ADR-013 | ComfyUI reuse or fresh? | Non-blocking | RESOLVED: reuse .48 (ADR-014) |
| Q5 | ADR-013 | Default unload tier | Non-blocking | OPEN: `soft` default + `hard` for FLUX.2↔Hunyuan3D (proceeding with this) |
| Q6 | ADR-013 | HF token for gated pulls | Non-blocking | OPEN: still required; non-blocking because .48-resident models work without it |
| Q7 | ADR-013 | gemma quant (Q5_K_M vs Q6_K) | Non-blocking | RESOLVED with caveat: Q5_K_M (Q6_K not published, requires local re-quant) |
| Q-014.1 | ADR-014 | **Salad add-on API model-lifecycle endpoints (load/free/unload)** | **Should be BLOCKING** (currently unclassified) | OPEN — the `hard`-tier lifecycle and `RecoveryController` step 5 depend on this. Fallback: `docker stop/start` if absent. |
| Q-014.2 | ADR-014 | FLUX.2 + Hunyuan3D-2.1 custom nodes already on .48 ComfyUI? | Non-blocking | OPEN — needed for D-014.1 provisioning; fallback (D-014.1 install+pin) handles it |
| Q-015.1 | ADR-015 | **Google OAuth client registration (GCP project / client credentials)** | **Should be BLOCKING for FR-37 / G-O3** (currently unclassified) | OPEN — browser OAuth consent flow cannot be implemented without a registered client ID + redirect URI |
| Q-015.2 | ADR-015 | Where should `vitrine-setup` run (.48 recommended)? | Non-blocking | OPEN — recommendation given; decision deferred to operator |
| Q-015.3 | ADR-015 | "No history" = single overwritten file vs. one active + manual archive? | Non-blocking | OPEN — DDD §7.1 invariant 1 treats it as single overwritten file; confirm with owner |

**Blocking open questions: 2** (Q-014.1, Q-015.1), currently unclassified as such.
**Non-blocking open questions: 10**, of which 6 resolved and 4 genuinely open.

---

## Traceability Integrity Assessment

### D1–D14 closure
Every delta D1–D14 traces to at least one FR and at least one gate (or is explicitly contract-only where no gate is needed, e.g. D6). No orphan deltas. D11 has a dual-ADR attribution issue (F-09) but does close.

### T1–T8 closure
T1–T8 all trace to FR-20..FR-28 and ADR-012 via the traceability table. T5 and T7 have no blocking gate (optional branch, A/B deferred) — this is intentional and documented. No orphan tooling gaps.

### FR-1..FR-40 coverage
All 40 FRs appear in the §9 commission table assigned to an ADR. All 40 appear in the §10 traceability table. No orphan FRs.

**One traceability anomaly**: The §10 table rows for ADR-013/014 operations use `O1–O6` labels in the Delta column — these collide with the §2 Objective IDs (see F-05). The actual Objectives O1–O20 all trace correctly through FRs.

### Gate coverage
27 gates total. All blocking gates (marked "Yes") trace to a delta or tooling gap. Non-blocking gates (G9, G15, G16, G-T2, G-O4) are correctly noted as such. Every FR with a `*` (high-risk) designation has a corresponding blocking gate.

**One gap**: FR-29 (manifest contract), FR-30 (VRAM-bounded run), FR-31 (docker mesh) have no gate in the PRD — this is intentional per §10 footnote ("operationalise" category) but worth noting that the manifest contract (FR-29) and the docker mesh (FR-31) have no automated verification, only structural/audit checks.

---

## Architectural Soundness Assessment

### Setup vs. agent hand-off (ADR-015 D-015.5)
The boundary is clean and consistently stated across ADR-015, DDD §7.1 invariant 3, and the `ProvisionStatus` value object. The `provision.status = "ready"` latch and `ProvisionReady` event are the correct boundary marker. One gap: `ObjectOfInterest.sam3_concept` is not a wizard field, but this is only explicit in the DDD (F-11).

### Serial model lifecycle VRAM logic (ADR-013 D-013.2) vs. hardware-aware selection (ADR-015 D-015.3)
Coherent. The D-015.3 `ModelSelection.serial_peak_estimate_gb` uses the serial-lifecycle peak (max single stage, not sum), which is the correct figure for the D-013.2 `hard`-tier design. The DDD §7.3 domain rule formalises this correctly: *"no selected stage model's `serial_peak_estimate_gb` may exceed the smallest GPU's `total_vram_gb`"*. The VRAM table in ADR-013 D-013.4 (FLUX.2 ~32 GB, Hunyuan3D ~16 GB, gemma Q5_K_M ~20 GB, SAM3 ~8 GB) is consistent with a 48 GB A6000 running any one of these serially with headroom. The `hard`-tier for FLUX.2↔Hunyuan3D transitions is architecturally justified given the fragmentation risk at 32 GB residual after FLUX.2.

### Circular dependencies between ADRs
None found. Dependency order is: ADR-009 → ADR-010 → ADR-011 → ADR-012 (additive); ADR-013 integrates 009/010/011/012; ADR-014 depends on ADR-013; ADR-015 depends on ADR-013 + ADR-014. This is a linear DAG with no cycles.

### Per-video vs. per-room reconstruction tension
Resolved cleanly in ADR-009 §(a) and DDD §1.4. The two-level loop (inner per-video, outer per-room) is architecturally correct and preserves the downstream pooled-frame contract exactly. The DDD `CaptureSession` invariant 2 (at most one non-null `local_scratch_path` at any instant) is the domain model of the retention ceiling.

### `pose_hint` backfill ordering dependency
ADR-009 introduces a second pass over sidecars after SfM to backfill `pose_hint`. This creates a new ordering dependency: sidecars must exist before the SfM completes. This is architecturally correct (sidecars are written before pooling; SfM runs on the pooled frame set; backfill runs after SfM). However, it is not represented as an explicit stage dependency in the optional DAG (FR-7). If FR-7 is implemented, the DAG must include `backfill_pose_hints` as a stage after SfM and before hull recon. This is not specified anywhere.

---

## Gaps — Items a Downstream Implementer Cannot Build From

### Gap G-I1: No `exhibit.toml` JSON Schema document specified
ADR-015 D-015.2 and FR-35 specify that a JSON Schema (`schema/exhibit.toml.schema.json`) is the "single source of truth" and drives the wizard's form generation. The PRD, ADR-015, and DDD all reference this schema but none of them define its structure. The schema must cover all blocks: `[exhibit]`, `[drive]`, `[[objects]]`, `[secrets]`, `[pipeline]`, `[oversight]`, `[models]`, `[models.vram_plan]`, `[provision]`. An implementer must infer this from the ADR-013 TOML example and the DDD `ExhibitManifest` aggregate. This is workable but unambiguous specification is missing.

### Gap G-I2: `pose_hint` backfill stage not placed in the stage sequence
As noted in the architectural soundness section: the `pose_hint` backfill (ADR-009 §Rationale, DDD §1.2/§1.3) is a new post-SfM stage with no placement in the `STAGE_NAMES` sequence (`stages.py:43-56`) or the optional DAG (FR-7). An implementer knows it must happen after SfM and before hull recon, but there is no specification of where in `stages.py` it inserts, what its ledger state transition is, or whether it is a `hard`/`soft` error if COLMAP fails to register a frame (the ADR says `pose_hint` stays null — but the backfill stage itself has no error contract).

### Gap G-I3: `ArtifactReport` value object not specified in DDD
As found in F-08: the `artifact_report` is described structurally in ADR-012 D-012.5 (typed labels + confidence + bbox) but never defined as a domain value object. The `ImageMetadataTag` DDD §1.3 mentions a `pose_hint` slot and `vlm` block only by reference. A VLM stage implementer cannot determine the exact field names or types from the planning documents alone.

### Gap G-I4: `vitrine-setup` binary scope boundary for provisioning — what triggers model downloads
ADR-015 D-015.5 says: *"Vitrine Onboarding (setup) … downloads & integrates models (HF pulls via the stored token; ComfyUI checkpoint/node ensure+pin via the ADR-014 Salad control API)."* However, it does not specify: (a) whether model downloads are triggered by the user clicking "Provision" in the wizard, or auto-triggered on manifest save; (b) what the progress stream format is (WebSocket vs polling, endpoint name, schema of progress events); (c) what "idempotent" means for a partial download (resume semantics). These are implementation-critical for the long-running provisioning UX. The agentbox pattern reference (`agentbox/setup/`) covers progress polling but `vitrine-setup` differs by triggering actual long-running downloads, not just validating config.

---

## Prioritised Remediation List

Priority 1 — Do before any implementation begins (prevents defects being baked in):
1. **F-01**: Fix FLUX.2-dev vs FLUX.1 Kontext in `adr-012` §Consequences and both stale flowchart nodes. (30 min, one author.)
2. **F-03**: Fix the flowchart entry-point paragraph re: unified agent-vlm (not Qwen+gemma split). Fix DDD §6 context map `FLUX.1-Fill` label. (20 min.)
3. **F-05**: Rename O1–O6 rows in PRD §10 to avoid collision with Objective O-numbers. (15 min.)
4. **F-11**: Add explicit note to ADR-015 D-015.5 that `sam3_concept` is not a wizard field. (10 min.)

Priority 2 — Do before architecture review sign-off:
5. **F-04**: Classify Q-014.1 as blocking; add `docker stop/start` fallback decision to ADR-013 D-013.2 and ADR-014. (ADR-014 author.)
6. **F-06**: Classify Q-015.1 as blocking for FR-37; add service-account fallback scope note. (ADR-015 author.)
7. **F-02**: Strike or annotate the "text-only MoE" paragraph in ADR-013 Context section. (5 min, same author.)
8. **G-I1**: Draft the `schema/exhibit.toml.schema.json` structure specification — or add a schema table to ADR-015 D-015.2. (Required before `vitrine-setup` implementation.)

Priority 3 — Do before implementation of affected components:
9. **F-07**: Reconcile `v2g:concept` between PRD FR-15 field list and ADR-011 schema table.
10. **F-08 / G-I3**: Define `ArtifactReport` value object in DDD §1.3 or §2.6.
11. **G-I2**: Specify `pose_hint` backfill as an explicit named stage in the stage sequence (ADR-009 Consequences or a new ADR-009 appendix).
12. **G-I4**: Add provisioning UX contract to ADR-015 (trigger, progress stream format, resume semantics).
13. **F-09**: Clarify D11 owner (ADR-010 = implementation, ADR-011 = gate) in PRD §9/§10.

Priority 4 — Cosmetic / polish:
14. **F-10**: Fix "FLUX.1 Kontext" in ADR-012 §Risks licensed-weights bullet.
15. **F-12**: Update ADR-009 Related Decisions to include ADR-012/013.
16. **F-13**: Update DDD §6 context map FLUX node label and add agent-vlm to EXT box.
