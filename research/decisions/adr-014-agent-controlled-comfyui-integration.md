# ADR-014: Agent-Controlled ComfyUI Integration (.48)

## Status

Proposed (2026-06-04). Depends on ADR-013 (`v2g-net` mesh, serial lifecycle, unified gemma-4 agent).

## Context

ADR-013 places the generative models on the **existing .48 ComfyUI** and wires it into `v2g-net`. The
owner's direction: *"use the existing ComfyUI, we will wire it there; create the PRD/ADR/DDD for
updating, connecting and agent-controlling that instance."* This ADR specifies those three things and
nothing more.

### What exists today

- **A real ComfyUI client** — `src/pipeline/comfyui_inpainter.py` (`ComfyUIInpainter`). It already
  talks to two surfaces on .48:
  - **Salad control-plane API** (`api_url`, default `http://192.168.2.48:3001`) — the owner confirms
    this is an **add-on API inside the ComfyUI container giving more control** (model probe/download,
    submission), *not* a cloud service. Methods: `probe_models()`, `download_model()`,
    `ensure_flux_fill_models()` / `ensure_flux_dev_models()`, `_submit_workflow()`.
  - **Direct ComfyUI graph API** (`comfyui_url`, default `http://192.168.2.48:8189`) — `object_info`
    introspection of available loaders/checkpoints.
  - An **`ImageServer`** (HTTP, ephemeral) that serves the orchestrator's input frames/masks back to
    ComfyUI over the LAN (`local_ip`, default `192.168.2.1`).
- Workflow **graph templates** in `src/pipeline/workflows/` (`flux_inpaint.json`,
  `flux_inpaint_vae_encode.json`, `sam3d_workflow.json`); the owner's FLUX.2 graphs
  (`workspace/ai-3d-fusion/workflows/glass-medallion-flux2*.json`) reference
  `flux2_dev_fp8mixed.safetensors` + `flux2-vae.safetensors` + `mistral_3_small_flux2_fp8.safetensors`.

### What is missing

1. **The client is hardcoded to `192.168.2.48:{3001,8189}`** (`__init__` defaults, lines 282-284) and
   to FLUX.1-Fill workflows; it has no FLUX.2 graph, no Hunyuan3D path, no version pinning.
2. **It is one-shot, not agentic.** `inpaint()` submits a graph and returns the first result. Nothing
   evaluates the output, and nothing closes the loop when the recovery is wrong (the exact failure the
   per-object occluded-face recovery of ADR-010 FR-11 must not silently pass).
3. **No update/connect contract** for standing the .48 instance into `v2g-net` with pinned nodes and
   verified checkpoints.

## Decision

### D-014.1 — Update & pin the .48 ComfyUI (the *updating* ask)

Bring the existing .48 ComfyUI to a known-good, pinned state via the Salad control API, idempotently
at run start (capability probe → ensure → fail-named-if-absent):

- **Pin** the ComfyUI commit and the required custom nodes (FLUX.2 nodes, Hunyuan3D-2.1 nodes) — record
  the pins in the ADR-012 T6 version-lock manifest. No `latest`.
- **Ensure checkpoints** through the existing `probe_models()` / `download_model()` surface:
  `flux2_dev_fp8mixed.safetensors`, `flux2-vae.safetensors`, `mistral_3_small_flux2_fp8.safetensors`,
  and Hunyuan3D-2.1 weights (already on .48 per ADR-013 Q3). A missing checkpoint **degrades to a
  declared fallback** (FLUX.1-Fill / Hunyuan3D-2.0) behind the capability probe — never a hard break.
- Generalise `ensure_flux_dev_models()` → `ensure_flux2_models()`; add `ensure_hunyuan3d_models()`.

### D-014.2 — Connect over `v2g-net` (the *connecting* ask)

- Replace the hardcoded `192.168.2.48:{3001,8189,8188}` defaults with **service-DNS endpoints**
  (`comfyui:8188` graph, `comfyui:3001` Salad control), sourced from `PipelineConfig`/manifest, with
  the literal-IP form retained only as a manifest override for the pre-mesh single-host case.
- The `ImageServer` binds on the orchestrator container's `v2g-net` address; `local_ip` becomes the
  orchestrator's **service name on the network**, so ComfyUI fetches inputs by DNS, not a hardcoded
  `192.168.2.1`.
- Both surfaces (graph + Salad control) are first-class: graph for **workflow execution**, Salad for
  **model lifecycle** (the ADR-013 D-013.2 `hard`-tier load/free of FLUX.2 ↔ Hunyuan3D runs through
  the Salad control API on this instance).

### D-014.3 — Agent control loop (the *agent-controlling* ask)

**Who runs the loop:** the in-container **Claude Code orchestrator** (the pipeline overseer;
`docs/architecture.md` §"Claude Code as Orchestrator"). It calls the **local gemma-4 vision tool**
(`agent-vlm`, ADR-013 D-013.5) for bulk seeing, and retains the judgment. The `RecoveryController` is a
stateless helper the orchestrator invokes — consistent with the "no hidden state machine; Claude
decides what to run next" architecture. Per recovery target (a `priority="key"` object from the
manifest object sub-list, ADR-010 / FR-29):

1. **Plan** — the orchestrator composes the FLUX.2 inpaint (occluded-face recovery) or Hunyuan3D (hull)
   graph from a template, parameterised by the object identity + the FR-27 `artifact_report` (which
   regions are blown-out/occluded) + FR-28 metadata.
2. **Submit** — `POST /prompt` (graph API); poll `/history/{id}`; fetch outputs via `/view`.
3. **Evaluate** — gemma-4 (the vision tool) **looks at** the generated image / mesh render and scores it
   against the object identity and artifact criteria (plausible recovered face? new artifacts?
   consistent with neighbouring views?); the orchestrator reads the score and may inspect edge cases
   itself.
4. **Decide** — the orchestrator chooses `accept` | `re-prompt` (adjust `denoise`/`guidance`/`seed`/mask,
   bounded retry budget) | `veto` (unrecoverable). The loop **annotates, never silently drops** (FR-28):
   every attempt, the gemma-4 verdict, and the reason are written to the per-video ledger + `v2g:*`
   lineage (ADR-009/011).
5. **Release** — Salad-control `free`/unload the stage model before the next stage (ADR-013 lifecycle).

Retry budget, confidence thresholds, and the accept/veto policy are config-driven (manifest
`[pipeline]` overrides) so a run is reproducible and the controller cannot loop unbounded.

### D-014.4 — Generative Recovery bounded context + ACL (the *DDD* ask)

The ComfyUI integration becomes an explicit **Generative Recovery** bounded context (DDD extension):

- **Anti-Corruption Layer** = `comfyui_inpainter.py` (generalised) + a `hunyuan3d` path on the same
  transport. It translates ComfyUI's prompt-graph + Salad control vocabulary into the domain language
  (`RecoveryRequest`, `RecoveryAttempt`, `RecoveryVerdict`) so ComfyUI's API shape never leaks into the
  pipeline core.
- **Domain policy** = the agent control loop (D-014.3): `RecoveryController` issues `RecoveryAttempt`s
  and consumes `VlmVerdict`s until `accept`/`veto`.
- **Relationship** = Conformist toward ComfyUI's API (we adapt to it) but Customer/Supplier toward the
  SceneAssembly context, to which it publishes accepted, identity-tagged hulls/textures.

Full DDD text lands in `research/ddd/v3-e2e-extensions.md` (new §). No change to existing aggregates —
`RecoveryAttempt`/`RecoveryVerdict` are value objects on the existing per-object recovery flow.

## Rationale

- The existing `ComfyUIInpainter` is already the right seam (two-surface client + image server); this
  ADR generalises rather than rewrites — minimal new surface.
- A VLM-in-the-loop is the only way the occluded-face recovery (ADR-010 FR-11) can be *verified*
  correct rather than hoped correct; the unified gemma-4 agent makes this one model, not two.
- Pinning + capability-probe + declared fallbacks keeps the run reproducible (ADR-012 T6) while
  tolerating a missing checkpoint.
- An explicit bounded context + ACL stops ComfyUI's API shape from contaminating the pipeline core and
  gives the agent loop a clean domain vocabulary.

## Consequences

**Positive:** verified per-object recovery; reuse of the owner's existing .48 ComfyUI + FLUX.2 assets;
host-portable endpoints; reproducible pins; agent loop fully audited in the ledger.

**Negative / costs:** generalising the client (FLUX.2 graph, Hunyuan3D path, DNS endpoints, retry
loop) + a `RecoveryController`; the VLM-evaluate step adds latency and a gemma-4 round-trip per attempt;
the Salad add-on API is a dependency of the .48 image that must be present and pinned.

**Neutral:** no new aggregate; the loop sits inside the existing per-object recovery flow.

## Alternatives considered

1. **Keep the one-shot client (no agent loop).** Rejected: cannot detect a bad recovery — defeats
   ADR-010's "correctly placed, plausible" contract.
2. **Stand up a fresh ComfyUI.** Rejected by the owner — reuse the existing .48 instance.
3. **Separate VLM judge ≠ agent reasoner.** Unnecessary now gemma-4 is multimodal (ADR-013 D-013.5);
   would double VRAM.
4. **Bypass Salad, use raw ComfyUI graph API only.** Rejected: lose the model probe/download control the
   Salad add-on provides, which the ADR-013 `hard`-tier lifecycle needs.

## Open Questions

- **Q-014.1** — Salad add-on API: confirm the model-lifecycle endpoints (load/free/unload) it exposes,
  so the ADR-013 `hard` tier drives it rather than `docker stop`.
- **Q-014.2** — Are the FLUX.2 + Hunyuan3D-2.1 **custom nodes** already installed on the .48 ComfyUI,
  or does D-014.1 need to install+pin them? (HF token from ADR-013 Q6 only matters for any gated pull.)

## Related

- **ADR-010** — per-object hull recovery + FLUX inpaint (this ADR verifies its output).
- **ADR-012** — FLUX.2-dev + Hunyuan3D-2.1 + T6 version pins (this ADR pins the .48 nodes/checkpoints).
- **ADR-013** — `v2g-net`, serial lifecycle (`hard` tier via Salad control), unified gemma-4 agent.
- **PRD-v3** — FR-32..FR-34 (ComfyUI update/connect/agent-control), operationalises FR-11/FR-27/FR-28.
- **DDD** — `research/ddd/v3-e2e-extensions.md` new Generative Recovery context + ACL.
- **Code** — `src/pipeline/comfyui_inpainter.py` (the ACL seam), `src/pipeline/workflows/`,
  `hunyuan3d_client.py`.
