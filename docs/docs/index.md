---
sidebar_position: 1
---

# Getting Started — Vitrine / LichtFeld Studio

> **Rename note (2026-06-04):** The pipeline product is renamed from **Video-to-Gaussian** to
> **Vitrine** (a museum display case; pairs with the upstream LichtFeld Studio engine). CLI/package
> id: `vitrine`. The web setup tool is **Vitrine Onboarding**. This rename is documentation-level
> only — code identifiers still say `video2splat` / `gaussian-toolkit`; a full code/repo rename is a
> separate scheduled follow-up (ADR-015).

## v3 Architecture (Proposed)

A redesigned end-to-end pipeline is under active design. It is not yet built. Key additions:

- **`exhibit.toml` manifest** — a single human-authored TOML file is the pipeline's one input, carrying exhibit identity, objects of interest, Drive source, secrets (as `env:` references only), and pipeline/oversight overrides. Decided in ADR-013.
- **Claude Code as orchestrator** — the in-container Claude Code agent drives the stateless pipeline stages; there is no hidden state machine. gemma-4-26B-A4B is a local vision tool it calls for bulk per-frame artifact triage, not a second orchestrator. ADR-013.
- **Serial model lifecycle** — a `ModelLifecycleManager` loads and unloads models stage-by-stage, bounding peak VRAM to the largest single stage rather than the sum of all stages. ADR-013.
- **`v2g-net` Docker mesh** — service-DNS endpoints replace hardcoded IP addresses; the orchestrator addresses peers as `http://comfyui:8188` etc. ADR-013.
- **Agent-controlled ComfyUI recovery** — the existing .48 ComfyUI (FLUX.2-dev + Hunyuan3D-2.1) is updated/pinned via the Salad add-on control API and driven by a `RecoveryController` plan→submit→VLM-evaluate→decide loop. ADR-014.
- **Vitrine Onboarding** — a frameworkless vanilla-JS + Rust/Axum setup wizard that edits the manifest, probes hardware for model selection, contains secrets server-side, and hands off to the Claude Code overseer. ADR-015.

See [v3 Pipeline Design](../architecture/v3-pipeline.md) and [Vitrine Onboarding](../onboarding.md) for details.

---