# ADR-001: Pipeline Architecture for Video-to-Scene Reconstruction

## Status

Accepted. Amended 2026-06-05 to record the **current** architecture.

> **Supersedes the original v1 proposal.** The first version of this ADR adopted a
> Gaussian-Grouping co-training + SuGaR-meshing + FLUX-background-inpaint pipeline.
> None of those three is used: **Gaussian Grouping** was replaced by post-hoc
> **SAM3** concept segmentation; **SuGaR** by the pluggable mesh backends (ADR-003:
> TSDF / MILo / CoMe / GaussianWrapping) plus image-to-3D hulls for key objects; and
> background-inpaint-and-retrain by per-object generative *recovery* of unseen faces.
> This record now documents what the pipeline actually is.

## Context

We convert a hand-held walkthrough video of an exhibit into a structured USD scene
with individually addressable, correctly-placed 3D objects. The existing stack:
LichtFeld Studio (3DGS training, native USD export, 70+ MCP tools), COLMAP (SfM),
ComfyUI (FLUX.2 / image-to-3D models), SAM3 (concept segmentation), Blender.

## Decision

**Reconstruct-then-decompose**, orchestrated by the in-container agent over the
LichtFeld MCP surface (no hidden state machine — see ADR-013). One run is:

```
Drive/video ingest → frame extraction + per-image metadata sidecar (ADR-009)
  → COLMAP SfM (ALIKED+LightGlue; SIFT fallback)              (ADR-012)
  → 3DGS training (LichtFeld: ImprovedGS+ / MRNF / MCMC)
  → .ksplat web delivery (splat-transform)                    (ADR-006)
  → SAM3 concept segmentation → key-item ranking              (ADR-010)
  → per-object hull: orbit render → FLUX.2 recovery → TRELLIS.2 / Hunyuan3D-2.1 (ADR-010/012/014)
  → environment mesh (CoMe default; MILo / GaussianWrapping / TSDF) (ADR-003/004/005)
  → USD scene graph (native LichtFeld export; v2g:* metadata)  (ADR-011)
```

Driven by a pre-run `exhibit.toml` manifest with `env:`-indirected secrets (ADR-013)
and validated by a SOTA model "idiot-check" before any run (ADR-012/013).

## Rationale

- **Reconstruct-then-decompose, not co-train.** A full Gaussian reconstruction is the
  shared reference for per-object pose, mask projection and recovery; it reuses the
  unmodified LichtFeld training path rather than a bespoke co-training fork.
- **SAM3 concept segmentation.** Text/visual concept prompts (4M concepts) identify
  exhibit objects directly; key-item ranking (`min_object_gaussians` + keyness) selects
  which enter the per-object path.
- **Image-to-3D hulls for key objects.** Orbiting a segmented object, recovering its
  unseen faces with FLUX.2, and lifting it to a watertight textured hull (TRELLIS.2
  primary, Hunyuan3D-2.1 fallback) gives a clean, placeable mesh — preferable to
  surface-extracting a partially-observed object subset.
- **CoMe as the default environment-mesh backend** (best indoor F1, ~3× faster than
  MILo); the ADR-003 interface keeps MILo / GaussianWrapping / TSDF interchangeable.
- **Native USD assembly.** LichtFeld's native `scene.export_usd` (v0.5.1+) is the base
  exporter; the custom assembler remains only for the `v2g:*` metadata / multi-object
  composition not yet covered natively (ADR-011).

## Consequences

- Dual representation (Gaussian | Mesh variant set) per object in the USD output.
- Mixed runtime (C++ core + Python pipeline + sidecar containers per CUDA ABI).
- Several SOTA models carry non-commercial licences (CoMe, FLUX.2-dev, SAM3D); the
  research/non-commercial posture is the default and is enforced by the idiot-check.
  A commercial build must swap them (CoMe→PGSR, FLUX.2→Qwen-Image-Edit).

## Related Decisions

- ADR-002: Upstream sync strategy (v0.5.2, one-way pull).
- ADR-003: Pluggable mesh-extraction backend interface.
- ADR-004 / ADR-005: CoMe (default) and GaussianWrapping mesh backends.
- ADR-006 / ADR-007: `.ksplat` web delivery; Fibonacci frame selection.
- ADR-009–ADR-011: Per-video ingest + provenance; key-item ranking + hull recovery; USD metadata.
- ADR-012: SOTA tooling (mesh/SfM/hull/inpaint models, version pins, licence posture).
- ADR-013–ADR-015: Manifest + serial lifecycle + `v2g-net`; agent-controlled ComfyUI; Vitrine onboarding.
