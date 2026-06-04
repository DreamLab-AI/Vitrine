# Work Order — SOTA Tooling Modernisation (in-container orchestrator)

**Status:** Active · **Created:** 2026-06-04 · **Owner:** in-container agentic
orchestrator (`CLAUDE_CONTAINER.md` → `/opt/gaussian-toolkit/CLAUDE.md`).
**Operationalises:** ADR-012 (SOTA tooling modernisation) + the native-USD /
plugin shift (T7) + the v3-e2e-closure PRD. **Supersedes assumptions in:**
ADR-013 Q3 (Hunyuan3D-2.1 location — now resolved, see §0).

This is a standing directive from the project owner: **use LichtFeld where
possible, always run clean SOTA, always check model versions and update, and
make these updates happen through research + implementation.** The host (agentbox)
has no GPU and does not perform the heavy integration — the in-container
orchestrator does, because it has the GPUs, the staged weights, the local
ComfyUI, and the LichtFeld MCP surface.

## 0. Known-good local assets (do NOT re-download)

Confirmed present on the host drive (`~` = `/home/john`):

| Asset | Path | Use |
|-------|------|-----|
| FLUX.2-dev | `~/comfyui-models-staging/diffusion_models/flux2_dev_fp8mixed.safetensors` | inpaint upgrade (item 1) |
| FLUX.2 VAE | `~/comfyui-models-staging/vae/flux2-vae.safetensors` | inpaint |
| FLUX.2 text encoder | `~/comfyui-models-staging/text_encoders/mistral_3_small_flux2_{fp8,bf16}.safetensors` | inpaint |
| FLUX.2 Turbo LoRAs | `~/comfyui-models-staging/loras/Flux2TurboComfyv2.safetensors`, `Flux_2-Turbo-LoRA_comfyui.safetensors` | speed |
| Hunyuan3D 2.1 | `~/comfyui-api-data/ComfyUI/comfy/ldm/hunyuan3dv2_1` | hull upgrade (item 2) |
| Hunyuan3D 2 MV | `~/comfyui-models-staging/hunyuan3d/hunyuan3d-dit-v2-mv` | hull fallback |
| SAM3D node + weights | `~/comfyui-api-data/ComfyUI/custom_nodes/comfyui-sam3dobjects`, `~/comfyui-models-staging/sam3d/` | hull fallback (item 3) |
| Local ComfyUI | `~/comfyui-api-data/ComfyUI` (a full install) | dev test target |

**Mounting:** the container only mounts the (empty) repo `./models-staging` →
`/models-staging:ro`. **First task:** make the host staging trees reachable in
the container — add bind mounts (or point `MODELS_STAGING_DIR`) for
`~/comfyui-models-staging` and the ComfyUI model dirs, then register them via
ComfyUI `extra_model_paths.yaml`.

## 1. The eight upgrades

Each: **research the current latest → check native/plugin coverage → implement
behind a capability probe → validate on a test job → pin → report.**

### Item 1 — Inpaint: FLUX.1-Fill → FLUX.2-dev  (ADR-012 T4 / FR-23)
- Current: `comfyui_inpainter.py:86-118` loads `flux1-fill-dev.safetensors`;
  `config.py InpaintConfig.model="flux-fill"`.
- Target: FLUX.2-dev using the staged fileset (§0). Add `workflows/flux2_inpaint.json`
  (FLUX.2 needs the Mistral-3 text encoder + FLUX.2 VAE; optional Turbo LoRA).
  Generalise `ensure_flux_*_models()` → `ensure_flux2_models()`. Probe → fallback
  to FLUX.1-Fill if FLUX.2 nodes/weights absent.

### Item 2 — Image-to-3D hull: Hunyuan3D 2.0 → 2.1  (ADR-012 T3 / FR-22)
- Current: `hunyuan3d_client.py:59-100` pins `tencent/Hunyuan3D-2` / `-2mv`.
- Target: Hunyuan3D 2.1 via the local `hunyuan3dv2_1` ldm + the `comfyui-sam3dobjects`
  node. Update model IDs/checkpoints; probe → degrade to 2.0.

### Item 3 — Wire the orphaned SAM3D fallback  (FR-22)
- Current: `sam3d_client.py` (~600 LOC, functional) is imported by no `src/`
  module; `config.hunyuan3d.fallback_sam3d=True` is dead.
- Target: call `sam3d_client` from the Hunyuan path when MV/SV fail. Node +
  weights are present (§0).

### Item 4 — COLMAP features: SIFT → ALIKED+LightGlue  (ADR-012 T2 / FR-21)
- Current: `config.py ReconstructConfig` SIFT + exhaustive matcher.
- Target: ALIKED feature + LightGlue matcher for indoor presets. **Prefer the
  LichtFeld COLMAP plugin** (`Lichtfeld-COLMAP-Plugin` / `lichtfeld-360-plugin`)
  if available; else build COLMAP with the GPU wheels. SIFT fallback retained.

### Item 5 — USD: custom Python → native LichtFeld export  (ADR-011 T7 / ADR-002)
- **USD is now native in LichtFeld Studio (v0.5.1+).** DONE (host-side wiring):
  `mcp_client.export_usd/sog/spz/html()` added; `ExportConfig.prefer_native_usd=True`;
  `stages._export_native_usd()` exports `scene_native.usd` best-effort (MCP ping →
  export → checkpoint-reload retry → fall back to the custom assembler, no regression).
- **REMAINING (needs live MCP — in-container): the customData-parity probe.** Run a
  real export and check whether native `scene.export_usd` can carry the ADR-011
  `v2g:*` per-object customData + `{Gaussian|Mesh}` variants + camera prims. If YES →
  retire `scripts/assemble_usd_scene.py` + `usd_assembler.py`. If NO → native = base
  scene, custom = composition/metadata layer. Resolves ADR-002 "USD deprecation blocked".

### Item 6 — Adopt LichtFeld plugins + native flags  (ADR-012 T7 / FR-26)
- `splat_ready` is referenced (`stages.py` `_run_colmap_direct` guard) but **not
  installed** → always falls through. Install + enable: splat_ready, PPISP,
  bilateral-grid, 3DGUT, pose-opt, ImprovedGS+. Wire the native flags into
  `train`. Prefer MCP/native over subprocess where a native path exists.

### Item 7 — Version pinning  (ADR-012 T6 / FR-25)
- Dockerfiles clone at HEAD: ComfyUI, Hunyuan3D, COLMAP, MILo, CoMe,
  GaussianWrapping, SAM3, `gaussian-toolkit`, and `pip install gsplat`.
- Pin every one to a tag/commit/checkpoint and record a lock manifest. No HEAD,
  no floating `latest`.

### Item 8 — Endpoint reconciliation  (ADR-013 FR-31 / SEC-04)
- Hardcoded `192.168.2.48:8189`(ComfyUI)/`:3001`(Salad) in ~11 sites:
  `config.py:135-137,152-153,171`; `comfyui_inpainter.py:282-284`;
  `hunyuan3d_client.py:192-193`; `sam3d_client.py:72-73`; `person_remover.py`.
  Plus an in-container ComfyUI on `:8188` (supervisord). Decide ONE source
  (Salad-tested workflows vs local ComfyUI) and route via config/manifest;
  remove literal IPs.

### Item 9 — Enable CoMe as the default mesh backend (project pivot, 2026-06-04)
- Decision: CoMe is the chosen GS-surface backend (config default `come`).
- Build with `INSTALL_COME=1` (flip the `docker-compose.consolidated.yml` default
  + `docker/Dockerfile.come`); run the `come` sidecar on GPU 1.
- **Verify `come_extractor.py` CLI flags + the two-phase train/extract invocation
  against the released CoMe repo** (ADR-004 follow-up — flags were inferred).
- Licence: CoMe ships no permissive LICENSE (Inria/MPII non-commercial) — confirm
  the deployment is non-commercial before enabling.
- Without this, the `come` default silently degrades to TSDF (non-SOTA).

### Item 10 — Refresh to current SOTA 3D model/workflow for 48 GB Ada
- Hardware: 2× RTX 6000 Ada (48 GB, sm_89, CUDA 12.x), serial model lifecycle.
- Re-verify, as of run time, the SOTA for: (a) image/multiview→3D mesh hull
  (currently Hunyuan3D-2.0/2.1), (b) generative recovery/inpaint (currently
  FLUX.x), (c) GS-surface meshing (CoMe) — "things have moved on", so confirm
  none is superseded. Prefer open-weights + a ComfyUI node, textured output,
  fits 48 GB. Pin whatever wins. (Host-side research seeds this; see §below.)

## 2. Validation gates (per item, before marking done)

Run a real test job end-to-end (`ingest → … → assemble_usd → validate`) and
confirm no regression against the `CLAUDE_CONTAINER.md` Quality Targets:
COLMAP ≥70% registration, training >10 MB PLY, ≥2 objects or full-scene,
per-object mesh >5K verts, final USD with room + objects + materials. For model
upgrades, A/B the new vs old output on the same input and keep the probe
fallback working.

## 3. Reporting

After each item: report via the REST API, record the pinned version/commit in
the lock manifest, and note in `docs/engineering-log.md` (the v3 entry that does
not yet exist — create it). Keep this file's checkboxes current:

- [ ] 0. Mount host staging weights into the container
- [ ] 1. FLUX.2-dev inpaint
- [ ] 2. Hunyuan3D-2.1 hull
- [ ] 3. SAM3D fallback wired
- [ ] 4. ALIKED+LightGlue (prefer LichtFeld plugin)
- [ ] 5. Native LichtFeld USD export
- [ ] 6. LichtFeld plugins + native flags
- [ ] 7. Version pins + lock manifest
- [ ] 8. Endpoint reconciliation
