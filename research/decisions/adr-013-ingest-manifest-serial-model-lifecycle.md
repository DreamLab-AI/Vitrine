# ADR-013: Pre-Run Ingest Manifest, Serial Model Lifecycle, and Docker-Network Model Mesh

## Status

Proposed — Open Questions resolved 2026-06-04. **Target host: 192.168.2.48** (develop scaffolding on
`agentbox`, deploy to .48). Oversight backend selectable (`[oversight].backend`, **default
`claude_code`** — user logs in inside the container; `gemma_local` optional with GPU-contention
tradeoff, D-013.6). gemma-4-26B-A4B is the multimodal artifact-VLM tool. ComfyUI agent control → ADR-014.

## Context

ADR-009..012 close the workflow-shape and tooling gaps but leave three operational gaps that block a
single-command, reproducible run on the GPU host:

1. **No pre-run input contract.** A run today is configured by hand-editing a JSON `PipelineConfig`
   (`config.py:255-263`, JSON-serialised dataclasses). There is no human-authored manifest that
   carries *what is being captured* (the exhibit), *what objects to decompose* (the SAM3 sub-list),
   *where the source footage lives* (the Drive URL), and *which credentials* the run may use (HF,
   Google Cloud). The object descriptions exist only inside `DecomposeConfig.sam3_concepts` /
   `descriptions`, with no exhibit-level framing and no per-object identity.

2. **No model lifecycle.** Every generative endpoint is a long-lived service hardcoded to a single
   host (`config.py:135-136,152-153` → `http://192.168.2.48:{3001,8189}`; `:171` → `localhost:8188`).
   Nothing loads the *most performant* model per stage and unloads it afterward. A naive run that
   keeps FLUX.2-dev (~32 GB fp8), Hunyuan3D-2.1, SAM3, a VLM, and a text reasoner all resident at once
   exceeds any single-GPU budget on this host.

3. **No threading topology.** The hardcoded `192.168.2.48:port` endpoints couple the pipeline to one
   machine's IP. There is no Docker network giving the orchestrator stable service names, and no use
   of container lifecycle primitives to reclaim VRAM between stages.

### What the audit found on this host (2026-06-04)

- **FLUX.2-dev is real and already staged.** `workspace/ai-3d-fusion/workflows/glass-medallion-flux2-dualgpu.json`
  references `flux2_dev_fp8mixed.safetensors` + `flux2-vae.safetensors` + the FLUX.2 text encoder
  `mistral_3_small_flux2_fp8.safetensors` (bf16 variant also present). The driving script targets
  `http://192.168.2.48:8188` (ComfyUI 0.3.75). This is the owner's "flux2 dev I think we downloaded" —
  confirmed, but it lives on the **192.168.2.48 GPU workstation**, which this `agentbox` container
  cannot route to.
- **The linked `gemma-4-26B-A4B-it-ara-abliterated` is multimodal** (26 B total / ~4 B active MoE,
  Arabic-tuned, abliterated). Architecture `Gemma4ForConditionalGeneration` (`model_type: gemma4`)
  ships a SigLIP vision tower + `mmproj` projector, so it **can** see frames and detect artifacts —
  it serves as *both* the FR-27 visual-analysis model and the FR-28 reasoner (the unified `agent-vlm`,
  D-013.5). A separate true VLM (Qwen2.5-VL / InternVL3) is therefore an **optional fallback**, not a
  requirement. *(An earlier draft of this Context bullet assumed gemma-4 was text-only; that
  assumption was corrected on 2026-06-04 by inspecting the model `config.json` — see D-013.5 and
  Q2-Resolved.)*
- **GPU reality differs from the brief.** `agentbox` sees 1×RTX A6000 (48 GB) + 2×Quadro RTX 6000
  (24 GB). The owner says "dual A6000"; the ai-3d-fusion README cites a single A6000 (48.5 GB) +
  376 GB RAM. The real serial-loading VRAM budget depends on **which host actually runs the
  pipeline** (see Open Questions Q1).
- **Hunyuan3D 2.1 was not found** on any reachable filesystem. The owner believes it is "set up" —
  presumably on .48 alongside ComfyUI.
- **No HF token** is present in any reachable env. Gated pulls (FLUX.2-dev, SAM3, Qwen-VL) are blocked.

## Decision

### D-013.1 — Pre-run TOML ingest manifest (`exhibit.toml`)

Introduce a single human-authored TOML file as the canonical pre-run input. A loader
(`src/pipeline/manifest.py`) parses it, resolves secrets from the environment, and **materialises**
the existing `PipelineConfig` (JSON snapshot for the run record). The manifest is the *source*; the
JSON `PipelineConfig` remains the *runtime artifact* — no dataclass rewrite, only an additive front
door. Authoritative schema:

```toml
schema_version = "1.0"

[exhibit]                                  # project-level identity → run record + USD metadata (ADR-011)
id          = "tate-2026-bronze"
name        = "Bronze Forms — Tate Modern"
venue       = "Tate Modern, London"
date        = "2026-06-01"
curator      = "..."
description = "Single-room bronze sculpture exhibit, mixed natural/spot lighting."

[drive]                                    # source footage location (ADR-009 ingest)
url            = "https://drive.google.com/drive/folders/<id>"
rclone_remote  = "gdrive"                  # reuses the existing rclone_conf secret
recursive      = true

# Object sub-list → DecomposeConfig.sam3_concepts/descriptions, but with stable identity per object.
# "key" objects trigger the ADR-010 hull-recon + per-object FLUX recovery path.
[[objects]]
id          = "obj-001"
name        = "Reclining Figure"
sam3_concept = "large bronze reclining human figure"
description = "Patinated bronze, ~2m, central plinth."
priority    = "key"                        # key | standard
expected_count = 1

[[objects]]
id          = "obj-002"
name        = "Maquette Set"
sam3_concept = "small bronze maquette on shelf"
priority    = "standard"
expected_count = 4

[secrets]                                  # env-var references only; never inline, never persisted to JSON
hf_token            = "env:HF_TOKEN"
gcloud_credentials  = "env:GOOGLE_APPLICATION_CREDENTIALS"   # path to service-account json
gcloud_project      = "dreamlab-v2g"

[pipeline]                                 # optional overrides onto PipelineConfig SOTA defaults (ADR-012)
mesh_backend = "milo"
matcher      = "aliked_lightglue"

[oversight]                                # who oversees the pipeline end-to-end (plans, recovers) — D-013.6
backend = "claude_code"                    # claude_code (DEFAULT) | gemma_local
# claude_code : the in-container Claude Code agent (ttyd :7681). No local GPU cost. Requires the user to
#               log in INSIDE the container once; the session persists in the claude-session OAuth volume.
# gemma_local : the local gemma-4 multimodal model also acts as overseer. On-host, no API key — BUT it
#               contends with FLUX.2 / Hunyuan3D / training for VRAM and must be evicted by the serial
#               lifecycle during heavy GPU stages (or pinned to a dedicated card). See D-013.6.
artifact_vlm = "gemma_local"               # the bulk per-frame artifact triage tool (FR-27) — transient,
                                           # loaded only for the artifact stage then unloaded; independent
                                           # of `backend`. Set to "claude_code" to triage via the overseer.
```

**Secret handling:** `env:NAME` indirection is mandatory for credentials; the loader resolves at parse
time and **strips secrets before** writing the JSON run snapshot, so provenance records never contain
tokens. A missing referenced env-var is a hard, named failure (not a silent empty string as
`InpaintConfig.hf_token=""` is today).

### D-013.2 — Serial model load/unload lifecycle (`ModelLifecycleManager`)

Stages run **serially** behind a lifecycle manager, so peak VRAM = `max(stage)` not `sum(stages)`.
Each stage declares a `ModelSpec(engine, checkpoint, vram_estimate_gb, gpu_affinity, endpoint)`. The
manager, as a context manager around each stage: (a) asserts free-VRAM headroom ≥ `vram_estimate`;
(b) loads/activates the model; (c) yields to the stage; (d) **unloads** and verifies reclamation.

Two unload tiers, selected per `ModelSpec.isolation`:

- **`soft` (default)** — in-process free: ComfyUI `POST /free {unload_models,free_memory}`;
  llama.cpp/vLLM model unload; `torch.cuda.empty_cache()`. Container stays warm. Fast (~seconds).
- **`hard`** — container lifecycle: `docker stop <svc>` after the stage, `docker start` before the
  next consumer. Guarantees full VRAM reclamation (driver-level) at the cost of cold-start latency.
  This is the "more docker primitives" the owner asked for, used selectively for the heavy
  back-to-back stages (FLUX.2 → Hunyuan3D) where soft-free leaves fragmentation.

### D-013.3 — Docker-network model mesh (`v2g-net`)

Replace hardcoded `192.168.2.48:port` / `localhost:port` endpoints with a user-defined bridge network
on the GPU host. Services resolve by DNS name; the orchestrator never hardcodes an IP again:

```
v2g-net (bridge, on the .48 host)
├── comfyui        :8188   existing .48 ComfyUI — FLUX.2-dev (fp8mixed) + Mistral-3 enc + FLUX.2 VAE
│                  :3001   + Salad add-on control-plane API (model probe/download/control) [ADR-014]
│                          + Hunyuan3D-2.1 nodes
├── agent-vlm      :8080   gemma-4-26B-A4B (multimodal) — unified artifact VLM (FR-27) + reasoner (FR-28)
├── gaussian-toolkit       orchestrator — addresses peers as http://comfyui:8188 etc.
├── milo  (sidecar)        device_ids ['1'] — docker exec
└── come  (sidecar)        device_ids ['1'] — docker exec  (gated, non-commercial)
```

Config endpoints become service URLs (`http://comfyui:8188`, Salad control API `http://comfyui:3001`),
overridable by the manifest `[pipeline]` block for the legacy single-host case. The lifecycle manager
(D-013.2 `hard` tier) owns `start`/`stop` of `comfyui` and `agent-vlm` on this network. The agentic
control loop over ComfyUI (both the prompt-graph API and the Salad control API) is specified in
**ADR-014**.

### D-013.4 — Per-stage model selection (most-performant-then-unload)

| Stage | Model (best on .48) | Engine / endpoint | ~VRAM | Unload tier |
|-------|----------------------|-------------------|-------|-------------|
| Frame **artifact VLM** + agent reasoning | **gemma-4-26B-A4B** (multimodal, unified) | `agent-vlm:8080` (llama.cpp mtmd or vLLM) | ~20 GB (Q5_K_M) / ~48 GB (BF16) | soft |
| SfM matching | ALIKED + LightGlue | in-proc torch | < 4 GB | soft |
| SfM fallback | COLMAP (pinned) / VGGT branch | native | mod | n/a |
| 3DGS training | LichtFeld / gsplat | native | scales w/ scene | n/a |
| Decomposition | SAM3 | in-proc torch | ~8 GB | soft |
| **Inpaint / occluded-face** | **FLUX.2-dev fp8mixed** + Mistral-3 enc + FLUX.2 VAE | `comfyui:8188` (existing .48) | ~32 GB | **hard** |
| Hull reconstruction | **Hunyuan3D-2.1** | `comfyui:8188` (existing .48) | ~16 GB | **hard** |
| Mesh extraction | MILo (default) / CoMe | sidecar `device_ids ['1']` | sidecar GPU | n/a |

The artifact VLM and reasoner are the **same** gemma-4 model (D-013.5) — one resident model, not two.
Quant: the repo ships **APEX Q5_K_M (~20 GB)** + BF16 (~48 GB); **Q6_K is not published** and would
have to be produced locally with `llama-quantize` from the BF16 GGUF. On the .48 A6000 (48 GB), Q5_K_M
+ `mmproj` leaves ample headroom; BF16 nearly fills a single 48 GB card (use the second card or vLLM
tensor-parallel if BF16 fidelity is wanted). The agentic ComfyUI control loop is specified in
**ADR-014**.

### D-013.5 — Unified multimodal agent (gemma-4-26B-A4B is vision-capable)

**Corrected 2026-06-04 by config inspection.** `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated` is
**multimodal**, not text-only: architecture `Gemma4ForConditionalGeneration` (`model_type: gemma4`)
with a SigLIP-style vision encoder (27-layer ViT, hidden 1152, patch 16), `image_token_id` /
`video_token_id` / `audio_token_id`, 280 vision soft-tokens per image, and a llama.cpp vision
projector `mmproj-gemma4-f16.gguf`. MoE: 128 experts, top-8, **26 B total / ~4 B active**, 256 K
context.

Therefore **one model serves both roles**: the containerised agent's **artifact VLM** (FR-27,
image-text-to-text artifact detection) *and* its **metadata-fused reasoner** (FR-28). This collapses
the previously planned `vlm:8081` + `reasoner:8080` split into a single `agent-vlm:8080` service,
freeing VRAM and matching the owner's intent of "a visual analysis model that works with the
containerised agent." Qwen2.5-VL / InternVL3 is demoted to an **optional fallback** behind a
capability probe (used only if the gemma-4 mtmd build is unavailable).

**Orchestrator vs. tool (architecture reconciliation).** The pipeline's *overseer* is the in-container
**Claude Code** agent (ttyd :7681; `docs/architecture.md` §"Claude Code as Orchestrator",
`CLAUDE_CONTAINER.md`) — it drives the stateless `stages.py` functions, calls MCP/ComfyUI, and works
around failures end-to-end; there is no hidden state machine. `gemma-4` is **not** a second
orchestrator — it is a **local vision tool the Claude orchestrator calls** for cheap, on-GPU,
high-volume per-frame artifact triage (hundreds of frames without a per-frame Claude-API round-trip).
Claude retains the judgment / accept-veto decisions; gemma-4 does the bulk seeing. "`agent-vlm`" names
this tool service, not the orchestrator.

**Engine caveat (verify):** the repo ships a llama.cpp `--mmproj` usage example, so vision via
llama.cpp/mtmd is claimed working — but `gemma4` vision is new; confirm the local llama.cpp build has
gemma4 mtmd support before pinning (otherwise serve via vLLM/SGLang from the BF16 safetensors).

### D-013.6 — Selectable oversight backend (`[oversight].backend`, default `claude_code`)

The pipeline *overseer* (plans stages, works around failures end-to-end) is selectable in the manifest:

- **`claude_code` (DEFAULT)** — the in-container Claude Code agent (the existing architecture,
  `docs/architecture.md` §"Claude Code as Orchestrator"). **No local GPU cost** — it runs over the
  Anthropic API, leaving all VRAM for FLUX.2 / Hunyuan3D / training. **Operational requirement:** the
  user must **log in to Claude Code inside the container once**; the session persists in the
  `claude-session` OAuth volume (`docker-compose.consolidated.yml:62`). A run started with this backend
  and no logged-in session fails fast with a clear "log in inside the container" message.
- **`gemma_local`** — the local gemma-4 multimodal model *also* serves as overseer. No API key, fully
  on-host — **but it creates GPU-contention tension**: a resident overseer competes with the heavy
  generative/training stages for VRAM. Under this backend the serial lifecycle (D-013.2) must **evict
  the overseer during heavy GPU stages** (FLUX.2 inpaint, Hunyuan3D, 3DGS training) and reload it
  between them, or pin it to a dedicated card (the .48 second GPU) so the main card stays free. This
  trades API independence for scheduling complexity and longer wall-clock from reload cycling.

**Orthogonal knob — `[oversight].artifact_vlm`.** Regardless of overseer, the bulk per-frame artifact
triage (FR-27) is a *transient tool* loaded only for the artifact stage and unloaded after — so
`artifact_vlm = "gemma_local"` is cheap even with `backend = "claude_code"` (the default, recommended
combination: Claude oversees with no GPU cost; gemma-4 does on-GPU bulk seeing only when that stage
runs). Set `artifact_vlm = "claude_code"` to triage via the overseer's own vision instead (higher API
cost on large frame sets).

## Rationale

- A TOML manifest is the right human-authored surface (comments, sections, arrays of tables for the
  object list) while preserving the JSON `PipelineConfig` as the machine run-record — additive, not a
  rewrite.
- Serial lifecycle is the only way the SOTA model set (FLUX.2 + Hunyuan3D-2.1 + VLM + reasoner) fits a
  single-host VRAM budget; it also bounds peak power and makes per-stage model choice free.
- A Docker network removes the last hardcoded IP and makes the heavy stages independently
  start/stoppable — the natural place to express "load the best model, then unload it."
- Pinning gemma as reasoner (not VLM) prevents a silent capability gap where the artifact stage would
  have no eyes.

## Consequences

**Positive:** single-command run from `exhibit.toml`; reproducible secret handling; peak VRAM bounded
to the largest single stage; host-portable endpoints; correct VLM/reasoner separation.

**Negative / costs:** cold-start latency on `hard` unloads (FLUX.2 ↔ Hunyuan3D); a new
`manifest.py` + `model_lifecycle.py` to build and test; the docker network and three model containers
must be stood up (fresh ComfyUI if not reusing .48); gated HF pulls still blocked on a token.

**Neutral:** no new DDD aggregate — this is ingest + infrastructure. Manifest fields map onto existing
`IngestConfig` / `DecomposeConfig` / `InpaintConfig` / `Hunyuan3DConfig`.

## Alternatives considered

1. **Keep JSON-only config, no manifest.** Rejected: no place for exhibit identity or per-object
   identity; secrets leak into the run snapshot.
2. **Keep all models resident (no lifecycle).** Rejected: exceeds single-GPU VRAM; defeats
   "most performant model per stage."
3. **Reuse the remote .48 ComfyUI as-is over the LAN.** Viable as a stopgap but keeps the hardcoded
   IP and gives no VRAM-reclamation control; folded into Q1/Q4 rather than rejected.
4. **Use a separate text-only reasoner + a distinct VLM (two models).** Rejected after confirming
   gemma-4-26B-A4B is multimodal (D-013.5): one unified `agent-vlm` serves both roles, halving VRAM.
   Qwen2.5-VL / InternVL3 is retained only as an optional fallback behind a capability probe.

## Open Questions — Resolved 2026-06-04

- **Q1 — Host & VRAM. RESOLVED.** Target deployment is **192.168.2.48**; the pipeline moves there once
  enough scaffolding is developed with the agents on `agentbox`. Development now happens on `agentbox`
  (no route to .48); the binding VRAM budget is .48's GPUs. *Implication:* build host-portable code
  here; do not hardcode agentbox GPUs.
- **Q2 — VLM vs reasoner. RESOLVED (assumption corrected).** gemma-4-26B-A4B **is** multimodal
  (D-013.5) → one unified `agent-vlm` model serves both artifact detection (FR-27) and reasoning
  (FR-28). No separate Qwen2.5-VL needed (optional fallback only).
- **Q3 — Hunyuan3D 2.1. RESOLVED.** Lives on the **.48 machine** alongside ComfyUI.
- **Q4 — ComfyUI reuse or fresh. RESOLVED.** **Reuse the existing .48 ComfyUI**; wire it into
  `v2g-net` and have the agent control it. The "Salad" surface the existing client targets is an
  **add-on control-plane API inside that container** (more control: model probe/download), not a cloud
  service. Updating + connecting + agent-controlling that instance is specified in **ADR-014**.
- **Q5 — Default unload tier. OPEN (default stands).** Proceeding with `soft` default + `hard` for
  FLUX.2↔Hunyuan3D unless told otherwise.
- **Q6 — HF token. OPEN.** Still required for gated pulls (SAM3; FLUX.2-dev/Hunyuan3D already staged on
  .48). Non-blocking for the .48-resident models.
- **Q7 — gemma quant. RESOLVED with caveat.** Target **Q5_K_M (APEX, ~20 GB)** — the requested **Q6_K
  is not published** in the repo and would need a local `llama-quantize` pass from the BF16 GGUF. Q5_K_M
  + `mmproj-gemma4-f16.gguf` fits the .48 A6000 with headroom and **is image-capable** (vision projector
  shipped). Verify the local llama.cpp build has `gemma4` mtmd support; else serve BF16 via vLLM.

## Related

- **ADR-009** — per-video ingest + capture metadata (manifest feeds this; `vlm` sidecar block).
- **ADR-010** — key-item hull reconstruction (manifest `priority="key"` triggers this path; FLUX.2 recovery).
- **ADR-011** — USD metadata enrichment (exhibit/object identity flows to USD).
- **ADR-012** — SOTA tooling (FLUX.2-dev amended from Kontext; D-012.5 VLM artifact analysis → D-013.4/5).
- **PRD-v3** — FR-27 (VLM artifact stage), FR-28 (metadata-aware scaffolding), `†` model-pull FRs.
- **config.py** — `PipelineConfig` (materialisation target), hardcoded endpoints `:135-136,152-153,171`.
