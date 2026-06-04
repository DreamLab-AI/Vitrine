# Vitrine Onboarding (Proposed)

> **Status: Proposed — not yet built.** Vitrine Onboarding is designed in ADR-015 (2026-06-04). The
> description below is the intended design; none of this tooling exists in the current codebase.

Vitrine Onboarding is the user-facing entry point to the Vitrine pipeline. It is a **schema-driven
web wizard** that takes a user from zero to a fully-provisioned, agent-ready pipeline run — without
editing a file or touching the command line. Its output is a valid [`exhibit.toml` manifest](architecture/v3-pipeline.md#exhibittoml-manifest) (ADR-013) that the Claude Code orchestrator consumes directly.

## What it is

- **Frontend:** frameworkless vanilla HTML/CSS/JavaScript — no framework, no build step.
- **Backend:** a single statically-linked **Rust/Axum** binary called `vitrine-setup`, listening on an ephemeral `127.0.0.1:0` port (localhost-only, no external exposure). It embeds the frontend via `rust-embed`.
- **Schema-driven forms:** a JSON Schema (`schema/exhibit.toml.schema.json`) is the single source of truth; the wizard walks it to generate nested form controls automatically. Adding a new manifest field requires only a schema edit.
- **TOML round-trip:** `toml_edit` parses, validates, and re-serialises the manifest, preserving comments and key order.
- **Re-entrant, no history:** on start the backend loads the existing `exhibit.toml` (if present) and populates all fields. There is exactly one active manifest; there is no list of past projects and no archive. Restarting the wizard resumes where you left off.

This is the same pattern as the `agentbox/setup/` tool (agentbox ADR-024), proven in production. ADR-015, section D-015.2.

## Wizard steps

### Step 1 — Exhibit

Name and describe the project. These fields populate the `[exhibit]` block in the manifest and flow
directly into USD scene metadata (ADR-011):

| Field | Manifest key | Notes |
|-------|-------------|-------|
| Exhibit ID | `exhibit.id` | Slug used in output paths and run records |
| Display name | `exhibit.name` | Human-readable title |
| Venue | `exhibit.venue` | Location (e.g. "Tate Modern, London") |
| Date | `exhibit.date` | Capture date (ISO 8601) |
| Curator | `exhibit.curator` | Optional contact |
| Description | `exhibit.description` | Free text; the agent uses this for context |

### Step 2 — Objects of Interest

Add the objects the pipeline will decompose and recover. Each becomes an `[[objects]]` entry:

| Field | Manifest key | Notes |
|-------|-------------|-------|
| Object ID | `objects[n].id` | Stable slug (e.g. `obj-001`) |
| Name | `objects[n].name` | Human label |
| SAM3 concept | `objects[n].sam3_concept` | Free-text description passed to SAM3 (e.g. "large bronze reclining human figure") |
| Priority | `objects[n].priority` | `key` or `standard`. Key objects trigger the full hull-reconstruction + FLUX.2 recovery path (ADR-010/014). |
| Expected count | `objects[n].expected_count` | Optional; the agent uses this to verify segmentation coverage. |

You can add, reorder, and remove objects freely. The wizard preserves entries already in the manifest.

### Step 3 — Hardware and Model Selection

The wizard calls a `/api/hardware` endpoint that probes the host:

- GPU count and per-GPU VRAM (via `nvidia-smi`)
- System RAM and available disk

It maps this against the ADR-013 per-stage model table and **recommends** the most-performant
model/quant that fits. You can accept the recommendation or override any field. Choices are written
to a `[models]` block in the manifest:

```toml
[models]                                   # written by Vitrine Onboarding
inpaint             = "flux2-dev-fp8mixed" # FLUX.2-dev | flux1-fill (fallback)
hull                = "hunyuan3d-2.1"      # | hunyuan3d-2.0 (fallback)
matcher             = "aliked_lightglue"   # | sift (fallback)
mesh                = "milo"               # | tsdf | come | gaussianwrap
artifact_vlm_quant  = "Q5_K_M"            # gemma-4 quantisation

[models.vram_plan]                         # informational; from /api/hardware
gpu0_total_gb          = 48
serial_peak_estimate_gb = 32               # max single stage (ADR-013)
```

Example recommendations by VRAM:

| GPU VRAM | Inpaint | Hull | Artifact VLM |
|----------|---------|------|-------------|
| 48 GB (A6000) | FLUX.2-dev fp8mixed | Hunyuan3D-2.1 | gemma-4 Q5_K_M (~20 GB) |
| 24 GB | FLUX.1-Fill fallback | Hunyuan3D-2.0 | gemma-4 Q5_K_M + hard unload |
| < 16 GB | FLUX.1-Fill fallback | TSDF mesh only | claude_code artifact_vlm |

ADR-015, section D-015.3.

### Step 4 — Secrets and Login

Credentials are handled with **server-side containment**: nothing ever enters the browser's JavaScript
memory, the TOML manifest, or a git-tracked file.

**HuggingFace token:**
- Pasted in the wizard form.
- Sent via `POST` to `vitrine-setup`.
- Written as a host keyring entry or Docker secret.
- Referenced in the manifest only as `hf_token = "env:HF_TOKEN"`.
- Displayed masked; never echoed back in full.

**Google Drive login:**
- The wizard launches Google's OAuth consent flow in the browser (Drive read + write scope).
- The **refresh token is stored server-side** by `vitrine-setup` (keyring/Docker secret), never in the browser or in the TOML.
- The manifest references it as `gcloud_credentials = "env:GOOGLE_APPLICATION_CREDENTIALS"` or an rclone remote name.
- The Rust backend proxies Drive calls server-side (`/api/proxy` pattern), so the browser never holds a token.

The `[secrets]` block in the manifest therefore contains only `env:` references:

```toml
[secrets]
hf_token           = "env:HF_TOKEN"
gcloud_credentials = "env:GOOGLE_APPLICATION_CREDENTIALS"
gcloud_project     = "dreamlab-v2g"
```

ADR-015, section D-015.4; ADR-013 secret-indirection rule.

**Google OAuth prerequisite (Open Question ADR-015 Q-015.1):** a DreamLab Google Cloud OAuth
client (consent screen, Drive scope) must be registered before the browser-login step can work.

### Step 5 — Provision and Hand-off

The final step is split into two distinct actors.

#### Vitrine Onboarding — deterministic provisioning

`vitrine-setup` performs reproducible, idempotent provisioning:

1. Validates the completed manifest against the JSON Schema.
2. Probes the `.48` ComfyUI via the **Salad add-on control API** (`comfyui:3001`) to ensure required checkpoints are present (FLUX.2-dev fp8mixed, Hunyuan3D-2.1 nodes). Missing checkpoints trigger a controlled download; unresolvable checkpoints fall back to declared alternatives.
3. Brings up `v2g-net` and starts the `agent-vlm` container with the selected gemma-4 quant.
4. Verifies readiness of all `v2g-net` services.
5. Writes `provision.status = "ready"` into the manifest.
6. Emits a hand-off event.

Progress is streamed to the browser UI. This step can take minutes (model downloads, container starts). ADR-015, section D-015.5.

#### The Claude Code overseer — interpretive scaffolding

On receiving the hand-off event, the **in-container Claude Code agent** (ttyd :7681) reads the
finalised manifest and performs the judgement work that provisioning cannot:

- Translates the free-text `[[objects]]` descriptions into SAM3 concept candidates and per-object recovery plans.
- Plans the run sequence, decides which mesh backend to invoke.
- Mediates failures end-to-end without a hidden state machine.

**The boundary:** setup makes the system runnable; the agent decides how to run it.

The `[oversight].backend = "claude_code"` setting (the default) requires the user to have logged in
to Claude Code inside the container at least once. The session persists in the `claude-session` OAuth
volume. A run with no logged-in session fails fast with a clear message. ADR-013, section D-013.6.

## Drive write-back

Finished artifacts (USD scene, `.ksplat`, per-object meshes, the run report) are uploaded back to
the **same Google Drive folder** the source video came from (or a `vitrine-output/` subfolder).
This requires the Drive write scope in the Step 4 OAuth grant. Configure in `[drive]`:

```toml
[drive]
url              = "https://drive.google.com/drive/folders/<id>"
rclone_remote    = "gdrive"
writeback        = true
writeback_subdir = "vitrine-output"   # optional; defaults to the source folder
```

ADR-015, section D-015.6.

## Re-entrant behaviour

Restarting `vitrine-setup` re-loads the existing `exhibit.toml` and populates all wizard fields.
Provisioned items (downloaded models, `v2g-net` state) are checked idempotently — already-present
resources are not re-downloaded. This means partial runs and interrupted provisioning steps can be
resumed without starting from scratch.

## Related design documents

- `research/decisions/adr-015-vitrine-web-onboarding.md` — authoritative design specification.
- `research/decisions/adr-013-ingest-manifest-serial-model-lifecycle.md` — `exhibit.toml` schema, lifecycle, oversight backend.
- `research/decisions/adr-014-agent-controlled-comfyui-integration.md` — Salad control API, ComfyUI provisioning.
- [v3 Pipeline Design](architecture/v3-pipeline.md) — `v2g-net` diagram and manifest-to-handoff flow.
- [Architecture (v3 section)](architecture.md#v3-end-to-end-architecture-proposed) — orchestrator/tool relationship, serial lifecycle.
