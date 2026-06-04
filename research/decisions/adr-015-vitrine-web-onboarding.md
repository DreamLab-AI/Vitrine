# ADR-015: "Vitrine" — Web Onboarding & Setup Tool (schema-driven manifest editor)

## Status

Proposed (2026-06-04). Depends on ADR-013 (`exhibit.toml` manifest, `v2g-net`, lifecycle, oversight
backend) and ADR-014 (ComfyUI provisioning). Models the **agentbox setup tool** design pattern.

## Context

The pipeline now has a single human-authored input — the ADR-013 `exhibit.toml` manifest — but no
ergonomic way to author it. The owner wants a **web onboarding page** that lets a user, with no
file-editing:

1. **Name & describe** the project (exhibit) and the **objects of interest** the agent will mediate.
2. **Select models by hardware** — probe GPUs, recommend the most-performant model/quant per stage.
3. **Paste tokens / log in via the browser** (HF token; Google for Drive read **and** write-back).
4. Have the setup **download & integrate** the required models and **scaffold** the project — using
   the **internal agent** for the interpretive part (distinct from deterministic setup).
5. **Persist the run in the TOML**; **re-entrant** — restarting reads the TOML, populates editable
   fields. **No history** of past projects (a single active manifest).
6. **Write outputs back to the Google Drive** folder the ingested video came from.

It also wants the project **renamed** — "Video-to-Gaussian" is rejected.

### The design to emulate (agentbox setup tool)

`/home/devuser/workspace/project/agentbox/setup/` is a proven pattern (agentbox ADR-024):

- **Frontend:** frameworkless vanilla HTML/CSS/JS (`frontend/dist/{index.html,app.js,style.css}`),
  no build step, DreamLab CSS design tokens.
- **Backend:** a single statically-linked **Rust + Axum** binary (`server/src/main.rs`), ephemeral
  `127.0.0.1:0` port. Endpoints `/api/config` (GET/POST), `/api/validate`, `/api/proxy/{path}`.
- **Schema-driven forms:** a **JSON Schema** (`schema/agentbox.toml.schema.json`) is the single source
  of truth; `renderFields()` walks it to generate nested form controls (enum→`<select>`, bool→toggle,
  number→constrained input). New fields = schema edit only.
- **Round-trip TOML:** `toml_edit::DocumentMut` parses→validates→re-serialises, preserving comments and
  key order. Server-side validation via `/api/validate`.
- **Secret containment:** the browser never sees credentials; the Rust server reads keys from
  filesystem/env and injects `Authorization: Bearer` on a server-side **proxy** (`/api/proxy/*`).
- It is a **config wizard, not a build orchestrator** — it edits the manifest; build is external.

## Decision

### D-015.1 — Rename the project to **Vitrine**

A *vitrine* is a museum display case (German/French/English) — it pairs with the upstream **LichtFeld
Studio** and names the product cleanly: agentic volumetric capture that turns a handheld walkthrough of
an exhibit into a structured, object-resolved 3D scene. CLI/package id: `vitrine`. The web tool is
**Vitrine Onboarding**. *Scope:* new docs adopt the name now; a full code/repo/remote rename is a
**separate, explicitly-scheduled follow-up** (high blast radius — not done silently here).

### D-015.2 — Onboarding tool = schema-driven `exhibit.toml` editor (agentbox pattern)

Build **Vitrine Onboarding** as a near-clone of the agentbox setup tool:

- **Frontend:** frameworkless vanilla JS/CSS, DreamLab tokens, a **wizard/stepper** (Exhibit →
  Objects → Hardware/Models → Secrets/Login → Provision & Hand-off).
- **Backend:** Rust + Axum single static binary (`vitrine-setup`), ephemeral localhost port; embeds
  the frontend via `rust-embed`.
- **Schema source of truth:** a JSON Schema of the ADR-013 manifest
  (`schema/exhibit.toml.schema.json`) drives form generation; `toml_edit` round-trips the file so
  comments/order survive edits. **Its output *is* the ADR-013 `exhibit.toml`** (extended below).
- **Re-entrant, no history:** on start the backend loads the existing `exhibit.toml` (if present) and
  populates fields; the user edits and re-saves the **same** file. There is exactly one active
  manifest — no past-project list, no archive (matches the owner's "no history required").

### D-015.3 — Hardware-aware model selection

A `/api/hardware` endpoint probes the host (GPU count, per-GPU VRAM via `nvidia-smi`; RAM; disk). The
wizard maps this to the ADR-013 D-013.4 per-stage table and **recommends** the most-performant
model/quant that fits (e.g. on a 48 GB A6000: FLUX.2-dev fp8mixed + Hunyuan3D-2.1 + gemma-4 Q5_K_M; on
24 GB: smaller quants / `hard` unload everywhere). The user accepts or overrides; the choices write to
a new manifest block:

```toml
[models]                                   # hardware-selected; written by Vitrine Onboarding (D-015.3)
inpaint   = "flux2-dev-fp8mixed"           # FLUX.2-dev | flux1-fill (fallback)
hull      = "hunyuan3d-2.1"                # | hunyuan3d-2.0 (fallback)
matcher   = "aliked_lightglue"             # | sift (fallback)
mesh      = "milo"                         # | tsdf | come | gaussianwrap
artifact_vlm_quant = "Q5_K_M"              # gemma-4 quant (Q6_K → local re-quant; see ADR-013 Q7)
[models.vram_plan]                         # informational; from /api/hardware
gpu0_total_gb = 48
serial_peak_estimate_gb = 32               # max single stage (ADR-013 D-013.2)
```

### D-015.4 — Secret entry & browser-based login (server-side containment)

Mirror agentbox's containment — **no credential ever enters the browser JS or the TOML**:

- **HF token:** pasted in the wizard → `POST` to the backend → written as a **Docker secret / host
  keyring entry**, referenced from the manifest only as `hf_token = "env:HF_TOKEN"` (ADR-013 secret
  indirection). The field shows masked, never echoed back in full.
- **Google login (browser OAuth):** the wizard launches Google's OAuth consent in the browser for the
  Drive scope; the **refresh token is stored server-side** (keyring/secret), never in the browser or
  TOML. The manifest references it as `gcloud_credentials = "env:GOOGLE_APPLICATION_CREDENTIALS"` or an
  rclone remote name. The Rust backend holds the token and proxies Drive calls (agentbox `/api/proxy`
  pattern). This is the "keys or a browser-based login which the web onboarding can handle."

### D-015.5 — Setup vs. internal agent (the hand-off boundary)

Two distinct actors, explicitly bounded (answers the owner's "scaffold … using the internal agent
(whatever that means vs the setup)"):

- **Vitrine Onboarding (setup) — deterministic provisioning.** Probes hardware, edits/validates the
  manifest, **downloads & integrates models** (HF pulls via the stored token; ComfyUI checkpoint/node
  ensure+pin via the ADR-014 Salad control API), brings up `v2g-net`, and verifies readiness. No
  interpretation, fully scriptable, idempotent. *Build/provision IS triggered here* — unlike agentbox,
  which only edits config. Progress streamed to the UI (agentbox polling+WebSocket hybrid).
- **Internal agent (the in-container Claude Code overseer, ADR-013 D-013.6) — interpretive
  scaffolding.** On hand-off it reads the finalised manifest and does the judgement work: turning the
  free-text **objects of interest** into SAM3 concept candidates + per-object recovery plans (ADR-010),
  planning the run, and mediating failures end-to-end. *"There is no hidden state machine"
  (`docs/architecture.md`).*

The boundary: **setup makes the system runnable; the agent decides how to run it.** Setup ends by
writing `provision.status = "ready"` into the manifest and emitting a hand-off event the overseer
consumes.

### D-015.6 — Output write-back to the source Google Drive

The manifest's `[drive]` block gains a write-back target so finished artifacts (USD scene, ksplat,
per-object meshes, the run report) are uploaded to the **same Drive folder** the source video came
from (or a `vitrine-output/` subfolder of it). This requires the **Drive write scope** in the D-015.4
OAuth grant. The Drive ACL (DDD) extends from read-only ingest to read+write.

```toml
[drive]
url            = "https://drive.google.com/drive/folders/<id>"   # source (ingest)
rclone_remote  = "gdrive"
writeback      = true                       # D-015.6 — upload outputs back to source
writeback_subdir = "vitrine-output"         # optional; defaults to the source folder
```

## Rationale

- Reusing the agentbox pattern is proven, dependency-light (one static binary, no npm), and already
  uses **TOML round-trip + JSON-Schema-driven forms + server-side secret containment** — exactly what
  an `exhibit.toml` editor needs.
- Making the onboarding output *be* the ADR-013 manifest means zero translation: the wizard, the CLI,
  and the agent all speak one file.
- The setup/agent split keeps provisioning deterministic and reproducible while leaving judgement to
  the overseer — consistent with the established "Claude Code decides what to run next" architecture.
- Browser OAuth + server-side token storage is the only way to do Drive read **and** write-back
  without leaking credentials into the manifest or the browser.

## Consequences

**Positive:** non-technical onboarding; one canonical manifest; hardware-correct model choices;
reproducible provisioning; outputs returned to the owner's Drive; secrets never in TOML/browser/git.

**Negative / costs:** a new Rust/Axum `vitrine-setup` binary + frontend to build; a JSON Schema to
maintain in lockstep with the manifest dataclasses; Google OAuth app registration + consent screen;
provisioning (model pulls) is long-running and needs robust progress/resume UX; the rename creates a
transitional period where code says "video2splat" and docs say "Vitrine".

**Neutral:** the tool is local-only (`127.0.0.1`), single active manifest — no multi-tenant concerns.

## Alternatives considered

1. **Hand-edit `exhibit.toml`.** Rejected by the onboarding requirement (non-technical users, token/
   login handling).
2. **A React/Next SPA.** Rejected: heavier, needs a build chain; agentbox proves vanilla JS suffices
   and matches the house pattern.
3. **Python/Flask backend** (the pipeline is Python). Viable, but the agentbox Rust/Axum static binary
   gives zero-dependency deployment and the `toml_edit` round-trip; chosen for consistency.
4. **Store secrets in the TOML.** Rejected: leaks into git/run-records; violates ADR-013 secret rule.
5. **Separate output bucket, not the source Drive.** Rejected by the owner's explicit "write outputs
   back to the Google Drive where the ingested video came from."

## Open Questions

- **Q-015.1** — Google OAuth: a DreamLab Google Cloud **OAuth client** (consent screen, Drive scope)
  must be registered. Who owns the GCP project / client credentials?
- **Q-015.2** — Where should `vitrine-setup` run — on .48 alongside the pipeline (recommended, so
  `/api/hardware` probes the real GPUs and model downloads land on the right host)?
- **Q-015.3** — Confirm "no history" literally means a single overwritten manifest (no per-exhibit
  files kept), vs. one active + manual archive.

## Related

- **ADR-013** — `exhibit.toml` schema (the tool's output), `v2g-net`, lifecycle, oversight backend.
- **ADR-014** — ComfyUI provisioning via Salad control API (setup uses it to ensure+pin models).
- **PRD-v3** — FR-35..FR-40 (onboarding wizard, hardware-aware selection, secret/login, provisioning
  hand-off, Drive write-back, rename).
- **DDD** — `research/ddd/v3-e2e-extensions.md` new Onboarding/Setup context + Drive write-back ACL.
- **Pattern** — `agentbox/setup/` (frontend `app.js`, backend `server/src/main.rs`, agentbox ADR-024).
