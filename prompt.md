# .48 Host Agent — v3 Validation & Pin-Resolution Megaprompt

You are operating on the GPU host (.48 / john@HP-Desktop). A container-side agent just
built and pushed a large "v3" upgrade to the Vitrine pipeline; your job is the host-only
half that a container physically cannot do: run the GPU/weights/pytest/cargo validation,
resolve the version pins against the real checkouts, and close (or precisely report) the
SOTA weight gaps. Verify everything yourself — do not trust prior claims.

## Repo & ground truth

- Repo: `~/githubs/gaussian/LichtFeld-Studio` (origin: `github.com:DreamLab-AI/Vitrine`, branch `main`)
- Expected HEAD after pull: `9629e5bd` ("docs(v3): engineering-log entry…"). Confirm with
  `git -C ~/githubs/gaussian/LichtFeld-Studio rev-parse HEAD` and a clean `git status`.
- The Python package root is `src/`. ALWAYS run pipeline modules from there:
  `cd ~/githubs/gaussian/LichtFeld-Studio/src && python -m pipeline.<x>`
- Live ComfyUI install: `~/comfyui-api-data/ComfyUI` (HTTP API `:8188`). Salad control-plane `:3001`.
- Default weight-staging search roots (override via `SOTA_MODEL_ROOTS`, `os.pathsep`-separated):
  `~/comfyui-models-staging`, `~/comfyui-api-data/ComfyUI/models`, `/models-staging`, `/opt/models`,
  `/opt/hf-cache`, `~/.cache/huggingface`.
- Hunyuan3D-2.1 checkpoints expected at `~/comfyui-api-data/ComfyUI/comfy/ldm/hunyuan3dv2_1`.
- SAM3D node: `~/comfyui-api-data/ComfyUI/custom_nodes/comfyui-sam3dobjects`.
- Component checkouts per `pins.lock.toml` live under `/opt/*` (gaussian-toolkit, milo, come,
  gaussianwrapping, comfyui, hunyuan3d-2, sam3-repo) and `~/.lichtfeld/plugins/splat_ready`.

## What was built (so you know what you're validating)

- **ADR-013 manifest**: `exhibit.toml` → `pipeline/manifest.py` resolves `env:NAME` secrets,
  strips them from the redacted JSON run-record, materialises a PipelineConfig overlay.
- **SOTA idiot-check** `pipeline/sota_registry.py` wired into `pipeline/preflight.py` (advisory;
  hard-fails only under `SOTA_STRICT`).
- **Serial VRAM lifecycle** (`pipeline/model_lifecycle.py`) + service-DNS endpoints
  (`pipeline/endpoints.py`, `V2G_*` env, legacy `192.168.2.48` only as fallback).
- **Agent-controlled ComfyUI** client (`pipeline/comfyui_control.py`, ADR-014).
- **SOTA model paths**: FLUX.2 inpaint + Hunyuan3D-2.1 PBR with graceful degradation
  (`comfyui_inpainter.py`, `hunyuan3d_client.py`, `workflows/*.json`). These DEGRADE cleanly
  when weights are absent — so "missing weights" is a capability gap, not a crash.
- **Rust/Axum onboarding wizard** in `onboarding/` (ADR-015, `:8088`).
- **`pins.lock.toml`** + `scripts/resolve_pins.sh` (work-order item 7).

NOT yet done (your job): pytest run, cargo build, live ComfyUI/endpoint checks, pin
resolution, weight-gap closure. The container has no pytest, no GPU, no cargo runtime parity.

## Tasks — do in this order, report findings as you go

1. **SANITY**: confirm HEAD/clean tree as above. If behind, `git pull --ff-only origin main`.

2. **PYTEST** (the container could not run these): from repo root,
   `python -m pytest tests/python/test_model_lifecycle.py tests/python/test_comfyui_control.py -q`
   then the broader suite `python -m pytest tests/python -q`. Report pass/fail counts. If a new
   suite fails, READ the failure and fix the source (not the test) only if it's a genuine bug;
   otherwise report it precisely. Do not delete or weaken tests to make them pass.

3. **SOTA IDIOT-CHECK** (discovery — drives everything downstream). From `src/`:
   ```bash
   python -m pipeline.sota_registry list            # what the registry targets + licences/VRAM
   python -m pipeline.sota_registry check           # PASS/WARN/FAIL per element vs THIS host
   python -m pipeline.sota_registry check --json     # machine-readable
   ```
   If your weights live somewhere non-default, re-run with
   `SOTA_MODEL_ROOTS=/path/one:/path/two python -m pipeline.sota_registry check`.
   Capture the full report. The FAIL/WARN lines are the authoritative list of what's missing
   (FLUX.2, Hunyuan-2.1, SAM3D, ALIKED+LightGlue, etc.) and the VRAM verdicts.

4. **RESOLVE PINS** (read-only by design — never fetches/checkouts):
   `bash scripts/resolve_pins.sh` → writes `pins.resolved.toml` with real `git rev-parse HEAD`
   per `/opt/*` repo and `pip show gsplat` version. Components absent on this host stay empty
   (expected). Review the summary table, then commit `pins.resolved.toml`.

5. **HOST SMOKE** of the manifest + preflight (from `src/`, with real env if you have tokens):
   ```bash
   HF_TOKEN=… GOOGLE_APPLICATION_CREDENTIALS=… python -m pipeline.manifest ../exhibit.example.toml
   python -m pipeline.preflight    # now includes the SOTA section
   ```
   Confirm: parse OK, secrets resolve, PipelineConfig validation OK, preflight green.

6. **ONBOARDING APP** (cargo runtime parity is host-only):
   ```bash
   cd onboarding && cargo build --release && (./target/release/vitrine-onboarding &)
   ```
   then `curl -s localhost:8088/api/health`, open `/`, POST a sample manifest to
   `/api/manifest`, and CONFIRM: it writes `exhibit.toml` with `env:` references only, diverts
   raw tokens to `.secrets.env` (verify `stat -c %a .secrets.env` == `600`), never echoes the
   token back. Kill the server after. Report any cargo warnings.

7. **LIVE ComfyUI + ENDPOINTS**:
   - `curl -s localhost:8188/system_stats` to confirm the live ComfyUI is up.
   - Exercise `pipeline/comfyui_control.py` against `:8188` (health, probe_models, free_vram) —
     a short python REPL/script is fine. Report which target models ComfyUI actually has.
   - Decide the endpoint wiring: if a docker service-DNS mesh (comfyui:8188, agent-vlm:8080,
     milo:8090, come:8091, control-plane:3001) is live, validate `pipeline.endpoints.from_env()`;
     otherwise set the `V2G_*` env vars to the real host addresses and document them.

8. **WEIGHT-GAP CLOSURE** — gated. Using the step-3 report, list each missing model with its
   on-disk size and licence (the registry default posture is research/non-commercial). DO NOT
   blind-pull tens of GB. For each gap: state size + source + target staging path, then stage
   it into the appropriate `SOTA_MODEL_ROOTS` dir (prefer `~/comfyui-api-data/ComfyUI/models` for
   ComfyUI-served models; Hunyuan-2.1 into `…/comfy/ldm/hunyuan3dv2_1`). Re-run step 3 to confirm
   the element flips to PASS. If a download is large/ambiguous or licence-restricted, stop and
   report rather than guess.

9. **(Optional, heavy — only if 2–8 are green and a small test dataset exists)** end-to-end smoke:
   drive `exhibit.example.toml` through the pipeline on a tiny input and confirm the serial VRAM
   lifecycle keeps peak = max(stage), not sum. Report peak VRAM and per-stage timings.

## Guardrails

- Read-only / non-destructive first. `resolve_pins.sh` must stay read-only; do not fetch/checkout
  `/opt/*` repos to "tidy" them.
- Never fabricate a commit SHA or version — unresolvable stays empty.
- Respect `BOUNDARIES.md`: do not modify upstream LichtFeld code on our branch.
- Commit small and targeted (`pins.resolved.toml`; any genuine source fix). Push to `origin main`
  over the host's SSH. Don't force-push.
- Licences: this is a research/non-commercial posture (CoMe = Inria/MPII non-commercial;
  GaussianWrapping has no licence → treat non-commercial). Flag, don't silently enable, any
  commercial-use path.

## Deliverable

A concise written report: pytest results; the full SOTA idiot-check verdict (PASS/WARN/FAIL
per element + VRAM); `pins.resolved.toml` summary (how many resolved); onboarding + ComfyUI +
endpoint validation results; the exact weight gaps with sizes/licences and which you staged;
and the HEAD you pushed. Then stop.
