# QE-04 Security Audit — Vitrine (Video-to-Gaussian / LichtFeld-Studio)

**Date:** 2026-06-04
**Branch:** feat/v2-upgrade-swarm
**Auditor:** Security auditor agent (claude-sonnet-4-6)
**Scope:** Credential / secret-handling surface; FINDING-006 remediation assessment; v3 onboarding threat surface
**Type:** Read-only code review + planning-document audit (no live execution)
**Prior audit:** research/decisions/v2-security-audit.md (2026-05-26)

---

## Executive Verdict

The v3 plan is **well-architected for secret containment** and constitutes a genuine remediation of FINDING-006 *in design*. However, the implementation does not yet exist: `manifest.py`, `model_lifecycle.py`, and `vitrine-setup/` are all absent from the codebase. The code that runs today retains the FINDING-006 pattern almost verbatim — `HF_TOKEN` is injected as a plain Docker environment variable, `InpaintConfig.hf_token` defaults to an empty string with no env-var resolution in the config loader, and `PipelineConfig.save()` would serialise any token value set at runtime directly into the JSON run snapshot without stripping it.

Two new issues join the prior findings: the `.env` file containing a real `HF_TOKEN` value is present on disk (not git-tracked, but co-located with the repo and loaded by `docker-compose.consolidated.yml` via `env_file: .env`), and multiple files hardcode the literal IP `192.168.2.48` as default service endpoints.

The onboarding design (ADR-015 / FR-37 / D-015.4) correctly identifies the key threat vectors and proposes sound mitigations, but leaves several implementation-level security details unspecified: OAuth state/PKCE for the Google consent flow, trust-boundary validation on `/api/proxy/*`, refresh-token storage format, and where `provision.status = "ready"` is written and whether it is itself a security boundary.

**Overall verdict: CONDITIONAL PASS on plan; FAIL on current code.** Gate G17 and G-O3 cannot be met by the code in the repository today.

---

## FINDING-006 Status

| Dimension | Status |
|-----------|--------|
| Remediated in the v3 plan (NFR-4, FR-37, D-013.1, D-015.4, G17, G-O3) | YES — plan is sound |
| Remediated in the current code | NO — `HF_TOKEN` still passed as plain Docker env var (line 19 `docker-compose.consolidated.yml`); `InpaintConfig.hf_token = ""` with no env-var resolution in `config.py:PipelineConfig.load()`; `PipelineConfig.save()` calls `asdict()` which would serialise any non-empty `hf_token` to the JSON run snapshot |

The v2 audit rated FINDING-006 as Medium (plaintext env var exposure of `ANTHROPIC_API_KEY` and `HF_TOKEN`). The v3 plan promotes this to blocking gate G17. The code has not been updated to implement the plan.

---

## Findings Table

| ID | Severity | Location | Finding | Remediation |
|----|----------|----------|---------|-------------|
| SEC-01 | **Critical** | `docker-compose.consolidated.yml:11-12, 19` | `env_file: .env` loads the `.env` file into the container; `HF_TOKEN=${HF_TOKEN}` then propagates it as a plain container environment variable. The `.env` file on disk contains a real `HF_TOKEN` value (37 chars, not a placeholder). Any process with Docker socket access can recover the token via `docker inspect gaussian-toolkit`. This is FINDING-006 unresolved for `HF_TOKEN`. | Replace with a Docker secret: `secrets: [hf_token]` + `file: ./secrets/hf_token` (gitignored), reference via `env HF_TOKEN=$(cat /run/secrets/hf_token)` in an entrypoint wrapper, or mount the secret file and read it at startup. Remove `HF_TOKEN` from `env_file`/`environment`. |
| SEC-02 | **High** | `docker-compose.consolidated.yml:21` | `ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}` passes the Anthropic API key as a plain Docker environment variable. FINDING-006 from the v2 audit. No change since that audit. | Same Docker secret pattern as SEC-01. |
| SEC-03 | **High** | `src/pipeline/config.py:138` + `config.py:252-258` | `InpaintConfig.hf_token: str = ""` is a dataclass field with no env-var resolution. `PipelineConfig.load()` deserialises it from JSON; `PipelineConfig.save()` calls `asdict()` and writes all fields to the JSON run snapshot without stripping secrets. If `hf_token` is set (e.g. by Claude Code resolving the `env:HF_TOKEN` indirection that ADR-013 plans), it will be written verbatim to the JSON run record and uploaded to the Drive `outputs/` folder. The v3 plan explicitly requires secrets to be stripped before the snapshot (D-013.1, FR-29). | Add a `to_public_dict()` method on `PipelineConfig` that clears `inpaint.hf_token` and any other secret field before serialisation. Use it in `save()` and wherever a run-record snapshot is written. |
| SEC-04 | **High** | `src/pipeline/comfyui_inpainter.py:282-284`, `src/pipeline/hunyuan3d_client.py:192-193`, `src/pipeline/config.py:135-136,152-153,171` | Hardcoded `192.168.2.48` IP addresses as default service endpoints appear in 11 locations across three files. ADR-013 FR-31 / D-013.3 plans to replace these with Docker-network DNS names (`comfyui:8188` etc.) but this is not implemented. A host that is not `192.168.2.48` will silently target the wrong machine; on a different network, the connection will fail or reach an unintended host on that address. | Replace all hardcoded `192.168.2.48` defaults with service-DNS names (`comfyui`, `agent-vlm`) overridable from the manifest `[pipeline]` block as planned in ADR-013 D-013.3. |
| SEC-05 | **Medium** | `src/pipeline/comfyui_inpainter.py:420-424` | When `self.hf_token` is set and the download URL contains `huggingface.co`, the token is embedded in the JSON payload sent to the ComfyUI Salad control API (`POST /download`). If ComfyUI's `/download` API forwards the auth token in server-side logs or error responses, the token may leak to a third-party service. The auth payload structure `{"type": "bearer", "token": "<value>"}` is non-standard. | Confirm with the Salad API specification that the token is not logged. Consider using a server-side proxy pattern (as planned in ADR-015 D-015.4) so the token is never transmitted to ComfyUI at all — instead have the orchestrator download the model directly using the token and push the binary to ComfyUI. |
| SEC-06 | **Medium** | `research/decisions/adr-015-vitrine-web-onboarding.md` (D-015.4) | The Google OAuth browser flow is described but the implementation plan omits: (a) `state` parameter for CSRF protection on the redirect callback, (b) PKCE (`code_challenge`/`code_verifier`) for the authorization code exchange, (c) explicit validation that the redirect URI matches `127.0.0.1:<ephemeral-port>`. Without a `state` parameter, a cross-site request can trigger the OAuth callback with an attacker-controlled `code`, potentially exchanging it for the operator's Google tokens via the Vitrine backend. The OAuth open question Q-015.1 (who registers the GCP client) is unresolved. | Implement `state` parameter generation (cryptographically random, stored server-side, verified on callback) and PKCE before the OAuth flow is built. Validate `redirect_uri` strictly to the `127.0.0.1` binding. Document this in the ADR-015 implementation notes. |
| SEC-07 | **Medium** | `research/decisions/adr-015-vitrine-web-onboarding.md` (D-015.4, D-015.2) | The `/api/proxy/*` endpoint pattern (agentbox model) injects `Authorization: Bearer` server-side. The ADR does not specify: (a) which upstream URLs the proxy is permitted to forward to (allowlist), (b) whether the proxy validates the path component to prevent SSRF, (c) whether the local browser origin is validated via CORS or same-origin check. Without an allowlist, a crafted request from any local process that can reach `127.0.0.1:<port>` could proxy arbitrary URLs using the operator's Google or HF token. | Define an explicit allowlist of proxy destinations (HF Hub domains, Google Drive API domain). Validate the `path` parameter server-side before forwarding. Add a `SameSite=Strict` or `Origin` check if the server is accessible to any local code. |
| SEC-08 | **Medium** | `docker-compose.consolidated.yml:43-47` | Five ports are bound to `0.0.0.0` (all interfaces): 7860 (Flask web UI), 7681 (ttyd web terminal), 8188 (ComfyUI), 45677 (LichtFeld MCP), 5901 (VNC). FINDING-009 from the v2 audit. No change since that audit. `ttyd` and VNC provide direct shell/desktop access; binding them to all interfaces on a host without a firewall exposes them to the LAN or internet. | Bind sensitive ports to `127.0.0.1`: `127.0.0.1:7681:7681`, `127.0.0.1:5901:5901`, `127.0.0.1:45677:45677`. The web UI (7860) and ComfyUI (8188) may remain on `0.0.0.0` only if LAN access is required and a firewall restricts the host. |
| SEC-09 | **Medium** | `docker-compose.consolidated.yml:161-173` | The `rclone_conf` Docker secret defaults to `${RCLONE_CONF_FILE:-./secrets/rclone.conf.example}`. The `.example` file is committed and contains a service-account template. If the operator forgets to set `RCLONE_CONF_FILE`, the placeholder is mounted as the secret — but `RCLONE_CONFIG` env var still points to `/run/secrets/rclone_conf`. `drive_ingestor.py` will pass this path to rclone, which will then fail with a config-parse or auth error. The failure mode is silent misconfiguration rather than a hard early error. A second concern: the RCLONE_CONFIG env var path `/run/secrets/rclone_conf` is itself visible via `docker inspect`, which reveals the secret's mount path even if not the content. | Add a startup check in `drive_ingestor.py` that validates the rclone config at launch: parse the file, assert a valid remote block exists, and fail fast with a clear error if the placeholder is mounted. Document this in the deployment guide. |
| SEC-10 | **Low** | `src/pipeline/comfyui_inpainter.py:280-284` | The class docstring example on line 14 hardcodes `api_url="http://192.168.2.48:3001"` in the usage example. Documentation examples with real internal IPs are not a vulnerability in themselves, but they increase the risk that an operator copies the example verbatim into production code on a different network where `192.168.2.48` resolves to an unrelated host. | Replace the docstring example IP with `http://comfyui:3001` (the planned service-DNS name) or `http://localhost:3001`. |
| SEC-11 | **Low** | `research/decisions/adr-015-vitrine-web-onboarding.md` (D-015.5) | `provision.status = "ready"` is written to the manifest by `vitrine-setup` to signal the hand-off to the Claude Code overseer. The ADR does not specify: (a) whether the manifest file is written atomically (rename-on-write), (b) whether the overseer validates the `provision.status` value cryptographically or merely checks the string. A race condition or a tampered manifest file could trick the overseer into starting a pipeline run prematurely or with incorrect provisioning. In the local-only single-user context this risk is low, but it is worth documenting. | Write the manifest atomically (write to a temp file, then rename). Document that `provision.status = "ready"` is a convenience signal, not a security boundary, and that the overseer must independently validate that required models exist and credentials resolve. |
| SEC-12 | **Info** | `src/pipeline/config.py:134` (InpaintConfig), `adr-013-ingest-manifest-serial-model-lifecycle.md` (D-013.1) | The v3 plan's secret indirection (`env:HF_TOKEN`) is implemented only in planning documents. The ADR-013 `manifest.py` loader (which would resolve `env:NAME` and strip secrets before the run snapshot) does not exist yet. Until it is built, the `env:HF_TOKEN` string would be parsed as a literal token value by the existing `PipelineConfig.load()`, silently failing with a malformed token at the HF API call rather than a clear startup error. | Build `manifest.py` as the first implementation priority for ADR-013. Ensure the loader fails fast (named error, not silent empty string) on missing env vars referenced via `env:`. |
| SEC-13 | **Info** | `src/pipeline/drive_ingestor.py:37` (CLI docstring) | The CLI usage example shows `--rclone-config /run/secrets/rclone.conf`. This is the correct path for a Docker secret mount. However, the `DriveIngestConfig.rclone_config` field defaults to `None`, which causes `_rclone_base()` to omit `--config` entirely, falling back to rclone's default config discovery. If rclone's default config does not have the required remote, the failure occurs at list-time, not startup, and the error message will reference rclone internals rather than the missing config. | Change the default to `os.environ.get("RCLONE_CONFIG")` so the Docker-compose-injected `RCLONE_CONFIG=/run/secrets/rclone_conf` is automatically honoured without requiring the operator to pass `--rclone-config` on the CLI. |

---

## Plan vs. Reality Gap Assessment

### Drive credential path (FINDING-006 / NFR-4 / G17)

**Plan claim**: Drive service-account credentials supplied as Docker secret (not env var); `rclone_conf` secret mounted at `/run/secrets/rclone_conf`; `RCLONE_CONFIG` env var points to the mount path.

**Code reality**: The `docker-compose.consolidated.yml` does implement the Docker secret for the rclone config (lines 75-80, 161-173). This is correct and represents partial progress over the v2 state. The rclone secret path is the right mitigation for the service-account file.

**Gap**: `HF_TOKEN` and `ANTHROPIC_API_KEY` are still passed as plain environment variables (lines 19, 21) loaded from the `.env` file via `env_file: .env`. The plan says no credential should be a plain env var; the compose file still passes two credentials this way. FINDING-006 is half-remediated: the Drive service-account is now a Docker secret; the HF and Anthropic tokens are not.

### v3 secret indirection design (FR-29 / ADR-013 D-013.1)

The `env:HF_TOKEN` manifest indirection, the `manifest.py` loader that resolves and strips secrets, and the `InpaintConfig.hf_token` population path from the manifest are all unimplemented. The current `PipelineConfig.save()` uses `asdict()` and would write any token value set at runtime directly to the JSON snapshot. The FR-29 requirement that secrets are "stripped before the JSON run snapshot" is not enforced by any code.

### Onboarding design soundness (FR-37 / ADR-015 D-015.4)

The core containment architecture is sound: Rust backend holds tokens, browser sees only masked references, manifest holds only `env:` indirection, proxy injects Bearer server-side. The design correctly mirrors the agentbox pattern.

Gaps requiring implementation attention:
1. OAuth CSRF state parameter and PKCE are not mentioned anywhere in the plan (SEC-06).
2. The `/api/proxy/*` URL allowlist is not specified (SEC-07).
3. The refresh-token storage format (keyring vs Docker secret vs file) is described only as "keyring/Docker secret" without specifying which, under what conditions, and at what path.
4. The Google OAuth client (Q-015.1) remains unregistered; without a registered client the redirect URI cannot be `127.0.0.1:<port>`, which is the key security property.

### `docker inspect` process-env visibility

Even after Docker secrets are used for file-based credentials, any secret also passed as an environment variable is visible via `docker inspect gaussian-toolkit` to any user with Docker socket access. This is the core of FINDING-006. The `HF_TOKEN` environment variable line in the compose file must be removed entirely, not just supplemented with a secret mount.

---

## Prioritised Fix List

| Priority | Finding | Action | Effort |
|----------|---------|--------|--------|
| 1 | SEC-01 (Critical) | Remove `HF_TOKEN` from `env_file` and `environment` in compose; mount as Docker secret instead | Small |
| 2 | SEC-02 (High) | Remove `ANTHROPIC_API_KEY` plain env var; use Docker secret or external secret injection | Small |
| 3 | SEC-03 (High) | Add `to_public_dict()` to `PipelineConfig` that zeros secret fields; use it in `save()` | Small |
| 4 | SEC-04 (High) | Replace hardcoded `192.168.2.48` defaults with service-DNS names in all three files | Medium |
| 5 | SEC-06 (Medium) | Specify and implement OAuth `state` parameter + PKCE in ADR-015 implementation plan | Small (spec), Medium (impl) |
| 6 | SEC-07 (Medium) | Define `/api/proxy/*` URL allowlist; add server-side path validation | Small |
| 7 | SEC-08 (Medium) | Bind ttyd (7681), VNC (5901), MCP (45677) to `127.0.0.1` in compose | Trivial |
| 8 | SEC-05 (Medium) | Audit Salad API log behaviour for auth token; consider server-side download proxy | Small |
| 9 | SEC-09 (Medium) | Add startup rclone config validation in `drive_ingestor.py`; fail fast on placeholder | Small |
| 10 | SEC-12 (Info) | Build `manifest.py` with `env:` resolution and named-error on missing vars — first ADR-013 deliverable | Medium |
| 11 | SEC-13 (Info) | Default `rclone_config` to `os.environ.get("RCLONE_CONFIG")` in `DriveIngestConfig` | Trivial |
| 12 | SEC-10 (Low) | Update docstring example URLs to service-DNS names | Trivial |
| 13 | SEC-11 (Low) | Write manifest atomically; document `provision.status` as non-cryptographic signal | Small |
