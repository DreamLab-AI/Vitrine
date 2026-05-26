# Upstream Sync Runbook — LichtFeld-Studio v0.5.2

**Date**: 2026-05-26
**Target**: MrNeRF/LichtFeld-Studio tag `v0.5.2` (released 2026-04-21)
**ADR references**: ADR-002, ADR-008
**Script**: `scripts/sync_upstream.sh`

---

## WARNING — ISOLATION POLICY (READ BEFORE EVERY SYNC)

This project is a **private, isolated fork**. The upstream sync is a **STRICT ONE-WAY PULL**.

| Rule | Detail |
|------|--------|
| NEVER push to upstream | `git push upstream <anything>` is forbidden. The `upstream` remote push URL is set to `DISABLED` by the script, so it will fail loudly if attempted. |
| NEVER open a PR against upstream | Do not open any pull request against `MrNeRF/LichtFeld-Studio`. |
| NEVER share fork-specific code upstream | `src/pipeline/`, `src/web/`, docker files, research docs — none of this goes upstream. |
| Sync direction | Inbound only: `git fetch upstream` then `git merge` into a local branch. |
| CI permissions | CI must have no credentials for the upstream remote. |

Violation of this policy would expose proprietary pipeline code to a public repository. There are no exceptions.

---

## 1. Divergence Facts

These numbers are grounded in the actual repository state as of 2026-05-26.

| Fact | Value |
|------|-------|
| Fork divergence commit | `2ced2313` "locking correctness" (2026-03-28 00:25 +01:00) |
| Our commits since divergence | 100 (on `main`) |
| Upstream master tip | `2ced2313` = `origin/master` tip (our fork's origin mirrors upstream) |
| Upstream commits ahead of our `main` | 0 on `origin/master`; ~410 on `upstream/master` once fetched |
| v0.5.0 tag | `bdd8f922` "update version v0.5.0" (2026-03-18) |
| v0.5.1 tag | Not present in local clone; will appear after `git fetch upstream --tags` |
| v0.5.2 tag | Not present in local clone; will appear after `git fetch upstream --tags` |
| Tags currently in local repo | v0.1.0 through v0.5.0 only |
| Upstream remote | Not configured yet — the script adds it |
| Upstream URL | `https://github.com/MrNeRF/LichtFeld-Studio.git` |

After `git fetch upstream --tags`, expect v0.5.1 and v0.5.2 to appear. The PRD (Section 5) documents ~410 upstream commits between our divergence and v0.5.2, and approximately 195 additional unreleased commits between v0.5.2 and `upstream/master`.

---

## 2. Pre-Merge Checklist

Complete all items before running the script with `DO_MERGE=1`.

### 2.1 State checks

- [ ] Branch is `main` (or a branch derived cleanly from `main`): `git rev-parse --abbrev-ref HEAD`
- [ ] Working tree is clean (no uncommitted changes): `git status --short`
- [ ] All pipeline tests pass on current `main`: `python -m pytest tests/ -q`
- [ ] Docker containers are healthy: `docker compose -f docker-compose.consolidated.yml ps`
- [ ] MCP server responds: `curl -s http://localhost:45677/health` (or equivalent)

### 2.2 Risk review

- [ ] Read ADR-008 (`research/decisions/adr-008-defer-vulkan-migration.md`).
  Confirm `UPSTREAM_REF=v0.5.2` (not `master`). The Vulkan migration on `upstream/master` is deferred.
- [ ] Review Section 5.2.1 of the PRD (known risk areas): `mcp_client.py`, `coordinate_transform.py`.
- [ ] Note: `coordinate_transform.py` is **safe for v0.5.2** — the coordinate-system cleanup (#1066) is `master`-only and is not in v0.5.2.
- [ ] Note: `eval/mcmc_indoor_reflective_params.json` is the only upstream-owned file currently modified in our fork. It will need conflict resolution.

### 2.3 Baseline capture

Record these before merging so post-merge validation has a comparison baseline.

```bash
# Record pipeline version fingerprint
git rev-parse HEAD > /tmp/pre-merge-sha.txt

# Capture current test results
python -m pytest tests/ -q --tb=no 2>&1 | tee /tmp/pre-merge-tests.txt

# Capture MCP server tool list
curl -s -X POST http://localhost:45677/rpc \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
    | python -m json.tool > /tmp/pre-merge-mcp-tools.json
```

---

## 3. Conflict Resolution Rules

These rules implement BOUNDARIES.md. When `git merge` stops with a conflict, apply the rule for the file's directory. "Accept upstream" means `git checkout --theirs <file>`. "Keep ours" means `git checkout --ours <file>`.

### 3.1 Directory classification table

| Path pattern | Owner | Conflict resolution | Notes |
|---|---|---|---|
| `src/core/**` | Upstream | Accept upstream | Core data structures, scene graph |
| `src/app/**` | Upstream | Accept upstream | Application entry point, GUI |
| `src/mcp/**` | Upstream | Accept upstream | Built-in MCP HTTP server |
| `src/rendering/**` | Upstream | Accept upstream | Rasterization, viewport |
| `src/training/**` | Upstream | Accept upstream | Training loop, optimizers |
| `src/geometry/**` | Upstream | Accept upstream | Spatial data structures |
| `src/io/**` | Upstream | Accept upstream | PLY/SOG/SPZ import-export |
| `src/sequencer/**` | Upstream | Accept upstream | Animation timeline |
| `src/visualizer/**` | Upstream | Accept upstream | GUI panels, assets |
| `src/python/**` | Upstream | Accept upstream | Embedded Python plugin runtime |
| `cmake/**` | Upstream | Accept upstream | Build system config |
| `external/**` | Upstream | Accept upstream | Git submodules |
| `eval/**` | Upstream | Accept upstream (then re-apply our eval params) | **Known conflict**: `eval/mcmc_indoor_reflective_params.json` |
| `tools/**` | Upstream | Accept upstream | CLI wrappers shipped by upstream |
| `tests/**` | Upstream | Accept upstream | Upstream test suite |
| `CMakeLists.txt` | Upstream | Accept upstream | Root build file |
| `vcpkg.json` | Upstream | Accept upstream | C++ dep manifest |
| `CONTRIBUTING.md` | Upstream | Accept upstream | |
| `LICENSE` | Upstream | Accept upstream | GPL-3.0 |
| `THIRD_PARTY_LICENSES.md` | Upstream | Accept upstream | |
| `src/pipeline/**` | Ours | Keep ours | All 28 pipeline modules |
| `src/web/**` | Ours | Keep ours | Flask web UI |
| `docker/**` | Ours | Keep ours | Container configs |
| `scripts/**` | Ours | Keep ours | Pipeline runners, test harnesses |
| `research/**` | Ours | Keep ours | Research and ADRs |
| `docs/**` | Ours | Keep ours | Engineering docs |
| `Dockerfile.consolidated` | Ours | Keep ours | Main consolidated Dockerfile |
| `docker-compose.consolidated.yml` | Ours | Keep ours | Two-container compose |
| `BOUNDARIES.md` | Ours | Keep ours | |
| `GAUSSIAN_TOOLKIT_README.md` | Ours | Keep ours | |
| `AGENTS.md` | Ours | Keep ours | |
| `CLAUDE_CONTAINER.md` | Ours | Keep ours | |
| `README.md` | Ours | Keep ours | Fork README |
| `.gitignore` | Merge both | Manual merge | Append upstream entries; keep ours; no duplicates |

### 3.2 Conflict resolution commands

```bash
# Accept upstream version (for upstream-owned files):
git checkout --theirs <file>
git add <file>

# Keep our version (for our-owned files):
git checkout --ours <file>
git add <file>

# Manual merge (for .gitignore):
# Open in editor, keep both sections, remove duplication, then:
git add .gitignore

# After resolving all conflicts:
git merge --continue
```

### 3.3 Known pre-identified conflict: eval/mcmc_indoor_reflective_params.json

This file exists in both our fork and upstream. Our copy contains tuned MCMC
parameters for indoor reflective scenes. Resolution:

1. Accept upstream: `git checkout --theirs eval/mcmc_indoor_reflective_params.json`
2. Inspect the diff: `git diff HEAD eval/mcmc_indoor_reflective_params.json`
3. If our custom params are still needed, add them back as a separate file
   `eval/mcmc_indoor_reflective_params.local.json` (ours-only, not tracked upstream).

---

## 4. Known Risk Areas

### 4.1 mcp_client.py vs Enhanced MCP (#984)

**Risk level**: Medium

PR #984 hardened the MCP server with more capabilities and changed some API
signatures. After the merge, `src/pipeline/mcp_client.py` may fail at runtime
if it calls tool endpoints or uses field names that changed.

**Action**:
1. After rebuilding LichtFeld and starting the MCP server, capture the current
   tool list: `curl -s -X POST http://localhost:45677/rpc -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'`
2. Compare with `/tmp/pre-merge-mcp-tools.json` captured in the pre-merge baseline.
3. For any tool whose signature changed, update `mcp_client.py` accordingly.
4. Run `scripts/test_orchestrator.py` to exercise all MCP calls.

### 4.2 coordinate_transform.py vs Coordinate Cleanup (#1066)

**Risk level**: SAFE for v0.5.2 (zero risk unless syncing to master)

PR #1066 (coordinate system cleanup) is on `upstream/master` only. It is **not**
included in the v0.5.2 tag. Issue #1104 (ERP+GUT training produces degenerate
flat-plane output) is the known regression introduced by #1066.

`src/pipeline/coordinate_transform.py` is safe after a v0.5.2 sync. This risk
is formally tracked in ADR-008 and must be re-evaluated before any future sync
to `upstream/master`.

### 4.3 CMakeLists.txt / vcpkg.json — accept upstream

**Risk level**: Low (build system, not our code)

Upstream updated the build files with new dependencies (USD, mesh libraries,
NanoGS, MRNF, etc.). Accept the upstream version. The Docker build process
handles fetching and building these dependencies automatically via vcpkg.

After accepting upstream `CMakeLists.txt` and `vcpkg.json`, rebuild the C++ core.
If the build fails, check `vcpkg install` output for missing system dependencies
and add them to `Dockerfile.consolidated` in the `apt-get install` layer.

### 4.4 Vulkan migration (v0.5.3-dev) — DEFERRED

**Risk level**: High (but does not apply to v0.5.2 sync)

Upstream `master` has removed the CUDA and OpenGL rendering backends entirely
(PRs #1170, #1234). Syncing to `master` instead of `v0.5.2` would break our
pipeline. This is deferred to ADR-008.

The script enforces this by rejecting any `UPSTREAM_REF` that does not look like
a version tag (`v0.5.2`, `v0.5.1`, etc.) unless `FORCE=1` is set.

### 4.5 eval/mcmc_indoor_reflective_params.json

**Risk level**: Low (data file, not code)

See Section 3.3 above. Our custom MCMC parameters must be preserved separately
after accepting the upstream version of this file.

---

## 5. Step-by-Step Procedure

### Step 1 — Confirm environment

```bash
cd /path/to/LichtFeld-Studio
git status          # must be clean
git branch          # must be on main (or your working branch)
git log --oneline -5
```

### Step 2 — Dry run (no state changes)

```bash
./scripts/sync_upstream.sh
```

The script will:
- Add the `upstream` remote (fetch only; push URL set to `DISABLED`)
- Fetch `v0.5.2` tag from upstream
- Print the conflict-zone report (files changed in both histories)
- Stop without making any changes

Review the conflict-zone output carefully. Compare it against the directory
classification table in Section 3.1 to pre-plan your resolution approach.

### Step 3 — Review the conflict zone

For each file listed in the "Potential conflict zones" section of the dry-run
output, decide in advance: accept upstream or keep ours. The classification
table in Section 3.1 covers all expected cases.

### Step 4 — Execute the merge

```bash
DO_MERGE=1 ./scripts/sync_upstream.sh
```

The script creates branch `sync/upstream-v0.5.2` from `main` and runs
`git merge --no-ff v0.5.2`.

If the merge completes cleanly, go to Step 6.
If the merge stops with conflicts, proceed to Step 5.

### Step 5 — Resolve conflicts (if any)

```bash
# See which files have conflicts
git status

# For each conflicted file, apply the rule from Section 3.1:
git checkout --theirs src/core/some_file.cpp   # upstream-owned
git add src/core/some_file.cpp

git checkout --ours src/pipeline/mcp_client.py  # our code
git add src/pipeline/mcp_client.py

# Special case: .gitignore — merge manually in editor, then:
git add .gitignore

# Verify no conflict markers remain
grep -rn "<<<<<<" . --include="*.py" --include="*.cpp" --include="*.h" \
    --include="*.json" --include="*.txt" --include="*.md"

# Complete the merge
git merge --continue
# (git will open your editor for the merge commit message)
```

### Step 6 — Rebuild and verify

```bash
# Rebuild the C++ core with upstream changes
docker compose -f docker-compose.consolidated.yml build gaussian-toolkit

# Start containers
docker compose -f docker-compose.consolidated.yml up -d

# Verify LichtFeld binary starts and MCP server is responsive
docker exec gaussian-toolkit-main ./LichtFeld_Studio --headless &
sleep 10
curl -s -X POST http://localhost:45677/rpc \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m json.tool

# Compare MCP tool list with pre-merge baseline
diff /tmp/pre-merge-mcp-tools.json <(curl -s -X POST http://localhost:45677/rpc \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m json.tool)
```

### Step 7 — Post-merge validation (see Section 6)

Run the full validation checklist. All items must pass before the sync branch
is merged into `main`.

### Step 8 — Internal PR

```bash
# Push the sync branch to YOUR fork (NOT to upstream)
git push origin sync/upstream-v0.5.2

# Open an internal PR: sync/upstream-v0.5.2 -> main
# Do NOT open a PR against MrNeRF/LichtFeld-Studio
```

---

## 6. Post-Merge Validation Checklist

All items must pass before merging `sync/upstream-v0.5.2` into `main`.

### 6.1 Core infrastructure

| Test | Command / Method | Pass Criteria |
|------|-----------------|---------------|
| Docker build | `docker compose -f docker-compose.consolidated.yml build` | Exit 0, no errors |
| Both containers start | `docker compose -f docker-compose.consolidated.yml up -d && docker compose ps` | Both `gaussian-toolkit` and `milo` show `running` |
| MCP server responds | `curl -s http://localhost:45677/health` (or `/rpc` tools/list) | Valid JSON response, no connection refused |
| No new MCP tool regressions | Diff against pre-merge MCP tool list | No tools removed; new tools from #984 present |

### 6.2 Pipeline end-to-end

| Test | Command / Method | Pass Criteria |
|------|-----------------|---------------|
| Full pipeline run | `python -m src.pipeline <test-video-path> --output /tmp/test-out/` | Completes without exception; USD file produced |
| Existing test suite | `python -m pytest tests/ -q` | Same or better pass rate vs pre-merge baseline |
| mcp_client.py MCP calls | `python scripts/test_orchestrator.py` | All MCP calls succeed with v0.5.2 server |

### 6.3 New upstream features

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Native USD export | Start LichtFeld, load a test splat via MCP, invoke USD export tool | `.usda` or `.usdc` file produced with correct scene hierarchy |
| Mesh import/export | Load a GLB mesh in LichtFeld, convert to splat, re-export | Round-trip succeeds without crash |
| MRNF densification | Initiate training via MCP with `densification_strategy: mrnf` | Training completes; output PLY produced |
| VRAM during training | Monitor `nvidia-smi` during a training run | No OOM; peak VRAM lower than or equal to pre-merge |
| NanoGS (#1014) | Invoke NanoGS training if CLI flag exposed | Compact PLY produced; no crash |

### 6.4 Regression checks

| Test | Method | Pass Criteria |
|------|--------|---------------|
| coordinate_transform.py | Run a COLMAP dataset through the pipeline end-to-end | Coordinate output matches pre-merge reference (no flat-plane degenerate output) |
| MILo sidecar | `docker exec milo python -c "import milo; print('ok')"` (or equivalent) | MILo sidecar unaffected by upstream C++ changes |
| Web UI | Browse to `http://localhost:7860`, upload video, submit job | Upload, processing, and download all succeed |
| USD assembler fallback | If native USD I/O does not support hierarchical prims, `usd_assembler.py` still works | USD scene file with correct per-object prim hierarchy |

---

## 7. Rollback Procedure

If post-merge validation fails and the failure cannot be quickly fixed:

### 7.1 Discard the sync branch (before pushing to origin)

```bash
# Return to main — no harm done if sync branch was never pushed
git checkout main
git branch -D sync/upstream-v0.5.2
```

`main` is unchanged. The upstream remote and its tags remain (harmless).

### 7.2 Rollback after pushing sync branch (before merging to main)

```bash
# The PR is still open, just close it without merging
git checkout main
git branch -D sync/upstream-v0.5.2
git push origin --delete sync/upstream-v0.5.2
```

`main` is unchanged.

### 7.3 Emergency rollback after merging to main

```bash
# Find the pre-merge commit SHA (from /tmp/pre-merge-sha.txt or git log)
PRE_MERGE_SHA=$(cat /tmp/pre-merge-sha.txt)

# Create a revert branch
git checkout -b revert/upstream-v0.5.2-sync main
git revert -m 1 $(git log --oneline main | grep "Sync upstream" | awk '{print $1}')
# Review the revert, then PR it to main
```

Avoid `git reset --hard` on a branch that has been pushed — use `git revert` to
preserve history.

### 7.4 What to preserve across rollback

- The `upstream` remote and its fetched tags (keeps future syncs fast)
- `/tmp/pre-merge-tests.txt` and `/tmp/pre-merge-mcp-tools.json` (baseline data)
- Any conflict resolution notes taken during the process

---

## 8. Future Sync Considerations

### Next sync: v0.5.3 and the Vulkan migration (ADR-008)

The v0.5.3 development line is a breaking change. Before the next upstream sync:

1. All ADR-008 trigger conditions must be satisfied (see
   `research/decisions/adr-008-defer-vulkan-migration.md`).
2. Specific checks required:
   - v0.5.3 released as a stable tagged version
   - Issue #1104 (ERP+GUT degenerate flat-plane) resolved upstream
   - Headless Vulkan validated in Docker (`mesa-vulkan-drivers`, GPU ICD)
   - `mcp_client.py` API compatibility verified against v0.5.3 MCP server
   - Python API audit: all "stale python api" upstream fixes reviewed
3. Add `vulkan-tools` and `mesa-vulkan-drivers` to `Dockerfile.consolidated`
   only after ADR-008 is accepted.
4. `src/pipeline/coordinate_transform.py` must be audited against #1066 before
   any sync that includes that commit.

### Monthly upstream watch

Track `upstream/master` via:

```bash
git fetch upstream --tags
git log --oneline v0.5.2..upstream/master
```

Look for: resolution of issue #1104, v0.5.3 release tag, API stability signals.
Log findings in `docs/engineering-log.md`.

### Merge base after v0.5.2 sync

After this sync is complete, the merge base for the next sync will be `v0.5.2`.
Update this runbook with the new divergence SHA and commit count at that time.
