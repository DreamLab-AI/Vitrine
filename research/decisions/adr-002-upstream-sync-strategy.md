# ADR-002: Upstream Sync Strategy — Stable v0.5.2 Tag with One-Way Pull Isolation

## Status

Accepted

## Context

Our fork (Gaussian Toolkit, branch `gaussian-toolkit`) diverged from MrNeRF/LichtFeld-Studio on 2026-03-28. As of 2026-05-26, upstream has shipped approximately 410 commits spanning two stable releases (v0.5.1, v0.5.2) and 195 additional unreleased commits toward v0.5.3. The v0.5.3 development line introduces a **complete, breaking removal** of the OpenGL and CUDA rendering backends in favour of Vulkan (#1170, #1234), an ImGui-to-RmlUi UI migration, and a coordinate-system cleanup (#1066) that directly risks breaking our `coordinate_transform.py`.

Upstream commits we most need are present in v0.5.2:
- Native USD import/export (#1032) — eliminates our custom `usd_assembler.py` workaround
- Native mesh support (#876, #879, #889) — mesh loading, mesh↔splat conversion, mesh picking
- MRNF densification (#1031) — new default densification strategy
- Enhanced MCP server (#984) — more automation tools for our agentic pipeline
- VRAM optimisations — critical for our dual-GPU setup
- Plugin marketplace (#914) — future distribution vehicle for our pipeline extensions

Two sync options were evaluated:

**Option A — Sync to v0.5.2 tag (2026-04-21)**: Stable release. Retains CUDA/OpenGL renderer. Delivers all high-value features above. Excludes Vulkan backend, asset manager, TCP event server, 8K image support, and the RmlUi UI.

**Option B — Sync to master (bleeding edge)**: Includes everything in Option A plus Vulkan, asset manager, TCP event server. Risk: coordinate cleanup (#1066, issue #1104 causes degenerate flat-plane output on ERP+GUT training), Python API changes, ImGui→RmlUi migration that may break any UI interactions, and v0.5.3 is not yet released.

This project is a private, production-oriented fork. The upstream is an open-source viewer/trainer by MrNeRF. We do not contribute back upstream and must not push to or open PRs against the upstream repository.

PRD Reference: Section 5.0 (Critical Decision: v0.5.2 Tag vs. Bleeding-Edge Master), Section 5.1–5.3, Section 11 (Risk Register).

## Decision

**Sync to the v0.5.2 stable tag. Defer the Vulkan migration (v0.5.3) to a separate, explicitly gated ADR-008.**

The merge process is:
1. Add the upstream remote (currently absent from the fork).
2. Create a dedicated sync branch: `sync/upstream-v0.5.2`.
3. Merge using `git merge v0.5.2 --no-ff`.
4. Resolve conflicts per BOUNDARIES.md rules: upstream directories (`src/core/`, `src/app/`, `src/mcp/`, `src/rendering/`, `src/training/`, etc.) accept upstream; our directories (`src/pipeline/`, `src/web/`, `docker/`) keep ours; build files (`CMakeLists.txt`, `vcpkg.json`) accept upstream then re-apply our additions; `README.md` keeps ours.
5. Rebuild the C++ core, verify the MCP server, run the full pipeline end-to-end.

**Isolation policy (binding constraint):** This fork is strictly one-way pull. We never push code, branches, or tags to `origin/MrNeRF/LichtFeld-Studio`. We never open pull requests against the upstream repository. Sync is always inbound: `git fetch upstream && git merge upstream/vX.Y.Z`. CI must have no upstream-push permissions.

## Rationale

- The high-value upstream features (USD I/O, mesh support, MRNF, VRAM fixes) are all present in v0.5.2 and absent from our current state; syncing is necessary.
- The Vulkan migration is a breaking change that cannot be validated until v0.5.3 is released and tested in a headless Docker container; proceeding with master today introduces unnecessary risk.
- BOUNDARIES.md already defines clean ownership boundaries that make the merge conflict surface predictable and manageable.
- Our pipeline uses LichtFeld via MCP for training, not for rendering. The CUDA renderer removal therefore does not block our training workflow, but it does block any rendering calls; separating the concerns lets us benefit from v0.5.2 now and absorb the Vulkan change carefully later.

## Consequences

### Positive
- Unlocks native USD I/O, mesh support, MRNF densification, enhanced MCP, and VRAM improvements within Weeks 1–2.
- Stable, released tag means known-good baseline; no in-progress upstream changes mid-integration.
- BOUNDARIES.md conflict-resolution rules are sufficient; no novel judgment calls needed.
- Upstream's coordinate-system cleanup (#1066) will not affect `coordinate_transform.py` until we later choose to sync to master.
- Clear, documented isolation policy prevents accidental upstream contribution and IP-boundary violations.

### Negative
- Approximately 195 commits worth of features (asset manager, TCP event server, 8K images, Vulkan-optimised selection tools) remain unavailable until ADR-008 is executed.
- The v0.5.3 `coordinate_transform.py` risk is deferred, not resolved; it must be addressed before any future master sync.
- Maintaining a fork that will diverge again means the next sync will carry its own conflict burden.

### Risks
- `mcp_client.py` may need minor updates due to enhanced MCP API changes in v0.5.2 (#984); post-merge integration testing required.
- If LichtFeld's native USD I/O does not support hierarchical scenes with per-object prims, `usd_assembler.py` deprecation is blocked and both paths coexist temporarily.

## Alternatives Considered

- **Sync to master immediately**: Rejected. Vulkan-only rendering in a headless container is unproven; CUDA renderer removal (#1234) may surface during training-side rendering calls; coordinate cleanup (#1066) is a known regression risk.
- **Do not sync (stay on fork as-is)**: Rejected. Missing USD I/O and MRNF densification are material capability gaps for the v2 pipeline; the VRAM pressure on dual-GPU is a production reliability concern.
- **Cherry-pick individual commits**: Rejected. The feature set we need spans hundreds of commits; cherry-picking creates a non-reproducible, hard-to-maintain history.

## Related Decisions

- ADR-001: Pipeline architecture (superseded by the v2 upgrade architecture defined in the PRD)
- ADR-008: Vulkan migration decision (deferred — trigger conditions defined there)
