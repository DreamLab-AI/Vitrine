# ADR-008: Defer the Vulkan Migration (v0.5.3)

## Status

Proposed / Deferred

## Context

Upstream LichtFeld Studio's development branch (unreleased v0.5.3) has completed a full rendering backend migration:
- OpenGL renderer: **removed** (PR #1170)
- CUDA renderer: **removed** (PR #1234)
- Vulkan renderer via VkSplat (FastGS ported to Vulkan): **sole rendering backend** (PR #1162)

Additional changes in v0.5.3 that interact with our pipeline:

| Change | PR | Risk to our fork |
|--------|----|-----------------|
| CUDA renderer removed | #1234 | Any pipeline path that invokes LichtFeld rendering (not just training) will break immediately |
| Coordinate system cleanup | #1066 | Known regression: issue #1104 reports ERP+GUT training produces degenerate flat-plane output. Our `coordinate_transform.py` may be affected |
| ImGui → RmlUi UI migration | Multiple | Any code that interacts with LichtFeld's UI layer must be updated; currently none in our pipeline, but a risk surface exists |
| Python API / MCP API changes | Multiple "stale python api" fixes | Our `mcp_client.py` calls the MCP server; changed API signatures will cause runtime errors |
| TCP event server | #1231 | New capability (not a risk), but replacing MCP-based progress monitoring with TCP events requires changes to `orchestrator.py` |
| CLI background mode | #1246 | New headless argument scheme; our CLI invocation in `mcp_client.py` may need updating |

Critically, v0.5.3 has **not been released** as a stable tag. The changes are on an active development branch where the Vulkan migration is still in progress.

Our pipeline uses LichtFeld primarily via the MCP server for training orchestration. It does not directly call LichtFeld's rendering API. However:
1. Some MCP tools internally invoke rendering to produce evaluation images; those will break with CUDA renderer removed.
2. Headless Vulkan in a Docker container requires `vulkan-tools`, `mesa-vulkan-drivers`, and a Vulkan-capable GPU driver; our current `Dockerfile.consolidated` does not include these.
3. The coordinate-system cleanup (#1066) and its known regression (issue #1104) are unresolved upstream; merging a broken coordinate system into our fork would propagate the regression to `coordinate_transform.py`.

The stable v0.5.2 sync (ADR-002) delivers all the high-value features we need. The Vulkan migration delivers future-proofing but no immediate capability gain for our video-to-3D pipeline.

PRD Reference: Section 2.2 (v0.5.3-dev table), Section 5.0 (Critical Decision), Section 7 (Priority Matrix: P3, Week 6+), Section 11 (Risk Register), Section 13 Questions 2, 12, 14.

## Decision

**Defer the Vulkan migration until all trigger conditions below are met. Do not merge upstream master or any v0.5.3-dev commits until this ADR is revisited and updated to Accepted.**

### Trigger conditions for revisiting this ADR

All of the following must be satisfied before the Vulkan migration is pursued:

1. **v0.5.3 is released as a stable tagged version** by MrNeRF. An unreleased development branch is not an acceptable sync target for production infrastructure.

2. **Coordinate-system regression is resolved upstream**: Issue #1104 (ERP+GUT degenerate flat-plane output introduced by #1066) must be closed as fixed, or a documented workaround for `coordinate_transform.py` must be available.

3. **Headless Vulkan validated in a Docker container**: A proof-of-concept `docker run` demonstrating that LichtFeld's Vulkan renderer starts and completes a training job headlessly on our GPU model (NVIDIA RTX, `sm_89` or compatible) with `mesa-vulkan-drivers`. This test must be documented before the merge.

4. **MCP API compatibility verified**: `mcp_client.py` must be tested against the v0.5.3 MCP server. Any changed tool signatures must be documented and the migration scope assessed.

5. **Python API audit complete**: All "stale python api" upstream fixes must be reviewed against our pipeline modules. A compatibility matrix must be produced.

### What happens in the interim

- The main working branch continues on v0.5.2 (per ADR-002).
- Upstream master is tracked in a dedicated branch (`upstream/master-watch`) and reviewed monthly for progress on the trigger conditions above.
- Any non-Vulkan features from v0.5.3 that are independently useful (e.g., asset manager, 8K image support) may be cherry-picked individually after review, provided they do not pull in Vulkan dependencies.
- The `Dockerfile.consolidated` must not add Vulkan system packages until this ADR transitions to Accepted.

## Rationale

- Our pipeline value is in training and mesh extraction, not in interactive rendering. The Vulkan migration is architecturally significant upstream but has near-zero day-one value for our headless Docker workflow.
- The coordinate-system regression (issue #1104) is a concrete, documented correctness risk. Merging it before it is fixed would require us to maintain a local patch, which conflicts with our BOUNDARIES.md policy of accepting upstream changes in upstream directories.
- Headless Vulkan in Docker is a non-trivial infrastructure requirement. GPU driver Vulkan ICD availability in our container base image (`nvcr.io/nvidia/cuda:12.8-devel-ubuntu24.04`) is not guaranteed; validating this before committing is responsible engineering.
- v0.5.3 is unreleased. Syncing to an in-progress development branch means we absorb the churn of an active feature branch rather than a tested release.

## Consequences

### Positive
- No coordinate-system regression risk in `coordinate_transform.py` until explicitly resolved.
- No Vulkan infrastructure setup work required until a stable release exists.
- No `mcp_client.py` API migration work in the near term.
- Decouples the high-value v0.5.2 sync (ADR-002) from the high-risk Vulkan migration; each can be validated independently.

### Negative
- Features exclusive to v0.5.3 (asset manager, TCP event server, 8K image support, Vulkan-optimised selection tools, RmlUi) remain unavailable until the migration is executed.
- The longer the deferral, the larger the future merge surface becomes as upstream continues to develop on the Vulkan baseline.
- If upstream stops maintaining the v0.5.2 tag (e.g., critical security fixes only applied to master), our risk calculus changes.

### Risks
- **Upstream abandons CUDA/OpenGL entirely**: If v0.5.3 becomes the de facto release and v0.5.2 stops receiving fixes, we will be forced to execute the Vulkan migration on a tighter timeline than planned. Mitigation: the `upstream/master-watch` branch provides continuous visibility.
- **Python API divergence compounds**: Each month on v0.5.2 while upstream evolves the Python API increases the eventual migration scope for `mcp_client.py`. Mitigation: the monthly watch review should flag API changes immediately.
- **TCP event server (#1231) offers a better monitoring path**: If the TCP event server proves superior to MCP-based progress monitoring, deferral means we continue on the inferior path longer. This is an accepted trade-off; the TCP server is an enhancement, not a fix.

## Alternatives Considered

- **Sync to master immediately**: Rejected. Unresolved coordinate regression (#1104), unvalidated headless Vulkan, unreleased tag — three concurrent risks that all require investigation before production merge. The risk matrix is unfavourable given that v0.5.2 provides all the immediate capability needs.
- **Maintain a local Vulkan patch set**: Rejected. This would require modifying upstream directories (violates BOUNDARIES.md) and creates an ongoing patch rebase burden.
- **Cherry-pick only Vulkan PRs (#1162, #1170, #1234) without the full v0.5.3 sync**: Rejected. These PRs are deeply interdependent; selective cherry-picking from an in-progress migration is high-risk and likely to produce a broken intermediate state.

## Related Decisions

- ADR-002: Upstream sync to v0.5.2 (this ADR documents why v0.5.3 is excluded from that sync)
- ADR-001: Pipeline architecture (Vulkan migration will require re-validation of any pipeline stage that interacts with the LichtFeld rendering layer)
