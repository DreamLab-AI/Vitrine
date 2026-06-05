# ADR-004: CoMe Integration via Dedicated Sidecar Container

## Status

Accepted Amended 2026-06-05.

## Amendment (2026-06-05) — CoMe is now the default mesh backend

The project pivoted to CoMe as the **default** environment-mesh backend
(`config.training.mesh_method = "come"`), superseding the original opt-in/gated
positioning and ADR-012's earlier MILo-default. Rationale: CoMe is the verified best on
indoor scenes (ScanNet++ F1 0.668) at ~3x MILo's speed.

Operational notes:
- The default routes directly to CoMe and **silently falls back to TSDF** unless the
  image is built with `INSTALL_COME=1` and the `come` sidecar is running.
- **Licence (confirmed): CC BY-NC-ND 4.0** — non-commercial *and* no-derivatives. The
  strictest licence in the stack; fine under the research/non-commercial posture, a hard
  blocker for a product (swap to PGSR).
- `come_extractor.py`'s CLI flags are still inferred and must be verified against the
  released CoMe repo before the default is trusted in production.

## Context

CoMe (Confidence-based Mesh Extraction, https://github.com/r4dl/CoMe) is a 3DGS training method that augments each Gaussian with a per-Gaussian confidence value and then extracts a mesh via marching tetrahedra. On an RTX 4090, CoMe completes training plus extraction in approximately 25 minutes total, compared to MILo's 69 minutes, at comparable or better F1 scores (0.521 on Tanks & Temples, 0.662 on ScanNet++). CoMe accepts standard COLMAP dataset format (images/ + sparse/), which is exactly the intermediate format our pipeline produces after the SfM stage.

CoMe's code was released on 2026-04-22 (initial public release). Key environmental constraints identified during integration planning:

- CoMe requires **Python 3.10** and **CUDA 12.1**, with SOF-based CUDA extensions.
- Our main container runs Python 3.12 / CUDA 12.8 / Ubuntu 24.04 — Python version mismatch prevents installation in the main container.
- The MILo sidecar runs Python 3.9 / CUDA 11.8 — both version and CUDA level are incompatible with CoMe.
- Therefore CoMe cannot share either existing container and requires a new sidecar.

**Licensing risk**: As of 2026-05-26, the CoMe repository does not include a LICENSE file. The SPDX identifier is NOASSERTION. The underlying SOF method (which CoMe builds on) carries a custom academic license. This creates an unresolved commercial-use risk.

PRD Reference: Section 3.1.4 (CoMe), Section 4.2 (Phase 2b: Mesh Extraction Backends), Section 9 (Updated Docker Architecture), Section 13 Question 7.

## Decision

**Integrate CoMe as the third mesh-extraction backend, deployed in a dedicated `come` sidecar container, mirroring the MILo sidecar pattern.**

Container specification:
- Base image: Ubuntu 22.04
- CUDA: 12.1
- Python: 3.10
- Conda environment: `come`
- Service name in `docker-compose.consolidated.yml`: `come`
- Dockerfile: `docker/Dockerfile.come`

The sidecar is invoked via `docker exec come python /opt/come/extract.py` (the same `subprocess`-based invocation pattern used by `milo_extractor.py`).

The pipeline module `src/pipeline/come_extractor.py` implements the ADR-003 uniform interface:

```python
@dataclass
class CoMeConfig:
    iterations: int = 30_000      # train iterations
    confidence_threshold: float = 0.5
    mesh_resolution: str = "default"
    train_timeout: int = 2400
    extract_timeout: int = 600

def is_come_available() -> bool:
    """Return True if the 'come' Docker service is running and healthy."""
    ...

def run_come(colmap_dir: Path, output_dir: Path, config: CoMeConfig) -> MeshResult:
    """Train CoMe Gaussians, extract mesh via marching tetrahedra, return MeshResult."""
    ...
```

CoMe produces geometry-only PLY meshes. A separate texturing pass (xatlas UV unwrap + multi-view reprojection) is required to produce textured output, consistent with MILo's post-processing flow.

**Licensing gate**: CoMe must not be used in any build where commercial distribution is intended until a LICENSE file is published and reviewed by the project lead. The `come` container is build-arg-gated (`--build-arg ENABLE_COME=1`) so it is opt-in and excluded from production images by default until the licensing situation is resolved. The `is_come_available()` guard will log a WARNING referencing this ADR when the container is found available in a non-development environment.

## Rationale

- The 3x speed improvement (25 min vs. 69 min) directly impacts pipeline throughput; at scale this is a material operating-cost difference.
- The sidecar pattern is already proven by MILo. Reusing the same invocation mechanism (`docker exec` + subprocess) means no new infrastructure or orchestration primitives.
- A build-arg gate is the minimum viable mechanism to prevent accidental inclusion in production images pending licence resolution; it imposes no runtime overhead.
- Deferring commercial use rather than blocking development use lets the engineering team benchmark CoMe now and adopt it the moment a licence is published.

## Consequences

### Positive
- Mesh extraction time target drops from ~69 min (MILo) to ~25 min (CoMe) on RTX 4090.
- Confidence mechanism reduces floater artefacts that TSDF and plain marching cubes struggle with.
- F1 scores on standard benchmarks are competitive with MILo.
- Container isolation means CoMe's Python 3.10 / CUDA 12.1 environment cannot destabilise the main container.

### Negative
- A third container increases Docker compose complexity and GPU memory scheduling overhead.
- The separate texturing pass (xatlas) adds latency relative to methods that produce textured output natively (SuGaR, MILo with texture).
- No LICENSE file as of integration date: commercial use is blocked until resolved; monitoring of the upstream repository is required.
- NOASSERTION SPDX status means automated licence scanners will flag this container in any compliance pipeline.

### Risks
- If CoMe's code never receives a permissive or clear commercial licence, the backend must remain development-only or be removed entirely. MILo remains the production fallback.
- CoMe code was released only 34 days before this ADR; the API may still change significantly in early releases.
- SOF dependency carries its own custom academic licence; the interaction between SOF's licence and any CoMe licence must be evaluated together.

## Alternatives Considered

- **Install CoMe in the MILo sidecar (CUDA 11.8)**: Rejected. CoMe requires CUDA 12.1 CUDA extensions; CUDA 11.8 is incompatible. Attempting a multi-CUDA sidecar increases image complexity without benefit.
- **Install CoMe in the main container (CUDA 12.8)**: Rejected. CoMe requires Python 3.10; main container runs Python 3.12. Python version downgrade would break 28 existing pipeline modules.
- **Defer CoMe entirely until licence is resolved**: Rejected. Development and benchmarking can proceed under development-only terms; the build-arg gate prevents accidental production use. Blocking integration would delay the benchmarking data needed to make the MILo→CoMe migration decision.

## Related Decisions

- ADR-003: Pluggable mesh-extraction backend architecture (defines the interface this ADR implements)
- ADR-005: GaussianWrapping integration (companion mesh backend using a different sidecar)
