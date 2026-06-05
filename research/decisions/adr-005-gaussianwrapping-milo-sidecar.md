# ADR-005: GaussianWrapping Integration Sharing the MILo Sidecar

## Status

Accepted Amended 2026-06-05.

## Amendment (2026-06-05) — optional thin-structure fallback only

GaussianWrapping remains an **optional** backend selected only for thin structures (via
`LICHTFELD_THIN_STRUCTURE=1` / the ADR-003 `auto` policy); the default is CoMe (ADR-004).
It loses to CoMe/PGSR on standard Tanks&Temples sampling. No LICENSE file as of writing —
treat as non-commercial. CLI flags remain inferred (verify post-build).

## Context

GaussianWrapping (https://github.com/diego1401/GaussianWrapping, 187 stars, last pushed 2026-05-19) reinterprets 3D Gaussians as "stochastic oriented surface elements" and extracts watertight, textured meshes with particular effectiveness on thin structures — bicycle spokes, wires, fences, railings, and thin architectural elements — that TSDF and standard marching cubes fail on. It accepts COLMAP-formatted datasets (images/ + sparse/), which is our standard pipeline intermediate.

Key environmental facts:
- GaussianWrapping supports **CUDA 11.8** and **Python 3.9**.
- The MILo sidecar already runs Ubuntu 22.04 / CUDA 11.8 / Python 3.9 — an exact environment match.
- GaussianWrapping provides two rasterisation backends: **RaDeGS** (higher quality) and a custom **median-depth** backend (faster, suitable for previews).
- GaussianWrapping also features "Primal Adaptive Meshing" for targeted high-resolution extraction in user-specified regions.

Sharing the MILo sidecar avoids the cost (image size, compose complexity, GPU scheduling overhead) of a third independent sidecar container for a backend that needs exactly the same runtime environment.

**Licensing note**: GaussianWrapping has no formal LICENSE file in its repository as of 2026-05-26. This is the same status as CoMe and must be treated identically.

PRD Reference: Section 3.1.5 (GaussianWrapping), Section 4.2 (Phase 2b), Section 6.4, Section 9 (Updated Docker Architecture), Section 13 Question 9.

## Decision

**Integrate GaussianWrapping into the existing MILo sidecar container (`docker/Dockerfile.milo`) and expose it as the fourth mesh-extraction backend via the ADR-003 uniform interface.**

Docker integration: append GaussianWrapping installation after the MILo installation steps in `docker/Dockerfile.milo`. GaussianWrapping and MILo do not conflict at the package level (both use CUDA 11.8 PyTorch); they are installed into separate directories (`/opt/milo` and `/opt/gaussianwrapping` respectively) and invoked via separate entry-point scripts. The sidecar service name remains `milo` in `docker-compose.consolidated.yml`; the compose label is updated to `milo-gw` as an alias.

The pipeline module `src/pipeline/gaussianwrapping_extractor.py` implements the ADR-003 uniform interface:

```python
@dataclass
class GaussianWrappingConfig:
    rasterizer: str = "radegs"   # "radegs" | "median_depth"
    adaptive_meshing: bool = False
    train_timeout: int = 2400
    extract_timeout: int = 900

def is_gaussianwrapping_available() -> bool:
    """Return True if the 'milo' sidecar is running and GaussianWrapping is installed at /opt/gaussianwrapping."""
    ...

def run_gaussianwrapping(colmap_dir: Path, output_dir: Path, config: GaussianWrappingConfig) -> MeshResult:
    """Run GaussianWrapping in the MILo sidecar; return MeshResult with textured PLY."""
    ...
```

`is_gaussianwrapping_available()` checks both that the `milo` service is healthy and that `/opt/gaussianwrapping` exists inside the sidecar (the latter guards against partially updated images).

**Rasteriser selection policy**: Default to `radegs` for production runs. Switch to `median_depth` automatically when `config.training.preview_mode = True` or when the `auto` backend selection policy is in speed-priority mode.

**Licensing gate**: Same build-arg gate as CoMe (`--build-arg ENABLE_GAUSSIANWRAPPING=1`). GaussianWrapping must not be included in commercial distribution images until a LICENSE file is reviewed. The `is_gaussianwrapping_available()` function logs a WARNING referencing this ADR when used outside development environments.

## Rationale

- Environment compatibility is exact: CUDA 11.8 / Python 3.9 is already the MILo sidecar specification. No new container is needed.
- Thin-structure scenes are a documented weakness of both TSDF and MILo's Delaunay triangulation approach. Providing a specialist backend for this failure mode materially improves pipeline robustness.
- Sharing the sidecar means GaussianWrapping costs zero additional memory overhead at idle: the `milo` container is already running when the pipeline is active.
- The `median_depth` rasteriser provides a useful preview-speed operating point without requiring separate infrastructure.

## Consequences

### Positive
- Four mesh-extraction backends available without four containers: TSDF (main), MILo (sidecar), CoMe (come sidecar), GaussianWrapping (milo sidecar).
- Thin-structure scenes that previously produced incomplete or inaccurate meshes now have a dedicated, validated extraction path.
- The MILo sidecar image size increase is bounded to the GaussianWrapping repository clone and its CUDA extension build (~1–2 GB additional layer).
- RaDeGS rasteriser is shared between MILo and GaussianWrapping; if it is already cached, the GaussianWrapping build step benefits.

### Negative
- The MILo sidecar Dockerfile becomes longer and carries two independent tools; maintenance burden increases.
- If GaussianWrapping introduces a conflicting pip dependency with MILo in the future, they would need to be separated into independent containers.
- `is_gaussianwrapping_available()` cannot detect a broken GaussianWrapping installation if the sidecar is healthy but the CUDA extension build failed; a smoke-test invocation at startup is recommended.
- No formal licence: same commercial-use block as CoMe; requires ongoing monitoring.

### Risks
- If the GaussianWrapping CUDA 11.8 extension build fails inside the MILo sidecar (e.g., due to a NVCC version edge case), it silently falls back to TSDF. The `--build-arg ENABLE_GAUSSIANWRAPPING=1` gate means this is a build-time failure rather than a runtime surprise.
- GaussianWrapping (187 stars) is research code that may change its interface. Pin to a specific commit in the Dockerfile.

## Alternatives Considered

- **Separate sidecar for GaussianWrapping**: Rejected. The environment is identical to the MILo sidecar; spinning up a third container for an identical runtime incurs unnecessary compose and scheduling overhead.
- **Install GaussianWrapping in the main container (CUDA 12.8)**: Rejected. GaussianWrapping's CUDA extensions target 11.8; the mismatch would require significant patching with uncertain outcome.
- **Use GaussianWrapping only offline (not in pipeline)**: Rejected. The value is in automated selection for thin-structure scenes; manual invocation defeats the purpose of the pluggable backend architecture.

## Related Decisions

- ADR-003: Pluggable mesh-extraction backend architecture (defines the interface this ADR implements)
- ADR-004: CoMe sidecar integration (companion backend; note different container and different licensing gate)
