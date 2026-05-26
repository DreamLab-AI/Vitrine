# ADR-003: Pluggable Mesh-Extraction Backend Architecture

## Status

Accepted

## Context

The v1 pipeline supported two mesh-extraction backends: TSDF (fast, lower-quality preview) and MILo (high-quality, ~69 min on RTX 4090). The v2 upgrade introduces two additional backends — CoMe (~25 min, confidence-based) and GaussianWrapping (thin-structure specialist) — raising the total to four. Without a principled architecture, adding each backend would require modifying `mesh_extractor.py`, `stages.py`, `orchestrator.py`, and `config.py` independently, creating tight coupling, brittle dispatch logic, and no clear extension contract for future backends.

Additionally, each backend carries different container requirements:
- TSDF: runs in the main container (CUDA 12.8, Python 3.12)
- MILo: requires a sidecar (CUDA 11.8, Python 3.9)
- CoMe: requires a different sidecar (CUDA 12.1, Python 3.10)
- GaussianWrapping: shares the MILo sidecar (CUDA 11.8)

The existing `milo_extractor.py` already defines a pattern that has proven workable:
- A dataclass (`MiloConfig`) holding all backend-specific parameters
- A standalone availability check (`is_milo_available() -> bool`)
- A standalone entry point (`run_milo(colmap_dir, output_dir, config) -> MeshResult`)

This pattern must be generalised and made the binding contract for all backends.

PRD Reference: Section 3.1.4 (CoMe), Section 3.1.5 (GaussianWrapping), Section 4.2 (Phase 2b), Section 8 (Updated Pipeline Architecture), Section 13 Question 9 (backend auto-selection heuristic).

## Decision

**Adopt a uniform extractor interface** modelled on `milo_extractor.py`. Every mesh-extraction backend must expose exactly three public symbols:

```python
@dataclass
class XConfig:
    """All backend-specific parameters with documented defaults."""
    ...

def is_X_available() -> bool:
    """Return True if the backend's sidecar/dependencies are reachable."""
    ...

def run_X(colmap_dir: Path, output_dir: Path, config: XConfig) -> MeshResult:
    """Run the backend; raise MeshExtractionError on failure."""
    ...
```

where `X` is the backend name (`tsdf`, `milo`, `come`, `gaussianwrapping`).

**Backend selection** is driven by a single key in the pipeline configuration:

```
config.training.mesh_method: "tsdf" | "milo" | "come" | "gaussianwrapping" | "auto"
```

When `mesh_method = "auto"`, `stages.py` applies the following decision policy:

| Condition | Selected backend |
|-----------|-----------------|
| `preview=True` or `quality_gate.mesh_speed_priority` | `tsdf` |
| Scene contains thin structures (bicycle, fence, wire, railing detected by SAM label heuristic) | `gaussianwrapping` |
| Speed is priority and CoMe sidecar is available | `come` |
| Default high-quality path | `milo` |
| CoMe unavailable and MILo unavailable | `tsdf` (fallback) |

Dispatch lives entirely in `stages.py` in a single `_select_mesh_backend()` function that calls each backend's `is_X_available()` guard before committing to a selection. `orchestrator.py` passes the resolved backend name as a stage parameter; it does not perform selection itself.

`mesh_extractor.py` is retained for TSDF only (it already implements the `is_tsdf_available` / `run_tsdf` / `TsdfConfig` contract implicitly) and will be refactored to expose the explicit interface.

New backend modules follow the naming convention `{name}_extractor.py` in `src/pipeline/`.

## Rationale

- Matching the established `milo_extractor.py` pattern means the interface is already battle-tested and developers have a concrete reference.
- Centralising dispatch in `stages.py` and keeping selection logic out of `orchestrator.py` respects the existing layer boundaries in the pipeline.
- The `is_X_available()` guard prevents silent failures when a sidecar container is not running; the caller gets an actionable error rather than a timeout.
- The `auto` heuristic encodes the cost/quality trade-offs agreed in the PRD priority matrix (TSDF < GaussianWrapping < CoMe < MILo in quality; inverted in speed), while remaining overridable by explicit config for reproducible runs.

## Consequences

### Positive
- Adding a fifth backend in the future requires writing one module (`{name}_extractor.py`) and one entry in the dispatch table; no changes to `orchestrator.py` or `config.py` schema.
- `is_X_available()` enables graceful degradation at runtime; the pipeline never hangs waiting for a dead sidecar.
- The `auto` policy captures expert knowledge in one auditable location.
- The interface is testable in isolation: each backend's `run_X` can be unit-tested against fixture data without a running pipeline.

### Negative
- `mesh_extractor.py` must be refactored to expose the explicit interface, requiring a coordinated rename of internal functions.
- The `auto` thin-structure heuristic depends on SAM label output; if SAM labels are absent or wrong, the wrong backend may be selected. Config override is the escape hatch.
- Four backends means four sidecar health checks on pipeline startup, adding ~1–2 s of latency to the preflight step.

### Risks
- If a future backend requires a fundamentally different input contract (e.g., pre-trained Gaussian weights rather than COLMAP directory), `run_X` signature may need versioning. Mitigation: `XConfig` absorbs optional fields; `run_X` ignores what it does not need.

## Alternatives Considered

- **Inheritance / abstract base class**: Rejected. Python ABC hierarchies across separate sidecar processes add import-time coupling without benefit; the function-level protocol is simpler and sufficient.
- **Plugin registry with entry-points**: Rejected. Over-engineering for a fixed four-backend set; the naming convention plus dispatch table in `stages.py` is more legible.
- **Single `mesh_extractor.py` with per-backend branches**: Rejected. This was the v1 approach and is what we are replacing; it makes backends hard to test and impossible to add without touching the core file.

## Related Decisions

- ADR-001: Pipeline architecture (this ADR extends the mesh-extraction layer defined there)
- ADR-004: CoMe integration — implements the interface defined here
- ADR-005: GaussianWrapping integration — implements the interface defined here
