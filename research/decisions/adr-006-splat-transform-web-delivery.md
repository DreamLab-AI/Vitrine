# ADR-006: splat-transform (PlayCanvas) for Web-Optimized Splat Delivery

## Status

Accepted

## Context

The v1 pipeline produces trained Gaussian splat files as raw `.ply` files. A typical scene produces 100+ MB of uncompressed PLY data. For web delivery this is prohibitive: large transfer sizes, no progressive loading, and no level-of-detail. Web-native splat viewers (PlayCanvas Engine, SuperSplat, 2Xplat) prefer compressed formats (`.splat`, `.ksplat`) that are significantly smaller and faster to parse.

`splat-transform` (https://github.com/playcanvas/splat-transform) is a JavaScript/CLI library published by PlayCanvas — the same organisation that ships the PlayCanvas 3D engine with native Gaussian splatting support. It provides:
- **compress**: Quantise SH coefficients, half-precision positions → significantly reduced file size
- **convert**: PLY ↔ `.splat` ↔ `.ksplat` format conversion
- **crop**: Remove Gaussians outside a bounding box (eliminates sky/ground noise)
- **filter**: Remove Gaussians by opacity threshold or scale (removes floaters post-training)
- **sort**: Reorder Gaussians for optimal front-to-back rendering order

This tool is production-ready: PlayCanvas ships it in their engine's build pipeline. It is actively maintained and uses an npm/Node.js toolchain.

The pipeline currently has no splat optimisation or web-delivery compression stage. Adding one requires:
1. Node.js in the main container
2. A new pipeline module (`splat_optimizer.py`)
3. A new stage entry in `stages.py` (`SPLAT_OPTIMIZE`, running after 3DGS training and before web delivery)
4. A new configuration namespace in `config.py`

PRD Reference: Section 3.1.2 (splat-transform), Section 4.2 (Phase 2 pipeline diagram), Section 6.1, Section 7 (Priority Matrix: P1, Low effort/Very low risk), Section 8 (Updated Pipeline Architecture).

## Decision

**Add `splat-transform` via npm to the main container and implement a `SPLAT_OPTIMIZE` pipeline stage.**

Installation in `Dockerfile.consolidated`:
```dockerfile
RUN apt-get install -y nodejs npm && \
    npm install -g @playcanvas/splat-transform@2.3.2
```

New pipeline module `src/pipeline/splat_optimizer.py` implements:
```python
@dataclass
class SplatOptConfig:
    crop_box: Optional[tuple[float, float, float, float, float, float]] = None
    opacity_threshold: float = 0.05
    max_scale: float = 5.0
    sort: bool = True
    compress: bool = True
    output_format: str = "ksplat"  # "ksplat" | "splat" | "ply"

class SplatOptimizer:
    def optimize(self, input_ply: Path, output_dir: Path, config: SplatOptConfig) -> SplatOptResult:
        """Pipeline: input.ply → crop → filter → sort → compress → output.ksplat"""
```

The `SPLAT_OPTIMIZE` stage runs immediately after the 3DGS training stage (LichtFeld or gsplat) and before the object segmentation stage. The compressed `.ksplat` file is the primary web-delivery artefact; the original `.ply` is retained as the source-of-truth for downstream mesh extraction.

`orchestrator.py` adds `SPLAT_OPTIMIZE` to the stage sequence. The stage is skippable via `config.pipeline.skip_splat_optimize = True` to preserve v1 behaviour.

CLI invocation (internal):
```bash
npx @playcanvas/splat-transform compress input.ply -o output.ksplat
npx @playcanvas/splat-transform crop input.ply --box "-10,-10,-10,10,10,10" -o cropped.ply
```

## Rationale

- PlayCanvas's production use of `splat-transform` in their engine pipeline is a strong maturity signal; this is not experimental research code.
- npm installation in the main container is low-risk: Node.js is a standard system package, and `splat-transform` has no CUDA or native extension dependencies.
- The PLY-first, compress-for-web approach preserves full fidelity for the mesh extraction backends while delivering a practical web asset.
- A web delivery size target of under 20 MB (vs. 100+ MB raw PLY) is achievable with compression + cropping; this directly improves the Flask web UI's download experience.

## Consequences

### Positive
- Web delivery size reduction from 100+ MB (raw PLY) to under 20 MB (compressed `.ksplat`).
- Floater removal via opacity/scale filtering produces cleaner scenes that are better for mesh extraction as well.
- `.ksplat` is natively supported by PlayCanvas Engine and SuperSplat without any additional viewer setup.
- Crop removes sky/ground clutter that inflates PLY size and degrades mesh extraction.
- The stage is purely additive and opt-out; no existing pipeline stage is changed.

### Negative
- Node.js adds approximately 100 MB to the main container image.
- The official package is `@playcanvas/splat-transform` (npm `playcanvas` org). An earlier draft of this ADR referenced an unofficial `@nicedoc/` re-publish; that was a supply-chain risk (see v2-security-audit FINDING-002) and has been corrected to the official package, pinned to a fixed version.
- The optimal crop bounding box is scene-dependent; auto-detection is not provided by `splat-transform` itself and must be estimated from the COLMAP sparse point cloud extent.

### Risks
- `.ksplat` format is a PlayCanvas-native format; if the PlayCanvas ecosystem loses momentum, format support in third-party viewers may erode. Mitigation: always retain the source `.ply`; `.ksplat` is a delivery artefact, not the master copy.
- `splat-transform` is a CLI tool; any breaking CLI interface change requires updating `splat_optimizer.py`. Pin the npm package version in the Dockerfile.

## Alternatives Considered

- **SuperSplat CLI**: SuperSplat (PlayCanvas's standalone editor) also offers compression. Rejected in favour of `splat-transform` because `splat-transform` is the library component, more suitable for CLI pipeline integration, and has a simpler dependency footprint.
- **Custom Python compressor**: Writing PLY quantisation in Python was considered. Rejected because `splat-transform` already implements this correctly and is maintained by domain experts; re-implementing it adds maintenance burden with no benefit.
- **Skip web compression entirely**: Rejected. 100+ MB download sizes are a material UX problem for the web UI; compression is P1 in the PRD priority matrix.

## Related Decisions

- ADR-001: Pipeline architecture (this ADR adds a new stage to the pipeline defined there)
- ADR-003: Pluggable mesh-extraction backends (the uncompressed PLY is still used by all backends; splat optimisation does not replace or modify the mesh path)
