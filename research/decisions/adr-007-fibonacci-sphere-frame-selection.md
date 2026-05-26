# ADR-007: Fibonacci-Sphere Frame Selection for Optimal Viewpoint Coverage

## Status

Accepted

## Context

The current `frame_selector.py` scores frames primarily by blur quality (Laplacian variance), exposure quality, and temporal subsampling. It does not account for viewpoint coverage: two frames that are sharp and well-exposed but taken from nearly identical camera positions provide redundant information for 3DGS training and COLMAP SfM. Gaps in angular coverage cause floaters and incomplete reconstructions in the corresponding scene regions.

The Fibonacci sphere (also called Fibonacci lattice on the sphere) is a well-known algorithm that distributes N points near-uniformly across the unit sphere with O(N log N) worst-case angular dispersion. It is used in rendering (importance sampling), computational geometry, and — relevantly — in cinema camera array design for optimal viewpoint coverage. The COLMAP Bridge Cinema 4D plugin (radiancefields.com) specifically uses Fibonacci sphere camera patterns for generating training data with optimal coverage; the same mathematical principle applies to scoring real-video frame sequences.

The algorithm requires only NumPy (already a pipeline dependency). No new dependencies, no container changes, no GPU requirements.

The enhancement applies to the existing `frame_selector.py` scoring function: frames are scored by their angular proximity to an under-covered Fibonacci sphere direction, with the quality score (blur + exposure) acting as a tiebreaker.

PRD Reference: Section 3.3.3 (COLMAP Bridge — Fibonacci concept), Section 4.2 (new `fibonacci_sampler.py` module), Section 6.2, Section 7 (Priority Matrix: P1, Low effort).

## Decision

**Add Fibonacci-sphere viewpoint scoring to `frame_selector.py` as an additional scoring term, weighted and combined with the existing blur and exposure quality scores.**

The core algorithm is:

```python
def fibonacci_sphere_score(camera_positions: np.ndarray, n_target: int) -> np.ndarray:
    """Score frames by coverage of a Fibonacci sphere distribution.

    Each frame is scored by how close its normalised camera direction is to
    the nearest under-represented Fibonacci sphere point. Higher score means
    the frame fills a coverage gap.

    Args:
        camera_positions: (N, 3) array of camera centres in world space.
        n_target: Target number of frames to select.

    Returns:
        (N,) array of coverage scores in [0, 1].
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_target)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_target)
    fib_points = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=-1)

    center = camera_positions.mean(axis=0)
    dirs = camera_positions - center
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Nearest-Fibonacci-point angular distance for each camera
    cos_sims = dirs @ fib_points.T          # (N, n_target)
    nearest = cos_sims.max(axis=1)          # (N,)
    scores = (nearest - nearest.min()) / (nearest.max() - nearest.min() + 1e-8)
    return scores
```

The combined frame score is:
```
score = w_quality * quality_score + w_coverage * fibonacci_sphere_score
```
with defaults `w_quality = 0.6`, `w_coverage = 0.4`, configurable via `config.frame_selector.coverage_weight`.

The algorithm is extracted into a standalone helper module `src/pipeline/fibonacci_sampler.py` and imported by `frame_selector.py`. This keeps `frame_selector.py` focused on orchestration and makes the algorithm independently testable.

Camera positions are obtained from the COLMAP sparse reconstruction (`cameras.bin` / `cameras.txt`) parsed by `colmap_parser.py`. When COLMAP output is not yet available (e.g., pre-SfM frame selection), the fallback is sequential temporal subsampling (v1 behaviour) with Fibonacci scoring applied in a second pass after SfM.

The setting `config.frame_selector.use_fibonacci = True` (default) enables this behaviour; setting to `False` restores the v1 quality-only scoring for reproducibility.

## Rationale

- Uniform angular coverage is the correct objective for multi-view reconstruction; quality scores alone do not capture it.
- The Fibonacci sphere is provably near-optimal for uniform sphere coverage and is O(N log N) to compute — negligible compared to COLMAP runtimes.
- The algorithm has zero new runtime dependencies and zero container changes; it is a pure Python/NumPy addition.
- Weighting quality and coverage rather than replacing quality scoring ensures we do not select a well-placed but blurry or overexposed frame over a slightly less well-placed but sharp one.

## Consequences

### Positive
- Improved viewpoint coverage reduces floaters in under-sampled scene regions.
- Better COLMAP reconstructions from the same video input, benefiting all downstream stages (training, mesh extraction).
- No new dependencies; the change is contained to `frame_selector.py` and a new `fibonacci_sampler.py`.
- The weight parameter allows tuning per-scene if needed; the default covers the common case.
- The algorithm is independently testable with synthetic camera positions.

### Negative
- Camera positions are only available after COLMAP SfM; for the pre-SfM frame selection pass, coverage scoring cannot be applied (falls back to v1).
- The combined score may de-prioritise frames that have excellent quality (very sharp, perfect exposure) but happen to be near existing coverage. In practice this is the correct trade-off, but it may surprise users who expect the "sharpest" frames to always win.
- The `coverage_weight` hyperparameter requires documentation and potentially per-scene tuning for videos with highly non-uniform camera motion (e.g., drone footage that circles one side only).

### Risks
- If COLMAP `cameras.bin` parsing fails or produces degenerate positions (all cameras at the same location), the Fibonacci score is undefined. `fibonacci_sampler.py` must guard against zero-norm directions and fall back to uniform coverage scores.
- The definition of "camera position" (camera centre in world space) assumes the COLMAP coordinate frame is meaningful as an angular reference. For forward-facing videos with very small baselines, the Fibonacci sphere model is a poor fit; the weight should be reduced for these scenes.

## Alternatives Considered

- **Farthest-point sampling on camera positions**: Would maximise pairwise angular distance between selected frames but is O(N²) and does not directly target the Fibonacci distribution. Rejected in favour of the Fibonacci approach which has a known optimality guarantee.
- **Replace quality scoring with coverage scoring**: Rejected. A perfectly placed but blurry frame contributes noise to training; quality remains a necessary filter.
- **Import a full camera-placement library**: Rejected. The Fibonacci sphere algorithm is 15 lines of NumPy; a library dependency would be disproportionate.

## Related Decisions

- ADR-001: Pipeline architecture (this ADR modifies the frame selection stage of the pipeline)
- ADR-002: Upstream sync (post-sync, MRNF densification benefits from better frame coverage; these improvements are complementary)
