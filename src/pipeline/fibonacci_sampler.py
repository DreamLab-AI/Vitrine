# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fibonacci-sphere viewpoint coverage scoring for frame selection.

Provides pure-NumPy utilities for scoring camera frames by how well they
cover a Fibonacci-sphere distribution of viewpoints.  No new runtime
dependencies are introduced beyond NumPy (already a pipeline dependency).

The Fibonacci sphere (golden-ratio spiral on the unit sphere) provides a
near-optimal uniform distribution of N points, and is used here as the
target distribution for viewpoint coverage.  Frames that fill angular
gaps in the Fibonacci distribution receive higher coverage scores.

ADR Reference: ADR-007 (Fibonacci-sphere frame selection decision).
PRD Reference: Section 3.3.3 (COLMAP Bridge concept), Section 4.2,
               Section 6.2 (algorithm spec).

Typical usage (integration into frame_selector.py)::

    from pipeline.fibonacci_sampler import (
        fibonacci_sphere,
        fibonacci_coverage_score,
        select_frames_by_coverage,
    )

    fib_pts = fibonacci_sphere(n_target)
    coverage = fibonacci_coverage_score(camera_positions, n_target)
    selected = select_frames_by_coverage(
        camera_positions, quality_scores, n_select=200
    )

The combined frame score follows ADR-007::

    score = 0.6 * quality_score + 0.4 * coverage_score
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Golden ratio constant -- the irrational number that drives the Fibonacci
# spiral and guarantees minimal angular discrepancy.
_GOLDEN_RATIO: float = (1.0 + np.sqrt(5.0)) / 2.0

# ADR-007 default weights for quality vs coverage.
_DEFAULT_COVERAGE_WEIGHT: float = 0.4
_DEFAULT_QUALITY_WEIGHT: float = 0.6


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n near-uniformly distributed unit vectors via the golden-ratio spiral.

    The Fibonacci sphere maps integer indices to sphere coordinates using the
    golden-ratio increment in azimuth and the arccosine of a linear ramp in
    elevation.  This produces a near-optimal low-discrepancy point set on
    S^2 with O(N log N) worst-case angular gap.

    Args:
        n: Number of points to generate.  Must be >= 1.

    Returns:
        Float64 array of shape ``(n, 3)`` where each row is a unit vector
        ``[x, y, z]`` on the unit sphere.

    Raises:
        ValueError: If ``n < 1``.

    Example::

        pts = fibonacci_sphere(100)
        assert pts.shape == (100, 3)
        norms = np.linalg.norm(pts, axis=1)
        assert np.allclose(norms, 1.0)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    indices = np.arange(n, dtype=np.float64)

    # Elevation: arccosine of a linearly-spaced ramp in [-1, 1].
    # Using (indices + 0.5) / n avoids the degenerate poles.
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)  # polar angle

    # Azimuth: golden-ratio increment gives irrational stepping.
    theta = 2.0 * np.pi * indices / _GOLDEN_RATIO  # azimuthal angle

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=-1)  # (n, 3)


def fibonacci_coverage_score(
    camera_positions: np.ndarray,
    n_target: int,
) -> np.ndarray:
    """Score each camera by how well it covers an under-represented Fibonacci direction.

    Algorithm:
        1. Normalise camera positions to the unit sphere around their centroid.
        2. Generate the Fibonacci sphere with ``n_target`` target points.
        3. Greedily assign each camera to its nearest Fibonacci point (by
           cosine similarity); each Fibonacci point can be claimed at most
           once.  Cameras that claim a previously unclaimed Fibonacci point
           receive a higher score.
        4. Scores are normalised to [0, 1] based on the minimum cosine
           similarity to the assigned (or nearest) Fibonacci point.

    The intent is to reward cameras that fill angular coverage gaps rather
    than cluster around already-represented directions.

    Args:
        camera_positions: Float array of shape ``(N, 3)`` containing camera
            centres in world space (e.g., COLMAP camera positions).
        n_target: Number of target viewpoints (Fibonacci sphere resolution).
            Typically the number of frames to be selected.

    Returns:
        Float64 array of shape ``(N,)`` with per-camera coverage scores in
        [0, 1].  Higher values indicate the camera covers an angular region
        not well-represented by other cameras.  Returns a uniform array of
        ones if all cameras are co-located (degenerate input).

    Raises:
        ValueError: If ``camera_positions`` is not a 2-D array with 3 columns.

    Notes:
        - When multiple cameras map to the same Fibonacci point, only the
          first (by input order) receives the "unique coverage" bonus; the
          rest receive a score proportional to their cosine similarity to the
          nearest Fibonacci point they can still claim.
        - For forward-facing video with a very small baseline, all cameras
          may normalise to nearly the same direction and the scores will be
          near-uniform.  Callers should reduce ``coverage_weight`` in that
          scenario (ADR-007 Risks section).
    """
    camera_positions = np.asarray(camera_positions, dtype=np.float64)

    if camera_positions.ndim != 2 or camera_positions.shape[1] != 3:
        raise ValueError(
            f"camera_positions must be shape (N, 3), got {camera_positions.shape}"
        )

    n_cameras = camera_positions.shape[0]

    # -- Step 1: normalise to unit sphere around centroid --------------------
    centroid = camera_positions.mean(axis=0)
    directions = camera_positions - centroid

    norms = np.linalg.norm(directions, axis=1, keepdims=True)

    # Guard against degenerate (co-located) cameras.
    if np.all(norms < 1e-9):
        logger.warning(
            "fibonacci_coverage_score: all cameras are co-located; "
            "returning uniform coverage scores"
        )
        return np.ones(n_cameras, dtype=np.float64)

    # Clamp near-zero norms to avoid division by zero for individual cameras.
    norms = np.maximum(norms, 1e-9)
    directions = directions / norms  # (N, 3), unit vectors

    # -- Step 2: generate Fibonacci target points ----------------------------
    fib_pts = fibonacci_sphere(n_target)  # (n_target, 3)

    # -- Step 3: compute cosine similarities between all cameras and Fibonacci
    # points.  Both are unit vectors so cos_sim = dot product.
    cos_sims = directions @ fib_pts.T  # (N, n_target)

    # -- Step 4: greedy assignment -- cameras sorted by their best cosine
    # similarity (most confident match first).
    claimed: set[int] = set()
    # Per-camera score: start at a baseline of nearest-fib cosine similarity.
    raw_scores = cos_sims.max(axis=1)  # (N,)  nearest fib-point similarity

    # Process cameras in order of their best match confidence (descending)
    # so high-confidence, unique-direction cameras claim their Fibonacci point.
    assignment_order = np.argsort(-raw_scores)  # descending

    assigned_fib = np.full(n_cameras, -1, dtype=np.int64)
    for cam_idx in assignment_order:
        # Find the best unclaimed Fibonacci point for this camera.
        sorted_fib = np.argsort(-cos_sims[cam_idx])
        for fib_idx in sorted_fib:
            if fib_idx not in claimed:
                claimed.add(fib_idx)
                assigned_fib[cam_idx] = fib_idx
                break
        # If all Fibonacci points are claimed, the camera keeps its nearest
        # (already set in raw_scores via cos_sims.max).

    # Cameras that claimed a unique Fibonacci point keep their cosine score.
    # Cameras that could not claim a unique point receive a lower base score
    # (their nearest Fibonacci similarity minus a penalty).
    # We express this as: cameras with unique assignments score higher.
    unique_mask = assigned_fib >= 0
    scores = raw_scores.copy()

    # Apply a modest penalty to cameras that share their closest Fibonacci
    # point direction with an already-claimed camera.
    non_unique_penalty = 0.3
    scores[~unique_mask] = np.maximum(
        0.0, scores[~unique_mask] - non_unique_penalty
    )

    # -- Step 5: normalise to [0, 1] -----------------------------------------
    s_min, s_max = scores.min(), scores.max()
    span = s_max - s_min
    if span < 1e-9:
        return np.ones(n_cameras, dtype=np.float64)

    return (scores - s_min) / span


def select_frames_by_coverage(
    camera_positions: np.ndarray,
    quality_scores: np.ndarray,
    n_select: int,
    coverage_weight: float = _DEFAULT_COVERAGE_WEIGHT,
) -> list[int]:
    """Select frame indices that balance viewpoint coverage and quality.

    Combines per-frame quality scores (blur, exposure) with Fibonacci-sphere
    coverage scores using the weights defined in ADR-007::

        combined = (1 - w) * quality + w * coverage

    where ``w`` is ``coverage_weight`` (default 0.4).

    Args:
        camera_positions: Float array of shape ``(N, 3)`` with camera centres
            in world space.  Obtain from COLMAP ``cameras.bin`` / ``cameras.txt``
            via ``colmap_parser.py``.
        quality_scores: Float array of shape ``(N,)`` with per-frame quality
            scores (e.g., blur + exposure).  Values should be in [0, 1] or
            will be normalised internally.
        n_select: Number of frames to select.  If ``n_select >= N``, all
            indices are returned in combined-score order.
        coverage_weight: Weight given to coverage vs. quality.  ADR-007
            default is 0.4.  Must be in [0, 1].

    Returns:
        List of ``n_select`` integer indices into the input arrays, sorted in
        descending combined-score order (best first).

    Raises:
        ValueError: If array shapes are inconsistent, ``n_select < 1``, or
            ``coverage_weight`` is outside [0, 1].

    Example::

        positions = np.random.randn(500, 3)
        quality = np.random.rand(500)
        selected = select_frames_by_coverage(positions, quality, n_select=150)
        assert len(selected) == 150
    """
    camera_positions = np.asarray(camera_positions, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    if camera_positions.ndim != 2 or camera_positions.shape[1] != 3:
        raise ValueError(
            f"camera_positions must be shape (N, 3), got {camera_positions.shape}"
        )
    if quality_scores.ndim != 1:
        raise ValueError(
            f"quality_scores must be 1-D, got shape {quality_scores.shape}"
        )
    n_cameras = camera_positions.shape[0]
    if quality_scores.shape[0] != n_cameras:
        raise ValueError(
            f"quality_scores length {quality_scores.shape[0]} does not match "
            f"camera_positions rows {n_cameras}"
        )
    if n_select < 1:
        raise ValueError(f"n_select must be >= 1, got {n_select}")
    if not (0.0 <= coverage_weight <= 1.0):
        raise ValueError(
            f"coverage_weight must be in [0, 1], got {coverage_weight}"
        )

    quality_weight = 1.0 - coverage_weight

    # Normalise quality scores to [0, 1].
    q_min, q_max = quality_scores.min(), quality_scores.max()
    q_span = q_max - q_min
    norm_quality = (
        (quality_scores - q_min) / q_span if q_span > 1e-9
        else np.ones(n_cameras, dtype=np.float64)
    )

    coverage = fibonacci_coverage_score(camera_positions, n_target=max(n_select, n_cameras))

    combined = quality_weight * norm_quality + coverage_weight * coverage

    # Return the top n_select indices sorted by descending combined score.
    k = min(n_select, n_cameras)
    top_indices = np.argsort(-combined)[:k]

    logger.info(
        "select_frames_by_coverage: selected %d/%d frames "
        "(coverage_weight=%.2f, quality_weight=%.2f)",
        k,
        n_cameras,
        coverage_weight,
        quality_weight,
    )

    return top_indices.tolist()


if __name__ == "__main__":
    """Self-test with synthetic camera positions."""

    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rng = np.random.default_rng(seed=42)

    # Synthetic: 300 cameras arranged roughly around a hemisphere
    # (simulating a drone orbiting a scene from varying heights).
    n_cams = 300
    theta = rng.uniform(0, 2 * np.pi, n_cams)
    phi = rng.uniform(np.pi / 6, np.pi / 2, n_cams)  # 30–90 deg elevation
    radius = rng.uniform(3.0, 5.0, n_cams)

    positions = np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        axis=-1,
    )  # (300, 3)

    # Synthetic quality scores: blur + exposure proxy.
    quality = rng.uniform(0.2, 1.0, n_cams)

    # Test 1: fibonacci_sphere
    pts = fibonacci_sphere(50)
    assert pts.shape == (50, 3), f"Expected (50,3) got {pts.shape}"
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9), "Fibonacci points must be unit vectors"
    print(f"fibonacci_sphere(50): OK — first point: {pts[0]}")

    # Test 2: fibonacci_coverage_score
    scores = fibonacci_coverage_score(positions, n_target=150)
    assert scores.shape == (n_cams,), f"Expected ({n_cams},) got {scores.shape}"
    assert scores.min() >= 0.0 and scores.max() <= 1.0 + 1e-9, (
        f"Scores out of range: [{scores.min():.4f}, {scores.max():.4f}]"
    )
    print(
        f"fibonacci_coverage_score: OK — "
        f"mean={scores.mean():.3f}, min={scores.min():.3f}, max={scores.max():.3f}"
    )

    # Test 3: select_frames_by_coverage
    n_select = 100
    selected = select_frames_by_coverage(positions, quality, n_select=n_select)
    assert len(selected) == n_select, f"Expected {n_select} selected, got {len(selected)}"
    assert len(set(selected)) == n_select, "Selected indices must be unique"
    print(f"select_frames_by_coverage: OK — selected {n_select} frames from {n_cams}")
    print(f"  First 10 selected indices: {selected[:10]}")

    # Test 4: degenerate input (all cameras at same position)
    degenerate = np.zeros((20, 3))
    degen_scores = fibonacci_coverage_score(degenerate, n_target=20)
    assert np.allclose(degen_scores, 1.0), "Degenerate input should yield uniform scores"
    print("fibonacci_coverage_score (degenerate): OK — uniform scores returned")

    print("\nAll self-tests passed.")
    sys.exit(0)
