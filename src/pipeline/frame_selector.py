"""Intelligent frame selection from oversampled video extraction.

Oversamples at high FPS (4-6fps), then curates the best subset by:
1. Dropping frames with people (via PersonRemover)
2. Scoring quality (blur, exposure, sharpness)
3. Rejecting redundant viewpoints (perceptual hash distance)
4. Selecting optimal coverage subset for COLMAP

Typical flow: 360 raw frames → person filter → quality score → 120-180 curated frames.

Optional Fibonacci-sphere coverage scoring (ADR-007): when
``SelectionConfig.use_fibonacci_coverage`` is True and camera positions are
supplied to ``FrameSelector.select()``, the quality composite score is blended
with a Fibonacci-sphere coverage score using ``coverage_weight`` (default 0.4).
When disabled or camera positions are unavailable, existing behaviour is
unchanged.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    path: str
    index: int
    blur_score: float = 0.0
    exposure_score: float = 0.0
    sharpness: float = 0.0
    has_people: bool = False
    people_coverage: float = 0.0
    is_duplicate: bool = False
    phash: Optional[str] = None
    composite_score: float = 0.0


@dataclass
class SelectionConfig:
    target_frames: int = 80
    min_frames: int = 60
    max_frames: int = 300
    blur_threshold: float = 50.0
    exposure_min: float = 30.0
    exposure_max: float = 230.0
    duplicate_hash_distance: int = 8
    people_max_coverage: float = 0.05
    oversample_fps: float = 4.0
    blur_weight: float = 0.4
    exposure_weight: float = 0.2
    diversity_weight: float = 0.4
    # Fibonacci-sphere coverage scoring (ADR-007).  Off by default.
    use_fibonacci_coverage: bool = False
    coverage_weight: float = 0.4  # ADR-007 default: 0.4 coverage / 0.6 quality


class FrameSelector:
    """Selects optimal frame subset from oversampled extraction."""

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()

    def score_frames(self, frame_dir: str) -> List[FrameScore]:
        """Score all frames in a directory for quality metrics."""
        from .frame_quality import FrameQualityAssessor

        frame_dir = Path(frame_dir)
        image_files = sorted(
            list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
        )

        if not image_files:
            logger.warning("No frames found in %s", frame_dir)
            return []

        assessor = FrameQualityAssessor()
        results = assessor.assess_directory(str(frame_dir))

        scores = []
        for i, (path, quality) in enumerate(zip(image_files, results)):
            score = FrameScore(
                path=str(path),
                index=i,
                blur_score=quality.blur_score,
                exposure_score=abs(quality.exposure_mean - 128),
                sharpness=quality.blur_score,
                phash=quality.phash if hasattr(quality, "phash") else None,
            )

            blur_norm = min(score.blur_score / 500.0, 1.0)
            exp_norm = 1.0 - min(score.exposure_score / 100.0, 1.0)
            score.composite_score = (
                self.config.blur_weight * blur_norm
                + self.config.exposure_weight * exp_norm
            )
            scores.append(score)

        return scores

    def mark_duplicates(self, scores: List[FrameScore]) -> List[FrameScore]:
        """Flag near-duplicate frames using perceptual hash distance."""
        threshold = self.config.duplicate_hash_distance

        for i, s in enumerate(scores):
            if s.is_duplicate or s.phash is None:
                continue
            for j in range(i + 1, len(scores)):
                if scores[j].is_duplicate or scores[j].phash is None:
                    continue
                dist = _hamming_distance(s.phash, scores[j].phash)
                if dist < threshold:
                    scores[j].is_duplicate = True

        return scores

    def mark_people(
        self, scores: List[FrameScore], manifest: Optional[dict] = None
    ) -> List[FrameScore]:
        """Mark frames that contain people using a person removal manifest."""
        if manifest is None:
            return scores

        actions = manifest.get("frames", {})
        for s in scores:
            fname = Path(s.path).name
            frame_info = actions.get(fname, {})
            if frame_info.get("action") == "dropped":
                s.has_people = True
                s.people_coverage = frame_info.get("coverage_pct", 100.0)
            elif frame_info.get("person_count", 0) > 0:
                s.has_people = True
                s.people_coverage = frame_info.get("coverage_pct", 0.0)

        return scores

    def select(
        self,
        scores: List[FrameScore],
        person_manifest: Optional[dict] = None,
        camera_positions: Optional[np.ndarray] = None,
    ) -> List[FrameScore]:
        """Select optimal subset of frames.

        Args:
            scores: Per-frame quality scores from :meth:`score_frames`.
            person_manifest: Optional person-removal manifest dict.
            camera_positions: Optional ``(N, 3)`` float array of camera
                centres in COLMAP world space, indexed parallel to *scores*.
                When provided and ``config.use_fibonacci_coverage`` is True,
                Fibonacci-sphere coverage scores are blended with the quality
                composite score (ADR-007).  When ``None`` or the feature is
                disabled, the existing quality-only path is used unchanged.
        """
        cfg = self.config

        if person_manifest:
            scores = self.mark_people(scores, person_manifest)

        scores = self.mark_duplicates(scores)

        # Filter: remove blurry, badly exposed, duplicate, people-heavy
        candidates = []
        for s in scores:
            if s.blur_score < cfg.blur_threshold:
                continue
            if s.has_people and s.people_coverage > cfg.people_max_coverage * 100:
                continue
            if s.is_duplicate:
                continue
            candidates.append(s)

        logger.info(
            "Filtered %d → %d candidates (blur/people/duplicate removal)",
            len(scores),
            len(candidates),
        )

        if len(candidates) < cfg.min_frames:
            logger.warning(
                "Only %d candidates after filtering (need %d). Relaxing thresholds.",
                len(candidates),
                cfg.min_frames,
            )
            candidates = [
                s for s in scores if not s.is_duplicate and s.blur_score > 20
            ]

        # -- Optional Fibonacci-coverage blending (ADR-007) ------------------
        if (
            cfg.use_fibonacci_coverage
            and camera_positions is not None
            and len(camera_positions) == len(scores)
        ):
            candidates = self._apply_fibonacci_coverage(
                scores, candidates, camera_positions, cfg
            )
        else:
            if cfg.use_fibonacci_coverage and camera_positions is None:
                logger.debug(
                    "Fibonacci coverage enabled but camera_positions not provided; "
                    "using quality-only selection"
                )
            # Sort by quality composite score (existing behaviour)
            candidates.sort(key=lambda s: s.composite_score, reverse=True)

        # Greedy diversity selection: pick top frames ensuring temporal spread
        selected = self._greedy_diverse_select(candidates, cfg.target_frames)

        # Sort selected by original index for temporal order
        selected.sort(key=lambda s: s.index)

        logger.info("Selected %d frames from %d candidates", len(selected), len(candidates))
        return selected

    def _apply_fibonacci_coverage(
        self,
        all_scores: List[FrameScore],
        candidates: List[FrameScore],
        camera_positions: np.ndarray,
        cfg: "SelectionConfig",
    ) -> List[FrameScore]:
        """Blend Fibonacci-sphere coverage with quality scores and re-rank candidates.

        Imports fibonacci_sampler lazily to avoid hard numpy-path issues on
        systems where the module is not yet installed.

        Returns *candidates* sorted descending by the blended score.
        """
        try:
            from pipeline.fibonacci_sampler import select_frames_by_coverage
        except ImportError:
            logger.warning(
                "fibonacci_sampler not importable; falling back to quality-only selection"
            )
            candidates.sort(key=lambda s: s.composite_score, reverse=True)
            return candidates

        # Build index map from FrameScore.index → position in all_scores
        index_to_pos = {s.index: i for i, s in enumerate(all_scores)}

        # Extract positions and quality scores for candidates only
        candidate_positions = []
        candidate_quality = []
        valid_candidates: List[FrameScore] = []
        for s in candidates:
            pos_idx = index_to_pos.get(s.index)
            if pos_idx is not None and pos_idx < len(camera_positions):
                candidate_positions.append(camera_positions[pos_idx])
                candidate_quality.append(s.composite_score)
                valid_candidates.append(s)

        if len(valid_candidates) < 2:
            logger.debug("Too few candidates for Fibonacci scoring; using quality sort")
            candidates.sort(key=lambda s: s.composite_score, reverse=True)
            return candidates

        cam_pos_arr = np.array(candidate_positions, dtype=np.float64)
        quality_arr = np.array(candidate_quality, dtype=np.float64)

        try:
            selected_indices = select_frames_by_coverage(
                cam_pos_arr,
                quality_arr,
                n_select=len(valid_candidates),
                coverage_weight=cfg.coverage_weight,
            )
            # Reorder valid_candidates by the combined-score ranking
            reordered = [valid_candidates[i] for i in selected_indices]
            logger.info(
                "Fibonacci coverage applied: coverage_weight=%.2f, "
                "%d candidates re-ranked",
                cfg.coverage_weight,
                len(reordered),
            )
            return reordered
        except Exception as exc:
            logger.warning(
                "Fibonacci coverage scoring failed (%s); using quality-only sort",
                exc,
            )
            candidates.sort(key=lambda s: s.composite_score, reverse=True)
            return candidates

    def _greedy_diverse_select(
        self, candidates: List[FrameScore], target: int
    ) -> List[FrameScore]:
        """Greedy selection ensuring temporal diversity."""
        if len(candidates) <= target:
            return candidates

        n = len(candidates)
        step = max(1, n // target)

        # First pass: evenly spaced by index
        selected_indices = set()
        by_index = sorted(range(len(candidates)), key=lambda i: candidates[i].index)

        for i in range(0, len(by_index), step):
            selected_indices.add(by_index[i])
            if len(selected_indices) >= target:
                break

        # Fill remaining slots with highest-quality unselected
        if len(selected_indices) < target:
            remaining = [
                i for i in range(len(candidates)) if i not in selected_indices
            ]
            remaining.sort(key=lambda i: candidates[i].composite_score, reverse=True)
            for i in remaining:
                selected_indices.add(i)
                if len(selected_indices) >= target:
                    break

        return [candidates[i] for i in selected_indices]

    def copy_selected(
        self, selected: List[FrameScore], output_dir: str
    ) -> List[str]:
        """Copy selected frames to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for s in selected:
            src = Path(s.path)
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            paths.append(str(dst))

        logger.info("Copied %d selected frames to %s", len(paths), output_dir)
        return paths


def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    if h1 is None or h2 is None:
        return 64
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return bin(i1 ^ i2).count("1")
    except (ValueError, TypeError):
        return 64
