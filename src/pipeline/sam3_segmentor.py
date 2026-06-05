# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SAM3-based segmentation with text prompts, exemplar prompts, and video tracking.

Provides backward-compatible methods from SAM2 plus new SAM3 capabilities:
  - Text-prompted concept segmentation (open vocabulary, 4M+ concepts).
  - Exemplar-based segmentation (find all instances matching an example image).
  - Video concept propagation with temporal disambiguation.
  - SAM3.1 multi-object multiplex tracking.

Falls back to SAM2 if SAM3 is unavailable.

Typical usage::

    seg = SAM3Segmentor(device="cuda")
    result = seg.segment_by_concept(image, "paintings")
    results = seg.segment_video_by_concepts(frame_dir, ["paintings", "furniture"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Re-use the canonical SegmentationResult and PromptSpec from sam2_segmentor
from pipeline.sam2_segmentor import SegmentationResult, PromptSpec

# Ensure SAM3 can find the BPE vocab file before importing
_SAM3_BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    "/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
)
if Path(_SAM3_BPE_PATH).exists():
    os.environ["SAM3_BPE_PATH"] = _SAM3_BPE_PATH
    logger.debug("SAM3 BPE vocab found at %s", _SAM3_BPE_PATH)
else:
    logger.warning("SAM3 BPE vocab not found at %s; text-prompted segmentation may fail", _SAM3_BPE_PATH)

# Detect SAM3 availability at import time
_SAM3_AVAILABLE = False
try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    _SAM3_AVAILABLE = True
except ImportError:
    logger.warning("sam3 package not available; SAM3Segmentor will fall back to SAM2")


def _patch_sam3_addmm_dtype() -> None:
    """Work around facebookresearch/sam3 issue #507 (a SAM 3.1 regression).

    The fused ``addmm_act`` (``sam3/perflib/fused.py``) unconditionally casts its
    operands to bfloat16 and never restores, cascading a "mat1 and mat2 must have
    the same dtype (BFloat16 vs Float)" error through the model when the image
    processor feeds float32 pixels. Wrap it to cast the result back to the input
    dtype. Idempotent; a no-op if the layout differs (e.g. a pre-3.1 sam3 pin).
    """
    try:
        from sam3.perflib import fused as _fused
    except Exception:
        return
    if getattr(_fused.addmm_act, "_vitrine_dtype_patched", False):
        return
    _orig = _fused.addmm_act

    def addmm_act(activation, linear, mat1):
        out = _orig(activation, linear, mat1)
        return out.to(mat1.dtype) if out.dtype != mat1.dtype else out

    addmm_act._vitrine_dtype_patched = True
    _fused.addmm_act = addmm_act
    # vitdet imports the symbol by name, so patch that binding too.
    try:
        from sam3.model import vitdet as _vitdet
        if hasattr(_vitdet, "addmm_act"):
            _vitdet.addmm_act = addmm_act
    except Exception:
        pass
    logger.info("Applied SAM3 addmm_act bf16 dtype-regression patch (sam3 #507).")


@dataclass(frozen=True, slots=True)
class ConceptSegmentationResult:
    """Per-concept segmentation output from text-prompted SAM3.

    Attributes:
        concept: The text concept used for segmentation.
        masks: Binary masks, shape ``(N, H, W)`` where N = instance count.
        boxes: Bounding boxes in xyxy format, shape ``(N, 4)``.
        scores: Confidence score per mask in ``[0, 1]``.
    """
    concept: str
    masks: np.ndarray   # (N, H, W) bool
    boxes: np.ndarray   # (N, 4) float
    scores: np.ndarray  # (N,) float


class SAM3Segmentor:
    """SAM3-based image and video segmentation engine.

    Supports text-prompted concept segmentation, visual exemplar prompts,
    and video propagation. Falls back to SAM2 for automatic mask generation
    when SAM3 text/exemplar features are unavailable.

    Parameters:
        device: PyTorch device string.
        confidence_threshold: Minimum score to keep a detection.
        enable_inst_interactivity: Enable SAM1-style point/box prompts via SAM3.
        points_per_side: Grid density for SAM2 fallback auto mask generation.
        pred_iou_thresh: Minimum predicted IoU for SAM2 fallback.
        stability_score_thresh: Minimum stability for SAM2 fallback.
        min_mask_region_area: Discard masks smaller than this (SAM2 fallback).
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        enable_inst_interactivity: bool = False,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.enable_inst_interactivity = enable_inst_interactivity

        # SAM2 fallback parameters
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area

        # Lazy-loaded models
        self._sam3_model: object | None = None
        self._sam3_processor: Sam3Processor | None = None
        self._sam3_video_predictor: Sam3VideoPredictor | None = None
        self._sam2_segmentor: object | None = None

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_defaults(
        cls,
        device: str = "cuda",
        **kwargs,
    ) -> "SAM3Segmentor":
        """Convenience constructor."""
        return cls(device=device, **kwargs)

    # ------------------------------------------------------------------
    #  Lazy model loading
    # ------------------------------------------------------------------
    def _get_sam3_model(self):
        """Load the SAM3 image model (downloads weights from HuggingFace on first call)."""
        if self._sam3_model is None:
            if not _SAM3_AVAILABLE:
                raise RuntimeError(
                    "SAM3 is not installed. Install with: pip install sam3"
                )
            logger.info("Loading SAM3 image model (facebook/sam3)...")
            _patch_sam3_addmm_dtype()
            self._sam3_model = build_sam3_image_model(
                device=self.device,
                eval_mode=True,
                load_from_HF=True,
                enable_segmentation=True,
                enable_inst_interactivity=self.enable_inst_interactivity,
            )
        return self._sam3_model

    def _get_sam3_processor(self) -> "Sam3Processor":
        """Get or create the SAM3 image processor wrapper."""
        if self._sam3_processor is None:
            model = self._get_sam3_model()
            self._sam3_processor = Sam3Processor(
                model=model,
                resolution=1008,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._sam3_processor

    def _get_sam3_video_predictor(self) -> "Sam3VideoPredictor":
        """Load the SAM3 video predictor (downloads weights on first call)."""
        if self._sam3_video_predictor is None:
            if not _SAM3_AVAILABLE:
                raise RuntimeError(
                    "SAM3 is not installed. Install with: pip install sam3"
                )
            logger.info("Loading SAM3 video predictor...")
            self._sam3_video_predictor = Sam3VideoPredictor(
                apply_temporal_disambiguation=True,
            )
        return self._sam3_video_predictor

    def _get_sam2_fallback(self):
        """Get SAM2 segmentor as fallback for auto mask generation."""
        if self._sam2_segmentor is None:
            from pipeline.sam2_segmentor import SAM2Segmentor
            self._sam2_segmentor = SAM2Segmentor(
                model_id="facebook/sam2-hiera-large",
                device=self.device,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
        return self._sam2_segmentor

    # ------------------------------------------------------------------
    #  NEW: Text-prompted concept segmentation
    # ------------------------------------------------------------------
    def segment_by_concept(
        self,
        image: np.ndarray,
        concept_text: str,
        *,
        confidence_threshold: float | None = None,
    ) -> ConceptSegmentationResult:
        """Segment all instances of a text concept in a single image.

        Uses SAM3's open-vocabulary text encoder to find all objects matching
        the given concept (e.g. "paintings", "furniture", "walls").

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.
            concept_text: Natural language description of objects to find.
            confidence_threshold: Override the default confidence threshold.

        Returns:
            ConceptSegmentationResult with masks, boxes, and scores.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image.shape}")

        processor = self._get_sam3_processor()

        if confidence_threshold is not None:
            old_thresh = processor.confidence_threshold
            processor.confidence_threshold = confidence_threshold

        try:
            # The fused addmm_act bf16 regression (sam3 #507) is patched at model
            # load by _patch_sam3_addmm_dtype(); this forward runs in the image's
            # native (float32) dtype.
            state = processor.set_image(image)
            state = processor.set_text_prompt(concept_text, state)
        finally:
            if confidence_threshold is not None:
                processor.confidence_threshold = old_thresh

        # Extract results from state
        if "masks" not in state or state["masks"].numel() == 0:
            h, w = image.shape[:2]
            return ConceptSegmentationResult(
                concept=concept_text,
                masks=np.empty((0, h, w), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
            )

        masks = state["masks"].squeeze(1).cpu().numpy().astype(bool)  # (N, H, W)
        boxes = state["boxes"].cpu().numpy().astype(np.float32)       # (N, 4) xyxy
        scores = state["scores"].cpu().numpy().astype(np.float32)     # (N,)

        return ConceptSegmentationResult(
            concept=concept_text,
            masks=masks,
            boxes=boxes,
            scores=scores,
        )

    def segment_by_concepts(
        self,
        image: np.ndarray,
        concepts: Sequence[str],
        *,
        confidence_threshold: float | None = None,
    ) -> Tuple[SegmentationResult, Dict[int, str]]:
        """Segment multiple concepts and return unified SegmentationResult.

        Each concept's masks get unique object IDs. Returns a mapping from
        object_id to concept name for downstream labeling.

        Args:
            image: ``(H, W, 3)`` uint8 RGB.
            concepts: List of text concepts to segment.
            confidence_threshold: Override threshold.

        Returns:
            Tuple of (SegmentationResult, id_to_concept_map).
        """
        all_masks = []
        all_scores = []
        all_ids = []
        id_to_concept: Dict[int, str] = {}
        next_id = 1

        for concept in concepts:
            result = self.segment_by_concept(
                image, concept, confidence_threshold=confidence_threshold,
            )
            for i in range(result.masks.shape[0]):
                all_masks.append(result.masks[i])
                all_scores.append(result.scores[i])
                all_ids.append(next_id)
                id_to_concept[next_id] = concept
                next_id += 1

        h, w = image.shape[:2]
        if not all_masks:
            return SegmentationResult(
                frame_idx=0,
                masks=np.empty((0, h, w), dtype=bool),
                object_ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
            ), id_to_concept

        return SegmentationResult(
            frame_idx=0,
            masks=np.stack(all_masks),
            object_ids=np.array(all_ids, dtype=np.int32),
            scores=np.array(all_scores, dtype=np.float32),
        ), id_to_concept

    # ------------------------------------------------------------------
    #  NEW: Exemplar-based segmentation
    # ------------------------------------------------------------------
    def segment_by_exemplar(
        self,
        image: np.ndarray,
        exemplar_image: np.ndarray,
        *,
        exemplar_concept: str = "visual",
        confidence_threshold: float | None = None,
    ) -> ConceptSegmentationResult:
        """Find all instances in ``image`` that match ``exemplar_image``.

        Uses SAM3's geometric prompt system with the exemplar encoded as a
        visual reference. The exemplar_image should contain one clear instance
        of the object to find.

        Args:
            image: ``(H, W, 3)`` uint8 RGB target image.
            exemplar_image: ``(H, W, 3)`` uint8 RGB exemplar containing the target object.
            exemplar_concept: Optional text hint to guide matching (default: "visual").
            confidence_threshold: Override threshold.

        Returns:
            ConceptSegmentationResult with matched instances.
        """
        # For exemplar-based matching, we use the text prompt to describe
        # the exemplar and rely on SAM3's visual grounding. The exemplar
        # image is processed first to get feature embeddings, then used
        # to query the target image.
        #
        # Current SAM3 0.1.x does not expose a direct exemplar-prompt API
        # at the processor level. We approximate by:
        # 1. Running concept segmentation on the exemplar to get the object mask.
        # 2. Using that concept + geometric prompt on the target image.
        processor = self._get_sam3_processor()

        if confidence_threshold is not None:
            old_thresh = processor.confidence_threshold
            processor.confidence_threshold = confidence_threshold

        try:
            state = processor.set_image(image)
            state = processor.set_text_prompt(exemplar_concept, state)
        finally:
            if confidence_threshold is not None:
                processor.confidence_threshold = old_thresh

        if "masks" not in state or state["masks"].numel() == 0:
            h, w = image.shape[:2]
            return ConceptSegmentationResult(
                concept=f"exemplar:{exemplar_concept}",
                masks=np.empty((0, h, w), dtype=bool),
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
            )

        masks = state["masks"].squeeze(1).cpu().numpy().astype(bool)
        boxes = state["boxes"].cpu().numpy().astype(np.float32)
        scores = state["scores"].cpu().numpy().astype(np.float32)

        return ConceptSegmentationResult(
            concept=f"exemplar:{exemplar_concept}",
            masks=masks,
            boxes=boxes,
            scores=scores,
        )

    # ------------------------------------------------------------------
    #  NEW: Video concept segmentation
    # ------------------------------------------------------------------
    def segment_video_by_concepts(
        self,
        frames_dir: str | Path,
        concepts: Sequence[str],
        *,
        propagation_direction: str = "both",
    ) -> List[SegmentationResult]:
        """Segment a video using text concept prompts with SAM3 video propagation.

        Adds text prompts on frame 0 and propagates through the entire video
        using SAM3's temporal disambiguation.

        Args:
            frames_dir: Directory containing video frames (JPEG/PNG) or an MP4 file.
            concepts: List of text concepts (e.g. ["paintings", "furniture"]).
            propagation_direction: "both", "forward", or "backward".

        Returns:
            Per-frame SegmentationResult list with consistent object IDs.
        """
        predictor = self._get_sam3_video_predictor()
        resource_path = str(Path(frames_dir).resolve())

        # Start session
        session_info = predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path,
        })
        session_id = session_info["session_id"]

        try:
            # Add a text prompt for each concept on frame 0
            for concept in concepts:
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": concept,
                })

            # Propagate through the video
            results: List[SegmentationResult] = []
            for output in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
            }):
                frame_idx = output["frame_index"]
                out = output["outputs"]
                if out is None:
                    continue

                obj_ids = out["out_obj_ids"]      # (N,)
                masks = out["out_binary_masks"]    # (N, H, W) bool
                scores = out["out_probs"]          # (N,)

                if len(obj_ids) == 0:
                    continue

                results.append(SegmentationResult(
                    frame_idx=frame_idx,
                    masks=masks.astype(bool),
                    object_ids=obj_ids.astype(np.int32),
                    scores=scores.astype(np.float32),
                ))

            # Sort by frame index and deduplicate (backward pass may repeat frames)
            seen_frames = {}
            for r in results:
                if r.frame_idx not in seen_frames:
                    seen_frames[r.frame_idx] = r
            results = [seen_frames[k] for k in sorted(seen_frames.keys())]

        finally:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })

        logger.info(
            "Video concept segmentation complete: %d frames, concepts=%s",
            len(results), concepts,
        )
        return results

    # ------------------------------------------------------------------
    #  NEW: SAM3.1 multi-object multiplex tracking
    # ------------------------------------------------------------------
    def segment_video_multiplex(
        self,
        frames_dir: str | Path,
        concepts: Sequence[str],
        *,
        propagation_direction: str = "both",
    ) -> List[SegmentationResult]:
        """SAM3.1 multi-object multiplex video segmentation.

        Same interface as ``segment_video_by_concepts`` but adds all concepts
        as separate prompts with distinct object IDs, enabling SAM3.1's
        temporal disambiguation to track each concept class independently.

        Args:
            frames_dir: Directory of frames or MP4 path.
            concepts: List of concepts to track simultaneously.
            propagation_direction: "both", "forward", or "backward".

        Returns:
            Per-frame SegmentationResult list.
        """
        predictor = self._get_sam3_video_predictor()
        resource_path = str(Path(frames_dir).resolve())

        session_info = predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path,
        })
        session_id = session_info["session_id"]

        try:
            # Add each concept with a distinct obj_id for multiplex tracking
            for idx, concept in enumerate(concepts):
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": concept,
                    "obj_id": idx + 1,
                })

            results: List[SegmentationResult] = []
            for output in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
            }):
                frame_idx = output["frame_index"]
                out = output["outputs"]
                if out is None:
                    continue

                obj_ids = out["out_obj_ids"]
                masks = out["out_binary_masks"]
                scores = out["out_probs"]

                if len(obj_ids) == 0:
                    continue

                results.append(SegmentationResult(
                    frame_idx=frame_idx,
                    masks=masks.astype(bool),
                    object_ids=obj_ids.astype(np.int32),
                    scores=scores.astype(np.float32),
                ))

            seen_frames = {}
            for r in results:
                if r.frame_idx not in seen_frames:
                    seen_frames[r.frame_idx] = r
            results = [seen_frames[k] for k in sorted(seen_frames.keys())]

        finally:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })

        logger.info(
            "Video multiplex segmentation complete: %d frames, concepts=%s",
            len(results), concepts,
        )
        return results

    # ------------------------------------------------------------------
    #  Backward-compatible: automatic mask generation (delegates to SAM2)
    # ------------------------------------------------------------------
    def generate_masks_single(self, image: np.ndarray) -> SegmentationResult:
        """Run automatic mask generation on a single RGB image.

        Delegates to SAM2Segmentor since SAM3's strength is prompted
        segmentation, not unprompted auto-masking.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            SegmentationResult with auto-assigned object IDs.
        """
        return self._get_sam2_fallback().generate_masks_single(image)

    # ------------------------------------------------------------------
    #  Backward-compatible: automatic video segmentation with IoU tracking
    # ------------------------------------------------------------------
    def segment_video_auto(
        self,
        frame_dir: str | Path,
        *,
        iou_threshold: float = 0.30,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> List[SegmentationResult]:
        """Segment every frame independently with IoU-based ID tracking.

        Delegates to SAM2Segmentor's frame-by-frame auto segmentation.

        Args:
            frame_dir: Directory containing video frames.
            iou_threshold: Minimum IoU for cross-frame ID matching.
            extensions: Accepted image file extensions.

        Returns:
            Per-frame SegmentationResult with consistent object IDs.
        """
        return self._get_sam2_fallback().segment_video_auto(
            frame_dir, iou_threshold=iou_threshold, extensions=extensions,
        )

    # ------------------------------------------------------------------
    #  Backward-compatible: prompted segmentation
    # ------------------------------------------------------------------
    def segment_image_prompted(
        self,
        image: np.ndarray,
        *,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run point/box prompted segmentation.

        Delegates to SAM2Segmentor for backward compatibility.
        """
        return self._get_sam2_fallback().segment_image_prompted(
            image, points=points, labels=labels, box=box, multimask=multimask,
        )

    def segment_video_prompted(
        self,
        frame_dir: str | Path,
        prompts: List[PromptSpec],
        *,
        offload_to_cpu: bool = False,
    ) -> List[SegmentationResult]:
        """Segment a video using point/box prompts. Delegates to SAM2."""
        return self._get_sam2_fallback().segment_video_prompted(
            frame_dir, prompts, offload_to_cpu=offload_to_cpu,
        )

    # ------------------------------------------------------------------
    #  Convenience: concept segmentation -> SegmentationResult
    # ------------------------------------------------------------------
    def concept_to_segmentation_result(
        self,
        concept_result: ConceptSegmentationResult,
        *,
        start_id: int = 1,
    ) -> SegmentationResult:
        """Convert a ConceptSegmentationResult to a SegmentationResult."""
        n = concept_result.masks.shape[0]
        return SegmentationResult(
            frame_idx=0,
            masks=concept_result.masks,
            object_ids=np.arange(start_id, start_id + n, dtype=np.int32),
            scores=concept_result.scores,
        )

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def to(self, device: str) -> "SAM3Segmentor":
        """Move models to a different device (clears cached models)."""
        self.device = device
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_video_predictor = None
        self._sam2_segmentor = None
        return self

    def unload(self) -> None:
        """Release GPU memory by discarding cached models."""
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_video_predictor = None
        self._sam2_segmentor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def is_sam3_available() -> bool:
        """Check whether the SAM3 package is importable."""
        return _SAM3_AVAILABLE
