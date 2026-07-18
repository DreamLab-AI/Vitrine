# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Image-edit alternate-view synthesis (ADR-025 D4 / PRD v4 R7, rung b).

Backsides of single-image-generated objects are model-hallucinated. The second
rung of the quality-escalation ladder synthesizes an ALTERNATE viewpoint of the
object crop ("show the back / rotate 180°") with an instruction-driven image
edit model, and feeds that edited image to the 3D generator as one more
best-of-N candidate. Each attempt remains ONE clean single image — this is
explicitly NOT the retired panel-stitching anti-pattern (ADR-025: multiview
panel conditioning is superseded; escalation is "image-edit second view fed as
an *alternative single-image attempt*, best-of-N selected").

Edit model: **Qwen-Image-Edit-2509** (Apache-2.0, commercial-safe — the only
edit model in the SOTA registry benchmarked for instruction-driven view
rotation). Executor: ComfyUI (a 2D stage, in-scope per ADR-014 as narrowed by
ADR-025). FLUX.2-Kontext-style editing is a licence-gated alternative; swap
the workflow template + model filenames via config to use it.

Staging honesty (sota_registry ground truth 2026-06-19): the Qwen text-encode
nodes are installed and ``qwen_image_vae`` is staged, but the Qwen-Image-Edit
DIFFUSION UNET is **not yet staged**. ``probe_edit_model()`` checks the live
``/object_info/UNETLoader`` list so callers can skip this rung gracefully
until the weights are pulled; nothing is faked.

The client mirrors the Trellis2Client plumbing (health_check, _upload_image,
_submit_prompt, _poll_completion, _download_file, _free_vram) so the two feel
identical to operate. Lineage records the edit model, the instruction, and
``surface: image-edit-inferred`` (ADR-025's observed-vs-inferred reporting
concept) so downstream consumers know these pixels were synthesized.

Usage::

    from pipeline.image_edit_view import ImageEditView
    editor = ImageEditView(comfyui_url="http://vitrine-comfyui:8188")
    if editor.probe_edit_model():
        res = editor.edit_view("object_crops/0001_vase.png", label="vase")
        # res.image_path is a single clean image -> another crop path for
        # Trellis2Client.reconstruct_from_image(...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / "workflows"
QWEN_EDIT_VIEW_WORKFLOW = WORKFLOW_DIR / "qwen_image_edit_view.json"

_CROP_PLACEHOLDER = "CROP_IMAGE_PLACEHOLDER"
_INSTRUCTION_PLACEHOLDER = "EDIT_INSTRUCTION_PLACEHOLDER"

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# The default alternate-view instruction. Identity-preserving and
# single-object by construction: the edit must show the SAME object, not
# restyle it, and must stay a clean isolated photograph (the exact contract
# the single-image generator was trained on).
DEFAULT_BACK_INSTRUCTION = (
    "Rotate the camera 180 degrees to directly behind this object and show "
    "its back side. Keep the exact same object: identical shape, materials, "
    "colors, proportions and lighting. The whole object centered and fully "
    "visible on a plain uniform light background. Do not add, remove or "
    "restyle anything."
)


@dataclass
class ImageEditResult:
    """Result of one instruction-driven alternate-view edit.

    ``image_data`` is the edited image byte-for-byte as ComfyUI produced it
    (matting, when applied, yields a *separate* RGBA byte string in
    ``image_data``; the raw download is kept in ``raw_image_data``).
    """
    image_data: Optional[bytes] = None       # the candidate image (RGBA if matted)
    raw_image_data: Optional[bytes] = None   # verbatim ComfyUI output (RGB)
    image_path: Optional[Path] = None        # where the candidate was written
    backend: str = "qwen-image-edit-comfyui"
    duration_seconds: float = 0.0
    prompt_id: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image_data).hexdigest() if self.image_data else ""


class ImageEditView:
    """Instruction-driven alternate-view client (Qwen-Image-Edit-2509 primary).

    Parameters
    ----------
    comfyui_url : str
        ComfyUI URL (e.g. ``http://vitrine-comfyui:8188``).
    timeout : int
        Maximum seconds per edit.
    diffusion_model / clip_model / vae_model : str
        Loader filenames injected into the workflow. The diffusion UNET is the
        canonical Comfy-Org release name but is NOT yet staged (registry ground
        truth 2026-06-19) — ``probe_edit_model()`` resolves/falls back against
        the live server before any submit.
    instruction : str
        Default edit instruction (``DEFAULT_BACK_INSTRUCTION``).
    steps / cfg / shift / denoise : sampling parameters
        Defaults follow the Comfy-Org Qwen-Image-Edit-2509 template
        (20 steps, cfg 2.5, AuraFlow shift 3.1, full denoise).
    matte_output : bool
        Re-matte the edited RGB output to RGBA via rembg when available. The
        edited view has a *different* silhouette from the source crop, so the
        source alpha can never be reused; without rembg the image is emitted
        opaque and flagged ``matted: false`` in lineage (the TRELLIS.2 native
        path mattes internally; the ComfyUI path prefers a real alpha).
    """

    def __init__(
        self,
        comfyui_url: str = "http://vitrine-comfyui:8188",
        timeout: int = 600,
        poll_interval: float = 2.0,
        diffusion_model: str = "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        clip_model: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        vae_model: str = "qwen_image_vae.safetensors",
        instruction: str = DEFAULT_BACK_INSTRUCTION,
        negative: str = "",
        steps: int = 20,
        cfg: float = 2.5,
        shift: float = 3.1,
        denoise: float = 1.0,
        seed: int = 42,
        matte_output: bool = True,
        workflow_path: str | Path | None = None,
    ):
        self.comfyui_url = comfyui_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.diffusion_model = diffusion_model
        self.clip_model = clip_model
        self.vae_model = vae_model
        self.instruction = instruction
        self.negative = negative
        self.steps = steps
        self.cfg = cfg
        self.shift = shift
        self.denoise = denoise
        self.seed = seed
        self.matte_output = matte_output
        self.workflow_path = Path(workflow_path) if workflow_path else QWEN_EDIT_VIEW_WORKFLOW
        self.session = requests.Session()

    @classmethod
    def from_config(cls, cfg: Any) -> "ImageEditView":
        """Construct from an ImageEditConfig, reading every field defensively."""
        return cls(
            comfyui_url=getattr(cfg, "comfyui_url", "http://vitrine-comfyui:8188"),
            timeout=getattr(cfg, "timeout", 600),
            diffusion_model=getattr(
                cfg, "diffusion_model", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
            clip_model=getattr(cfg, "clip_model", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            vae_model=getattr(cfg, "vae_model", "qwen_image_vae.safetensors"),
            # Empty string in config means "use the built-in back-view
            # instruction" (keeps config.py free of an import on this module).
            instruction=getattr(cfg, "instruction", "") or DEFAULT_BACK_INSTRUCTION,
            negative=getattr(cfg, "negative", ""),
            steps=getattr(cfg, "steps", 20),
            cfg=getattr(cfg, "cfg", 2.5),
            shift=getattr(cfg, "shift", 3.1),
            denoise=getattr(cfg, "denoise", 1.0),
            seed=getattr(cfg, "seed", 42),
            matte_output=getattr(cfg, "matte_output", True),
        )

    # ------------------------------------------------------------------
    # ComfyUI interaction (native API — mirrors Trellis2Client)
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.comfyui_url}/system_stats", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def probe_edit_model(self) -> Optional[str]:
        """Resolve the edit UNET against the live server, or None if absent.

        The registry ground truth (2026-06-19) is that the Qwen-Image-Edit
        diffusion UNET is NOT staged. Rather than submit a workflow that will
        fail node validation, probe ``/object_info/UNETLoader``: exact config
        name first, then any staged name containing both "qwen" and "edit"
        (fuzzy match tolerates fp8/bf16 variant renames). On a fuzzy hit the
        client adopts the staged filename. None = skip this escalation rung.
        """
        try:
            r = self.session.get(
                f"{self.comfyui_url}/object_info/UNETLoader", timeout=10)
            data = r.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("probe_edit_model: cannot query UNETLoader: %s", exc)
            return None
        names = (
            data.get("UNETLoader", {}).get("input", {}).get("required", {})
            .get("unet_name", [[]])[0]
        )
        if not isinstance(names, list):
            return None
        if self.diffusion_model in names:
            return self.diffusion_model
        for name in names:
            low = str(name).lower()
            if "qwen" in low and "edit" in low:
                logger.info("probe_edit_model: adopting staged edit UNET %r "
                            "(config asked for %r)", name, self.diffusion_model)
                self.diffusion_model = str(name)
                return self.diffusion_model
        logger.info("probe_edit_model: no Qwen-Image-Edit UNET staged "
                    "(%d UNETs listed) — image-edit escalation unavailable",
                    len(names))
        return None

    def _upload_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            r = self.session.post(
                f"{self.comfyui_url}/upload/image", files=files, timeout=30,
            )
        r.raise_for_status()
        return r.json().get("name", image_path.name)

    def _submit_prompt(self, prompt: dict) -> str:
        r = self.session.post(
            f"{self.comfyui_url}/prompt", json={"prompt": prompt}, timeout=30,
        )
        data = r.json()
        if data.get("error") or data.get("node_errors"):
            node_errors = data.get("node_errors", {})
            details = "; ".join(
                f"node {nid}: {e.get('errors', e)}" for nid, e in node_errors.items()
            ) if node_errors else str(data.get("error"))
            raise RuntimeError(f"ComfyUI validation error: {details}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return prompt_id

    def _poll_completion(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                r = self.session.get(
                    f"{self.comfyui_url}/history/{prompt_id}", timeout=60,
                )
                hist = r.json()
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                logger.debug("History poll transient error: %s", e)
                continue
            if prompt_id not in hist:
                if time.monotonic() - last_log > 30:
                    logger.info("Waiting for image-edit prompt %s...", prompt_id[:8])
                    last_log = time.monotonic()
                continue
            entry = hist[prompt_id]
            status = entry.get("status", {}).get("status_str", "unknown")
            if status == "success":
                return entry
            if status == "error":
                messages = entry.get("status", {}).get("messages", [])
                raise RuntimeError(f"Image-edit execution error: {messages}")
            if time.monotonic() - last_log > 30:
                logger.info("Image-edit %s status: %s", prompt_id[:8], status)
                last_log = time.monotonic()
        raise TimeoutError(f"Image-edit prompt {prompt_id} timed out after {self.timeout}s")

    def _free_vram(self) -> None:
        """POST /free to unload models + free VRAM (serial lifecycle, ADR-013).
        Best-effort, never raises — the ~40 GB edit model must not stay
        co-resident with TRELLIS.2."""
        try:
            self.session.post(
                f"{self.comfyui_url}/free",
                json={"unload_models": True, "free_memory": True}, timeout=30,
            )
            logger.info("freed ComfyUI VRAM after image-edit generation")
        except requests.RequestException as e:  # noqa: BLE001
            logger.warning("free_vram failed: %s", e)

    def _download_file(self, filename: str, subfolder: str = "") -> bytes:
        for file_type in ("output", "temp"):
            r = self.session.get(
                f"{self.comfyui_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": file_type},
                timeout=120,
            )
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
        raise FileNotFoundError(f"Cannot download {subfolder}/{filename} from ComfyUI")

    def _extract_image_refs(self, history: dict) -> list[tuple[str, str]]:
        """Scan ComfyUI history outputs for downloadable image references.

        SaveImage registers ``outputs[node]["images"] = [{filename, subfolder,
        type}]``; this scan is robust to the exact ui key and returns
        (filename, subfolder) pairs for anything with an image suffix.
        """
        refs: list[tuple[str, str]] = []
        outputs = history.get("outputs", {})

        def add(fname: str, sub: str = "") -> None:
            if fname and fname.lower().endswith(_IMAGE_SUFFIXES):
                refs.append((Path(fname).name, sub or (
                    str(Path(fname).parent) if "/" in fname else sub)))

        for _node_id, node_output in outputs.items():
            for _key, items in node_output.items():
                if not isinstance(items, list):
                    if isinstance(items, str):
                        add(items)
                    continue
                for item in items:
                    if isinstance(item, str):
                        add(item)
                    elif isinstance(item, dict):
                        add(item.get("filename", ""), item.get("subfolder", ""))
        return refs

    # ------------------------------------------------------------------
    # Image pre/post-processing (best-effort, never fatal)
    # ------------------------------------------------------------------

    def _prepare_input(self, crop_path: Path) -> tuple[Path, bool]:
        """Flatten an RGBA crop onto white before upload, when possible.

        ComfyUI ``LoadImage`` drops alpha (the premultiplied-ghost failure mode
        the 2026-07-09 audit documented on the retired panel path), so an RGBA
        crop must be composited onto a clean background *before* the edit model
        sees it. Best-effort: if Pillow is unavailable or the file cannot be
        decoded, the original path is used and the fact recorded in lineage.
        Returns (path_to_upload, flattened).
        """
        try:
            from PIL import Image  # noqa: PLC0415 — optional dependency
            with Image.open(crop_path) as im:
                if im.mode != "RGBA":
                    return crop_path, False
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                flat_path = crop_path.with_name(crop_path.stem + "__flat_rgb.png")
                bg.save(flat_path, format="PNG")
                return flat_path, True
        except Exception as exc:  # noqa: BLE001 — non-fatal preprocessing
            logger.debug("RGBA flatten skipped for %s: %s", crop_path.name, exc)
            return crop_path, False

    def _matte_output(self, image_data: bytes) -> Optional[bytes]:
        """RGBA matte of the edited view via rembg, or None when unavailable.

        The edited view has a NEW silhouette (the object was rotated), so the
        source crop's alpha can never be reused. rembg is the same fallback
        matting backend the object_crops stage uses (ObjectCropsConfig.matting).
        """
        try:
            from rembg import remove  # noqa: PLC0415 — optional dependency
            return remove(image_data)
        except Exception as exc:  # noqa: BLE001 — non-fatal postprocessing
            logger.warning("rembg matte of edited view unavailable/failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Workflow construction
    # ------------------------------------------------------------------

    def _build_prompt(self, uploaded_name: str, instruction: str,
                      seed: int, label: str) -> dict:
        with open(self.workflow_path) as f:
            prompt = json.load(f)
        prompt = {k: v for k, v in prompt.items() if not k.startswith("_")}

        for node in prompt.values():
            ins = node.get("inputs", {})
            if ins.get("image") == _CROP_PLACEHOLDER:
                ins["image"] = uploaded_name
            if ins.get("prompt") == _INSTRUCTION_PLACEHOLDER:
                ins["prompt"] = instruction

        # Runtime parameters (node ids per qwen_image_edit_view.json).
        if "1" in prompt:
            prompt["1"]["inputs"]["unet_name"] = self.diffusion_model
        if "2" in prompt:
            prompt["2"]["inputs"]["clip_name"] = self.clip_model
        if "3" in prompt:
            prompt["3"]["inputs"]["vae_name"] = self.vae_model
        if "21" in prompt:
            prompt["21"]["inputs"]["prompt"] = self.negative
        if "40" in prompt:
            prompt["40"]["inputs"]["shift"] = self.shift
        if "50" in prompt:
            prompt["50"]["inputs"].update({
                "seed": seed, "steps": self.steps, "cfg": self.cfg,
                "denoise": self.denoise,
            })
        if "70" in prompt:
            prompt["70"]["inputs"]["filename_prefix"] = f"vitrine_editview_{label}"
        return prompt

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def edit_view(
        self,
        crop_path: str | Path,
        instruction: str | None = None,
        seed: int | None = None,
        label: str = "object",
        provenance: dict | None = None,
        output_path: str | Path | None = None,
    ) -> ImageEditResult:
        """Synthesize ONE alternate-view image of the object crop.

        The result is a single clean image (never a panel/grid) intended to be
        fed to ``Trellis2Client.reconstruct_from_image`` as one additional
        best-of-N candidate. ``provenance`` (the object_crops manifest entry)
        is folded into the lineage so the synthesized view still traces back
        to the source observation. Every synthesized pixel is flagged
        ``surface: image-edit-inferred``.
        """
        crop_path = Path(crop_path)
        if not crop_path.exists():
            raise FileNotFoundError(f"Crop not found: {crop_path}")
        seed = self.seed if seed is None else seed
        instruction = instruction or self.instruction
        safe = "".join(c if c.isalnum() else "_" for c in label)[:40] or "object"

        logger.info("Image-edit alternate view: %s (seed=%d) — %r",
                    crop_path.name, seed, instruction[:60])

        t0 = time.monotonic()
        # Serial lifecycle: begin from a clean GPU (a prior stage that crashed
        # before its own /free leaves stale models resident).
        self._free_vram()

        upload_path, flattened = self._prepare_input(crop_path)
        uploaded = self._upload_image(upload_path)
        prompt = self._build_prompt(uploaded, instruction, seed=seed, label=safe)
        prompt_id = self._submit_prompt(prompt)
        logger.info("Submitted image-edit prompt %s", prompt_id)

        history = self._poll_completion(prompt_id)
        elapsed = time.monotonic() - t0

        result = ImageEditResult(
            backend="qwen-image-edit-comfyui",
            duration_seconds=elapsed,
            prompt_id=prompt_id,
        )
        for fname, sub in self._extract_image_refs(history):
            try:
                result.raw_image_data = self._download_file(fname, sub)
                break
            except (FileNotFoundError, requests.RequestException) as e:
                logger.warning("Could not download %s/%s: %s", sub, fname, e)

        matted = False
        if result.raw_image_data is not None:
            result.image_data = result.raw_image_data
            if self.matte_output:
                rgba = self._matte_output(result.raw_image_data)
                if rgba is not None:
                    result.image_data = rgba
                    matted = True
            out = Path(output_path) if output_path else crop_path.with_name(
                crop_path.stem + "__editview.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(result.image_data)
            result.image_path = out
            logger.info("Image-edit view complete in %.1fs -> %s (matted=%s)",
                        elapsed, out.name, matted)
        else:
            result.error = "No retrievable image in edit outputs"
            logger.warning("Image-edit: %s (history outputs scanned)", result.error)

        # Serial lifecycle: free the edit model so the 3D generator has VRAM.
        self._free_vram()

        result.lineage = {
            "edit_model": "Qwen-Image-Edit-2509",
            "diffusion_checkpoint": self.diffusion_model,
            "instruction": instruction,
            "surface": "image-edit-inferred",
            "source_crop": str(crop_path),
            "executor": result.backend,
            "seed": seed,
            "steps": self.steps,
            "cfg": self.cfg,
            "input_flattened": flattened,
            "matted": matted,
            **({"source": provenance} if provenance else {}),
        }
        return result
