# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ComfyUI-based inpainting client for the SaladTechnologies comfyui-api.

Connects to a remote ComfyUI API (SaladTechnologies v1.17.x) and submits
FLUX Fill or fallback VAEEncodeForInpaint workflows to recover clean
backgrounds from training images with object masks removed.

Usage::

    from pipeline.comfyui_inpainter import ComfyUIInpainter

    inpainter = ComfyUIInpainter(api_url="http://192.168.2.48:3001")
    result = inpainter.inpaint(image, mask, prompt="clean empty background")
"""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    logger.warning("Pillow not installed; image conversion will require numpy arrays")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InpaintResult:
    """Result from a ComfyUI inpainting run."""
    image: np.ndarray  # H x W x 3, uint8
    workflow_id: str
    elapsed_s: float
    model_used: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Tracked model availability on the remote server."""
    diffusion_models: list[str] = field(default_factory=list)
    clip_models: list[str] = field(default_factory=list)
    vae_models: list[str] = field(default_factory=list)
    loras: list[str] = field(default_factory=list)
    controlnets: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)


class ComfyUIError(Exception):
    """Error from ComfyUI API interaction."""
    pass


class ModelNotFoundError(ComfyUIError):
    """Required model is not available on the server."""
    pass


# ---------------------------------------------------------------------------
# Model download specifications
# ---------------------------------------------------------------------------

FLUX_FILL_MODELS = {
    "diffusion_model": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors",
        "model_type": "diffusion_models",
        "filename": "flux1-fill-dev.safetensors",
    },
    "clip_l": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "model_type": "clip",
        "filename": "clip_l.safetensors",
    },
    "t5xxl": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
        "model_type": "clip",
        "filename": "t5xxl_fp16.safetensors",
    },
    "vae": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "model_type": "vae",
        "filename": "ae.safetensors",
    },
}

FLUX_DEV_MODELS = {
    "diffusion_model": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "model_type": "diffusion_models",
        "filename": "flux1-dev.safetensors",
    },
    "clip_l": FLUX_FILL_MODELS["clip_l"],
    "t5xxl": FLUX_FILL_MODELS["t5xxl"],
    "vae": FLUX_FILL_MODELS["vae"],
}

# FLUX.2-dev fileset (item 1, SOTA modernisation). There is NO FLUX.2 "Fill"
# checkpoint: masked recovery is achieved via InpaintModelConditioning +
# ReferenceLatent on the base FLUX.2-dev diffusion model, paired with the
# Mistral-3 text encoder and the FLUX.2 VAE. Filenames match the staged assets
# (see work-order §0). URLs are best-effort for the auto-download path; when the
# weights are already staged locally the probe finds them and no download runs.
FLUX2_DEV_MODELS = {
    "diffusion_model": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        "model_type": "diffusion_models",
        "filename": "flux2_dev_fp8mixed.safetensors",
    },
    "text_encoder": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
        "model_type": "clip",
        "filename": "mistral_3_small_flux2_fp8.safetensors",
    },
    "vae": {
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        "model_type": "vae",
        "filename": "flux2-vae.safetensors",
    },
}


# ---------------------------------------------------------------------------
# Workflow templates
# ---------------------------------------------------------------------------

_WORKFLOW_DIR = Path(__file__).parent / "workflows"


def _load_workflow(name: str) -> dict[str, Any]:
    """Load a workflow JSON template from the workflows directory."""
    path = _WORKFLOW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Workflow template not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_pil(img: Any) -> "Image.Image":
    """Convert numpy array or PIL Image to PIL Image."""
    if not _HAS_PIL:
        raise ImportError("Pillow is required for image conversion")
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.ndim == 2:
            return Image.fromarray(img, mode="L")
        if img.ndim == 3 and img.shape[2] == 4:
            return Image.fromarray(img, mode="RGBA")
        return Image.fromarray(img, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_numpy(img: Any) -> np.ndarray:
    """Convert PIL Image or numpy array to H x W x 3 uint8 numpy array."""
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        return img[:, :, :3] if img.shape[2] > 3 else img
    if _HAS_PIL and isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    raise TypeError(f"Unsupported image type: {type(img)}")


def _image_to_png_bytes(img: Any) -> bytes:
    """Convert image to PNG bytes."""
    pil_img = _to_pil(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _image_to_data_uri(img: Any) -> str:
    """Convert image to a data URI (not used; images served via temp HTTP)."""
    png_bytes = _image_to_png_bytes(img)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _mask_to_rgb(mask: Any) -> Any:
    """Ensure mask is an RGB image with white=masked, black=keep."""
    pil = _to_pil(mask)
    if pil.mode == "L":
        return pil.convert("RGB")
    if pil.mode == "RGBA":
        return pil.convert("RGB")
    return pil


# ---------------------------------------------------------------------------
# Temporary image server for feeding images to ComfyUI
# ---------------------------------------------------------------------------

class _ImageServingHandler(BaseHTTPRequestHandler):
    """Serves images from an in-memory dict keyed by path."""

    images: dict[str, bytes] = {}

    def do_GET(self):
        path = self.path.lstrip("/")
        if path in self.images:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.images[path])))
            self.end_headers()
            self.wfile.write(self.images[path])
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs


class ImageServer:
    """Ephemeral HTTP server to serve images to the remote ComfyUI."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        self._handler_class = type(
            "_Handler",
            (_ImageServingHandler,),
            {"images": {}},
        )
        self._server = HTTPServer((host, port), self._handler_class)
        self.port = self._server.server_address[1]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Image server started on port %d", self.port)

    def add_image(self, name: str, img: Any) -> str:
        """Add an image and return its URL path component."""
        self._handler_class.images[name] = _image_to_png_bytes(img)
        return name

    def get_url(self, name: str, local_ip: str) -> str:
        """Get full URL for a served image."""
        return f"http://{local_ip}:{self.port}/{name}"

    def shutdown(self):
        self._server.shutdown()


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class ComfyUIInpainter:
    """Client for submitting inpainting workflows to a SaladTechnologies ComfyUI API.

    Parameters
    ----------
    api_url : str
        Base URL of the SaladTechnologies comfyui-api (e.g. ``http://192.168.2.48:3001``).
    comfyui_url : str
        Direct ComfyUI URL for object_info queries (e.g. ``http://192.168.2.48:8189``).
    local_ip : str
        IP address of this machine reachable from the ComfyUI server,
        used to serve input images. Set to the LAN IP.
    hf_token : str | None
        HuggingFace token for downloading gated models (FLUX).
    timeout : float
        HTTP request timeout in seconds.
    denoise : float
        Denoise strength for inpainting (0.0-1.0). Range 0.6-0.85 recommended.
    steps : int
        Number of sampling steps.
    guidance : float
        FLUX guidance scale.
    seed : int
        Random seed for reproducibility. -1 for random.
    """

    def __init__(
        self,
        api_url: str = "http://192.168.2.48:3001",
        comfyui_url: str = "http://192.168.2.48:8189",
        local_ip: str = "192.168.2.1",
        hf_token: str | None = None,
        timeout: float = 300.0,
        denoise: float = 0.75,
        steps: int = 28,
        guidance: float = 30.0,
        seed: int = 42,
        prefer_flux2: bool = True,
        flux2_diffusion: str = "flux2_dev_fp8mixed.safetensors",
        flux2_vae: str = "flux2-vae.safetensors",
        flux2_text_encoder: str = "mistral_3_small_flux2_fp8.safetensors",
        flux2_guidance: float = 4.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.comfyui_url = comfyui_url.rstrip("/")
        self.local_ip = local_ip
        self.hf_token = hf_token
        self.timeout = timeout
        self.denoise = denoise
        self.steps = steps
        self.guidance = guidance
        self.seed = seed

        # FLUX.2 capability config (item 1). Filenames default to the staged
        # asset names; callers may override (e.g. from InpaintConfig).
        self.prefer_flux2 = prefer_flux2
        self.flux2_diffusion = flux2_diffusion
        self.flux2_vae = flux2_vae
        self.flux2_text_encoder = flux2_text_encoder
        self.flux2_guidance = flux2_guidance

        self._model_info: ModelInfo | None = None
        self._image_server: ImageServer | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> "ComfyUIInpainter":
        """Construct from an InpaintConfig, reading fields defensively.

        Tolerant of agent D's field rename: every FLUX.2 field is read via
        ``getattr`` with a staged-asset default, so this works whether or not the
        new config fields have landed. ``prefer_flux2`` is enabled by default
        but disabled automatically if the legacy ``model`` field explicitly pins
        a FLUX.1 variant (``"flux-fill"`` / ``"flux1-..."``).
        """
        legacy_model = str(getattr(cfg, "model", "") or "").lower()
        prefer_flux2 = getattr(cfg, "prefer_flux2", None)
        if prefer_flux2 is None:
            # Honour an explicit FLUX.1 pin; otherwise default to FLUX.2.
            prefer_flux2 = not legacy_model.startswith(("flux-fill", "flux1", "flux-1"))
        return cls(
            api_url=getattr(cfg, "comfyui_api_url", "http://192.168.2.48:3001"),
            comfyui_url=getattr(cfg, "comfyui_direct_url", "http://192.168.2.48:8189"),
            local_ip=getattr(cfg, "local_ip", "192.168.2.1"),
            hf_token=getattr(cfg, "hf_token", "") or None,
            denoise=getattr(cfg, "denoise", 0.75),
            steps=getattr(cfg, "steps", 28),
            guidance=getattr(cfg, "guidance", 30.0),
            prefer_flux2=bool(prefer_flux2),
            flux2_diffusion=getattr(cfg, "flux2_diffusion", "flux2_dev_fp8mixed.safetensors"),
            flux2_vae=getattr(cfg, "flux2_vae", "flux2-vae.safetensors"),
            flux2_text_encoder=getattr(
                cfg, "flux2_text_encoder", "mistral_3_small_flux2_fp8.safetensors"
            ),
            flux2_guidance=getattr(cfg, "flux2_guidance", 4.0),
        )

    # ----- HTTP helpers -----

    def _request(
        self,
        method: str,
        url: str,
        data: dict | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request and return parsed JSON."""
        timeout = timeout or self.timeout
        body = json.dumps(data).encode("utf-8") if data else None
        req = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise ComfyUIError(
                f"HTTP {e.code} from {url}: {body_text}"
            ) from e
        except URLError as e:
            raise ComfyUIError(f"Connection error to {url}: {e}") from e

    def _get(self, url: str, timeout: float | None = None) -> dict[str, Any]:
        return self._request("GET", url, timeout=timeout)

    def _post(self, url: str, data: dict, timeout: float | None = None) -> dict[str, Any]:
        return self._request("POST", url, data=data, timeout=timeout)

    # ----- Model discovery -----

    def probe_models(self) -> ModelInfo:
        """Query the remote ComfyUI for available models."""
        info = ModelInfo()

        # Try object_info endpoints for each loader
        loaders = {
            "UNETLoader": ("unet_name", "diffusion_models"),
            "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
            "DualCLIPLoader": ("clip_name1", "clip_models"),
            "VAELoader": ("vae_name", "vae_models"),
            "LoraLoader": ("lora_name", "loras"),
            "ControlNetLoader": ("control_net_name", "controlnets"),
        }

        for node_name, (input_name, attr_name) in loaders.items():
            try:
                resp = self._get(f"{self.comfyui_url}/object_info/{node_name}", timeout=10)
                node_data = resp.get(node_name, {})
                required = node_data.get("input", {}).get("required", {})
                names = required.get(input_name, [[]])[0]
                if isinstance(names, list):
                    setattr(info, attr_name, names)
            except Exception as e:
                logger.debug("Could not probe %s: %s", node_name, e)

        # Also try DualCLIPLoader clip_name2
        try:
            resp = self._get(f"{self.comfyui_url}/object_info/DualCLIPLoader", timeout=10)
            clip2 = resp.get("DualCLIPLoader", {}).get("input", {}).get(
                "required", {}
            ).get("clip_name2", [[]])[0]
            if isinstance(clip2, list):
                for name in clip2:
                    if name not in info.clip_models:
                        info.clip_models.append(name)
        except Exception:
            pass

        self._model_info = info
        logger.info(
            "Models found: %d diffusion, %d clip, %d vae, %d checkpoints, %d loras",
            len(info.diffusion_models),
            len(info.clip_models),
            len(info.vae_models),
            len(info.checkpoints),
            len(info.loras),
        )
        return info

    def _ensure_models_probed(self) -> ModelInfo:
        if self._model_info is None:
            return self.probe_models()
        return self._model_info

    # ----- Model downloading -----

    def download_model(
        self,
        url: str,
        model_type: str,
        filename: str | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Download a model to the remote ComfyUI server via the /download endpoint."""
        payload: dict[str, Any] = {
            "url": url,
            "model_type": model_type,
            "wait": wait,
        }
        if filename:
            payload["filename"] = filename
        if self.hf_token and "huggingface.co" in url:
            payload["auth"] = {
                "type": "bearer",
                "token": self.hf_token,
            }

        logger.info(
            "Downloading model: %s → %s/%s (wait=%s)",
            url.split("/")[-1], model_type, filename or "auto", wait,
        )
        timeout = 600.0 if wait else 30.0
        return self._post(f"{self.api_url}/download", payload, timeout=timeout)

    def ensure_flux_fill_models(self) -> bool:
        """Ensure all FLUX Fill models are downloaded. Returns True if ready."""
        info = self._ensure_models_probed()

        needed = []
        if "flux1-fill-dev.safetensors" not in info.diffusion_models:
            needed.append(FLUX_FILL_MODELS["diffusion_model"])
        if "clip_l.safetensors" not in info.clip_models:
            needed.append(FLUX_FILL_MODELS["clip_l"])
        if "t5xxl_fp16.safetensors" not in info.clip_models:
            needed.append(FLUX_FILL_MODELS["t5xxl"])
        if "ae.safetensors" not in info.vae_models:
            needed.append(FLUX_FILL_MODELS["vae"])

        if not needed:
            logger.info("All FLUX Fill models already available")
            return True

        logger.info("Need to download %d models for FLUX Fill", len(needed))
        for model_spec in needed:
            self.download_model(
                url=model_spec["url"],
                model_type=model_spec["model_type"],
                filename=model_spec["filename"],
                wait=True,
            )

        # Re-probe
        self.probe_models()
        return True

    def ensure_flux_dev_models(self) -> bool:
        """Ensure FLUX Dev models (fallback) are downloaded."""
        info = self._ensure_models_probed()

        needed = []
        if "flux1-dev.safetensors" not in info.diffusion_models:
            needed.append(FLUX_DEV_MODELS["diffusion_model"])
        if "clip_l.safetensors" not in info.clip_models:
            needed.append(FLUX_DEV_MODELS["clip_l"])
        if "t5xxl_fp16.safetensors" not in info.clip_models:
            needed.append(FLUX_DEV_MODELS["t5xxl"])
        if "ae.safetensors" not in info.vae_models:
            needed.append(FLUX_DEV_MODELS["vae"])

        if not needed:
            logger.info("All FLUX Dev models already available")
            return True

        logger.info("Need to download %d models for FLUX Dev", len(needed))
        for model_spec in needed:
            self.download_model(
                url=model_spec["url"],
                model_type=model_spec["model_type"],
                filename=model_spec["filename"],
                wait=True,
            )

        self.probe_models()
        return True

    def ensure_flux2_models(self) -> bool:
        """Ensure the FLUX.2-dev fileset is available. Returns True if ready.

        Generalises the FLUX.1 ``ensure_flux_*_models()`` helpers to the FLUX.2
        stack: the fp8-mixed diffusion model, the FLUX.2 VAE, and the Mistral-3
        text encoder. When the weights are already staged on the server the
        probe finds them and no download is issued. If the SaladTechnologies
        ``/download`` endpoint is unavailable, missing weights are reported and
        the caller falls back to the FLUX.1 path.
        """
        info = self._ensure_models_probed()

        needed = []
        if self.flux2_diffusion not in info.diffusion_models:
            spec = dict(FLUX2_DEV_MODELS["diffusion_model"])
            spec["filename"] = self.flux2_diffusion
            needed.append(spec)
        if self.flux2_text_encoder not in info.clip_models:
            spec = dict(FLUX2_DEV_MODELS["text_encoder"])
            spec["filename"] = self.flux2_text_encoder
            needed.append(spec)
        if self.flux2_vae not in info.vae_models:
            spec = dict(FLUX2_DEV_MODELS["vae"])
            spec["filename"] = self.flux2_vae
            needed.append(spec)

        if not needed:
            logger.info("All FLUX.2 models already available")
            return True

        logger.info("Need to download %d models for FLUX.2", len(needed))
        for model_spec in needed:
            try:
                self.download_model(
                    url=model_spec["url"],
                    model_type=model_spec["model_type"],
                    filename=model_spec["filename"],
                    wait=True,
                )
            except ComfyUIError as e:
                logger.warning(
                    "FLUX.2 model download failed for %s: %s",
                    model_spec["filename"], e,
                )
                return False

        self.probe_models()
        return self.has_flux2_models()

    def has_flux2_models(self) -> bool:
        """Capability probe: True if the full FLUX.2 fileset is present.

        The FLUX.2 inpaint path additionally requires the ComfyUI
        ``InpaintModelConditioning`` + ``ReferenceLatent`` nodes (the same nodes
        the FLUX.1-Fill path relies on, plus ReferenceLatent). The server-side
        node presence is validated implicitly at submit time; here we gate on
        weight availability, which is the dominant differentiator.
        """
        info = self._ensure_models_probed()
        return (
            self.flux2_diffusion in info.diffusion_models
            and self.flux2_text_encoder in info.clip_models
            and self.flux2_vae in info.vae_models
        )

    # ----- Image server management -----

    def _ensure_image_server(self) -> ImageServer:
        """Start or return the ephemeral image server."""
        if self._image_server is None:
            self._image_server = ImageServer()
        return self._image_server

    # ----- Workflow construction -----

    def _build_flux2_workflow(
        self,
        image_url: str,
        mask_url: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a FLUX.2 masked-recovery inpaint workflow from the template.

        FLUX.2 has no dedicated 'Fill' checkpoint; masked recovery uses
        InpaintModelConditioning (pixels + mask + VAE) feeding a ReferenceLatent
        into the base FLUX.2-dev diffusion model, with the Mistral-3 text encoder
        and FLUX.2 VAE. Model filenames are injected from the client config so a
        rename in agent D's config (or a different staged variant) is honoured.
        """
        wf = _load_workflow("flux2_inpaint.json")
        graph = wf["prompt"]

        # Inject model filenames (config-driven, defensive against renames)
        graph["1"]["inputs"]["unet_name"] = self.flux2_diffusion
        graph["2"]["inputs"]["clip_name"] = self.flux2_text_encoder
        graph["3"]["inputs"]["vae_name"] = self.flux2_vae

        # Set image URLs
        graph["4"]["inputs"]["image"] = image_url
        graph["5"]["inputs"]["image"] = mask_url

        # Set prompts
        graph["7"]["inputs"]["text"] = prompt
        graph["8"]["inputs"]["text"] = negative_prompt

        # FLUX.2 guidance (distinct scale from FLUX.1-Fill's ~30)
        graph["9"]["inputs"]["guidance"] = self.flux2_guidance

        # Set sampler parameters
        graph["13"]["inputs"]["denoise"] = denoise or self.denoise
        graph["13"]["inputs"]["steps"] = steps or self.steps
        graph["13"]["inputs"]["seed"] = seed if seed is not None else self.seed

        return wf

    def _build_flux_fill_workflow(
        self,
        image_url: str,
        mask_url: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a FLUX Fill inpainting workflow from the template."""
        wf = _load_workflow("flux_inpaint.json")
        graph = wf["prompt"]

        # Set image URLs
        graph["4"]["inputs"]["image"] = image_url
        graph["5"]["inputs"]["image"] = mask_url

        # Set prompts
        graph["7"]["inputs"]["text"] = prompt
        graph["8"]["inputs"]["text"] = negative_prompt

        # Set guidance
        graph["9"]["inputs"]["guidance"] = self.guidance

        # Set sampler parameters
        graph["12"]["inputs"]["denoise"] = denoise or self.denoise
        graph["12"]["inputs"]["steps"] = steps or self.steps
        graph["12"]["inputs"]["seed"] = seed if seed is not None else self.seed

        return wf

    def _build_flux_vae_encode_workflow(
        self,
        image_url: str,
        mask_url: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float | None = None,
        steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build a FLUX VAEEncodeForInpaint workflow (fallback for non-Fill model)."""
        wf = _load_workflow("flux_inpaint_vae_encode.json")
        graph = wf["prompt"]

        # Set image URLs
        graph["4"]["inputs"]["image"] = image_url
        graph["5"]["inputs"]["image"] = mask_url

        # Set prompts
        graph["8"]["inputs"]["text"] = prompt
        graph["9"]["inputs"]["text"] = negative_prompt

        # Set guidance
        graph["10"]["inputs"]["guidance"] = self.guidance

        # Set sampler parameters
        graph["12"]["inputs"]["denoise"] = denoise or self.denoise
        graph["12"]["inputs"]["steps"] = steps or self.steps
        graph["12"]["inputs"]["seed"] = seed if seed is not None else self.seed

        return wf

    def _select_workflow(
        self,
        image_url: str,
        mask_url: str,
        prompt: str,
        negative_prompt: str,
        denoise: float | None,
        steps: int | None,
        seed: int | None,
    ) -> tuple[dict[str, Any], str]:
        """Select the best available workflow and return (workflow, model_name)."""
        info = self._ensure_models_probed()

        # Strategy 0: FLUX.2-dev masked recovery (item 1). Capability probe — if
        # the FLUX.2 weights/nodes are present, prefer this SOTA path; otherwise
        # fall through to the unchanged FLUX.1-Fill strategies below.
        if self.prefer_flux2 and self.has_flux2_models():
            logger.info(
                "Inpaint path: FLUX.2-dev (masked recovery via "
                "InpaintModelConditioning + ReferenceLatent; model=%s)",
                self.flux2_diffusion,
            )
            wf = self._build_flux2_workflow(
                image_url, mask_url, prompt, negative_prompt,
                denoise, steps, seed,
            )
            return wf, self.flux2_diffusion
        if self.prefer_flux2:
            logger.info(
                "Inpaint path: FLUX.2 weights absent (need %s/%s/%s) — "
                "falling back to FLUX.1 path",
                self.flux2_diffusion, self.flux2_text_encoder, self.flux2_vae,
            )

        # Strategy 1: FLUX Fill (purpose-built inpainting model)
        has_fill = "flux1-fill-dev.safetensors" in info.diffusion_models
        has_clip = (
            "clip_l.safetensors" in info.clip_models
            and "t5xxl_fp16.safetensors" in info.clip_models
        )
        has_vae = "ae.safetensors" in info.vae_models

        if has_fill and has_clip and has_vae:
            logger.info("Using FLUX Fill workflow (flux1-fill-dev)")
            wf = self._build_flux_fill_workflow(
                image_url, mask_url, prompt, negative_prompt,
                denoise, steps, seed,
            )
            return wf, "flux1-fill-dev"

        # Strategy 2: FLUX Dev with VAEEncodeForInpaint
        has_dev = "flux1-dev.safetensors" in info.diffusion_models
        if has_dev and has_clip and has_vae:
            logger.info("Using FLUX Dev + VAEEncodeForInpaint workflow")
            wf = self._build_flux_vae_encode_workflow(
                image_url, mask_url, prompt, negative_prompt,
                denoise, steps, seed,
            )
            return wf, "flux1-dev"

        # Strategy 3: Any available FLUX-compatible diffusion model
        flux_models = [
            m for m in info.diffusion_models
            if "flux" in m.lower()
        ]
        if flux_models and has_clip and has_vae:
            model_name = flux_models[0]
            logger.info("Using FLUX model %s with VAEEncodeForInpaint", model_name)
            wf = self._build_flux_vae_encode_workflow(
                image_url, mask_url, prompt, negative_prompt,
                denoise, steps, seed,
            )
            wf["prompt"]["1"]["inputs"]["unet_name"] = model_name
            return wf, model_name

        raise ModelNotFoundError(
            f"No suitable inpainting model found. "
            f"Available diffusion models: {info.diffusion_models}, "
            f"CLIP: {info.clip_models}, VAE: {info.vae_models}. "
            f"Run ensure_flux_fill_models() to download required models."
        )

    # ----- Submission and result handling -----

    def _submit_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Submit a workflow to the SaladTechnologies API and wait for result."""
        workflow_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "prompt": workflow.get("prompt", workflow),
            "id": workflow_id,
        }

        logger.info("Submitting workflow %s to %s/prompt", workflow_id, self.api_url)
        result = self._post(
            f"{self.api_url}/prompt",
            payload,
            timeout=self.timeout,
        )

        return {
            "id": workflow_id,
            "result": result,
        }

    def _decode_result_images(self, result: dict[str, Any]) -> list[np.ndarray]:
        """Extract images from the API response."""
        images_b64 = result.get("result", {}).get("images", [])
        if not images_b64:
            raise ComfyUIError(
                f"No images in response. Full result: {json.dumps(result, indent=2)[:500]}"
            )

        decoded = []
        for b64_str in images_b64:
            img_bytes = base64.b64decode(b64_str)
            if _HAS_PIL:
                pil_img = Image.open(io.BytesIO(img_bytes))
                decoded.append(np.array(pil_img.convert("RGB")))
            else:
                raise ImportError("Pillow required to decode result images")

        return decoded

    # ----- Public API -----

    def health_check(self) -> bool:
        """Check if the ComfyUI API is healthy and ready."""
        try:
            h = self._get(f"{self.api_url}/health", timeout=5)
            r = self._get(f"{self.api_url}/ready", timeout=5)
            healthy = h.get("status") == "healthy"
            ready = r.get("status") == "ready"
            logger.info("ComfyUI API health=%s ready=%s", healthy, ready)
            return healthy and ready
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    def inpaint(
        self,
        image: Any,
        mask: Any,
        prompt: str = "clean empty background, photorealistic, high quality",
        negative_prompt: str = "artifacts, distortion, blurry, text, watermark",
        denoise: float | None = None,
        steps: int | None = None,
        seed: int | None = None,
        auto_download: bool = False,
    ) -> InpaintResult:
        """Run inpainting on an image using a mask.

        Parameters
        ----------
        image : PIL.Image.Image | np.ndarray
            Source image (RGB).
        mask : PIL.Image.Image | np.ndarray
            Binary mask where white (255) = area to inpaint, black (0) = keep.
        prompt : str
            Text prompt describing the desired inpainted content.
        negative_prompt : str
            Negative prompt for unwanted content.
        denoise : float | None
            Override denoise strength (0.0-1.0).
        steps : int | None
            Override sampling steps.
        seed : int | None
            Override random seed.
        auto_download : bool
            If True, automatically download missing models.

        Returns
        -------
        InpaintResult
            Contains the inpainted image and metadata.
        """
        t0 = time.monotonic()

        if auto_download:
            ready = False
            if self.prefer_flux2:
                try:
                    ready = self.ensure_flux2_models()
                except Exception as e:
                    logger.warning("FLUX.2 ensure failed: %s", e)
            if not ready:
                try:
                    self.ensure_flux_fill_models()
                except Exception as e:
                    logger.warning("FLUX Fill download failed, trying FLUX Dev: %s", e)
                    try:
                        self.ensure_flux_dev_models()
                    except Exception as e2:
                        logger.error("Model download failed: %s", e2)

        # Ensure mask is RGB (white=inpaint)
        mask_rgb = _mask_to_rgb(mask)

        # Start image server and register images
        server = self._ensure_image_server()
        img_name = server.add_image("source.png", image)
        mask_name = server.add_image("mask.png", mask_rgb)

        image_url = server.get_url(img_name, self.local_ip)
        mask_url = server.get_url(mask_name, self.local_ip)

        logger.info("Serving source at %s, mask at %s", image_url, mask_url)

        # Select and build workflow
        workflow, model_used = self._select_workflow(
            image_url, mask_url, prompt, negative_prompt,
            denoise, steps, seed,
        )

        # Submit and wait for result
        result = self._submit_workflow(workflow)

        # Decode result images
        images = self._decode_result_images(result)

        elapsed = time.monotonic() - t0
        logger.info(
            "Inpainting complete in %.1fs using %s, got %d images",
            elapsed, model_used, len(images),
        )

        return InpaintResult(
            image=images[0],
            workflow_id=result["id"],
            elapsed_s=elapsed,
            model_used=model_used,
            stats=result.get("result", {}).get("stats", {}),
        )

    def inpaint_batch(
        self,
        items: list[tuple[Any, Any, str]],
        negative_prompt: str = "artifacts, distortion, blurry, text, watermark",
        denoise: float | None = None,
        steps: int | None = None,
        auto_download: bool = False,
    ) -> list[InpaintResult]:
        """Inpaint multiple images sequentially.

        Parameters
        ----------
        items : list[tuple[image, mask, prompt]]
            List of (image, mask, prompt) tuples.
        negative_prompt : str
            Shared negative prompt.
        denoise : float | None
            Override denoise strength.
        steps : int | None
            Override steps.
        auto_download : bool
            Auto-download models if needed (only checked once).

        Returns
        -------
        list[InpaintResult]
        """
        results = []
        for i, (image, mask, prompt) in enumerate(items):
            logger.info("Inpainting batch item %d/%d", i + 1, len(items))
            result = self.inpaint(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                denoise=denoise,
                steps=steps,
                seed=self.seed + i if self.seed >= 0 else -1,
                auto_download=auto_download and i == 0,
            )
            results.append(result)
        return results

    def close(self):
        """Shut down the image server."""
        if self._image_server is not None:
            self._image_server.shutdown()
            self._image_server = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
