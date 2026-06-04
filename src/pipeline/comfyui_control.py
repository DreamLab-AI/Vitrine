# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent-controlled ComfyUI client (ADR-014).

A thin, dependency-light control surface so the in-container Claude Code
orchestrator can drive the existing .48 ComfyUI instance and its "Salad"
add-on control-plane API. This is the transport seam of the Generative
Recovery bounded context (ADR-014 D-014.4): it speaks ComfyUI's prompt-graph
and the Salad control vocabulary and nothing higher-level. The agent loop
(plan / submit / evaluate / decide / release, ADR-014 D-014.3) lives in the
orchestrator and uses this class to act.

Two distinct HTTP surfaces, kept explicit (ADR-014 D-014.2):

Stock ComfyUI graph API (``comfyui_url``, default service ``comfyui:8188``)
    ``/prompt``          submit a prompt graph for execution
    ``/history/{id}``    poll an executed prompt's outputs
    ``/object_info``     introspect available nodes / checkpoints / loaders
    ``/view``            fetch a single output file (image / mesh / etc.)
    ``/free``            unload models and free VRAM between stages
    ``/system_stats``    liveness / device probe

Salad control-plane API (``control_url``, default service ``comfyui:3001``)
    ``{control_url}/models``           enumerate present model files
    ``{control_url}/models/download``  request a model download (model lifecycle)

The Salad surface is an add-on API *inside* the ComfyUI container giving model
probe / download / lifecycle control (ADR-014: it is NOT a cloud service). When
``control_url`` is ``None`` the client operates against the stock graph API
only; model-lifecycle calls degrade gracefully.

Transport: uses ``requests`` if importable, otherwise the stdlib
``urllib.request``. Every network call carries an explicit timeout and raises
``ComfyUIControlError`` on any HTTP or transport failure.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

try:
    import requests as _requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover - exercised only where requests absent
    _requests = None  # type: ignore
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Route table — single auditable block (ADR-014 D-014.2)
# ---------------------------------------------------------------------------
# Stock ComfyUI graph API routes (joined onto ``comfyui_url``).
ROUTE_PROMPT = "/prompt"
ROUTE_HISTORY = "/history"            # appended with /{prompt_id}
ROUTE_OBJECT_INFO = "/object_info"
ROUTE_VIEW = "/view"
ROUTE_FREE = "/free"
ROUTE_SYSTEM_STATS = "/system_stats"

# Salad control-plane API routes (joined onto ``control_url``).
ROUTE_CONTROL_MODELS = "/models"
ROUTE_CONTROL_MODEL_DOWNLOAD = "/models/download"


class ComfyUIControlError(RuntimeError):
    """Raised on any HTTP status error or transport failure to ComfyUI/control-plane."""


class ComfyUIControl:
    """Agent-facing control client for the .48 ComfyUI + Salad control plane.

    Parameters
    ----------
    comfyui_url : str
        Base URL of the stock ComfyUI graph API (e.g. ``http://comfyui:8188``).
    control_url : str | None
        Base URL of the Salad control-plane API (e.g. ``http://comfyui:3001``).
        When ``None``, model-lifecycle operations fall back to graph-API probing
        only and downloads are unavailable.
    timeout : float
        Default per-request HTTP timeout in seconds.
    client_id : str | None
        ComfyUI client id sent with ``/prompt`` submissions. Auto-generated if
        not supplied, so ``/history`` correlation stays stable for one client.
    """

    def __init__(
        self,
        comfyui_url: str,
        control_url: str | None = None,
        timeout: float = 30.0,
        client_id: str | None = None,
    ) -> None:
        self.comfyui_url = comfyui_url.rstrip("/")
        self.control_url = control_url.rstrip("/") if control_url else None
        self.timeout = timeout
        self.client_id = client_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Low-level HTTP transport
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict | None = None,
        timeout: float | None = None,
        raw: bool = False,
    ) -> Any:
        """Perform an HTTP request, returning parsed JSON (or raw bytes if ``raw``).

        Raises
        ------
        ComfyUIControlError
            On any HTTP status error or transport-level failure.
        """
        timeout = self.timeout if timeout is None else timeout

        if _HAS_REQUESTS:
            return self._request_requests(method, url, json_body, timeout, raw)
        return self._request_urllib(method, url, json_body, timeout, raw)

    def _request_requests(
        self,
        method: str,
        url: str,
        json_body: dict | None,
        timeout: float,
        raw: bool,
    ) -> Any:
        try:
            resp = _requests.request(
                method, url, json=json_body, timeout=timeout
            )
        except _requests.exceptions.RequestException as exc:  # transport
            raise ComfyUIControlError(f"Transport error for {method} {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise ComfyUIControlError(
                f"HTTP {resp.status_code} for {method} {url}: {resp.text[:500]}"
            )
        if raw:
            return resp.content
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise ComfyUIControlError(
                f"Non-JSON response for {method} {url}: {resp.text[:200]}"
            ) from exc

    def _request_urllib(
        self,
        method: str,
        url: str,
        json_body: dict | None,
        timeout: float,
        raw: bool,
    ) -> Any:
        body = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        req = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover - best-effort body read
                pass
            raise ComfyUIControlError(
                f"HTTP {exc.code} for {method} {url}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            raise ComfyUIControlError(f"Transport error for {method} {url}: {exc}") from exc

        if raw:
            return payload
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ComfyUIControlError(
                f"Non-JSON response for {method} {url}: {payload[:200]!r}"
            ) from exc

    def _get(self, url: str, *, timeout: float | None = None, raw: bool = False) -> Any:
        return self._request("GET", url, timeout=timeout, raw=raw)

    def _post(self, url: str, json_body: dict, *, timeout: float | None = None) -> Any:
        return self._request("POST", url, json_body=json_body, timeout=timeout)

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Return True if the ComfyUI graph API answers ``/system_stats`` (or root)."""
        try:
            self._get(f"{self.comfyui_url}{ROUTE_SYSTEM_STATS}", timeout=min(self.timeout, 10.0))
            return True
        except ComfyUIControlError:
            pass
        try:
            self._get(f"{self.comfyui_url}/", timeout=min(self.timeout, 10.0))
            return True
        except ComfyUIControlError as exc:
            logger.debug("health probe failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Model discovery / lifecycle (Salad control plane)
    # ------------------------------------------------------------------

    def probe_models(self) -> dict:
        """Enumerate available models/nodes via ``/object_info`` and the control plane.

        Returns a dict with:
          ``object_info``  raw ComfyUI ``/object_info`` payload (nodes + loaders)
          ``checkpoints``  flat list of checkpoint filenames found in loaders
          ``control``      control-plane ``/models`` payload (only when ``control_url`` set)
        """
        result: dict[str, Any] = {"object_info": {}, "checkpoints": [], "control": None}

        try:
            object_info = self._get(f"{self.comfyui_url}{ROUTE_OBJECT_INFO}")
            if isinstance(object_info, dict):
                result["object_info"] = object_info
                result["checkpoints"] = self._extract_checkpoints(object_info)
        except ComfyUIControlError as exc:
            logger.debug("object_info probe failed: %s", exc)

        if self.control_url:
            try:
                result["control"] = self._get(f"{self.control_url}{ROUTE_CONTROL_MODELS}")
            except ComfyUIControlError as exc:
                logger.debug("control-plane /models probe failed: %s", exc)

        return result

    @staticmethod
    def _extract_checkpoints(object_info: dict) -> list[str]:
        """Pull checkpoint/diffusion model filenames out of an ``/object_info`` payload."""
        loaders = {
            "CheckpointLoaderSimple": "ckpt_name",
            "UNETLoader": "unet_name",
            "VAELoader": "vae_name",
            "DiffusionModelLoader": "model_name",
        }
        found: list[str] = []
        for node_name, input_name in loaders.items():
            node = object_info.get(node_name)
            if not isinstance(node, dict):
                continue
            required = node.get("input", {}).get("required", {})
            spec = required.get(input_name)
            if isinstance(spec, list) and spec and isinstance(spec[0], list):
                for name in spec[0]:
                    if isinstance(name, str) and name not in found:
                        found.append(name)
        return found

    def _model_present(self, name: str, probe: dict | None = None) -> bool:
        """Check whether ``name`` is present in a probe result (control plane + checkpoints)."""
        probe = probe if probe is not None else self.probe_models()

        if name in probe.get("checkpoints", []):
            return True

        control = probe.get("control")
        if isinstance(control, dict):
            for value in control.values():
                if isinstance(value, list) and name in value:
                    return True
            if name in control.get("present", []) or name in control.get("models", []):
                return True
        elif isinstance(control, list):
            for item in control:
                if item == name:
                    return True
                if isinstance(item, dict) and item.get("name") == name:
                    return True
        return False

    def ensure_model(
        self,
        name: str,
        download_url: str | None = None,
        poll_timeout: float = 1800,
    ) -> bool:
        """Ensure ``name`` is present, downloading via the control plane if missing.

        Returns True if the model is present (already, or after a successful
        download + poll); False if it could not be made present within
        ``poll_timeout``.
        """
        if self._model_present(name):
            logger.info("model already present: %s", name)
            return True

        if not self.control_url:
            logger.warning(
                "model %s missing and no control_url configured; cannot download", name
            )
            return False

        spec: dict[str, Any] = {"name": name}
        if download_url:
            spec["url"] = download_url
        logger.info("requesting control-plane download: %s", name)
        self._post(f"{self.control_url}{ROUTE_CONTROL_MODEL_DOWNLOAD}", spec)

        deadline = time.monotonic() + poll_timeout
        interval = 5.0
        while time.monotonic() < deadline:
            if self._model_present(name):
                logger.info("model became present after download: %s", name)
                return True
            time.sleep(interval)
        logger.error("model %s not present after %.0fs", name, poll_timeout)
        return False

    # ------------------------------------------------------------------
    # Workflow submission + result retrieval (graph API)
    # ------------------------------------------------------------------

    def submit_workflow(self, graph: dict) -> str:
        """POST a prompt graph to ``/prompt`` and return the assigned ``prompt_id``."""
        payload = {"prompt": graph, "client_id": self.client_id}
        resp = self._post(f"{self.comfyui_url}{ROUTE_PROMPT}", payload)
        if not isinstance(resp, dict) or "prompt_id" not in resp:
            raise ComfyUIControlError(
                f"/prompt did not return a prompt_id: {json.dumps(resp)[:300]}"
            )
        prompt_id = str(resp["prompt_id"])
        logger.info("submitted workflow, prompt_id=%s", prompt_id)
        return prompt_id

    def wait(self, prompt_id: str, timeout: float = 1800, poll: float = 2.0) -> dict:
        """Poll ``/history/{prompt_id}`` until the entry appears; return it.

        Raises
        ------
        TimeoutError
            If the history entry does not appear within ``timeout`` seconds.
        """
        url = f"{self.comfyui_url}{ROUTE_HISTORY}/{prompt_id}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self._get(url)
            if isinstance(history, dict) and prompt_id in history:
                entry = history[prompt_id]
                if isinstance(entry, dict):
                    logger.info("history ready for prompt_id=%s", prompt_id)
                    return entry
            time.sleep(poll)
        raise TimeoutError(
            f"prompt {prompt_id} did not complete within {timeout}s"
        )

    def download_outputs(self, history_entry: dict, dest_dir: str) -> list[str]:
        """Fetch all output files referenced by a history entry into ``dest_dir``.

        Parses ``history_entry['outputs']`` for nodes carrying ``images`` (and
        the analogous ``gifs`` / ``files`` / ``meshes`` lists used by 3D nodes),
        GETs each via ``/view``, writes it, and returns the saved file paths.
        """
        os.makedirs(dest_dir, exist_ok=True)
        saved: list[str] = []
        outputs = history_entry.get("outputs", {})
        if not isinstance(outputs, dict):
            return saved

        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for key in ("images", "gifs", "files", "meshes"):
                items = node_output.get(key, [])
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    saved.append(self._fetch_view(item, dest_dir))
        return saved

    def _fetch_view(self, item: dict, dest_dir: str) -> str:
        """GET a single output file from ``/view`` and write it under ``dest_dir``."""
        params = {"filename": item["filename"]}
        if item.get("subfolder"):
            params["subfolder"] = item["subfolder"]
        if item.get("type"):
            params["type"] = item["type"]
        url = f"{self.comfyui_url}{ROUTE_VIEW}?{urlencode(params)}"
        content = self._get(url, raw=True)
        if not isinstance(content, (bytes, bytearray)):
            content = bytes(content)
        out_path = os.path.join(dest_dir, os.path.basename(item["filename"]))
        with open(out_path, "wb") as fh:
            fh.write(content)
        logger.info("saved output %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # VRAM lifecycle (graph API)
    # ------------------------------------------------------------------

    def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """POST ``/free`` to unload models and free VRAM between pipeline stages."""
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        self._post(f"{self.comfyui_url}{ROUTE_FREE}", payload)
        logger.info(
            "freed VRAM (unload_models=%s, free_memory=%s)", unload_models, free_memory
        )
