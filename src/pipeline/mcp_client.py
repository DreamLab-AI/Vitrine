# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Typed Python wrapper for LichtFeld Studio's JSON-RPC 2.0 MCP API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async support is optional; fall back gracefully.
# ---------------------------------------------------------------------------
try:
    import aiohttp

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


class McpError(Exception):
    """Error returned by the LichtFeld MCP server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP error {code}: {message}")


class McpConnectionError(McpError):
    """Could not reach the MCP endpoint."""

    def __init__(self, detail: str) -> None:
        super().__init__(-32000, detail)


@dataclass
class TrainingState:
    """Snapshot of training progress."""
    running: bool
    iteration: int
    max_iterations: int
    loss: float
    psnr: float
    ssim: float
    elapsed_s: float
    num_gaussians: int


@dataclass
class RenderResult:
    """Result of a render capture call."""
    path: str
    width: int
    height: int
    format: str


@dataclass
class SelectionResult:
    """Result of a selection operation."""
    count: int
    description: str


@dataclass
class LossHistory:
    """Loss history from training."""
    iterations: list[int]
    losses: list[float]


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------

class McpClient:
    """Synchronous HTTP client for LichtFeld MCP (JSON-RPC 2.0)."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:45677/mcp",
        timeout: float = 30.0,
        training_timeout: float = 600.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.training_timeout = training_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _build_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        return payload

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> Any:
        effective_timeout = timeout or self.timeout
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urlopen(req, timeout=effective_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if "error" in data and data["error"] is not None:
                    err = data["error"]
                    raise McpError(
                        err.get("code", -1),
                        err.get("message", "Unknown error"),
                        err.get("data"),
                    )
                return data.get("result")
            except (TimeoutError, OSError, URLError, HTTPError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "MCP request attempt %d/%d failed: %s",
                        attempt, self.max_retries, exc,
                    )
                    time.sleep(self.retry_delay * attempt)

        raise McpConnectionError(f"Failed after {self.max_retries} attempts: {last_error}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        params: dict[str, Any] = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments
        return self._post(self._build_request("tools/call", params), timeout=timeout)

    # ------------------------------------------------------------------
    # Typed convenience methods
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            self._post(self._build_request("ping"), timeout=2.0)
            return True
        except (McpError, McpConnectionError):
            return False

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._post(self._build_request("tools/list"))
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return result if isinstance(result, list) else []

    def load_dataset(
        self,
        path: str,
        images_folder: str = "images",
        max_iterations: int = 30000,
        strategy: str = "mcmc",
    ) -> dict[str, Any]:
        return self.call_tool("scene.load_dataset", {
            "path": path,
            "images_folder": images_folder,
            "max_iterations": max_iterations,
            "strategy": strategy,
        })

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        return self.call_tool("scene.load_checkpoint", {"path": path})

    def save_checkpoint(self, path: str) -> dict[str, Any]:
        return self.call_tool("scene.save_checkpoint", {"path": path})

    def save_ply(self, path: str) -> dict[str, Any]:
        return self.call_tool("scene.save_ply", {"path": path})

    # ------------------------------------------------------------------
    # Native scene export (LichtFeld v0.5.1+: USD / SOG / SPZ / HTML).
    # These export the currently-loaded scene — call load_checkpoint() first
    # if the training session is no longer resident.
    # ------------------------------------------------------------------

    def export_usd(self, path: str, **extra: Any) -> dict[str, Any]:
        """Export the loaded scene to USD via native scene.export_usd (v0.5.1+)."""
        return self.call_tool("scene.export_usd", {"path": path, **extra})

    def export_sog(self, path: str, **extra: Any) -> dict[str, Any]:
        return self.call_tool("scene.export_sog", {"path": path, **extra})

    def export_spz(self, path: str, **extra: Any) -> dict[str, Any]:
        return self.call_tool("scene.export_spz", {"path": path, **extra})

    def export_html(self, path: str, **extra: Any) -> dict[str, Any]:
        return self.call_tool("scene.export_html", {"path": path, **extra})

    def export_status(self) -> dict[str, Any]:
        return self.call_tool("scene.export_status")

    def training_start(self) -> dict[str, Any]:
        return self.call_tool("training.start")

    def training_get_state(self) -> TrainingState:
        raw = self.call_tool("training.get_state")
        return TrainingState(
            running=raw.get("running", False),
            iteration=raw.get("iteration", 0),
            max_iterations=raw.get("max_iterations", 0),
            loss=raw.get("loss", float("inf")),
            psnr=raw.get("psnr", 0.0),
            ssim=raw.get("ssim", 0.0),
            elapsed_s=raw.get("elapsed_s", 0.0),
            num_gaussians=raw.get("num_gaussians", 0),
        )

    def training_get_loss_history(self, last_n: int = 100) -> LossHistory:
        raw = self.call_tool("training.get_loss_history", {"last_n": last_n})
        return LossHistory(
            iterations=raw.get("iterations", []),
            losses=raw.get("losses", []),
        )

    def render_capture(
        self,
        width: int = 1920,
        height: int = 1080,
        output_path: str | None = None,
        camera_index: int | None = None,
    ) -> RenderResult:
        args: dict[str, Any] = {"width": width, "height": height}
        if output_path:
            args["output_path"] = output_path
        if camera_index is not None:
            args["camera_index"] = camera_index
        raw = self.call_tool("render.capture", args)
        return RenderResult(
            path=raw.get("path", ""),
            width=raw.get("width", width),
            height=raw.get("height", height),
            format=raw.get("format", "png"),
        )

    def selection_by_description(self, description: str) -> SelectionResult:
        raw = self.call_tool("selection.by_description", {"description": description})
        return SelectionResult(
            count=raw.get("count", 0),
            description=description,
        )

    def selection_get(self) -> dict[str, Any]:
        return self.call_tool("selection.get")

    def selection_clear(self) -> dict[str, Any]:
        return self.call_tool("selection.clear")

    def plugin_invoke(self, plugin: str, action: str, params: dict[str, Any] | None = None) -> Any:
        args: dict[str, Any] = {"plugin": plugin, "action": action}
        if params:
            args["params"] = params
        return self.call_tool("plugin.invoke", args)

    def plugin_list(self) -> list[dict[str, Any]]:
        result = self.call_tool("plugin.list")
        return result if isinstance(result, list) else []

    def ask_advisor(self, question: str) -> str:
        raw = self.call_tool("training.ask_advisor", {"question": question})
        return raw.get("answer", str(raw)) if isinstance(raw, dict) else str(raw)

    def wait_training_complete(self, poll_interval: float = 5.0) -> TrainingState:
        """Block until training finishes, polling state."""
        deadline = time.monotonic() + self.training_timeout
        while time.monotonic() < deadline:
            state = self.training_get_state()
            if not state.running:
                return state
            logger.info(
                "Training: iter %d/%d  loss=%.6f  psnr=%.2f",
                state.iteration, state.max_iterations, state.loss, state.psnr,
            )
            time.sleep(poll_interval)
        raise McpError(-32001, f"Training did not complete within {self.training_timeout}s")


# ---------------------------------------------------------------------------
# Async client (requires aiohttp)
# ---------------------------------------------------------------------------

class AsyncMcpClient:
    """Async HTTP client for LichtFeld MCP (JSON-RPC 2.0).

    Falls back to sync if aiohttp is not installed.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:45677/mcp",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        if not _HAS_AIOHTTP:
            raise ImportError("aiohttp is required for AsyncMcpClient: pip install aiohttp")
        self.endpoint = endpoint
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._request_id = 0
        self._session: aiohttp.ClientSession | None = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post(self, payload: dict[str, Any], timeout: float | None = None) -> Any:
        import asyncio

        session = await self._ensure_session()
        effective_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else self.timeout

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    timeout=effective_timeout,
                ) as resp:
                    data = await resp.json()
                if "error" in data and data["error"] is not None:
                    err = data["error"]
                    raise McpError(err.get("code", -1), err.get("message", "Unknown error"), err.get("data"))
                return data.get("result")
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)

        raise McpConnectionError(f"Failed after {self.max_retries} attempts: {last_error}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        params: dict[str, Any] = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": params,
        }
        return await self._post(payload, timeout=timeout)

    async def ping(self) -> bool:
        try:
            payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": "ping"}
            await self._post(payload, timeout=2.0)
            return True
        except (McpError, McpConnectionError):
            return False
