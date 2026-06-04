# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serial model load/unload lifecycle (ADR-013 D-013.2).

Stages run **serially** behind :class:`ModelLifecycleManager` so peak VRAM is
``max(stage)`` not ``sum(stages)``. Each stage declares a :class:`ModelSpec`;
the manager, as a context manager around the stage:

  (a) asserts free-VRAM headroom >= ``spec.vram_estimate_gb`` (best-effort —
      logs a WARNING and continues when VRAM cannot be measured, e.g. no GPU);
  (b) for ``isolation == "hard"``: ``docker start <docker_service>`` before the
      stage (cold-start, driver-level VRAM guarantees);
  (c) yields control to the stage body;
  (d) on exit unloads —
        * ``soft`` (default): ``POST {endpoint}/free`` with
          ``{"unload_models": true, "free_memory": true}`` AND
          ``torch.cuda.empty_cache()`` when torch is importable;
        * ``hard``: ``docker stop <docker_service>``.

**Robustness invariant:** every subprocess and network call is guarded
(try/except + timeouts). The module imports and runs cleanly with *no* GPU,
*no* docker, and *no* network — every such absence degrades to a logged
warning. Unload NEVER raises; a failed stage exception still propagates.

Quant/VRAM figures live in ``sota_registry.REGISTRY``; this module is the
runtime executor. ``gpu_vram_gb()`` (totals) is reused from that registry;
``free_vram_gb()`` here queries per-GPU *free* memory.
"""

from __future__ import annotations

import json
import logging
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from pipeline.sota_registry import gpu_vram_gb

logger = logging.getLogger(__name__)

#: Subprocess wall-clock guards (seconds).
_NVIDIA_SMI_TIMEOUT = 10
_DOCKER_TIMEOUT = 120
#: HTTP soft-free guard (seconds).
_HTTP_TIMEOUT = 30

_VALID_ISOLATION = ("soft", "hard")


# ---------------------------------------------------------------------------
#  Model spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """One stage's model, sized for the serial VRAM budget (ADR-013 D-013.2).

    Attributes
    ----------
    name : human/registry name (e.g. ``"FLUX.2-dev"``).
    engine : serving engine — ``comfyui`` | ``llama.cpp`` | ``vllm`` |
        ``sidecar`` | ``native`` (informational; drives operator choices).
    endpoint : base service URL for the soft ``/free`` call (resolve via
        :mod:`pipeline.endpoints`). May be empty for native/sidecar stages.
    vram_estimate_gb : peak single-model VRAM; the headroom assertion target
        and the per-stage value behind the ``peak = max(stage)`` invariant.
    isolation : ``"soft"`` (in-process free, container stays warm) or
        ``"hard"`` (``docker start``/``stop`` for driver-level reclamation).
    gpu_affinity : optional device index / ids hint (informational).
    docker_service : container name for ``hard`` isolation start/stop.
    checkpoint : optional checkpoint filename (provenance).
    """

    name: str
    engine: str
    endpoint: str = ""
    vram_estimate_gb: float = 0.0
    isolation: str = "soft"
    gpu_affinity: Optional[str] = None
    docker_service: Optional[str] = None
    checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.isolation not in _VALID_ISOLATION:
            raise ValueError(
                f"ModelSpec.isolation must be one of {_VALID_ISOLATION}, "
                f"got {self.isolation!r}"
            )
        if self.isolation == "hard" and not self.docker_service:
            raise ValueError(
                f"ModelSpec {self.name!r}: isolation='hard' requires "
                f"docker_service (the container to start/stop)"
            )


# ---------------------------------------------------------------------------
#  VRAM probing
# ---------------------------------------------------------------------------

def free_vram_gb() -> list[float]:
    """Per-GPU *free* VRAM in GB via ``nvidia-smi``. Empty list if unavailable.

    Mirrors :func:`pipeline.sota_registry.gpu_vram_gb` (which reports totals) but
    queries ``memory.free``. Never raises — returns ``[]`` on any failure so the
    no-GPU path degrades to a warning rather than an exception."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT,
        )
        if out.returncode != 0:
            return []
        return [round(float(x.strip()) / 1024.0, 1)
                for x in out.stdout.splitlines() if x.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def peak_vram_estimate(specs: "list[ModelSpec]") -> float:
    """Serial-lifecycle invariant: peak VRAM is ``max(stage)``, not ``sum``.

    Returns the largest ``vram_estimate_gb`` across ``specs`` (0.0 if empty).
    This is the whole point of D-013.2 — only one model is resident at a time,
    so the budget is bounded by the single heaviest stage."""
    if not specs:
        return 0.0
    return max(s.vram_estimate_gb for s in specs)


# ---------------------------------------------------------------------------
#  HTTP soft-free (requests if importable, else urllib — both guarded)
# ---------------------------------------------------------------------------

def _post_free(endpoint: str) -> bool:
    """POST ``{endpoint}/free`` with the ComfyUI unload payload. Returns True on
    a 2xx response, False otherwise. Never raises — any network/timeout/parse
    failure is logged at WARNING and reported as False."""
    if not endpoint or not endpoint.strip():
        logger.warning("soft unload: no endpoint set; skipping /free")
        return False
    url = endpoint.rstrip("/") + "/free"
    payload = {"unload_models": True, "free_memory": True}

    try:
        import requests  # type: ignore
    except ImportError:
        requests = None  # type: ignore

    if requests is not None:
        try:
            resp = requests.post(url, json=payload, timeout=_HTTP_TIMEOUT)
            ok = 200 <= resp.status_code < 300
            if not ok:
                logger.warning("soft unload: %s -> HTTP %s", url, resp.status_code)
            return ok
        except Exception as exc:  # noqa: BLE001 — unload must never raise
            logger.warning("soft unload: POST %s failed: %s", url, exc)
            return False

    # urllib fallback — same guard discipline.
    import urllib.error
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            status = getattr(resp, "status", None) or resp.getcode()
            ok = 200 <= int(status) < 300
            if not ok:
                logger.warning("soft unload: %s -> HTTP %s", url, status)
            return ok
    except Exception as exc:  # noqa: BLE001 — unload must never raise
        logger.warning("soft unload: POST %s failed: %s", url, exc)
        return False


def _empty_torch_cache() -> bool:
    """Call ``torch.cuda.empty_cache()`` if torch is importable and CUDA is
    available. Returns True if the cache was flushed. Never raises."""
    try:
        import torch  # type: ignore
    except ImportError:
        logger.debug("soft unload: torch not importable; skipping empty_cache")
        return False
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        logger.debug("soft unload: torch CUDA unavailable; skipping empty_cache")
        return False
    except Exception as exc:  # noqa: BLE001 — unload must never raise
        logger.warning("soft unload: torch.cuda.empty_cache failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
#  Docker primitives (hard isolation — guarded)
# ---------------------------------------------------------------------------

def _docker(action: str, service: str) -> bool:
    """Run ``docker <action> <service>`` (``start``/``stop``). Returns True on
    success. Never raises — missing docker, missing container, or timeout all
    degrade to a logged WARNING and False."""
    try:
        out = subprocess.run(
            ["docker", action, service],
            capture_output=True, text=True, timeout=_DOCKER_TIMEOUT,
        )
        if out.returncode == 0:
            logger.info("docker %s %s: ok", action, service)
            return True
        logger.warning("docker %s %s failed (rc=%s): %s",
                       action, service, out.returncode,
                       (out.stderr or "").strip())
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("docker %s %s unavailable: %s", action, service, exc)
        return False


# ---------------------------------------------------------------------------
#  The lifecycle manager
# ---------------------------------------------------------------------------

class ModelLifecycleManager:
    """Context-manager driver for the serial model lifecycle (ADR-013 D-013.2).

    Usage::

        mgr = ModelLifecycleManager()
        with mgr.stage(spec):
            run_stage(...)        # model resident here
        # model unloaded on exit (soft /free or hard docker stop)

    All container/network/GPU operations are best-effort and logged; the manager
    is safe to use on a host with no GPU, no docker, and no network.
    """

    def __init__(self, *, headroom_margin_gb: float = 0.0) -> None:
        #: Extra GB required on top of ``spec.vram_estimate_gb`` before a stage
        #: is considered to fit (0.0 = exact headroom).
        self.headroom_margin_gb = headroom_margin_gb

    # -- VRAM headroom check ------------------------------------------------

    def assert_headroom(self, spec: ModelSpec) -> bool:
        """Assert free-VRAM headroom >= ``spec.vram_estimate_gb`` (+ margin).

        Returns True if there is enough measured headroom on at least one GPU.
        When VRAM cannot be measured (no nvidia-smi / no GPU) this logs a WARNING
        and returns True (best-effort — do not block a CPU/dev host). When VRAM
        *is* measured and no GPU has enough free, logs a WARNING and returns
        False, but does **not** raise — the caller chose serial scheduling, so we
        surface the risk without aborting the run."""
        need = spec.vram_estimate_gb + self.headroom_margin_gb
        if need <= 0:
            return True
        free = free_vram_gb()
        if not free:
            logger.warning(
                "VRAM headroom: cannot measure free memory (no GPU / nvidia-smi)"
                " — proceeding for %s (needs ~%.1f GB)", spec.name, need)
            return True
        best = max(free)
        if best >= need:
            logger.info("VRAM headroom: %s needs ~%.1f GB, best free %.1f GB — ok",
                        spec.name, need, best)
            return True
        totals = gpu_vram_gb()
        logger.warning(
            "VRAM headroom: %s needs ~%.1f GB but best free is %.1f GB "
            "(totals=%s) — serial unload of the prior stage may be incomplete",
            spec.name, need, best, totals or "unknown")
        return False

    # -- unload tiers -------------------------------------------------------

    def unload(self, spec: ModelSpec) -> None:
        """Unload the model for ``spec`` per its isolation tier. Never raises.

        ``soft`` → POST ``/free`` + ``torch.cuda.empty_cache()``.
        ``hard`` → ``docker stop <docker_service>``.
        """
        try:
            if spec.isolation == "hard":
                if spec.docker_service:
                    _docker("stop", spec.docker_service)
                else:  # pragma: no cover — guarded by ModelSpec.__post_init__
                    logger.warning("hard unload: %s has no docker_service", spec.name)
                return
            # soft
            _post_free(spec.endpoint)
            _empty_torch_cache()
        except Exception as exc:  # noqa: BLE001 — unload must never raise
            logger.warning("unload(%s) unexpected error (suppressed): %s",
                           spec.name, exc)

    # -- context manager ----------------------------------------------------

    @contextmanager
    def stage(self, spec: ModelSpec) -> Iterator[ModelSpec]:
        """Serial-lifecycle context manager around one stage.

        Asserts headroom, (for ``hard``) starts the container, yields ``spec``,
        and unloads on exit (always — even if the stage body raises). The stage
        exception, if any, propagates; the unload itself never raises."""
        self.assert_headroom(spec)
        if spec.isolation == "hard" and spec.docker_service:
            _docker("start", spec.docker_service)
        try:
            logger.info("stage start: %s (engine=%s, isolation=%s, ~%.1f GB)",
                        spec.name, spec.engine, spec.isolation, spec.vram_estimate_gb)
            yield spec
        finally:
            logger.info("stage end: %s — unloading (%s)", spec.name, spec.isolation)
            self.unload(spec)
