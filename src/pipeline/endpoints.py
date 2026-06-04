# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runtime service-URL resolver for the v2g-net model mesh (ADR-013 D-013.3).

ADR-013 replaces the hardcoded ``192.168.2.48:port`` / ``localhost:port``
endpoints (see ``config.py`` InpaintConfig / Hunyuan3DConfig) with a
user-defined Docker bridge network (``v2g-net``) on the GPU host, where every
service resolves by DNS name. The orchestrator never hardcodes an IP again.

Resolution order for each service, highest precedence first:

  1. an explicit ``override`` argument (e.g. the manifest ``[pipeline]`` block
     for the legacy single-host case);
  2. the matching ``V2G_*`` environment variable;
  3. the ``v2g-net`` DNS default (``http://comfyui:8188`` etc.).

The legacy ``192.168.2.48`` IPs are retained ONLY as documented fallback
constants (``LEGACY_SINGLE_HOST``) for the single-host stopgap — they are never
used automatically.

CLI:  python -m pipeline.endpoints
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
#  v2g-net DNS service defaults (ADR-013 D-013.3 topology table)
# ---------------------------------------------------------------------------

#: ComfyUI prompt-graph API — FLUX.2-dev + Hunyuan3D-2.1 nodes.
DEFAULT_COMFYUI_URL = "http://comfyui:8188"
#: ComfyUI Salad add-on control-plane API (model probe/download/control, ADR-014).
DEFAULT_COMFYUI_API_URL = "http://comfyui:3001"
#: Unified gemma-4-26B-A4B multimodal artifact VLM + reasoner (D-013.5).
DEFAULT_AGENT_VLM_URL = "http://agent-vlm:8080"
#: MILo mesh-extraction sidecar (device_ids ['1']).
DEFAULT_MILO_URL = "http://milo:8090"
#: CoMe mesh-extraction sidecar (device_ids ['1'], gated/non-commercial).
DEFAULT_COME_URL = "http://come:8091"


#: Environment-variable names, one per service field.
ENV_VARS: dict[str, str] = {
    "comfyui_url": "V2G_COMFYUI_URL",
    "comfyui_api_url": "V2G_COMFYUI_API_URL",
    "agent_vlm_url": "V2G_AGENT_VLM_URL",
    "milo_url": "V2G_MILO_URL",
    "come_url": "V2G_COME_URL",
}


#: Legacy single-host fallbacks (the old hardcoded .48 box). Documented for the
#: stopgap "reuse the existing .48 ComfyUI over the LAN" path (ADR-013 Q4) — the
#: resolver never selects these automatically; set the matching V2G_* env var or
#: pass an override to use them.
LEGACY_SINGLE_HOST: dict[str, str] = {
    # InpaintConfig.comfyui_direct_url / Hunyuan3DConfig.comfyui_url
    "comfyui_url": "http://192.168.2.48:8189",
    # InpaintConfig.comfyui_api_url / Hunyuan3DConfig.api_url
    "comfyui_api_url": "http://192.168.2.48:3001",
}


# ---------------------------------------------------------------------------
#  Resolver
# ---------------------------------------------------------------------------

@dataclass
class Endpoints:
    """Resolved service URLs for one pipeline run.

    Fields mirror the ``v2g-net`` services. Build with :meth:`from_env` (the
    normal path) or construct directly for tests / single-host overrides.
    """

    comfyui_url: str = DEFAULT_COMFYUI_URL
    comfyui_api_url: str = DEFAULT_COMFYUI_API_URL
    agent_vlm_url: str = DEFAULT_AGENT_VLM_URL
    milo_url: str = DEFAULT_MILO_URL
    come_url: str = DEFAULT_COME_URL

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Endpoints":
        """Build from the ``V2G_*`` environment variables, falling back to the
        v2g-net DNS defaults. Pass ``env`` to resolve against an explicit mapping
        (defaults to :data:`os.environ`). Empty/whitespace values are ignored so
        an exported-but-blank var does not blank a service URL."""
        src = os.environ if env is None else env
        defaults = cls()
        values: dict[str, str] = {}
        for field_name, var in ENV_VARS.items():
            raw = src.get(var)
            values[field_name] = raw.strip() if (raw and raw.strip()) \
                else getattr(defaults, field_name)
        return cls(**values)

    def as_dict(self) -> dict[str, str]:
        """Return the resolved URLs keyed by service field name."""
        return {f: getattr(self, f) for f in ENV_VARS}

    def resolve(self, service: str, override: str | None = None) -> str:
        """Resolve a single service URL.

        Precedence: explicit ``override`` > already-resolved field on this
        instance (env or DNS default). ``service`` is a field name
        (``comfyui_url``, ``comfyui_api_url``, ``agent_vlm_url``, ``milo_url``,
        ``come_url``) or a bare alias (``comfyui``, ``agent-vlm``/``agent_vlm``,
        ``milo``, ``come``). A non-empty ``override`` always wins. Raises
        ``KeyError`` for an unknown service so misconfiguration fails fast."""
        if override is not None and override.strip():
            return override.strip()
        key = _canonical_service(service)
        if key not in ENV_VARS:
            raise KeyError(
                f"unknown service {service!r}; known: {sorted(ENV_VARS)}"
            )
        return getattr(self, key)


_ALIASES: dict[str, str] = {
    "comfyui": "comfyui_url",
    "comfyui_direct": "comfyui_url",
    "comfyui_api": "comfyui_api_url",
    "salad": "comfyui_api_url",
    "agent-vlm": "agent_vlm_url",
    "agent_vlm": "agent_vlm_url",
    "vlm": "agent_vlm_url",
    "milo": "milo_url",
    "come": "come_url",
}


def _canonical_service(service: str) -> str:
    """Normalise a service name/alias to an :class:`Endpoints` field name."""
    s = service.strip().lower()
    if s in ENV_VARS:
        return s
    return _ALIASES.get(s, s)


def resolve(service: str, override: str | None = None,
            env: dict[str, str] | None = None) -> str:
    """Module-level convenience: resolve one service against the environment.

    Equivalent to ``Endpoints.from_env(env).resolve(service, override)``."""
    return Endpoints.from_env(env).resolve(service, override)


def _main() -> int:
    ep = Endpoints.from_env()
    width = max(len(k) for k in ENV_VARS)
    for field_name, url in ep.as_dict().items():
        print(f"{field_name:<{width}}  {url}  (env: {ENV_VARS[field_name]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
