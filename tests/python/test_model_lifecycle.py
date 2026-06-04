# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the serial model lifecycle + endpoint resolver (ADR-013 D-013.2/3).

All GPU / docker / network effects are monkeypatched — no real GPU, docker, or
network is touched. The robustness invariant (every effect guarded, unload never
raises) is verified by stubbing each effect to *raise* and asserting the manager
still completes cleanly.
"""

from __future__ import annotations

import pytest

from pipeline import endpoints as ep
from pipeline import model_lifecycle as ml
from pipeline.model_lifecycle import ModelLifecycleManager, ModelSpec


# ---------------------------------------------------------------------------
#  ModelSpec validation
# ---------------------------------------------------------------------------

def test_modelspec_defaults_soft():
    spec = ModelSpec(name="gemma-4", engine="llama.cpp",
                     endpoint="http://agent-vlm:8080", vram_estimate_gb=20.0)
    assert spec.isolation == "soft"
    assert spec.docker_service is None


def test_modelspec_invalid_isolation_raises():
    with pytest.raises(ValueError):
        ModelSpec(name="x", engine="comfyui", isolation="medium")


def test_modelspec_hard_requires_docker_service():
    with pytest.raises(ValueError):
        ModelSpec(name="flux", engine="comfyui", isolation="hard")
    # with a service it is fine
    spec = ModelSpec(name="flux", engine="comfyui", isolation="hard",
                     docker_service="comfyui")
    assert spec.docker_service == "comfyui"


# ---------------------------------------------------------------------------
#  peak_vram_estimate  (serial invariant: max, not sum)
# ---------------------------------------------------------------------------

def test_peak_vram_estimate_returns_max_not_sum():
    specs = [
        ModelSpec("vlm", "llama.cpp", vram_estimate_gb=20.0),
        ModelSpec("flux", "comfyui", vram_estimate_gb=32.0, isolation="hard",
                  docker_service="comfyui"),
        ModelSpec("hunyuan", "comfyui", vram_estimate_gb=16.0, isolation="hard",
                  docker_service="comfyui"),
    ]
    assert ml.peak_vram_estimate(specs) == 32.0
    # explicitly NOT the sum
    assert ml.peak_vram_estimate(specs) != sum(s.vram_estimate_gb for s in specs)


def test_peak_vram_estimate_empty():
    assert ml.peak_vram_estimate([]) == 0.0


# ---------------------------------------------------------------------------
#  soft unload posts /free  (and flushes torch cache)
# ---------------------------------------------------------------------------

def test_soft_unload_posts_free(monkeypatch):
    calls = {"free_url": None, "free_payload": None, "torch": 0}

    def fake_post_free(endpoint):
        calls["free_url"] = endpoint.rstrip("/") + "/free"
        calls["free_payload"] = {"unload_models": True, "free_memory": True}
        return True

    def fake_empty():
        calls["torch"] += 1
        return True

    # never measure real VRAM in tests
    monkeypatch.setattr(ml, "free_vram_gb", lambda: [])
    monkeypatch.setattr(ml, "_post_free", fake_post_free)
    monkeypatch.setattr(ml, "_empty_torch_cache", fake_empty)
    # docker must NOT be touched for a soft spec
    monkeypatch.setattr(ml, "_docker",
                        lambda *a, **k: pytest.fail("docker called for soft"))

    spec = ModelSpec(name="gemma-4", engine="llama.cpp",
                     endpoint="http://agent-vlm:8080", vram_estimate_gb=20.0)
    mgr = ModelLifecycleManager()
    with mgr.stage(spec):
        pass

    assert calls["free_url"] == "http://agent-vlm:8080/free"
    assert calls["free_payload"] == {"unload_models": True, "free_memory": True}
    assert calls["torch"] == 1


# ---------------------------------------------------------------------------
#  hard isolation: docker start before yield, stop after
# ---------------------------------------------------------------------------

def test_hard_isolation_docker_start_then_stop(monkeypatch):
    order = []

    def fake_docker(action, service):
        order.append((action, service))
        return True

    monkeypatch.setattr(ml, "free_vram_gb", lambda: [])
    monkeypatch.setattr(ml, "_docker", fake_docker)
    # soft-free helpers must NOT be touched for a hard spec
    monkeypatch.setattr(ml, "_post_free",
                        lambda *a, **k: pytest.fail("/free called for hard"))
    monkeypatch.setattr(ml, "_empty_torch_cache",
                        lambda *a, **k: pytest.fail("torch flush for hard"))

    spec = ModelSpec(name="FLUX.2-dev", engine="comfyui",
                     endpoint="http://comfyui:8188", vram_estimate_gb=32.0,
                     isolation="hard", docker_service="comfyui")
    mgr = ModelLifecycleManager()
    inside = []
    with mgr.stage(spec):
        inside.append(list(order))  # snapshot mid-stage

    # start happened before the body ran, stop after
    assert inside[0] == [("start", "comfyui")]
    assert order == [("start", "comfyui"), ("stop", "comfyui")]


def test_hard_unload_still_stops_when_stage_raises(monkeypatch):
    order = []
    monkeypatch.setattr(ml, "free_vram_gb", lambda: [])
    monkeypatch.setattr(ml, "_docker",
                        lambda action, service: order.append((action, service)) or True)

    spec = ModelSpec(name="flux", engine="comfyui", vram_estimate_gb=32.0,
                     isolation="hard", docker_service="comfyui")
    mgr = ModelLifecycleManager()
    with pytest.raises(RuntimeError):
        with mgr.stage(spec):
            raise RuntimeError("stage boom")
    # stop still ran despite the exception
    assert ("stop", "comfyui") in order


# ---------------------------------------------------------------------------
#  Robustness: never raises with no GPU / no docker / no network
# ---------------------------------------------------------------------------

def test_no_gpu_no_docker_no_network_never_raises(monkeypatch):
    # nvidia-smi, docker, requests, urllib all blow up — manager must survive.
    def boom_run(*a, **k):
        raise OSError("no nvidia-smi / no docker here")

    monkeypatch.setattr(ml.subprocess, "run", boom_run)

    # force the urllib path (no requests) and make it raise too
    import builtins
    real_import = builtins.__import__

    def no_requests_import(name, *a, **k):
        if name == "requests":
            raise ImportError("no requests")
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_requests_import)

    def boom_urlopen(*a, **k):
        raise OSError("no network")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", boom_urlopen)

    soft = ModelSpec(name="vlm", engine="llama.cpp",
                     endpoint="http://agent-vlm:8080", vram_estimate_gb=20.0)
    hard = ModelSpec(name="flux", engine="comfyui",
                     endpoint="http://comfyui:8188", vram_estimate_gb=32.0,
                     isolation="hard", docker_service="comfyui")

    mgr = ModelLifecycleManager()
    # both tiers complete without raising
    with mgr.stage(soft):
        pass
    with mgr.stage(hard):
        pass

    # the probing helpers also degrade to empty / False, never raise
    assert ml.free_vram_gb() == []
    assert ml._post_free("http://x") is False
    assert ml._empty_torch_cache() is False
    assert ml._docker("stop", "x") is False


def test_assert_headroom_warns_and_returns_false_when_tight(monkeypatch):
    # measured free is small; spec wants a lot -> False (but no raise)
    monkeypatch.setattr(ml, "free_vram_gb", lambda: [5.0, 4.0])
    monkeypatch.setattr(ml, "gpu_vram_gb", lambda: [48.0, 24.0])
    spec = ModelSpec(name="flux", engine="comfyui", vram_estimate_gb=32.0)
    mgr = ModelLifecycleManager()
    assert mgr.assert_headroom(spec) is False


def test_assert_headroom_ok_when_free_enough(monkeypatch):
    monkeypatch.setattr(ml, "free_vram_gb", lambda: [40.0, 10.0])
    spec = ModelSpec(name="flux", engine="comfyui", vram_estimate_gb=32.0)
    mgr = ModelLifecycleManager()
    assert mgr.assert_headroom(spec) is True


def test_assert_headroom_no_gpu_best_effort_true(monkeypatch):
    monkeypatch.setattr(ml, "free_vram_gb", lambda: [])
    spec = ModelSpec(name="flux", engine="comfyui", vram_estimate_gb=32.0)
    mgr = ModelLifecycleManager()
    assert mgr.assert_headroom(spec) is True


# ---------------------------------------------------------------------------
#  _post_free urllib path (no requests) — monkeypatched 200 response
# ---------------------------------------------------------------------------

def test_post_free_urllib_success(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_requests(name, *a, **k):
        if name == "requests":
            raise ImportError("no requests")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_requests)

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getcode(self):
            return 200

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["timeout"] = timeout
        return FakeResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert ml._post_free("http://comfyui:8188") is True
    assert captured["url"] == "http://comfyui:8188/free"
    import json as _json
    assert _json.loads(captured["data"].decode()) == {
        "unload_models": True, "free_memory": True}


def test_post_free_empty_endpoint_false():
    assert ml._post_free("") is False
    assert ml._post_free("   ") is False


# ---------------------------------------------------------------------------
#  Endpoints resolver (D-013.3)
# ---------------------------------------------------------------------------

def test_endpoints_dns_defaults():
    e = ep.Endpoints.from_env(env={})
    assert e.comfyui_url == "http://comfyui:8188"
    assert e.comfyui_api_url == "http://comfyui:3001"
    assert e.agent_vlm_url == "http://agent-vlm:8080"
    assert e.milo_url == "http://milo:8090"
    assert e.come_url == "http://come:8091"


def test_endpoints_env_override():
    env = {
        "V2G_COMFYUI_URL": "http://10.0.0.5:8188",
        "V2G_AGENT_VLM_URL": "  http://vlm.local:8080  ",  # trimmed
        "V2G_MILO_URL": "   ",  # blank -> default
    }
    e = ep.Endpoints.from_env(env=env)
    assert e.comfyui_url == "http://10.0.0.5:8188"
    assert e.agent_vlm_url == "http://vlm.local:8080"
    assert e.milo_url == "http://milo:8090"  # blank ignored
    assert e.comfyui_api_url == "http://comfyui:3001"  # untouched default


def test_endpoints_resolve_precedence():
    e = ep.Endpoints.from_env(env={})
    # override wins
    assert e.resolve("comfyui", override="http://override:9999") == "http://override:9999"
    # aliases
    assert e.resolve("comfyui") == "http://comfyui:8188"
    assert e.resolve("agent-vlm") == "http://agent-vlm:8080"
    assert e.resolve("salad") == "http://comfyui:3001"
    assert e.resolve("comfyui_api_url") == "http://comfyui:3001"
    # blank override ignored
    assert e.resolve("milo", override="  ") == "http://milo:8090"


def test_endpoints_resolve_unknown_raises():
    e = ep.Endpoints.from_env(env={})
    with pytest.raises(KeyError):
        e.resolve("does-not-exist")


def test_module_resolve_helper():
    assert ep.resolve("come", env={}) == "http://come:8091"
    assert ep.resolve("come", override="http://x:1", env={}) == "http://x:1"


def test_legacy_constants_present_but_not_default():
    # legacy IPs documented, never auto-selected
    assert ep.LEGACY_SINGLE_HOST["comfyui_url"] == "http://192.168.2.48:8189"
    assert ep.LEGACY_SINGLE_HOST["comfyui_api_url"] == "http://192.168.2.48:3001"
    e = ep.Endpoints.from_env(env={})
    assert "192.168.2.48" not in e.comfyui_url
    assert "192.168.2.48" not in e.comfyui_api_url
