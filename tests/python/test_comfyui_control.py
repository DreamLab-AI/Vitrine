# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.comfyui_control with a monkeypatched HTTP layer.

No real network is performed: every test replaces ``ComfyUIControl._request``
with a recording fake so route paths, methods, and payloads are asserted
directly. ADR-014 routes (stock graph API vs Salad control plane) are verified
by inspecting the URLs the client constructs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

try:
    import pytest
except ImportError:  # standalone fallback where pytest is not installed
    class _PytestShim:
        """Minimal pytest.raises shim so the suite runs without pytest installed."""

        @staticmethod
        @contextmanager
        def raises(exc_type):
            try:
                yield
            except exc_type:
                return
            except Exception as other:  # noqa: BLE001
                raise AssertionError(
                    f"expected {exc_type.__name__}, got {type(other).__name__}: {other}"
                )
            raise AssertionError(f"{exc_type.__name__} was not raised")

    pytest = _PytestShim()  # type: ignore

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.comfyui_control import (  # noqa: E402
    ComfyUIControl,
    ComfyUIControlError,
    ROUTE_PROMPT,
    ROUTE_HISTORY,
    ROUTE_FREE,
    ROUTE_OBJECT_INFO,
    ROUTE_CONTROL_MODELS,
)

COMFY = "http://comfyui:8188"
CONTROL = "http://comfyui:3001"


class _Recorder:
    """Records calls and returns scripted responses keyed by (method, route fragment)."""

    def __init__(self, scripted=None, error=None):
        self.calls = []
        self.scripted = scripted or []
        self.error = error

    def __call__(self, method, url, *, json_body=None, timeout=None, raw=False):
        self.calls.append(
            {"method": method, "url": url, "json_body": json_body,
             "timeout": timeout, "raw": raw}
        )
        if self.error is not None:
            raise self.error
        for matcher, response in self.scripted:
            if matcher(method, url, json_body):
                return response() if callable(response) else response
        return {}


def _make(monkeypatch, recorder, control_url=CONTROL):
    client = ComfyUIControl(COMFY, control_url=control_url, timeout=5.0, client_id="cid-1")
    monkeypatch.setattr(client, "_request", recorder)
    return client


def test_submit_workflow_posts_to_prompt_and_returns_id(monkeypatch):
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "POST" and u.endswith(ROUTE_PROMPT),
         {"prompt_id": "abc-123", "number": 1}),
    ])
    client = _make(monkeypatch, rec)

    graph = {"3": {"class_type": "KSampler", "inputs": {}}}
    prompt_id = client.submit_workflow(graph)

    assert prompt_id == "abc-123"
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{COMFY}{ROUTE_PROMPT}"
    assert call["json_body"] == {"prompt": graph, "client_id": "cid-1"}


def test_submit_workflow_raises_without_prompt_id(monkeypatch):
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "POST", {"error": "bad graph"}),
    ])
    client = _make(monkeypatch, rec)
    with pytest.raises(ComfyUIControlError):
        client.submit_workflow({})


def test_wait_polls_history_and_returns_when_ready(monkeypatch):
    state = {"n": 0}

    def history_response():
        state["n"] += 1
        if state["n"] < 3:
            return {}  # not ready yet
        return {"pid-9": {"outputs": {"7": {"images": [{"filename": "r.png"}]}}}}

    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "GET" and f"{ROUTE_HISTORY}/pid-9" in u, history_response),
    ])
    client = _make(monkeypatch, rec)
    monkeypatch.setattr("pipeline.comfyui_control.time.sleep", lambda *_: None)

    entry = client.wait("pid-9", timeout=30, poll=0.01)

    assert entry == {"outputs": {"7": {"images": [{"filename": "r.png"}]}}}
    assert state["n"] >= 3
    assert all(c["url"] == f"{COMFY}{ROUTE_HISTORY}/pid-9" for c in rec.calls)


def test_wait_times_out(monkeypatch):
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "GET", {}),  # never ready
    ])
    client = _make(monkeypatch, rec)

    seq = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr("pipeline.comfyui_control.time.monotonic", lambda: next(seq))
    monkeypatch.setattr("pipeline.comfyui_control.time.sleep", lambda *_: None)

    with pytest.raises(TimeoutError):
        client.wait("pid-x", timeout=10, poll=0.01)


def test_free_vram_posts_to_free(monkeypatch):
    rec = _Recorder()
    client = _make(monkeypatch, rec)

    client.free_vram(unload_models=True, free_memory=True)

    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{COMFY}{ROUTE_FREE}"
    assert call["json_body"] == {"unload_models": True, "free_memory": True}


def test_ensure_model_returns_true_when_present_via_object_info(monkeypatch):
    object_info = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["flux2_dev_fp8mixed.safetensors"]]}}
        }
    }
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "GET" and u.endswith(ROUTE_OBJECT_INFO), object_info),
        (lambda m, u, b: m == "GET" and u.endswith(ROUTE_CONTROL_MODELS), {"present": []}),
    ])
    client = _make(monkeypatch, rec)

    present = client.ensure_model("flux2_dev_fp8mixed.safetensors")

    assert present is True
    # No download POST should have been issued.
    assert all(c["method"] != "POST" for c in rec.calls)


def test_ensure_model_returns_true_when_present_via_control_plane(monkeypatch):
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "GET" and u.endswith(ROUTE_OBJECT_INFO), {}),
        (lambda m, u, b: m == "GET" and u.endswith(ROUTE_CONTROL_MODELS),
         {"present": ["flux2-vae.safetensors"]}),
    ])
    client = _make(monkeypatch, rec)
    assert client.ensure_model("flux2-vae.safetensors") is True


def test_transport_error_raises_comfyui_control_error(monkeypatch):
    rec = _Recorder(error=ComfyUIControlError("Transport error for POST .../prompt: refused"))
    client = _make(monkeypatch, rec)
    with pytest.raises(ComfyUIControlError):
        client.submit_workflow({"1": {}})


def test_health_true_on_system_stats(monkeypatch):
    rec = _Recorder(scripted=[
        (lambda m, u, b: u.endswith("/system_stats"), {"system": {"os": "linux"}}),
    ])
    client = _make(monkeypatch, rec)
    assert client.health() is True


def test_health_false_when_all_probes_fail(monkeypatch):
    rec = _Recorder(error=ComfyUIControlError("down"))
    client = _make(monkeypatch, rec)
    assert client.health() is False


def test_download_outputs_writes_view_files(monkeypatch, tmp_path):
    rec = _Recorder(scripted=[
        (lambda m, u, b: m == "GET" and "/view" in u, b"PNGBYTES"),
    ])
    client = _make(monkeypatch, rec)

    history_entry = {
        "outputs": {
            "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
        }
    }
    paths = client.download_outputs(history_entry, str(tmp_path))

    assert len(paths) == 1
    assert Path(paths[0]).read_bytes() == b"PNGBYTES"
    view_call = [c for c in rec.calls if "/view" in c["url"]][0]
    assert view_call["raw"] is True
    assert "filename=out.png" in view_call["url"]


# ---------------------------------------------------------------------------
# Standalone runner so the suite is exercisable where pytest is not installed.
# ---------------------------------------------------------------------------

def _run_standalone() -> int:
    class _MP:
        """Minimal monkeypatch shim: setattr on objects, modules, and dotted paths."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, name_or_value, value=None):
            if isinstance(target, str):
                mod_path, attr = target.rsplit(".", 1)
                obj = sys.modules[mod_path] if mod_path in sys.modules else None
                if obj is None:
                    parts = mod_path.split(".")
                    obj = sys.modules[parts[0]]
                    for p in parts[1:]:
                        obj = getattr(obj, p)
                old = getattr(obj, attr)
                self._undo.append((obj, attr, old))
                setattr(obj, attr, name_or_value)
            else:
                old = getattr(target, name_or_value, None)
                self._undo.append((target, name_or_value, old))
                setattr(target, name_or_value, value)

        def undo(self):
            for obj, attr, old in reversed(self._undo):
                setattr(obj, attr, old)
            self._undo.clear()

    class _Tmp:
        def __init__(self, base):
            self._base = base

        def __str__(self):
            return self._base

    import tempfile

    tests = [
        (test_submit_workflow_posts_to_prompt_and_returns_id, False),
        (test_submit_workflow_raises_without_prompt_id, False),
        (test_wait_polls_history_and_returns_when_ready, False),
        (test_wait_times_out, False),
        (test_free_vram_posts_to_free, False),
        (test_ensure_model_returns_true_when_present_via_object_info, False),
        (test_ensure_model_returns_true_when_present_via_control_plane, False),
        (test_transport_error_raises_comfyui_control_error, False),
        (test_health_true_on_system_stats, False),
        (test_health_false_when_all_probes_fail, False),
        (test_download_outputs_writes_view_files, True),
    ]
    failures = 0
    for fn, needs_tmp in tests:
        mp = _MP()
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as td:
                    fn(mp, _Tmp(td))
            else:
                fn(mp)
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        finally:
            mp.undo()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
