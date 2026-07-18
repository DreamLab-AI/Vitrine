# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.pixal3d_client (single-image, PRD v4 R8).

All HTTP and GPU calls are mocked: no live ComfyUI/native service, no GPU,
and — critically — no Pixal3D weights (they are NOT staged; the client is a
contract scaffold). The tests pin the drop-in contract instead:

* Pixal3DResult is field-compatible with Trellis2Result, so the generator
  chain and eval harness can swap generators without code changes.
* GLB bytes are returned verbatim (PRD v4 R6).
* Lineage records generator=Pixal3D, the TRELLIS.2-backbone note, and the
  MIT licence (ADR-025 amendment 2026-07-09).
* With no executor available the client raises loudly (never fakes).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Minimal trimesh stub when the real one is absent (keeps CI slim).
try:
    import trimesh as _trimesh_real  # noqa: F401
except ImportError:
    _stub = types.ModuleType("trimesh")
    _stub.Trimesh = MagicMock
    _stub.Scene = MagicMock
    _stub.load = MagicMock(return_value=MagicMock())
    _util = types.ModuleType("trimesh.util")
    _util.concatenate = MagicMock(return_value=MagicMock())
    _stub.util = _util
    sys.modules["trimesh"] = _stub
    sys.modules["trimesh.util"] = _util

from pipeline.pixal3d_client import (  # noqa: E402
    Pixal3DClient,
    Pixal3DResult,
    PIXAL3D_SI_WORKFLOW,
)
from pipeline.trellis2_client import Trellis2Result  # noqa: E402

GLB_BYTES = b"glTF-FAKE-PIXAL-PAYLOAD" * 10


@pytest.fixture()
def crop(tmp_path) -> Path:
    p = tmp_path / "0001_vase.png"
    p.write_bytes(b"\x89PNG-fake")
    return p


@pytest.fixture()
def workflow(tmp_path) -> Path:
    """A placeholder single-image workflow (stands in for the env-build-time
    pixal3d_single_image_pbr.json, which does not exist yet)."""
    wf = {
        "_meta": {"note": "comment keys must be stripped"},
        "10": {"class_type": "LoadImage",
               "inputs": {"image": "CROP_IMAGE_PLACEHOLDER"}},
        "40": {"class_type": "Pixal3DImageToShape",
               "inputs": {"seed": 0, "resolution": "512"}},
        "60": {"class_type": "Pixal3DBakePBR",
               "inputs": {"texture_size": 512, "target_face_count": 1}},
        "70": {"class_type": "ExportGLB",
               "inputs": {"filename_prefix": "x"}},
    }
    p = tmp_path / "pixal3d_single_image_pbr.json"
    p.write_text(json.dumps(wf))
    return p


def _client(**kw) -> Pixal3DClient:
    c = Pixal3DClient(comfyui_url="http://comfy:8188", **kw)
    c._load_glb = MagicMock(return_value=None)   # no real GLB parsing in unit tests
    return c


# ---------------------------------------------------------------------------
# Drop-in contract: Pixal3DResult must be Trellis2Result-compatible
# ---------------------------------------------------------------------------

def test_result_is_field_compatible_with_trellis2result():
    t2_fields = {f.name for f in dataclasses.fields(Trellis2Result)}
    px_fields = {f.name for f in dataclasses.fields(Pixal3DResult)}
    assert t2_fields <= px_fields, f"missing fields: {t2_fields - px_fields}"
    # The derived properties the generator chain + eval harness consume.
    for prop in ("glb_sha256", "vertex_count", "face_count", "has_texture"):
        assert isinstance(getattr(Pixal3DResult, prop), property)


def test_public_surface_mirrors_trellis2client():
    from pipeline.trellis2_client import Trellis2Client
    for name in ("reconstruct_from_image", "from_config", "health_check"):
        assert callable(getattr(Pixal3DClient, name))
        assert callable(getattr(Trellis2Client, name))
    import inspect
    px_sig = inspect.signature(Pixal3DClient.reconstruct_from_image)
    t2_sig = inspect.signature(Trellis2Client.reconstruct_from_image)
    assert list(px_sig.parameters) == list(t2_sig.parameters)


def test_result_sha_empty_without_glb():
    assert Pixal3DResult().glb_sha256 == ""


# ---------------------------------------------------------------------------
# No-executor fail-fast (weights/nodes not staged — never fake a result)
# ---------------------------------------------------------------------------

def test_shipped_workflow_does_not_exist_yet():
    # Honesty pin: the ComfyUI workflow is VERIFY-ON-ENV-BUILD. If this file
    # appears, the fail-fast test below and the scaffold docstrings must be
    # revisited (and this test updated) as part of the env build.
    assert not PIXAL3D_SI_WORKFLOW.exists()


def test_no_executor_raises_loudly(crop):
    client = _client()                      # no native_url, no workflow file
    with pytest.raises(RuntimeError, match="VERIFY-ON-ENV-BUILD"):
        client.reconstruct_from_image(crop)
    # Fail-fast means NO HTTP was attempted.
    client.session = MagicMock()
    with pytest.raises(RuntimeError):
        client.reconstruct_from_image(crop)
    client.session.post.assert_not_called()
    client.session.get.assert_not_called()


def test_missing_crop_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _client().reconstruct_from_image(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# Workflow prompt construction (ComfyUI scaffold executor)
# ---------------------------------------------------------------------------

def test_build_prompt_substitutes_crop_and_parameters(workflow):
    client = _client(workflow_path=workflow, resolution="1024_cascade",
                     texture_size=2048, face_count_high=123_456)
    prompt = client._build_prompt("uploaded_vase.png", seed=7, label="vase")

    assert prompt["10"]["inputs"]["image"] == "uploaded_vase.png"
    assert prompt["40"]["inputs"]["seed"] == 7
    assert prompt["40"]["inputs"]["resolution"] == "1024_cascade"
    assert prompt["60"]["inputs"]["texture_size"] == 2048
    assert prompt["60"]["inputs"]["target_face_count"] == 123_456
    assert prompt["70"]["inputs"]["filename_prefix"] == "vitrine_object_vase"
    # Comment keys never reach the ComfyUI API.
    assert not any(k.startswith("_") for k in prompt)


# ---------------------------------------------------------------------------
# ComfyUI executor (mocked session)
# ---------------------------------------------------------------------------

def _mock_comfy_session(history_outputs: dict) -> MagicMock:
    session = MagicMock()

    def post(url, **kw):
        resp = MagicMock(status_code=200)
        if url.endswith("/upload/image"):
            resp.json.return_value = {"name": "uploaded_vase.png"}
        elif url.endswith("/prompt"):
            resp.json.return_value = {"prompt_id": "p123"}
        else:  # /free
            resp.json.return_value = {}
        return resp

    def get(url, **kw):
        resp = MagicMock(status_code=200)
        if "/history/" in url:
            resp.json.return_value = {
                "p123": {"status": {"status_str": "success"},
                         "outputs": history_outputs},
            }
        elif "/view" in url:
            resp.content = GLB_BYTES
        return resp

    session.post.side_effect = post
    session.get.side_effect = get
    return session


def test_comfyui_executor_returns_glb_bytes_verbatim(crop, workflow):
    client = _client(workflow_path=workflow, poll_interval=0.001)
    client.session = _mock_comfy_session({
        "80": {"result": [{"filename": "vitrine_object_vase.glb", "subfolder": ""}]},
    })
    result = client.reconstruct_from_image(crop, label="vase",
                                           provenance={"source_frame": "f1.jpg"})
    assert result.glb_data == GLB_BYTES              # byte-identical (R6)
    assert result.glb_sha256                          # hash recorded
    assert result.backend == "pixal3d-comfyui-single-image"
    # Lineage: generator + backbone + licence (ADR-025 amendment).
    assert result.lineage["conditioning"] == "single-image"
    assert result.lineage["generator"] == "Pixal3D"
    assert "TRELLIS.2" in result.lineage["generator_backbone"]
    assert result.lineage["licence"] == "MIT"
    assert result.lineage["generator_repo"] == "TencentARC/Pixal3D"
    assert result.lineage["source"]["source_frame"] == "f1.jpg"
    # Only ONE image was uploaded — single-crop conditioning (ADR-025).
    uploads = [c for c in client.session.post.call_args_list
               if c.args and c.args[0].endswith("/upload/image")]
    assert len(uploads) == 1


def test_comfyui_executor_reports_missing_glb(crop, workflow):
    client = _client(workflow_path=workflow, poll_interval=0.001)
    client.session = _mock_comfy_session({})          # no outputs at all
    result = client.reconstruct_from_image(crop, label="vase")
    assert result.glb_data is None
    assert result.error


# ---------------------------------------------------------------------------
# Native-service executor (mirrored PRD v4 R5 contract)
# ---------------------------------------------------------------------------

def test_native_executor_decodes_high_low_pair(crop):
    client = _client(native_url="http://native:8403")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "glb_high_b64": base64.b64encode(GLB_BYTES).decode(),
        "glb_low_b64": base64.b64encode(b"low-poly").decode(),
        "lineage": {"timings": {"run_s": 30.0}},
    }
    client.session.post = MagicMock(return_value=resp)

    result = client.reconstruct_from_image(crop, label="vase")
    assert result.glb_data == GLB_BYTES
    assert result.glb_low_data == b"low-poly"
    assert result.backend == "pixal3d-native-single-image"
    assert result.lineage["timings"] == {"run_s": 30.0}
    assert result.lineage["generator"] == "Pixal3D"
    url = client.session.post.call_args.args[0]
    assert url == "http://native:8403/generate"


def test_native_executor_error_path(crop):
    client = _client(native_url="http://native:8403")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": "OOM at 1536"}
    client.session.post = MagicMock(return_value=resp)

    result = client.reconstruct_from_image(crop, label="vase")
    assert result.glb_data is None
    assert result.error == "OOM at 1536"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

def test_from_config_reads_all_fields():
    cfg = types.SimpleNamespace(
        comfyui_url="http://c:8188", native_url="http://n:8403", timeout=99,
        resolution="512", texture_size=1024, seed=1, ss_steps=2, shape_steps=3,
        tex_steps=4, face_count_high=5, face_count_low=6,
    )
    client = Pixal3DClient.from_config(cfg)
    assert client.native_url == "http://n:8403"
    assert client.timeout == 99
    assert (client.ss_steps, client.shape_steps, client.tex_steps) == (2, 3, 4)
    assert (client.face_count_high, client.face_count_low) == (5, 6)


def test_from_config_defensive_defaults():
    client = Pixal3DClient.from_config(types.SimpleNamespace())
    assert client.comfyui_url == "http://vitrine-comfyui:8188"
    assert client.native_url == ""
    assert client.resolution == "1536_cascade"
    assert client.workflow_path == PIXAL3D_SI_WORKFLOW
