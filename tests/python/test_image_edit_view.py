# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for pipeline.image_edit_view (ADR-025 D4 / PRD v4 R7 rung b).

All HTTP and GPU calls are mocked: no live ComfyUI, no Qwen weights. The tests
pin the escalation-rung contract: ONE crop + an instruction in, ONE edited
single image out (never a panel), lineage flags every synthesized pixel as
``surface: image-edit-inferred``, and the client degrades gracefully when the
edit UNET is not staged (the current registry ground truth).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.image_edit_view import (  # noqa: E402
    DEFAULT_BACK_INSTRUCTION,
    ImageEditResult,
    ImageEditView,
    QWEN_EDIT_VIEW_WORKFLOW,
)

EDITED_PNG = b"\x89PNG-EDITED-FAKE-BYTES" * 8
RGBA_PNG = b"\x89PNG-RGBA-MATTED-FAKE" * 8


@pytest.fixture()
def crop(tmp_path) -> Path:
    p = tmp_path / "0001_vase.png"
    p.write_bytes(b"\x89PNG-fake")            # undecodable -> flatten falls back
    return p


def _client(**kw) -> ImageEditView:
    c = ImageEditView(comfyui_url="http://comfy:8188", poll_interval=0.001, **kw)
    return c


def _mock_comfy_session(history_outputs: dict) -> MagicMock:
    session = MagicMock()

    def post(url, **kw):
        resp = MagicMock(status_code=200)
        if url.endswith("/upload/image"):
            resp.json.return_value = {"name": "uploaded_vase.png"}
        elif url.endswith("/prompt"):
            resp.json.return_value = {"prompt_id": "e123"}
        else:  # /free
            resp.json.return_value = {}
        return resp

    def get(url, **kw):
        resp = MagicMock(status_code=200)
        if "/history/" in url:
            resp.json.return_value = {
                "e123": {"status": {"status_str": "success"},
                         "outputs": history_outputs},
            }
        elif "/view" in url:
            resp.content = EDITED_PNG
        return resp

    session.post.side_effect = post
    session.get.side_effect = get
    return session


# ---------------------------------------------------------------------------
# Workflow template + prompt construction
# ---------------------------------------------------------------------------

def test_shipped_workflow_is_single_image_single_output():
    graph = json.loads(QWEN_EDIT_VIEW_WORKFLOW.read_text())
    nodes = {n["class_type"] for k, n in graph.items() if not k.startswith("_")}
    # One conditioning image in, one edited image out — never a panel set.
    loads = [n for k, n in graph.items()
             if not k.startswith("_") and n["class_type"] == "LoadImage"]
    assert len(loads) == 1
    saves = [n for k, n in graph.items()
             if not k.startswith("_") and n["class_type"] == "SaveImage"]
    assert len(saves) == 1
    # Instruction-driven Qwen edit conditioning (runtime-verified node name).
    assert "TextEncodeQwenImageEditPlus" in nodes
    # The retired panel-path nodes must never reappear here.
    assert "ImageBatch" not in nodes
    assert "Trellis2MultiViewImageToShape" not in nodes
    # The scaffold is honestly marked until the UNET is staged + e2e-verified.
    assert "_scaffold" in graph


def test_build_prompt_substitutes_image_instruction_and_parameters():
    client = _client(steps=8, cfg=3.0, shift=2.5, denoise=0.9, negative="blurry",
                     diffusion_model="qwen_custom.safetensors",
                     clip_model="clip_custom.safetensors",
                     vae_model="vae_custom.safetensors")
    prompt = client._build_prompt("uploaded_vase.png",
                                  "show the back of this object",
                                  seed=7, label="vase")

    assert prompt["10"]["inputs"]["image"] == "uploaded_vase.png"
    assert prompt["20"]["inputs"]["prompt"] == "show the back of this object"
    assert prompt["21"]["inputs"]["prompt"] == "blurry"          # negative
    assert prompt["1"]["inputs"]["unet_name"] == "qwen_custom.safetensors"
    assert prompt["2"]["inputs"]["clip_name"] == "clip_custom.safetensors"
    assert prompt["3"]["inputs"]["vae_name"] == "vae_custom.safetensors"
    assert prompt["40"]["inputs"]["shift"] == 2.5
    assert prompt["50"]["inputs"]["seed"] == 7
    assert prompt["50"]["inputs"]["steps"] == 8
    assert prompt["50"]["inputs"]["cfg"] == 3.0
    assert prompt["50"]["inputs"]["denoise"] == 0.9
    assert prompt["70"]["inputs"]["filename_prefix"] == "vitrine_editview_vase"
    # Comment/scaffold keys never reach the ComfyUI API.
    assert not any(k.startswith("_") for k in prompt)


# ---------------------------------------------------------------------------
# edit_view happy path
# ---------------------------------------------------------------------------

def test_edit_view_returns_single_edited_image(crop, tmp_path):
    client = _client()
    client.session = _mock_comfy_session({
        "70": {"images": [{"filename": "vitrine_editview_vase_00001_.png",
                           "subfolder": "", "type": "output"}]},
    })
    client._matte_output = MagicMock(return_value=None)   # rembg absent path
    out = tmp_path / "alt" / "vase_back.png"

    result = client.edit_view(crop, label="vase", seed=99,
                              provenance={"source_frame": "f1.jpg"},
                              output_path=out)

    assert result.error is None
    assert result.raw_image_data == EDITED_PNG            # verbatim download
    assert result.image_data == EDITED_PNG                # unmatted fallback
    assert result.image_path == out
    assert out.read_bytes() == EDITED_PNG
    assert result.image_sha256
    # Lineage: edit model + instruction + inferred-surface flag + provenance.
    assert result.lineage["edit_model"] == "Qwen-Image-Edit-2509"
    assert result.lineage["instruction"] == DEFAULT_BACK_INSTRUCTION
    assert result.lineage["surface"] == "image-edit-inferred"
    assert result.lineage["source_crop"] == str(crop)
    assert result.lineage["seed"] == 99
    assert result.lineage["matted"] is False
    assert result.lineage["source"]["source_frame"] == "f1.jpg"
    # Exactly ONE image uploaded — single-image contract, no panels.
    uploads = [c for c in client.session.post.call_args_list
               if c.args and c.args[0].endswith("/upload/image")]
    assert len(uploads) == 1
    # Serial VRAM lifecycle: /free before AND after the edit.
    frees = [c for c in client.session.post.call_args_list
             if c.args and c.args[0].endswith("/free")]
    assert len(frees) == 2


def test_edit_view_mattes_output_when_rembg_available(crop):
    client = _client()
    client.session = _mock_comfy_session({
        "70": {"images": [{"filename": "v.png", "subfolder": ""}]},
    })
    client._matte_output = MagicMock(return_value=RGBA_PNG)

    result = client.edit_view(crop, label="vase")
    assert result.raw_image_data == EDITED_PNG             # kept verbatim
    assert result.image_data == RGBA_PNG                   # matted candidate
    assert result.lineage["matted"] is True
    client._matte_output.assert_called_once_with(EDITED_PNG)


def test_edit_view_custom_instruction_lands_in_lineage_and_prompt(crop):
    client = _client()
    client.session = _mock_comfy_session({
        "70": {"images": [{"filename": "v.png", "subfolder": ""}]},
    })
    client._matte_output = MagicMock(return_value=None)

    result = client.edit_view(crop, instruction="show the left side", label="vase")
    assert result.lineage["instruction"] == "show the left side"
    submits = [c for c in client.session.post.call_args_list
               if c.args and c.args[0].endswith("/prompt")]
    graph = submits[0].kwargs["json"]["prompt"]
    assert graph["20"]["inputs"]["prompt"] == "show the left side"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_edit_view_missing_crop_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _client().edit_view(tmp_path / "nope.png")


def test_edit_view_reports_missing_output(crop):
    client = _client()
    client.session = _mock_comfy_session({})               # no outputs at all
    result = client.edit_view(crop, label="vase")
    assert result.image_data is None
    assert result.image_path is None
    assert result.error


def test_submit_prompt_surfaces_node_errors(crop):
    client = _client()
    session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"error": "invalid prompt",
                              "node_errors": {"1": {"errors": ["unet not found"]}}}
    session.post.return_value = resp
    client.session = session
    with pytest.raises(RuntimeError, match="unet not found"):
        client._submit_prompt({"1": {}})


# ---------------------------------------------------------------------------
# Staging probe (the UNET is NOT staged yet — registry ground truth)
# ---------------------------------------------------------------------------

def _object_info_session(unet_names: list[str]) -> MagicMock:
    session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "UNETLoader": {"input": {"required": {"unet_name": [unet_names]}}},
    }
    session.get.return_value = resp
    return session


def test_probe_edit_model_exact_hit():
    client = _client()
    client.session = _object_info_session(
        ["flux2_dev_fp8mixed.safetensors", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"])
    assert client.probe_edit_model() == "qwen_image_edit_2509_fp8_e4m3fn.safetensors"


def test_probe_edit_model_fuzzy_hit_adopts_staged_name():
    client = _client()
    client.session = _object_info_session(["Qwen-Image-Edit-2509_bf16.safetensors"])
    assert client.probe_edit_model() == "Qwen-Image-Edit-2509_bf16.safetensors"
    assert client.diffusion_model == "Qwen-Image-Edit-2509_bf16.safetensors"


def test_probe_edit_model_absent_returns_none():
    client = _client()
    client.session = _object_info_session(["flux2_dev_fp8mixed.safetensors"])
    assert client.probe_edit_model() is None


def test_probe_edit_model_connection_error_returns_none():
    import requests
    client = _client()
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("down")
    client.session = session
    assert client.probe_edit_model() is None


# ---------------------------------------------------------------------------
# Input preparation (RGBA flatten) — best-effort, never fatal
# ---------------------------------------------------------------------------

def test_prepare_input_flattens_rgba_onto_white(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "crop.png"
    im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))          # fully transparent
    im.putpixel((4, 4), (200, 10, 10, 255))               # one opaque pixel
    im.save(p)

    path, flattened = _client()._prepare_input(p)
    assert flattened is True
    assert path != p
    with Image.open(path) as flat:
        assert flat.mode == "RGB"
        assert flat.getpixel((0, 0)) == (255, 255, 255)   # transparent -> white
        assert flat.getpixel((4, 4)) == (200, 10, 10)     # object preserved


def test_prepare_input_falls_back_on_undecodable_bytes(crop):
    path, flattened = _client()._prepare_input(crop)
    assert path == crop
    assert flattened is False


def test_matte_output_returns_none_without_rembg(monkeypatch):
    # Force the guarded import to fail even if rembg happens to be installed.
    monkeypatch.setitem(sys.modules, "rembg", None)
    assert _client()._matte_output(EDITED_PNG) is None


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_from_config_reads_fields():
    cfg = types.SimpleNamespace(
        comfyui_url="http://c:8188", timeout=99,
        diffusion_model="d.safetensors", clip_model="c.safetensors",
        vae_model="v.safetensors", instruction="show the back",
        negative="ugly", steps=4, cfg=1.0, shift=3.0, denoise=0.8,
        seed=5, matte_output=False,
    )
    client = ImageEditView.from_config(cfg)
    assert client.comfyui_url == "http://c:8188"
    assert client.timeout == 99
    assert client.diffusion_model == "d.safetensors"
    assert client.instruction == "show the back"
    assert (client.steps, client.cfg, client.shift, client.denoise) == (4, 1.0, 3.0, 0.8)
    assert client.matte_output is False


def test_from_config_defaults_when_fields_absent():
    client = ImageEditView.from_config(types.SimpleNamespace())
    assert client.instruction == DEFAULT_BACK_INSTRUCTION
    assert client.diffusion_model == "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
    assert client.matte_output is True


def test_result_sha_empty_without_image():
    assert ImageEditResult().image_sha256 == ""
