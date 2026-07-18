# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for scripts/trellis2_native_service (PRD v4 R5).

Hermetic: ``_load_pipeline``/``_generate`` are mocked, so no model, no GPU,
no native trellis2 package needed — just Flask's test client. The tests pin
the HTTP contract the pipeline client (pipeline.trellis2_client,
``_generate_native``) already speaks: multipart ``image`` + form params in;
``{glb_high_b64, glb_low_b64, lineage}`` out; the 400/503/500 taxonomy.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("flask", reason="native-service tests need flask")
pytest.importorskip("PIL", reason="native-service tests need pillow")


def _load_service():
    """Import the service by path — scripts/ is not a package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "trellis2_native_service.py"
    spec = importlib.util.spec_from_file_location("trellis2_native_service", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["trellis2_native_service"] = module
    spec.loader.exec_module(module)
    return module


service = _load_service()

TIMINGS = {"run_s": 1.2, "to_glb_s": 3.4}


@pytest.fixture()
def client():
    service.app.config["TESTING"] = True
    return service.app.test_client()


@pytest.fixture()
def png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(buf, "PNG")
    return buf.getvalue()


def _post(client, png_bytes, **form):
    data = {"image": (io.BytesIO(png_bytes), "crop.png"), **form}
    return client.post("/generate", data=data, content_type="multipart/form-data")


# ---------------------------------------------------------------------------
# /health vs /ready
# ---------------------------------------------------------------------------

def test_health_is_liveness_only(client):
    resp = client.get("/health")
    assert resp.status_code == 200                    # up even with no env
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["pipeline_loaded"] is False
    assert body["service"] == "trellis2-native"


def test_ready_503_until_pipeline_loaded(client, monkeypatch):
    monkeypatch.setattr(service, "_pipeline", None)
    monkeypatch.setattr(service, "_native_available", lambda: False)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["ready"] is False
    assert body["native_package_available"] is False
    assert body["error"]


def test_ready_200_once_loaded(client, monkeypatch):
    monkeypatch.setattr(service, "_pipeline", MagicMock())
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.get_json()["ready"] is True


# ---------------------------------------------------------------------------
# /generate happy path (the client contract)
# ---------------------------------------------------------------------------

def test_generate_happy_path_returns_glb_pair_and_lineage(client, png_bytes, monkeypatch):
    gen = MagicMock(return_value=(b"HIGH-GLB", b"LOW-GLB", TIMINGS))
    monkeypatch.setattr(service, "_generate", gen)

    # Exactly the form fields pipeline.trellis2_client._generate_native sends.
    resp = _post(client, png_bytes,
                 seed="7", resolution="1024_cascade", texture_size="2048",
                 ss_steps="9", shape_steps="10", tex_steps="11",
                 face_count_high="100000", face_count_low="5000", label="vase")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert base64.b64decode(payload["glb_high_b64"]) == b"HIGH-GLB"   # verbatim (R6)
    assert base64.b64decode(payload["glb_low_b64"]) == b"LOW-GLB"

    lin = payload["lineage"]
    assert lin["generator"] == "TRELLIS.2-4B"
    assert lin["executor"] == "native-service"
    assert lin["conditioning"] == "single-image"
    assert lin["seed"] == 7
    assert lin["resolution"] == "1024_cascade"
    assert lin["label"] == "vase"
    assert lin["timings"] == TIMINGS
    # Reproducibility: the input crop is hash-recorded with its dimensions.
    assert lin["input"]["sha256"] == hashlib.sha256(png_bytes).hexdigest()
    assert (lin["input"]["width"], lin["input"]["height"]) == (8, 8)
    assert lin["artifact_bytes"] == {"glb_high": 8, "glb_low": 7}

    # Params landed on the generator parsed + typed; image is matted RGBA.
    image, params = gen.call_args.args
    assert image.mode == "RGBA"
    assert params["texture_size"] == 2048
    assert (params["ss_steps"], params["shape_steps"], params["tex_steps"]) == (9, 10, 11)
    assert (params["face_count_high"], params["face_count_low"]) == (100_000, 5_000)


def test_generate_no_params_uses_client_defaults(client, png_bytes, monkeypatch):
    gen = MagicMock(return_value=(b"H", b"L", {}))
    monkeypatch.setattr(service, "_generate", gen)
    assert _post(client, png_bytes).status_code == 200
    params = gen.call_args.args[1]
    assert params["seed"] == 42
    assert params["resolution"] == "1536_cascade"
    assert params["texture_size"] == 4096
    assert (params["face_count_high"], params["face_count_low"]) == (500_000, 20_000)
    assert params["label"] == "object"


# ---------------------------------------------------------------------------
# Bad input -> 400
# ---------------------------------------------------------------------------

def test_generate_missing_image_400(client):
    resp = client.post("/generate", data={"seed": "7"},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "image" in resp.get_json()["error"]


def test_generate_undecodable_image_400(client):
    resp = client.post("/generate",
                       data={"image": (io.BytesIO(b"not-a-png"), "crop.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "decode" in resp.get_json()["error"]


def test_generate_empty_image_400(client):
    resp = client.post("/generate",
                       data={"image": (io.BytesIO(b""), "crop.png")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Bad params default/clamp (never 400 — the crop still generates)
# ---------------------------------------------------------------------------

def test_generate_bad_params_default_and_clamp(client, png_bytes, monkeypatch):
    gen = MagicMock(return_value=(b"H", b"L", {}))
    monkeypatch.setattr(service, "_generate", gen)
    resp = _post(client, png_bytes,
                 seed="banana",              # unparseable -> default
                 texture_size="999999",      # out of range -> clamped
                 ss_steps="0",               # below floor -> clamped
                 resolution="4096_mega",     # unknown -> default
                 face_count_high="1000",
                 face_count_low="999999999", # -> clamped, then <= high
                 label="weird label!!")      # sanitized
    assert resp.status_code == 200
    params = gen.call_args.args[1]
    assert params["seed"] == 42
    assert params["texture_size"] == 8192
    assert params["ss_steps"] == 1
    assert params["resolution"] == "1536_cascade"
    assert params["face_count_high"] == 1000
    assert params["face_count_low"] == 1000     # low can never exceed high
    assert params["label"] == "weird_label__"


# ---------------------------------------------------------------------------
# Error taxonomy: 503 env-not-ready vs 500 generation failure
# ---------------------------------------------------------------------------

def test_generate_env_not_ready_503(client, png_bytes, monkeypatch):
    monkeypatch.setattr(service, "_generate", MagicMock(
        side_effect=service.EnvNotReadyError("trellis2 package not importable")))
    resp = _post(client, png_bytes)
    assert resp.status_code == 503
    assert "trellis2" in resp.get_json()["error"]


def test_generate_lazy_import_error_maps_to_503(client, png_bytes, monkeypatch):
    # CUDA extensions import lazily inside pipeline.run — env, not a bug.
    monkeypatch.setattr(service, "_generate", MagicMock(
        side_effect=ImportError("No module named 'nvdiffrast'")))
    resp = _post(client, png_bytes)
    assert resp.status_code == 503
    assert "nvdiffrast" in resp.get_json()["error"]


def test_generate_failure_500(client, png_bytes, monkeypatch):
    monkeypatch.setattr(service, "_generate", MagicMock(
        side_effect=RuntimeError("CUDA out of memory")))
    resp = _post(client, png_bytes)
    assert resp.status_code == 500
    assert "CUDA out of memory" in resp.get_json()["error"]


def test_load_pipeline_absent_package_raises_env_not_ready(monkeypatch):
    monkeypatch.setattr(service, "_pipeline", None)
    # Deterministic even on a machine that HAS trellis2 installed.
    monkeypatch.setitem(sys.modules, "trellis2", None)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", None)
    with pytest.raises(service.EnvNotReadyError, match="native TRELLIS.2 env"):
        service._load_pipeline()


# ---------------------------------------------------------------------------
# Every error path is JSON (the client always parses the body)
# ---------------------------------------------------------------------------

def test_unknown_route_is_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_wrong_method_is_json_405(client):
    resp = client.get("/generate")
    assert resp.status_code == 405
    assert "error" in resp.get_json()
