#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thin HTTP service wrapping the NATIVE TRELLIS.2 pipeline (PRD v4 R5).

ADR-025 D2: the 3D half of object generation moves off ComfyUI onto the
upstream microsoft/TRELLIS.2 pipeline behind this small, stable HTTP
contract. One matted object crop in; a high-poly + decimated low-poly PBR
GLB pair out, plus lineage. The Vitrine client (pipeline.trellis2_client,
``native_url`` config) already speaks this contract — flipping the config
from ComfyUI to this service requires no pipeline change.

Contract
--------
GET  /health   — liveness: is the process up? Always 200 while serving.
    -> 200 {"status": "ok", "pipeline_loaded": bool, "model_path": str, ...}

GET  /ready    — readiness: is the pipeline loaded and able to generate?
    -> 200 {"ready": true, ...}
    -> 503 {"ready": false, "native_package_available": bool, "error": str}

POST /generate   (multipart/form-data)
    image: PNG/JPEG file — RGBA preferred (alpha = object matte)
    seed, resolution, texture_size, ss_steps, shape_steps, tex_steps,
    face_count_high, face_count_low, label: form fields (all optional;
    unparseable values fall back to defaults, out-of-range values clamp)
    -> 200 {"glb_high_b64": str, "glb_low_b64": str, "lineage": {...}}

Error taxonomy (all errors are JSON ``{"error": str}``):
    400  bad input        — missing / empty / undecodable ``image``
    413  oversized upload — > MAX_UPLOAD_MB
    503  env not ready    — native package absent or pipeline failed to load
    500  generation fail  — the loaded pipeline raised mid-generation

Environment
-----------
Runs in its OWN pinned env (ADR-021 vendoring discipline), NOT ComfyUI's:
the upstream TRELLIS.2 checkout + its CUDA extensions (nvdiffrast, CuMesh,
flash-attn, o_voxel). VRAM >= 24 GB for 1024_cascade; 1536_cascade is the
hero setting. Model dir defaults to the staged tree; override with
TRELLIS2_MODEL_PATH. Serve with:

    python3 scripts/trellis2_native_service.py --host 127.0.0.1 --port 8402

The service starts fine WITHOUT the native package (health stays green so
orchestration can distinguish "down" from "not ready"); /generate and
/ready return 503 until the pinned env is stood up.

NOTE (scaffold status): the HTTP surface and client contract are final; the
two functions marked VERIFY-ON-ENV-BUILD wrap the upstream API exactly as
documented (``pipeline.run(image)`` -> ``to_glb`` twice) and must be smoke-
tested against the pinned upstream checkout when the env is stood up
(PRD v4 R5 acceptance; see also the R9 eval harness).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import logging
import os
import threading
import time

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("trellis2-native")

SERVICE_NAME = "trellis2-native"
SERVICE_VERSION = "1.0.0"
CONTRACT = "trellis2-native/v1"
MAX_UPLOAD_MB = 64

MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "microsoft/TRELLIS.2-4B")

# LoadTrellis2Models-compatible resolutions (runtime-verified node pack set).
ALLOWED_RESOLUTIONS = ("512", "1024", "1024_cascade", "1536_cascade")
DEFAULT_RESOLUTION = "1536_cascade"

# name -> (default, lower bound, upper bound); unparseable -> default,
# out-of-range -> clamped. Defaults mirror pipeline.trellis2_client.
INT_FIELDS = {
    "seed": (42, 0, 2**31 - 1),
    "texture_size": (4096, 256, 8192),
    "ss_steps": (12, 1, 100),
    "shape_steps": (12, 1, 100),
    "tex_steps": (12, 1, 100),
    "face_count_high": (500_000, 1_000, 4_000_000),
    "face_count_low": (20_000, 100, 4_000_000),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_pipeline = None            # loaded lazily on first /generate (or --preload)
_pipeline_lock = threading.Lock()


class EnvNotReadyError(RuntimeError):
    """The native TRELLIS.2 environment is absent or failed to load (-> 503)."""


def _native_available() -> bool:
    """Cheap importability probe — never actually imports the package."""
    if _pipeline is not None:
        return True
    try:
        return importlib.util.find_spec("trellis2") is not None
    except (ImportError, ValueError):  # broken parent package on sys.path
        return False


def _load_pipeline():
    """Load the upstream TRELLIS.2 image-to-3D pipeline once (thread-safe).

    Raises ``EnvNotReadyError`` for anything that goes wrong at load time —
    an absent package and a failed weight load are both deployment problems,
    not request problems, and map to HTTP 503.

    VERIFY-ON-ENV-BUILD: import path + class name per the pinned upstream
    checkout (microsoft/TRELLIS.2). The documented API is
    ``Trellis2ImageTo3DPipeline.from_pretrained(...)`` with ``.run(image)``.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        t0 = time.monotonic()
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline  # upstream repo
        except ImportError as exc:
            raise EnvNotReadyError(
                "native TRELLIS.2 env not available in this interpreter: "
                f"{exc}. Stand up the pinned env per PRD v4 R5 / ADR-021, or "
                "leave trellis2.native_url empty to use the ComfyUI executor."
            ) from exc
        try:
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_PATH)
            pipe.cuda()
        except Exception as exc:  # noqa: BLE001 — weights/CUDA failures are env problems
            raise EnvNotReadyError(
                f"TRELLIS.2 pipeline failed to load from {MODEL_PATH}: {exc}"
            ) from exc
        _pipeline = pipe
        logger.info("TRELLIS.2 pipeline loaded from %s in %.1fs",
                    MODEL_PATH, time.monotonic() - t0)
    return _pipeline


def _generate(image, params: dict):
    """Run single-image generation + the high/low to_glb pair.

    VERIFY-ON-ENV-BUILD: ``run()`` kwargs and the o-voxel ``to_glb``
    signature (decimation target, texture_size, remesh/UV options) per the
    pinned checkout — including where ``resolution`` lands (model-load-time
    in the ComfyUI port) and the texture-stage steps kwarg for
    ``tex_steps``. Returns (glb_high_bytes, glb_low_bytes, timings).
    """
    pipeline = _load_pipeline()
    t0 = time.monotonic()
    outputs = pipeline.run(
        image,
        seed=params["seed"],
        sparse_structure_sampler_params={"steps": params["ss_steps"]},
        slat_sampler_params={"steps": params["shape_steps"]},
    )
    t_run = time.monotonic() - t0

    # One supported call per artifact: high-poly hero + decimated game-res
    # low-poly, both with the PBR bake (ADR-025 D2 — never re-export).
    def _to_glb(face_count: int) -> bytes:
        glb = outputs.to_glb(
            decimation_target=face_count,
            texture_size=params["texture_size"],
        )
        buf = io.BytesIO()
        glb.export(buf, file_type="glb")
        return buf.getvalue()

    t1 = time.monotonic()
    glb_high = _to_glb(params["face_count_high"])
    glb_low = _to_glb(params["face_count_low"])
    t_glb = time.monotonic() - t1
    return glb_high, glb_low, {"run_s": round(t_run, 1), "to_glb_s": round(t_glb, 1)}


def _parse_params(form) -> dict:
    """Parse + validate form fields: default on garbage, clamp on range.

    Bad parameters never 400 — a crop with a typo'd seed should still
    generate (the client sends well-formed fields anyway; this guards
    hand-rolled curl calls).
    """
    params: dict = {}
    for name, (default, lo, hi) in INT_FIELDS.items():
        raw = form.get(name)
        if raw is None or str(raw).strip() == "":
            params[name] = default
            continue
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            logger.warning("unparseable %s=%r — defaulting to %d", name, raw, default)
            params[name] = default
            continue
        clamped = max(lo, min(hi, value))
        if clamped != value:
            logger.warning("%s=%d out of range [%d, %d] — clamped to %d",
                           name, value, lo, hi, clamped)
        params[name] = clamped

    # The low-poly pair can never exceed the high-poly hero.
    if params["face_count_low"] > params["face_count_high"]:
        logger.warning("face_count_low %d > face_count_high %d — clamping",
                       params["face_count_low"], params["face_count_high"])
        params["face_count_low"] = params["face_count_high"]

    resolution = form.get("resolution", DEFAULT_RESOLUTION)
    if resolution not in ALLOWED_RESOLUTIONS:
        logger.warning("unknown resolution %r — defaulting to %s",
                       resolution, DEFAULT_RESOLUTION)
        resolution = DEFAULT_RESOLUTION
    params["resolution"] = resolution

    label = str(form.get("label", "object"))
    params["label"] = "".join(c if c.isalnum() else "_" for c in label)[:40] or "object"
    return params


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness only: the process is up. Never 503s — use /ready for that."""
    return jsonify({
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "contract": CONTRACT,
        "pipeline_loaded": _pipeline is not None,
        "model_path": MODEL_PATH,
        "device": str(getattr(_pipeline, "device", "cuda")) if _pipeline is not None else None,
    })


@app.get("/ready")
def ready():
    """Readiness: 200 only when the pipeline is loaded and can generate."""
    if _pipeline is not None:
        return jsonify({"ready": True, "model_path": MODEL_PATH})
    available = _native_available()
    return jsonify({
        "ready": False,
        "native_package_available": available,
        "error": (
            "pipeline not loaded yet (first /generate or --preload loads it)"
            if available else
            "native 'trellis2' package not importable in this interpreter — "
            "stand up the pinned env per PRD v4 R5 / ADR-021"
        ),
    }), 503


@app.post("/generate")
def generate():
    # -- input validation (400) --------------------------------------------
    if "image" not in request.files:
        return jsonify({"error": "multipart field 'image' is required"}), 400
    upload = request.files["image"]
    data = upload.read()
    if not data:
        return jsonify({"error": "empty 'image' upload"}), 400
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:  # noqa: BLE001 — PIL raises a zoo of decode errors
        return jsonify({"error": f"could not decode image {upload.filename!r}: {exc}"}), 400

    params = _parse_params(request.form)
    logger.info("generate '%s': %dx%d seed=%d res=%s tex=%d faces=%d/%d",
                params["label"], image.width, image.height, params["seed"],
                params["resolution"], params["texture_size"],
                params["face_count_high"], params["face_count_low"])

    # -- generation (503 env / 500 failure) --------------------------------
    t0 = time.monotonic()
    try:
        glb_high, glb_low, timings = _generate(image, params)
    except EnvNotReadyError as exc:
        return jsonify({"error": str(exc)}), 503
    except ImportError as exc:
        # Lazy CUDA-extension imports inside the pipeline (nvdiffrast, CuMesh,
        # o_voxel) surface here — still an environment problem, not a bug.
        return jsonify({"error": (
            f"native TRELLIS.2 env incomplete: {exc}. Stand up the pinned "
            "env per PRD v4 R5 / ADR-021."
        )}), 503
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the service
        logger.exception("generation failed for '%s'", params["label"])
        return jsonify({"error": f"generation failed: {exc}"}), 500
    total_s = round(time.monotonic() - t0, 1)
    logger.info("generate '%s' done in %.1fs (high=%dB low=%dB)",
                params["label"], total_s, len(glb_high), len(glb_low))

    # -- response ------------------------------------------------------------
    # Lineage is complete on its own: the client folds it into the asset
    # record, so everything needed to reproduce this artifact rides along.
    return jsonify({
        "glb_high_b64": base64.b64encode(glb_high).decode("ascii"),
        "glb_low_b64": base64.b64encode(glb_low).decode("ascii"),
        "lineage": {
            "generator": "TRELLIS.2-4B",
            "executor": "native-service",
            "service": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "contract": CONTRACT,
            "model_path": MODEL_PATH,
            "conditioning": "single-image",
            **params,
            "input": {
                "filename": upload.filename or "",
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": image.width,
                "height": image.height,
            },
            "artifact_bytes": {"glb_high": len(glb_high), "glb_low": len(glb_low)},
            "timings": timings,
            "service_total_s": total_s,
        },
    })


# JSON on every error path — the client always parses the body (ADR-025 D2).

@app.errorhandler(404)
def _not_found(_e):
    return jsonify({"error": "not found — endpoints: /health /ready /generate"}), 404


@app.errorhandler(405)
def _method_not_allowed(_e):
    return jsonify({"error": "method not allowed"}), 405


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({"error": f"upload too large (max {MAX_UPLOAD_MB} MB)"}), 413


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thin HTTP service wrapping the native TRELLIS.2 pipeline (PRD v4 R5)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (loopback by default, ADR-022)")
    parser.add_argument("--port", type=int, default=8402)
    parser.add_argument("--preload", action="store_true",
                        help="Load the pipeline at startup instead of first request")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # Inside a container 0.0.0.0 is legitimate (the compose publish is the
        # boundary, pinned to host loopback) — but flag it either way.
        logger.warning("binding non-loopback %s — ensure the host publish is "
                       "loopback-pinned (ADR-022)", args.host)

    if not _native_available():
        logger.warning("native 'trellis2' package NOT importable — serving "
                       "anyway; /generate and /ready return 503 until the "
                       "pinned env is stood up (PRD v4 R5 / ADR-021)")

    if args.preload:
        try:
            _load_pipeline()
        except EnvNotReadyError as exc:
            logger.error("--preload failed: %s", exc)
            raise SystemExit(1) from exc

    # threaded=False: one generation at a time — serial GPU lifecycle (ADR-013).
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
