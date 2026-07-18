# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pixal3D client — SINGLE-image object generation (PRD v4 R8 gated eval).

Pixal3D (TencentARC, May 2026, SIGGRAPH 2026, **MIT** — ADR-025 amendment
2026-07-09 corrected the original "non-MIT" claim) is pixel-aligned generation
on the TRELLIS.2 backbone: it back-projects pixel features from the
conditioning image into 3D for near-reconstruction input fidelity. It is the
likely TRELLIS.2 successor and is evaluated head-to-head against TRELLIS.2 on
the R9 crop set (see ``research/2026-07-pixal3d-eval.md``) before any
adoption decision.

STATUS — HONEST SCAFFOLD, NOT A LIVE-VERIFIED GENERATOR
    Pixal3D weights are NOT staged in ``data/comfyui/models/`` and no service
    or node pack for it exists in this environment yet. This client is
    validated against the *mirrored contract* (``Trellis2Client``'s public
    surface + the PRD v4 R5 native-service HTTP contract), so it drops into
    the ``stages._generate_object_from_crop`` generator chain unchanged the
    day an executor exists. Every point where the real Pixal3D API could not
    be confirmed is marked ``VERIFY-ON-ENV-BUILD``.

Two executors, selected by config (mirroring ``trellis2_client``):

* ``native_url`` set — a thin HTTP service wrapping the native Pixal3D
  pipeline (a future ``scripts/pixal3d_native_service.py``), speaking the
  SAME contract as ``scripts/trellis2_native_service.py``: multipart
  ``image`` + form params → ``{glb_high_b64, glb_low_b64?, lineage}``.
  Preferred (PRD v4 R5 discipline: native pipelines for the 3D half).
  VERIFY-ON-ENV-BUILD: the underlying Pixal3D python API (assumed
  ``pipeline.run(image)``-shaped since it sits on the TRELLIS.2 backbone).
* ``native_url`` empty — a ComfyUI workflow executor. As of 2026-07-10 there
  is NO runtime-verified Pixal3D node pack; this path fails fast with a
  clear error unless a workflow JSON (``pixal3d_single_image_pbr.json``)
  has been authored and verified at env build. VERIFY-ON-ENV-BUILD: node
  pack existence, node class names, and node ids.

The returned GLB bytes are the artifact: callers persist them verbatim
(hash-recorded) and must not re-export through trimesh (PRD v4 R6).

Usage::

    from pipeline.pixal3d_client import Pixal3DClient
    client = Pixal3DClient(native_url="http://gaussian-toolkit:8403")
    result = client.reconstruct_from_image("object_crops/0001_vase.png")
    Path("vase.glb").write_bytes(result.glb_data)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import trimesh

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / "workflows"
# VERIFY-ON-ENV-BUILD: this workflow does not exist yet. It must be authored
# against a runtime-verified Pixal3D node pack (none confirmed as of
# 2026-07-10) before the ComfyUI executor can run.
PIXAL3D_SI_WORKFLOW = WORKFLOW_DIR / "pixal3d_single_image_pbr.json"

_CROP_PLACEHOLDER = "CROP_IMAGE_PLACEHOLDER"

# Provenance pins (docs/audit/object-pipeline-audit-2026-07-09.md §B.1).
PIXAL3D_HF_REPO = "TencentARC/Pixal3D"
PIXAL3D_PAPER = "arXiv:2605.10922"
PIXAL3D_LICENCE = "MIT"  # ADR-025 amendment 2026-07-09 (GitHub LICENSE + HF card)
PIXAL3D_BACKBONE = "TRELLIS.2 (pixel-aligned conditioning)"


@dataclass
class Pixal3DResult:
    """Result of a Pixal3D single-image object generation.

    Field-for-field compatible with ``trellis2_client.Trellis2Result`` so the
    two generators are interchangeable in ``stages._generate_object_from_crop``
    and ``eval/objects/run_eval.py``. ``glb_data`` is the generator's PBR GLB
    byte-for-byte; ``mesh`` is a trimesh view for stats/placement only —
    never re-export it (PRD v4 R6).
    """
    mesh: Optional[trimesh.Trimesh] = None
    glb_data: Optional[bytes] = None
    glb_low_data: Optional[bytes] = None   # decimated game-res pair (native service)
    backend: str = "pixal3d-single-image"
    duration_seconds: float = 0.0
    prompt_id: str = ""
    output_paths: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def glb_sha256(self) -> str:
        return hashlib.sha256(self.glb_data).hexdigest() if self.glb_data else ""

    @property
    def vertex_count(self) -> int:
        return 0 if self.mesh is None else len(self.mesh.vertices)

    @property
    def face_count(self) -> int:
        return 0 if self.mesh is None else len(self.mesh.faces)

    @property
    def has_texture(self) -> bool:
        return (
            self.mesh is not None
            and getattr(self.mesh.visual, "kind", None) in ("texture", "vertex")
        )


class Pixal3DClient:
    """Client for Pixal3D single-image object generation (contract scaffold).

    Mirrors ``Trellis2Client``'s public surface exactly:
    ``reconstruct_from_image(image_path, seed, label, provenance)`` returning
    a result with ``glb_data`` / ``mesh`` / ``face_count`` / ``lineage`` /
    ``error`` / ``glb_sha256``.

    Parameters
    ----------
    comfyui_url : str
        ComfyUI URL for the (currently unavailable) node-pack executor.
    native_url : str
        Native-pipeline service URL (future ``scripts/pixal3d_native_service.py``,
        same HTTP contract as the TRELLIS.2 native service). Empty selects the
        ComfyUI workflow — which fails fast until a verified workflow exists.
    timeout : int
        Maximum seconds per generation.
    resolution : str
        Structure resolution ladder. VERIFY-ON-ENV-BUILD: assumed to share
        the TRELLIS.2 backbone ladder (``512`` … ``1536_cascade``).
    texture_size : int
        PBR texture resolution (default 4096).
    seed : int
        Generation seed (re-rolls are the first escalation rung, PRD v4 R7).
    ss_steps / shape_steps / tex_steps : int
        Sampling steps for the sparse-structure / shape / texture stages
        (TRELLIS.2-backbone naming; VERIFY-ON-ENV-BUILD for Pixal3D's actual
        knob names — the native service maps them).
    face_count_high / face_count_low : int
        Remesh targets for the high-poly artifact and (native service only)
        the decimated low-poly pair.
    """

    def __init__(
        self,
        comfyui_url: str = "http://vitrine-comfyui:8188",
        native_url: str = "",
        timeout: int = 1800,
        poll_interval: float = 2.0,
        resolution: str = "1536_cascade",
        texture_size: int = 4096,
        seed: int = 42,
        ss_steps: int = 12,
        shape_steps: int = 12,
        tex_steps: int = 12,
        face_count_high: int = 500_000,
        face_count_low: int = 20_000,
        workflow_path: str | Path | None = None,
    ):
        self.comfyui_url = comfyui_url.rstrip("/")
        self.native_url = native_url.rstrip("/") if native_url else ""
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.resolution = resolution
        self.texture_size = texture_size
        self.seed = seed
        self.ss_steps = ss_steps
        self.shape_steps = shape_steps
        self.tex_steps = tex_steps
        self.face_count_high = face_count_high
        self.face_count_low = face_count_low
        self.workflow_path = Path(workflow_path) if workflow_path else PIXAL3D_SI_WORKFLOW
        self.session = requests.Session()

    @classmethod
    def from_config(cls, cfg: Any) -> "Pixal3DClient":
        """Construct from a Pixal3DConfig, reading every field defensively."""
        return cls(
            comfyui_url=getattr(cfg, "comfyui_url", "http://vitrine-comfyui:8188"),
            native_url=getattr(cfg, "native_url", ""),
            timeout=getattr(cfg, "timeout", 1800),
            resolution=getattr(cfg, "resolution", "1536_cascade"),
            texture_size=getattr(cfg, "texture_size", 4096),
            seed=getattr(cfg, "seed", 42),
            ss_steps=getattr(cfg, "ss_steps", 12),
            shape_steps=getattr(cfg, "shape_steps", 12),
            tex_steps=getattr(cfg, "tex_steps", 12),
            face_count_high=getattr(cfg, "face_count_high", 500_000),
            face_count_low=getattr(cfg, "face_count_low", 20_000),
            workflow_path=getattr(cfg, "workflow_path", None) or None,
        )

    # ------------------------------------------------------------------
    # Shared HTTP plumbing (mirrors Trellis2Client; ComfyUI native API is
    # executor-generic and stable — these calls are NOT Pixal3D-specific)
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        try:
            if self.native_url:
                r = self.session.get(f"{self.native_url}/health", timeout=10)
            else:
                r = self.session.get(f"{self.comfyui_url}/system_stats", timeout=10)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _upload_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            r = self.session.post(
                f"{self.comfyui_url}/upload/image", files=files, timeout=30,
            )
        r.raise_for_status()
        return r.json().get("name", image_path.name)

    def _submit_prompt(self, prompt: dict) -> str:
        r = self.session.post(
            f"{self.comfyui_url}/prompt", json={"prompt": prompt}, timeout=30,
        )
        data = r.json()
        if data.get("error") or data.get("node_errors"):
            node_errors = data.get("node_errors", {})
            details = "; ".join(
                f"node {nid}: {e.get('errors', e)}" for nid, e in node_errors.items()
            ) if node_errors else str(data.get("error"))
            raise RuntimeError(f"ComfyUI validation error: {details}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return prompt_id

    def _poll_completion(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                r = self.session.get(
                    f"{self.comfyui_url}/history/{prompt_id}", timeout=60,
                )
                hist = r.json()
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                logger.debug("History poll transient error: %s", e)
                continue
            if prompt_id not in hist:
                if time.monotonic() - last_log > 30:
                    logger.info("Waiting for Pixal3D prompt %s...", prompt_id[:8])
                    last_log = time.monotonic()
                continue
            entry = hist[prompt_id]
            status = entry.get("status", {}).get("status_str", "unknown")
            if status == "success":
                return entry
            if status == "error":
                messages = entry.get("status", {}).get("messages", [])
                raise RuntimeError(f"Pixal3D execution error: {messages}")
            if time.monotonic() - last_log > 30:
                logger.info("Pixal3D %s status: %s", prompt_id[:8], status)
                last_log = time.monotonic()
        raise TimeoutError(f"Pixal3D prompt {prompt_id} timed out after {self.timeout}s")

    def _free_vram(self) -> None:
        """POST /free to unload models + free VRAM (serial lifecycle, ADR-013).
        Best-effort, never raises. No-op for the native service."""
        if self.native_url:
            return
        try:
            self.session.post(
                f"{self.comfyui_url}/free",
                json={"unload_models": True, "free_memory": True}, timeout=30,
            )
            logger.info("freed ComfyUI VRAM after Pixal3D generation")
        except requests.RequestException as e:  # noqa: BLE001
            logger.warning("free_vram failed: %s", e)

    def _download_file(self, filename: str, subfolder: str = "") -> bytes:
        for file_type in ("output", "temp"):
            r = self.session.get(
                f"{self.comfyui_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": file_type},
                timeout=120,
            )
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
        raise FileNotFoundError(f"Cannot download {subfolder}/{filename} from ComfyUI")

    def _extract_glb_refs(self, history: dict) -> list[tuple[str, str]]:
        """Scan ComfyUI history outputs for downloadable GLB references.

        Robust to the exact ui key (``3d``, ``gltf``, ``result``, ``meshes``,
        ``text``): returns (filename, subfolder) pairs for anything ending in
        .glb — same tolerant scan as the TRELLIS.2 client, since export nodes
        frequently terminate in a Preview3D whose ui output carries the file.
        """
        refs: list[tuple[str, str]] = []
        outputs = history.get("outputs", {})

        def add(fname: str, sub: str = "") -> None:
            if fname and fname.lower().endswith(".glb"):
                refs.append((Path(fname).name, sub or str(Path(fname).parent) if "/" in fname else sub))

        for _node_id, node_output in outputs.items():
            for _key, items in node_output.items():
                if not isinstance(items, list):
                    if isinstance(items, str):
                        add(items)
                    continue
                for item in items:
                    if isinstance(item, str):
                        add(item)
                    elif isinstance(item, dict):
                        fn = item.get("filename") or item.get("model_file") or ""
                        add(fn, item.get("subfolder", ""))
        return refs

    def _load_glb(self, data: bytes) -> Optional[trimesh.Trimesh]:
        """Trimesh view of a GLB for stats/placement — NOT for re-export."""
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            scene = trimesh.load(tmp.name, file_type="glb", force="scene")
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        if isinstance(scene, trimesh.Trimesh):
            return scene
        return None

    # ------------------------------------------------------------------
    # Workflow construction (ComfyUI executor — SCAFFOLD)
    # ------------------------------------------------------------------

    def _build_prompt(self, uploaded_name: str, seed: int, label: str) -> dict:
        """Load + parameterize the Pixal3D workflow.

        VERIFY-ON-ENV-BUILD: no runtime-verified Pixal3D node pack exists as
        of 2026-07-10, so node ids/class names cannot be pinned the way
        ``trellis2_client._build_prompt`` pins them. Parameter injection is
        therefore *input-name driven* (any node input literally named
        ``seed`` / ``texture_size`` / ``target_face_count`` /
        ``filename_prefix`` is set) — re-pin to explicit node ids once the
        workflow JSON is authored against verified nodes.
        """
        with open(self.workflow_path) as f:
            prompt = json.load(f)
        prompt = {k: v for k, v in prompt.items() if not k.startswith("_")}

        for node in prompt.values():
            ins = node.get("inputs", {})
            if ins.get("image") == _CROP_PLACEHOLDER:
                ins["image"] = uploaded_name
            if "seed" in ins:
                ins["seed"] = seed
            if "resolution" in ins:
                ins["resolution"] = self.resolution
            if "texture_size" in ins:
                ins["texture_size"] = self.texture_size
            if "target_face_count" in ins:
                ins["target_face_count"] = self.face_count_high
            if "filename_prefix" in ins:
                ins["filename_prefix"] = f"vitrine_object_{label}"
        return prompt

    # ------------------------------------------------------------------
    # Executors
    # ------------------------------------------------------------------

    def _generate_native(self, image_path: Path, seed: int, label: str) -> Pixal3DResult:
        """POST the crop to the native-pipeline service.

        Contract (mirrors ``scripts/trellis2_native_service.py``, PRD v4 R5):
        multipart ``image`` + form params; JSON response
        ``{glb_high_b64, glb_low_b64?, lineage}``. A future
        ``scripts/pixal3d_native_service.py`` must honour this contract.
        VERIFY-ON-ENV-BUILD: the wrapped Pixal3D python API itself.
        """
        t0 = time.monotonic()
        with open(image_path, "rb") as f:
            r = self.session.post(
                f"{self.native_url}/generate",
                files={"image": (image_path.name, f, "image/png")},
                data={
                    "seed": str(seed),
                    "resolution": self.resolution,
                    "texture_size": str(self.texture_size),
                    "ss_steps": str(self.ss_steps),
                    "shape_steps": str(self.shape_steps),
                    "tex_steps": str(self.tex_steps),
                    "face_count_high": str(self.face_count_high),
                    "face_count_low": str(self.face_count_low),
                    "label": label,
                },
                timeout=self.timeout,
            )
        r.raise_for_status()
        payload = r.json()
        result = Pixal3DResult(
            backend="pixal3d-native-single-image",
            duration_seconds=time.monotonic() - t0,
            lineage=payload.get("lineage", {}),
        )
        if payload.get("glb_high_b64"):
            result.glb_data = base64.b64decode(payload["glb_high_b64"])
            result.mesh = self._load_glb(result.glb_data)
        if payload.get("glb_low_b64"):
            result.glb_low_data = base64.b64decode(payload["glb_low_b64"])
        if result.glb_data is None:
            result.error = payload.get("error", "native service returned no GLB")
        return result

    def _generate_comfyui(self, image_path: Path, seed: int, label: str) -> Pixal3DResult:
        """Run the single-image workflow on ComfyUI (SCAFFOLD executor)."""
        t0 = time.monotonic()
        # Begin from a clean GPU (serial lifecycle, ADR-013).
        self._free_vram()

        uploaded = self._upload_image(image_path)
        prompt = self._build_prompt(uploaded, seed=seed, label=label)
        prompt_id = self._submit_prompt(prompt)
        logger.info("Submitted Pixal3D single-image prompt %s", prompt_id)

        history = self._poll_completion(prompt_id)
        elapsed = time.monotonic() - t0
        logger.info("Pixal3D completed in %.1fs", elapsed)

        result = Pixal3DResult(
            backend="pixal3d-comfyui-single-image",
            duration_seconds=elapsed,
            prompt_id=prompt_id,
        )
        for fname, sub in self._extract_glb_refs(history):
            try:
                data = self._download_file(fname, sub)
                result.glb_data = data
                result.output_paths[f"{sub}/{fname}" if sub else fname] = fname
                result.mesh = self._load_glb(data)
                if result.mesh is not None:
                    logger.info("Loaded Pixal3D object: %d verts, %d faces",
                                result.vertex_count, result.face_count)
                    break
            except (FileNotFoundError, requests.RequestException) as e:
                logger.warning("Could not download %s/%s: %s", sub, fname, e)

        if result.glb_data is None:
            result.error = "No retrievable GLB in Pixal3D outputs"
            logger.warning("Pixal3D: %s (history outputs scanned)", result.error)

        # Serial lifecycle: free the models so the next object has full VRAM.
        self._free_vram()
        return result

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def reconstruct_from_image(
        self,
        image_path: str | Path,
        seed: int | None = None,
        label: str = "object",
        provenance: dict | None = None,
    ) -> Pixal3DResult:
        """Generate a PBR-textured object GLB from ONE matted crop.

        Drop-in for the ADR-025 generator chain: same signature and result
        shape as ``Trellis2Client.reconstruct_from_image``. ``provenance``
        (the object_crops manifest entry) is folded into the result lineage.
        Backsides are model-completed (``surface: inferred``); Pixal3D's
        pixel-aligned conditioning is expected to improve *front* fidelity
        specifically — the head-to-head eval measures exactly that.

        Raises loudly (never fakes a result) when no executor is available:
        the generator chain in ``stages._generate_object_from_crop`` catches
        and falls through to TRELLIS.2 / Hunyuan3D.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Crop not found: {image_path}")
        if not self.native_url and not self.workflow_path.exists():
            # Fail fast BEFORE any HTTP: there is nothing to execute against.
            raise RuntimeError(
                "Pixal3D has no available executor: native_url is unset and "
                f"no ComfyUI workflow exists at {self.workflow_path} "
                "(no runtime-verified Pixal3D node pack as of 2026-07-10 — "
                "VERIFY-ON-ENV-BUILD; stand up scripts/pixal3d_native_service.py "
                "or author + verify pixal3d_single_image_pbr.json)")
        seed = self.seed if seed is None else seed
        safe = "".join(c if c.isalnum() else "_" for c in label)[:40] or "object"

        logger.info("Pixal3D single-image generation: %s (%s/%d, seed=%d)",
                    image_path.name, self.resolution, self.texture_size, seed)

        if self.native_url:
            result = self._generate_native(image_path, seed, safe)
        else:
            result = self._generate_comfyui(image_path, seed, safe)

        result.lineage = {
            "conditioning": "single-image",
            "crop": str(image_path),
            "generator": "Pixal3D",
            "generator_backbone": PIXAL3D_BACKBONE,
            "generator_repo": PIXAL3D_HF_REPO,
            "generator_paper": PIXAL3D_PAPER,
            "licence": PIXAL3D_LICENCE,
            "executor": result.backend,
            "resolution": self.resolution,
            "seed": seed,
            "surface": "observed-front/inferred-back",
            **({"source": provenance} if provenance else {}),
            **result.lineage,
        }
        return result
