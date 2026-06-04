# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SOTA model/tooling registry + environment "idiot-check" (config system core).

This is the single source of truth for which SOTA model/tool each pipeline
element uses, web-verified 2026-06-04 for 2x RTX 6000 Ada (48 GB, sm_89).
``check_environment()`` validates the live host against the registry so a run
can fail fast on a misconfiguration instead of silently degrading mid-pipeline:

  * are the chosen checkpoints actually present at a staged path?
  * does the model fit our GPU VRAM (serial lifecycle)?
  * is the ComfyUI node / tool pinned to a commit?
  * licence posture (research vs commercial) — warn or fail accordingly?
  * known wiring caveats (CoMe CLI inferred, Hunyuan untextured, splat_ready
    dead reference, native USD not called, ...).

Posture default is RESEARCH/non-commercial (user decision 2026-06-04): the best
model wins regardless of licence and non-commercial licences are WARN, not FAIL.
Set ``commercial_use=True`` to make any non-commercial model a hard FAIL.

CLI:  python -m pipeline.sota_registry check [--commercial] [--json]
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
#  Licences
# ---------------------------------------------------------------------------

class Licence(str, Enum):
    MIT = "MIT"
    APACHE2 = "Apache-2.0"
    TENCENT_COMMUNITY = "Tencent-Community"   # commercial use permitted
    NONCOMMERCIAL = "Non-Commercial"
    NC_ND = "CC-BY-NC-ND-4.0"                 # non-commercial AND no-derivatives
    NATIVE = "native"                         # part of LichtFeld / no extra weight
    NONE = "no-licence-file"


_COMMERCIAL_OK = {Licence.MIT, Licence.APACHE2, Licence.TENCENT_COMMUNITY, Licence.NATIVE}


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
#  Where staged weights live (host + container). Override with SOTA_MODEL_ROOTS
#  (os.pathsep-separated) to point at the real model trees.
# ---------------------------------------------------------------------------

def staged_roots() -> list[Path]:
    env = os.environ.get("SOTA_MODEL_ROOTS")
    if env:
        roots = [Path(p) for p in env.split(os.pathsep) if p]
    else:
        home = Path.home()
        roots = [
            home / "comfyui-models-staging",
            home / "comfyui-api-data" / "ComfyUI" / "models",
            Path("/models-staging"),
            Path("/opt/models"),
            Path("/opt/hf-cache"),
            home / ".cache" / "huggingface",
        ]
    return [r for r in roots if r.exists()]


# ---------------------------------------------------------------------------
#  Model / tool spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """One concrete SOTA choice for a pipeline element."""
    element: str
    name: str
    version: str                              # tag / commit / release date
    licence: Licence = Licence.NONE
    vram_gb: float = 0.0                       # peak single-model VRAM
    serving: str = ""                          # comfyui | llama.cpp | sidecar | native-mcp | plugin
    repo: str = ""
    node_repo: str = ""
    node_commit: str = ""                      # "" => UNPINNED (warn — ADR-012 T6)
    checkpoints: list[str] = field(default_factory=list)  # filenames expected on disk
    requires_staged: bool = True               # False for sidecar-trained (CoMe/MILo) or native
    caveats: list[str] = field(default_factory=list)

    @property
    def commercial_ok(self) -> bool:
        return self.licence in _COMMERCIAL_OK


@dataclass
class Element:
    key: str
    title: str
    primary: ModelSpec
    fallbacks: list[ModelSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  THE REGISTRY  (web-verified 2026-06-04 — see memory/project-sota-scorecard)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Element] = {
    "inpaint": Element(
        "inpaint", "Generative recovery / inpainting",
        ModelSpec(
            "inpaint", "FLUX.2-dev", "2025-11-25", Licence.NONCOMMERCIAL, 40.0, "comfyui",
            repo="black-forest-labs/FLUX.2-dev",
            checkpoints=["flux2_dev_fp8mixed.safetensors", "flux2-vae.safetensors",
                         "mistral_3_small_flux2_fp8.safetensors"],
            caveats=["no FLUX.2 'Fill' checkpoint — masked recovery via "
                     "InpaintModelConditioning + reference latent",
                     "fp8 diffusion (34GB)+encoder(17GB) exceeds 48GB co-resident → "
                     "needs ComfyUI weight-streaming / serial load",
                     "src/ is still 100% FLUX.1 — needs flux2_inpaint.json + config branch"],
        ),
        [
            ModelSpec("inpaint", "Qwen-Image-Edit-2509", "2025-09", Licence.APACHE2, 40.0,
                      "comfyui", repo="Qwen/Qwen-Image-Edit-2509",
                      checkpoints=["qwen_image_vae.safetensors"],
                      caveats=["commercial-safe alternative; true inpaint DiffSynth ControlNet"]),
            ModelSpec("inpaint", "FLUX.1-Fill-dev", "2024-11", Licence.NONCOMMERCIAL, 23.0,
                      "comfyui", repo="black-forest-labs/FLUX.1-Fill-dev",
                      checkpoints=["flux1-fill-dev.safetensors"],
                      caveats=["current wired default; purpose-built inpaint; lowest VRAM"]),
        ],
    ),
    "vlm": Element(
        "vlm", "Local artifact-analysis VLM",
        ModelSpec(
            "vlm", "gemma-4-26B-A4B-it", "2026-04-02", Licence.APACHE2, 28.0, "llama.cpp",
            repo="ggml-org/gemma-4-26b-a4b-it-GGUF",
            checkpoints=["gemma-4-26B-A4B-it-Q8_0.gguf", "mmproj-gemma-4-26B-A4B-it-bf16.gguf"],
            caveats=["Q5_K_M not published — staged Q8_0 (~28GB) + mmproj-bf16 (fits 48GB serial)",
                     "no VLM code in src/ — build agent-vlm client + RecoveryController"],
        ),
        [ModelSpec("vlm", "Qwen3-VL-30B-A3B-Instruct-FP8", "2025-10-04", Licence.APACHE2, 31.0,
                   "vllm", repo="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", requires_staged=False)],
    ),
    "hull": Element(
        "hull", "Per-object 3D hull (image -> textured mesh)",
        ModelSpec(
            "hull", "TRELLIS.2-4B", "2025-12-16", Licence.MIT, 24.0, "comfyui",
            repo="microsoft/TRELLIS.2-4B",
            node_repo="PozzettiAndrea/ComfyUI-TRELLIS2",
            checkpoints=["TRELLIS.2-4B"],
            caveats=["MIT + PBR textured + sharp topology; single-image input",
                     "NOT staged — pull weights; pin a ComfyUI node commit after smoke test"],
        ),
        [
            ModelSpec("hull", "Hunyuan3D-2.1", "2025-06-13", Licence.TENCENT_COMMUNITY, 29.0,
                      "comfyui", repo="tencent/Hunyuan3D-2.1",
                      node_repo="visualbruno/ComfyUI-Hunyuan3d-2-1",
                      checkpoints=["hunyuan3d-dit-v2-1", "hunyuan3d-paintpbr-v2-1"],
                      caveats=["multiview (2mv) variant matches our orbit renderer",
                               "only 2.0-mv WEIGHTS + 2.1 CODE staged — 2.1 weights not pulled",
                               "current workflow has NO paint node => untextured (must add PBR node)"]),
            ModelSpec("hull", "SAM-3D-Objects", "2025", Licence.NONCOMMERCIAL, 32.0, "comfyui",
                      node_repo="PozzettiAndrea/ComfyUI-SAM3DObjects",
                      node_commit="cc2ba08d7e53f767d7115f5a3a3cb9bb76fc2746",
                      checkpoints=["ss_generator.ckpt"],
                      caveats=["staged but Meta non-commercial; slat_generator.ckpt is CORRUPT — re-pull",
                               "sam3d_client.py is orphaned; fallback_sam3d flag is dead"]),
        ],
    ),
    "gs_mesh": Element(
        "gs_mesh", "Gaussian-splatting -> surface mesh",
        ModelSpec(
            "gs_mesh", "CoMe", "2026-04-22", Licence.NC_ND, 20.0, "sidecar",
            repo="r4dl/CoMe", requires_staged=False,
            caveats=["VERIFIED best indoor (ScanNet++ F1 0.668, ~18min, 3x faster than MILo)",
                     "CC BY-NC-ND: non-commercial AND no-derivatives",
                     "come_extractor.py CLI flags INFERRED/unverified; 'iterations' not plumbed",
                     "needs INSTALL_COME=1 + come sidecar running, else silently -> TSDF",
                     "config-vs-policy mismatch: config default 'come' but _select_mesh_backend "
                     "still treats MILo as default-high-quality"],
        ),
        [
            ModelSpec("gs_mesh", "PGSR", "2024 (TVCG)", Licence.NONE, 16.0, "sidecar",
                      repo="zju3dv/PGSR", requires_staged=False,
                      caveats=["commercial-shippable rival; T&T 0.53 (edges CoMe on outdoor)"]),
            ModelSpec("gs_mesh", "MILo", "2025 (SIGGRAPH Asia)", Licence.NONCOMMERCIAL, 16.0, "sidecar",
                      repo="Anttwo/MILo", requires_staged=False,
                      caveats=["quality fallback; 10x fewer mesh verts (CAD/USD-friendly)"]),
            ModelSpec("gs_mesh", "TSDF", "in-process", Licence.NATIVE, 8.0, "sidecar",
                      requires_staged=False, caveats=["always-available last-resort fallback"]),
        ],
    ),
    "training": Element(
        "training", "3DGS training strategy",
        ModelSpec(
            "training", "ImprovedGS+ (igs+)", "native v0.5.0", Licence.NATIVE, 24.0, "native-mcp",
            requires_staged=False,
            caveats=["native to LichtFeld; -26.8% time / -13.3% Gaussians vs MCMC",
                     "config default is still 'mrnf' — switch to 'igs+'",
                     "PPISP compiled in our core (parameters.hpp:145-155) but DISABLED; "
                     "bilateral-grid / 3DGUT / pose-opt native flags NOT wired"],
        ),
        [ModelSpec("training", "MCMC", "native", Licence.NATIVE, 24.0, "native-mcp",
                   requires_staged=False)],
    ),
    "sfm": Element(
        "sfm", "SfM / feature matching",
        ModelSpec(
            "sfm", "ALIKED+LightGlue (COLMAP 4.1)", "plugin", Licence.NATIVE, 8.0, "plugin",
            node_repo="alexmgee/lichtfeld-360-plugin",
            requires_staged=False,
            caveats=["via LichtFeld plugin (360 or Lichtfeld-COLMAP-Plugin) — pin a commit",
                     "splat_ready is a DEAD reference (~/.lichtfeld/plugins/splat_ready absent) "
                     "-> silently falls to SIFT; install+pin or delete",
                     "config matcher is still SIFT 'exhaustive'"],
        ),
        [
            ModelSpec("sfm", "VGGT (neural SfM)", "2025 (CVPR)", Licence.NONCOMMERCIAL, 35.0,
                      "sidecar", repo="facebookresearch/vggt", requires_staged=False,
                      caveats=["~80 frames/24GB -> ~160/48GB; 4K orbits need chunking "
                               "(FastVGGT / VGGT-X)"]),
            ModelSpec("sfm", "COLMAP SIFT", "4.x", Licence.NATIVE, 4.0, "sidecar",
                      requires_staged=False, caveats=["universal fallback"]),
        ],
    ),
    "usd": Element(
        "usd", "USD scene assembly",
        ModelSpec(
            "usd", "native scene.export_usd", "LichtFeld v0.5.1", Licence.NATIVE, 0.0, "native-mcp",
            requires_staged=False,
            caveats=["WIRED: mcp_client.export_usd() added; assemble_usd tries native first (best-effort)",
                     "PROBE PENDING: does native export carry ADR-011 v2g:* customData? if yes, "
                     "retire scripts/assemble_usd_scene.py; else native=base scene, custom=composition"],
        ),
        [],
    ),
    "engine": Element(
        "engine", "LichtFeld engine",
        ModelSpec(
            "engine", "LichtFeld Studio v0.5.2", "2025-04-21 (latest stable)", Licence.NATIVE,
            0.0, "native", requires_staged=False,
            caveats=["correct pin; v0.5.3/Vulkan unreleased -> ADR-008 deferral valid"],
        ),
        [],
    ),
}


# ---------------------------------------------------------------------------
#  Environment probing
# ---------------------------------------------------------------------------

def find_checkpoint(filename: str, roots: Optional[list[Path]] = None) -> Optional[Path]:
    """Locate a checkpoint file (or a repo/dir of that name) under the staged
    roots. Matches an exact file OR a directory of that name. Returns None if
    not found. Never raises."""
    roots = roots if roots is not None else staged_roots()
    for root in roots:
        direct = root / filename
        if direct.exists():
            return direct
        try:
            for p in root.rglob(filename):
                return p
        except OSError:
            continue
    return None


def gpu_vram_gb() -> list[float]:
    """Per-GPU total VRAM in GB via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        return [round(float(x.strip()) / 1024.0, 1) for x in out.stdout.splitlines() if x.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


# ---------------------------------------------------------------------------
#  The idiot-check
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    element: str
    model: str
    status: Status
    message: str


def _check_spec(spec: ModelSpec, *, is_primary: bool, commercial_use: bool,
                roots: list[Path], max_gpu_gb: float) -> list[Finding]:
    out: list[Finding] = []

    def add(status: Status, msg: str) -> None:
        out.append(Finding(spec.element, spec.name, status, msg))

    # 1. Licence posture
    if not spec.commercial_ok:
        if commercial_use:
            add(Status.FAIL, f"licence {spec.licence.value} forbids commercial use")
        else:
            add(Status.WARN, f"licence {spec.licence.value} (research-only — OK in non-commercial posture)")

    # 2. Weights present
    if spec.requires_staged and spec.checkpoints:
        missing = [c for c in spec.checkpoints if find_checkpoint(c, roots) is None]
        if not missing:
            add(Status.PASS, f"all {len(spec.checkpoints)} checkpoint(s) staged")
        elif is_primary:
            add(Status.FAIL, f"primary missing {len(missing)}/{len(spec.checkpoints)} "
                             f"checkpoint(s): {', '.join(missing)}")
        else:
            add(Status.WARN, f"fallback missing: {', '.join(missing)}")

    # 3. VRAM fit (serial lifecycle: one model resident)
    if spec.vram_gb and max_gpu_gb:
        if spec.vram_gb > max_gpu_gb:
            add(Status.FAIL, f"needs ~{spec.vram_gb:.0f}GB > {max_gpu_gb:.0f}GB GPU")
        elif spec.vram_gb > 0.85 * max_gpu_gb:
            add(Status.WARN, f"needs ~{spec.vram_gb:.0f}GB of {max_gpu_gb:.0f}GB — "
                             f"tight; rely on serial load / offload")

    # 4. Pinning (ADR-012 T6)
    if spec.node_repo and not spec.node_commit:
        add(Status.WARN, f"ComfyUI node {spec.node_repo} is UNPINNED — pin a commit")

    # 5. Caveats are informational warnings
    for c in spec.caveats:
        add(Status.WARN, c)

    return out


def check_environment(*, commercial_use: bool = False,
                      elements: Optional[list[str]] = None) -> dict:
    """Run the full idiot-check. Returns a structured report:
    {posture, gpus, overall, elements: {key: {title, findings:[...]}}}.
    Never raises."""
    roots = staged_roots()
    gpus = gpu_vram_gb()
    max_gpu = max(gpus) if gpus else 0.0

    keys = elements or list(REGISTRY.keys())
    report: dict = {
        "posture": "commercial" if commercial_use else "research/non-commercial",
        "gpus_gb": gpus,
        "staged_roots": [str(r) for r in roots],
        "elements": {},
    }
    worst = Status.PASS
    for key in keys:
        el = REGISTRY.get(key)
        if el is None:
            continue
        findings: list[Finding] = []
        findings += _check_spec(el.primary, is_primary=True, commercial_use=commercial_use,
                                roots=roots, max_gpu_gb=max_gpu)
        for fb in el.fallbacks:
            findings += _check_spec(fb, is_primary=False, commercial_use=commercial_use,
                                    roots=roots, max_gpu_gb=max_gpu)
        for f in findings:
            if f.status == Status.FAIL:
                worst = Status.FAIL
            elif f.status == Status.WARN and worst != Status.FAIL:
                worst = Status.WARN
        report["elements"][key] = {
            "title": el.title,
            "primary": el.primary.name,
            "findings": [{"model": f.model, "status": f.status.value, "message": f.message}
                         for f in findings],
        }
    report["overall"] = worst.value
    return report


def format_report(report: dict) -> str:
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    lines = [
        "SOTA environment idiot-check",
        f"  posture : {report['posture']}",
        f"  GPUs    : {report['gpus_gb'] or 'none detected'} GB",
        f"  overall : {report['overall']}",
        "",
    ]
    for key, el in report["elements"].items():
        lines.append(f"[{key}] {el['title']}  (primary: {el['primary']})")
        for f in el["findings"]:
            lines.append(f"    {icon.get(f['status'], '?')} {f['status']:4} {f['model']}: {f['message']}")
        lines.append("")
    return "\n".join(lines)


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SOTA registry / environment idiot-check")
    ap.add_argument("command", choices=["check", "list"], nargs="?", default="check")
    ap.add_argument("--commercial", action="store_true",
                    help="commercial posture: non-commercial models FAIL instead of WARN")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--element", action="append", help="restrict to element key(s)")
    args = ap.parse_args(argv)

    if args.command == "list":
        for key, el in REGISTRY.items():
            print(f"{key:9} {el.primary.name:28} {el.primary.licence.value:18} "
                  f"~{el.primary.vram_gb:.0f}GB  {el.primary.serving}")
        return 0

    report = check_environment(commercial_use=args.commercial, elements=args.element)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 1 if report["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(_main())
