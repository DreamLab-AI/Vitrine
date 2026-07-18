# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pipeline configuration with typed defaults and JSON persistence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_SECRET_KEY_RE = re.compile(r"(token|secret|password|api[_-]?key|apikey)", re.IGNORECASE)


def _redact_secrets(value: Any) -> Any:
    """Recursively blank secret-named string fields (e.g. ``hf_token``) so they
    are never written into a persisted config snapshot (SEC-01/SEC-03 — these
    snapshots are uploaded to Drive alongside outputs). Redacted to "" (their
    default) so the file round-trips and runtime re-resolves from env/secrets.
    """
    if isinstance(value, dict):
        return {
            k: ("" if (_SECRET_KEY_RE.search(k) and isinstance(v, str) and v) else _redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


@dataclass
class EndpointsConfig:
    """Service endpoints on the ``v2g-net`` Docker bridge (ADR-013 / D-013.3).

    DNS-name defaults replace the historic hardcoded ``localhost:port`` IPs
    so the orchestrator is host-portable. The manifest ``[pipeline]`` block may
    override the comfyui_* / agent_llm_url keys for the legacy single-host case.
    """
    comfyui_url: str = "http://comfyui:8188"          # FLUX.2 + Hunyuan3D prompt-graph API
    comfyui_api_url: str = "http://comfyui:3001"      # Salad add-on control-plane API (ADR-014)
    # Local agent LLM — DiffusionGemma 26B-A4B (Gemma-4 MoE), OpenAI-compatible,
    # TEXT-ONLY. Host-served (not a v2g-net container); override V2G_AGENT_LLM_URL.
    agent_llm_url: str = "http://localhost:8084"
    agent_llm_model: str = "diffusiongemma-26B-A4B-it-Q8_0"
    milo_url: str = "http://milo:8090"                # MILo mesh sidecar
    come_url: str = "http://come:8091"                # CoMe mesh sidecar


@dataclass
class OversightConfig:
    """Pipeline overseer selection (ADR-013 / D-013.6).

    ``backend`` chooses who plans stages and recovers from failures end-to-end.
    ``artifact_vlm`` is the orthogonal bulk per-frame artifact-triage tool (FR-27),
    a transient model loaded only for the artifact stage then unloaded.

    ``diffusiongemma`` is the local host-served reasoner (ADR-013). NOTE it is
    TEXT-ONLY, so it cannot do *visual* per-frame triage — for the artifact_vlm
    role it reasons over text/metadata (captions, gate numbers) only; ``claude_code``
    remains the vision-capable default for that role.
    """
    backend: str = "claude_code"        # claude_code (DEFAULT, no GPU cost) | diffusiongemma
    artifact_vlm: str = "claude_code"   # claude_code (vision-capable) | diffusiongemma (text-only)


@dataclass
class IngestConfig:
    """Frame extraction parameters."""
    fps: float = 2.0  # 2fps for walkthroughs (SplatReady recommends 1.0)
    max_image_size: int = 2000
    min_frames: int = 20
    max_frames: int = 500
    blur_threshold: float = 100.0
    exposure_range: tuple[float, float] = (0.1, 0.9)
    target_frames: int = 120  # more frames = better COLMAP coverage

    # Fibonacci-sphere frame selection (ADR-007). Requires camera positions
    # from COLMAP (post-SfM second pass). Off by default for backward
    # compatibility; when enabled the existing quality-only path is unchanged
    # if camera positions are unavailable.
    use_fibonacci_coverage: bool = False
    coverage_weight: float = 0.4  # ADR-007 default: 0.4 coverage / 0.6 quality


@dataclass
class ReconstructConfig:
    """COLMAP / SplatReady reconstruction parameters."""
    method: str = "colmap"
    colmap_exe: str = "/usr/local/bin/colmap"
    use_fisheye: bool = False
    min_scale: float = 0.5
    matcher: str = "exhaustive"  # exhaustive for video (not sequential)
    single_camera: bool = True  # all frames share one camera model
    undistort: bool = True  # run image_undistorter before training
    # SfM feature extractor (work-order item 4): "sift" (universal fallback) or
    # "aliked" (ALIKED+LightGlue, SOTA for indoor presets). prefer_lichtfeld_plugin
    # routes via the LichtFeld COLMAP plugin when available, else builds COLMAP.
    feature_extractor: str = "sift"  # "sift" | "aliked"
    prefer_lichtfeld_plugin: bool = False


@dataclass
class TrainingConfig:
    """Gaussian splatting training parameters."""
    max_iterations: int = 30000
    iterations: int = 30000
    # Training strategy: "igs+" (ImprovedGS+, native v0.5.0 default — -26.8% time /
    # -13.3% Gaussians vs MCMC), "mcmc", or "mrnf". See work-order item 6.
    strategy: str = "igs+"
    sh_degree: int = 3
    lichtfeld_binary: str = "/opt/gaussian-toolkit/build/LichtFeld-Studio"
    target_psnr: float = 25.0
    target_ssim: float = 0.85
    convergence_window: int = 500
    convergence_threshold: float = 0.001
    checkpoint_interval: int = 5000
    # Mesh backend: "come" (default — project pivot 2026-06-04: ~3x faster than
    # MILo at comparable F1, confidence-based), "milo" (high-quality), "tsdf"
    # (fast fallback), "gaussianwrapping" (thin structures), or "auto".
    # train() routes a concrete backend directly and falls back to LichtFeld/TSDF
    # when the come sidecar is unavailable — so this default is SAFE but will
    # silently degrade to TSDF unless the image is built with INSTALL_COME=1 and
    # the come sidecar is running. See ADR-004 + work-order-sota-modernisation.md.
    mesh_method: str = "come"

    # When True the ADR-003 auto-selection policy is applied regardless of the
    # mesh_method value, unless mesh_method is set to a concrete backend name.
    # False by default for backward compatibility.
    mesh_backend_auto: bool = False

    # ADR-003: when True, the "auto" policy prefers the faster CoMe backend
    # (~25 min) over MILo (~69 min) when the CoMe sidecar is available. When
    # False (default), MILo is the default high-quality path. See ADR-003.
    mesh_speed_priority: bool = False

    # Native LichtFeld training flags (work-order item 6). PPISP is compiled into
    # the core (parameters.hpp:145-155) but disabled; bilateral-grid / 3DGUT /
    # pose-opt native flags are not yet wired. Off by default for backward compat.
    ppisp_enabled: bool = False
    bilateral_grid: bool = False
    use_3dgut: bool = False
    pose_opt: bool = False

    # Scene type preset: "default" or "indoor_reflective"
    scene_preset: str = "default"

    # Indoor-reflective scene overrides (applied when scene_preset == "indoor_reflective")
    # These suppress SH overfitting on specular highlights, use MCMC stochastic
    # relocation for better coverage of reflective surfaces, reduce iterations to
    # avoid overfitting, and set a white background to match bright gallery walls.
    indoor_sh_degree: int = 1
    indoor_strategy: str = "mcmc"
    indoor_iterations: int = 15000
    indoor_bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    indoor_opacity_reg: float = 0.01
    indoor_scale_reg: float = 0.01

    def resolved_iterations(self) -> int:
        """Return iteration count, respecting scene preset."""
        if self.scene_preset == "indoor_reflective":
            return self.indoor_iterations
        return self.iterations

    def resolved_strategy(self) -> str:
        """Return strategy name, respecting scene preset."""
        if self.scene_preset == "indoor_reflective":
            return self.indoor_strategy
        return self.strategy

    def resolved_sh_degree(self) -> int:
        """Return SH degree, respecting scene preset."""
        if self.scene_preset == "indoor_reflective":
            return self.indoor_sh_degree
        return self.sh_degree


@dataclass
class DecomposeConfig:
    """Scene decomposition parameters."""
    selection_method: str = "by_description"
    min_object_gaussians: int = 100
    descriptions: list[str] = field(default_factory=list)
    use_sam3: bool = True
    sam3_confidence_threshold: float = 0.5
    sam3_concepts: list[str] = field(default_factory=lambda: [
        "paintings", "frames", "sculptures", "furniture",
        "walls", "floor", "ceiling", "fixtures", "doorways",
    ])
    sam3_fallback_to_sam2: bool = True
    sam3_bpe_path: str = "/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    sam3_checkpoint_path: str = ""  # auto-detect from HF cache

    # Open-vocabulary object discovery (object_discovery.py). When enabled, the
    # segment stage first samples a spread of representative frames and asks a
    # *pluggable vision overseer* to enumerate the distinct physical objects in
    # the scene (open vocabulary). The discovered labels REPLACE the static
    # ``sam3_concepts`` list for that run. Off by default so existing Stage 6
    # behaviour (fixed concept list) is unchanged unless explicitly opted in.
    #
    # NOTE the default overseer is the ``claude_code`` agent path, because the
    # local DiffusionGemma reasoner (agent_llm.py) is TEXT-ONLY and cannot read
    # pixels (ADR-013). With no live vision overseer wired, discovery falls back
    # to the static ``sam3_concepts`` and logs a WARN — it never blocks the run.
    use_open_vocab_discovery: bool = False
    discovery_overseer: str = "claude_code"   # claude_code (vision) | static
    discovery_num_frames: int = 8             # representative frames to sample
    discovery_min_confidence: float = 0.3     # drop discoveries below this
    discovery_max_objects: int = 24           # cap concept-list size for SAM3


@dataclass
class MeshConfig:
    """Mesh extraction parameters."""
    min_vertices: int = 100
    max_vertices: int = 500_000
    watertight_check: bool = True
    normal_consistency_threshold: float = 0.8


@dataclass
class InpaintConfig:
    """Background inpainting parameters."""
    method: str = "comfyui"
    comfyui_api_url: str = "http://localhost:3001"
    comfyui_direct_url: str = "http://localhost:8189"
    local_ip: str = "192.168.2.1"
    hf_token: str = ""
    # Inpaint model (work-order item 1): "flux2" (FLUX.2-dev, staged default) or
    # "flux-fill" (FLUX.1-Fill-dev fallback). FLUX.2 needs the Mistral-3 text
    # encoder + FLUX.2 VAE alongside the diffusion checkpoint.
    model: str = "flux2"
    flux2_diffusion: str = "flux2_dev_fp8mixed.safetensors"
    flux2_vae: str = "flux2-vae.safetensors"
    flux2_text_encoder: str = "mistral_3_small_flux2_fp8.safetensors"
    denoise: float = 0.75
    steps: int = 28
    guidance: float = 30.0
    auto_download_models: bool = True
    blend_radius: float = 2.0
    iterations: int = 10000


@dataclass
class ObjectCropsConfig:
    """Best-frame object crop extraction (ADR-025 D1 / PRD v4 R3).

    For each SAM3 object the stage selects the best observing source frame
    (silhouette area x sharpness x centrality x edge clearance), crops with
    padding, mattes, and square-pads to >= ``out_size``. The crop is the ONLY
    conditioning input the 3D generator receives.
    """
    pad_frac: float = 0.12             # bbox padding as a fraction of its long side
    out_size: int = 1024               # minimum square output size (upscaled if smaller)
    candidates: int = 12               # top-by-area frames scored fully (sharpness needs decode)
    min_mask_area_px: int = 400        # ignore frames where the object is tiny
    # Matting: "auto" uses the SAM silhouette as alpha, falling back to rembg
    # when the mask is box-shaped (the SAM3 boxes bug, PRD v4 R1); "mask" and
    # "rembg" force a backend; "none" emits an opaque crop (flagged in lineage).
    matting: str = "auto"
    # A mask filling >= this fraction of its own bbox is treated as a box, not
    # a silhouette (the R1 defect signature).
    boxlike_fill_threshold: float = 0.95


@dataclass
class Hunyuan3DConfig:
    """Hunyuan3D 2.0 mesh reconstruction parameters."""
    enabled: bool = True
    # v2g-net ComfyUI service (was dead localhost:8189/3001 — master-audit gap #5).
    comfyui_url: str = "http://vitrine-comfyui:8188"
    api_url: str = "http://vitrine-comfyui:8188"
    quality: str = "standard"
    multiview: bool = True
    turbo: bool = False
    # Hunyuan3D 2.1 upgrade (work-order item 2). Probe → degrade to 2.0 if the
    # 2.1 ldm/weights are absent. fallback_* paths handle MV/SV failure.
    model_version: str = "2.1"
    dit_checkpoint: str = "hunyuan3d-dit-v2-1.safetensors"
    paint_checkpoint: str = "hunyuan3d-paintpbr-v2-1.safetensors"
    fallback_singleview: bool = True
    fallback_sam3d: bool = True
    timeout: int = 600
    seed: int = 42
    num_views: int = 4
    render_size: int = 512
    camera_distance: float = 2.5


@dataclass
class Trellis2Config:
    """TRELLIS.2-4B object generation, SINGLE-image conditioning (ADR-025).

    The generator is conditioned on one clean, matted best-frame crop from the
    ``object_crops`` stage — never on splat renders (ADR-025 supersedes the
    ADR-015/017 multiview-panel path). ``native_url`` selects the thin HTTP
    service wrapping the native TRELLIS.2 pipeline (PRD v4 R5); when empty the
    single-image ComfyUI workflow is used (runtime-verified node pack, interim
    executor until the native service env is stood up).
    """
    enabled: bool = True
    comfyui_url: str = "http://vitrine-comfyui:8188"
    # Thin native-pipeline service (scripts/trellis2_native_service.py). Empty
    # string = use the ComfyUI single-image workflow instead.
    native_url: str = ""
    # LoadTrellis2Models resolution: 512 | 1024 | 1024_cascade | 1536_cascade.
    resolution: str = "1536_cascade"
    texture_size: int = 4096           # PBR texture resolution
    timeout: int = 1800                # TRELLIS.2 + PBR is slow; time is not a constraint
    seed: int = 42
    # Quality-escalation ladder (ADR-025 D4 / PRD v4 R7). best_of_n > 1 runs
    # that many seed re-rolls and keeps the candidate whose FRONT silhouette
    # best matches the observed crop (+ mesh sanity); all candidate scores are
    # recorded in the winner's lineage. Default 1 = single-shot (unchanged).
    best_of_n: int = 3          # SOTA-quality default (R7a): 3 seed re-rolls,
                                # keep the best by silhouette+sanity. Time is
                                # not a constraint on the reference host; drop
                                # to 1 for a fast single-shot run.
    seeds: list[int] = field(default_factory=list)  # explicit seeds; else derived
    # Rung (b) of the escalation ladder (ADR-025 D4 / PRD v4 R7): when the best
    # seed re-roll scores below ``escalation_threshold``, synthesize ONE
    # image-edit alternate view ("show the back", Qwen-Image-Edit-2509) and add
    # it as an additional single-image best-of-N candidate — never panel
    # synthesis. Off by default (the Qwen edit UNET is not staged yet; the rung
    # also self-gates at runtime via ImageEditView.probe_edit_model()).
    image_edit_escalation: bool = False
    escalation_threshold: float = 0.55
    ss_steps: int = 12                 # sparse-structure sampling steps
    shape_steps: int = 12              # shape diffusion steps
    tex_steps: int = 12                # texture diffusion steps
    # High/low asset pair (native service to_glb x2, PRD v4 R5).
    face_count_high: int = 500_000
    face_count_low: int = 20_000


@dataclass
class Pixal3DConfig:
    """Pixal3D single-image object generation (PRD v4 R8 — gated eval).

    Pixel-aligned generation on the TRELLIS.2 backbone (TencentARC, MIT —
    ADR-025 amendment). DISABLED by default: weights not staged, no executor
    stood up; the R8 head-to-head eval (research/2026-07-pixal3d-eval.md) gates
    any promotion. When enabled, stages tries Pixal3D FIRST, falling through to
    TRELLIS.2 on failure. ``native_url`` selects the thin HTTP service (mirrors
    the TRELLIS.2 R5 contract); empty selects the ComfyUI workflow, which fails
    fast until a verified node pack + workflow JSON exist (VERIFY-ON-ENV-BUILD).
    """
    enabled: bool = False
    comfyui_url: str = "http://vitrine-comfyui:8188"
    native_url: str = ""
    resolution: str = "1536_cascade"   # VERIFY-ON-ENV-BUILD: assumed backbone ladder
    texture_size: int = 4096
    timeout: int = 1800
    seed: int = 42
    ss_steps: int = 12
    shape_steps: int = 12
    tex_steps: int = 12
    face_count_high: int = 500_000
    face_count_low: int = 20_000
    workflow_path: str = ""            # "" = default (not authored yet)


@dataclass
class ImageEditConfig:
    """Instruction-driven alternate-view edit (ADR-025 D4 / PRD v4 R7 rung b).

    Qwen-Image-Edit-2509 (Apache-2.0, commercial-safe) on ComfyUI — a 2D stage,
    in-scope per ADR-014 as narrowed by ADR-025. Loader filenames are the
    canonical Comfy-Org release names; the diffusion UNET is NOT staged yet, so
    ImageEditView.probe_edit_model() resolves/gates at runtime.
    """
    comfyui_url: str = "http://vitrine-comfyui:8188"
    diffusion_model: str = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
    clip_model: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae_model: str = "qwen_image_vae.safetensors"
    instruction: str = ""      # empty -> DEFAULT_BACK_INSTRUCTION (no import cycle)
    negative: str = ""
    steps: int = 20
    cfg: float = 2.5
    shift: float = 3.1
    denoise: float = 1.0
    timeout: int = 600
    seed: int = 42
    matte_output: bool = True  # rembg RGBA matte of the edited view


@dataclass
class ViewCompletionConfig:
    """RETIRED from the object arc (ADR-025 supersedes ADR-017).

    FLUX.2 panel completion conditioned the generator on synthesized views of a
    partial splat — an anti-pattern the 2026-07-09 audit measured as strictly
    worse than one clean source-frame crop. The dataclass is retained only so
    existing config JSONs with a ``view_completion`` section still load; the
    pipeline no longer reads it. Occlusion escalation is now seed re-rolls +
    image-edit alternative attempts (PRD v4 R7).
    """
    enabled: bool = False
    comfyui_url: str = "http://vitrine-comfyui:8188"
    generator: str = "flux2"           # flux2 | qwen-image-edit (commercial-safe)
    gap_threshold: float = 0.02        # coverage <= this -> synthesise
    keep_threshold: float = 0.10       # coverage >= this -> keep real render
    steps: int = 28
    guidance: float = 4.0
    seed: int = 42
    max_references: int = 2
    timeout: int = 600


@dataclass
class PersonRemovalConfig:
    """Person detection and removal parameters."""
    enabled: bool = False  # Disabled by default until runtime deps verified
    method: str = "opencv"
    comfyui_url: str = "http://localhost:8188"
    flux_endpoint: str = ""
    confidence: float = 0.5
    dilation_px: int = 15
    drop_threshold: float = 0.50
    flag_threshold: float = 0.15
    comfyui_timeout: float = 120.0


@dataclass
class DeliveryConfig:
    """Web-delivery optimisation via PlayCanvas splat-transform (ADR-006).

    All options default to off/conservative values so existing pipeline runs
    are unaffected until explicitly enabled.
    """
    # Enable the post-training splat optimisation step. When True and
    # is_splat_transform_available() returns True, stages._optimize_splat()
    # is called after a successful train; failure is non-fatal.
    enable_splat_optimize: bool = False
    output_format: str = "ksplat"  # "ksplat" | "sog" | "glb" | "ply" | "compressed-ply"
    opacity_min_threshold: float = 0.05  # discard floaters below this opacity
    max_scale: float | None = None  # discard oversized Gaussians (None = no limit)
    sort: bool = True  # Morton-order sort for front-to-back rendering
    generate_html_viewer: bool = False

    # Optional extra LichtFeld-native splat exports (additive, off by default).
    # Each format is exported via the MCP scene.export_* tool after training.
    # Accepted values: "spz", "sog", "html", "rad".
    # "usdz_nurec" is also supported but requires a proprietary NVIDIA NuRec
    # licence — do NOT enable for commercial deliverables without licence review.
    # Unknown values are skipped with a warning (non-fatal).
    lichtfeld_extra_formats: list[str] = field(default_factory=list)
    lichtfeld_extra_formats_sh_degree: int = 3


@dataclass
class ExportConfig:
    """USD / final export parameters."""
    format: str = "usd"
    include_materials: bool = True
    coordinate_system: str = "right_handed_y_up"
    # Prefer LichtFeld's native USD export (scene.export_usd, v0.5.1+) over the
    # custom scripts/assemble_usd_scene.py path. When True, stages.assemble_usd()
    # tries native export first (best-effort, additive); the custom assembler
    # stays as fallback and still supplies multi-object hierarchy + ADR-011
    # v2g:* customData until the native-customData parity probe confirms parity.
    prefer_native_usd: bool = True


@dataclass
class QualityConfig:
    """Quality gate thresholds."""
    gate1_min_psnr: float = 20.0
    gate1_min_ssim: float = 0.75
    gate2_min_mesh_vertices: int = 500
    gate2_normal_consistency: float = 0.7
    gate2_roundtrip_psnr: float = 18.0
    final_min_psnr: float = 22.0


@dataclass
class RetryConfig:
    """Retry behaviour per stage."""
    max_retries: int = 3
    parameter_adjustments: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "RECONSTRUCT": {"min_scale": 0.25, "matcher": "sequential"},
        "RECONSTRUCT:2": {"matcher": "vocab_tree"},
        "QUALITY_GATE_1": {},
        "MESH_OBJECTS": {"max_vertices": 1_000_000},
        "RETRAIN_BG": {"max_iterations": 50000},
    })


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    mcp_endpoint: str = "http://127.0.0.1:45677/mcp"
    mcp_timeout: float = 30.0
    mcp_training_timeout: float = 600.0
    status_file: str = "pipeline_status.json"
    output_dir: str = "./output"

    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    oversight: OversightConfig = field(default_factory=OversightConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    person_removal: PersonRemovalConfig = field(default_factory=PersonRemovalConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    object_crops: ObjectCropsConfig = field(default_factory=ObjectCropsConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    trellis2: Trellis2Config = field(default_factory=Trellis2Config)
    pixal3d: Pixal3DConfig = field(default_factory=Pixal3DConfig)
    image_edit: ImageEditConfig = field(default_factory=ImageEditConfig)
    view_completion: ViewCompletionConfig = field(default_factory=ViewCompletionConfig)
    hunyuan3d: Hunyuan3DConfig = field(default_factory=Hunyuan3DConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        # SEC-01/SEC-03: redact secret-named fields (e.g. inpaint.hf_token)
        # before persisting — run snapshots are uploaded to Drive with outputs.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _redact_secrets(self.to_dict())
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        cfg = cls()
        direct_fields = {
            "mcp_endpoint", "mcp_timeout", "mcp_training_timeout",
            "status_file", "output_dir",
        }
        for k in direct_fields:
            if k in data:
                setattr(cfg, k, data[k])

        sub_map: dict[str, type] = {
            "endpoints": EndpointsConfig,
            "oversight": OversightConfig,
            "ingest": IngestConfig,
            "person_removal": PersonRemovalConfig,
            "reconstruct": ReconstructConfig,
            "training": TrainingConfig,
            "decompose": DecomposeConfig,
            "object_crops": ObjectCropsConfig,
            "mesh": MeshConfig,
            "trellis2": Trellis2Config,
            "pixal3d": Pixal3DConfig,
            "image_edit": ImageEditConfig,
            "view_completion": ViewCompletionConfig,
            "hunyuan3d": Hunyuan3DConfig,
            "inpaint": InpaintConfig,
            "delivery": DeliveryConfig,
            "export": ExportConfig,
            "quality": QualityConfig,
            "retry": RetryConfig,
        }
        for key, klass in sub_map.items():
            if key in data and isinstance(data[key], dict):
                sub = klass()
                for field_name, value in data[key].items():
                    if hasattr(sub, field_name):
                        current = getattr(sub, field_name)
                        if isinstance(current, tuple) and isinstance(value, list):
                            value = tuple(value)
                        setattr(sub, field_name, value)
                setattr(cfg, key, sub)
        return cfg

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ingest.fps <= 0:
            errors.append("ingest.fps must be positive")
        if self.ingest.min_frames < 3:
            errors.append("ingest.min_frames must be >= 3")
        if self.ingest.min_frames > self.ingest.max_frames:
            errors.append("ingest.min_frames must be <= max_frames")
        if not (0.0 <= self.ingest.coverage_weight <= 1.0):
            errors.append("ingest.coverage_weight must be in [0, 1]")
        if self.training.max_iterations < 1000:
            errors.append("training.max_iterations must be >= 1000")
        if self.training.target_psnr < 10:
            errors.append("training.target_psnr must be >= 10")
        _valid_strategies = {"mrnf", "mcmc", "igs+"}
        if self.training.strategy not in _valid_strategies:
            errors.append(
                f"training.strategy must be one of {sorted(_valid_strategies)}, "
                f"got '{self.training.strategy}'"
            )
        _valid_oversight_backends = {"claude_code", "diffusiongemma"}
        if self.oversight.backend not in _valid_oversight_backends:
            errors.append(
                f"oversight.backend must be one of {sorted(_valid_oversight_backends)}, "
                f"got '{self.oversight.backend}'"
            )
        _valid_artifact_vlm = {"claude_code", "diffusiongemma"}
        if self.oversight.artifact_vlm not in _valid_artifact_vlm:
            errors.append(
                f"oversight.artifact_vlm must be one of {sorted(_valid_artifact_vlm)}, "
                f"got '{self.oversight.artifact_vlm}'"
            )
        _valid_mesh_methods = {"tsdf", "milo", "come", "gaussianwrapping", "auto"}
        if self.training.mesh_method not in _valid_mesh_methods:
            errors.append(
                f"training.mesh_method must be one of {sorted(_valid_mesh_methods)}, "
                f"got '{self.training.mesh_method}'"
            )
        _valid_delivery_formats = {"ksplat", "sog", "glb", "ply", "compressed-ply"}
        if self.delivery.output_format not in _valid_delivery_formats:
            errors.append(
                f"delivery.output_format must be one of {sorted(_valid_delivery_formats)}, "
                f"got '{self.delivery.output_format}'"
            )
        if not (0.0 <= self.delivery.opacity_min_threshold <= 1.0):
            errors.append("delivery.opacity_min_threshold must be in [0, 1]")
        if self.quality.gate1_min_psnr < 5:
            errors.append("quality.gate1_min_psnr must be >= 5")
        if self.retry.max_retries < 0:
            errors.append("retry.max_retries must be >= 0")
        if not self.mcp_endpoint.startswith("http"):
            errors.append("mcp_endpoint must be an HTTP URL")
        return errors
