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


@dataclass
class TrainingConfig:
    """Gaussian splatting training parameters."""
    max_iterations: int = 30000
    iterations: int = 30000
    strategy: str = "mrnf"
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
    comfyui_api_url: str = "http://192.168.2.48:3001"
    comfyui_direct_url: str = "http://192.168.2.48:8189"
    local_ip: str = "192.168.2.1"
    hf_token: str = ""
    model: str = "flux-fill"
    denoise: float = 0.75
    steps: int = 28
    guidance: float = 30.0
    auto_download_models: bool = True
    blend_radius: float = 2.0
    iterations: int = 10000


@dataclass
class Hunyuan3DConfig:
    """Hunyuan3D 2.0 mesh reconstruction parameters."""
    enabled: bool = True
    comfyui_url: str = "http://192.168.2.48:8189"
    api_url: str = "http://192.168.2.48:3001"
    quality: str = "standard"
    multiview: bool = True
    turbo: bool = False
    fallback_singleview: bool = True
    fallback_sam3d: bool = True
    timeout: int = 600
    seed: int = 42
    num_views: int = 4
    render_size: int = 512
    camera_distance: float = 2.5


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

    ingest: IngestConfig = field(default_factory=IngestConfig)
    person_removal: PersonRemovalConfig = field(default_factory=PersonRemovalConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
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
            "ingest": IngestConfig,
            "person_removal": PersonRemovalConfig,
            "reconstruct": ReconstructConfig,
            "training": TrainingConfig,
            "decompose": DecomposeConfig,
            "mesh": MeshConfig,
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
