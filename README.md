# Vitrine

Video-to-structured-3D-scene pipeline built on [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio). Point it at a folder of videos and it produces a richly-annotated **USD scene graph** — a textured environment mesh plus individually-reconstructed, correctly-placed 3D hulls of the key objects in the scene — with full video→frame→object lineage, and a compressed Gaussian splat (`.ksplat`) for web delivery.

> This is an **isolated fork**. Upstream sync is one-way pull only; we never push to or open PRs against the upstream repository. See [BOUNDARIES.md](BOUNDARIES.md).

## Pipeline

```
Drive/video ingest  →  frame extraction + per-image metadata sidecar
   →  COLMAP SfM (ALIKED+LightGlue, SIFT fallback)
   →  3DGS training (LichtFeld: ImprovedGS+ / MRNF / MCMC)
   →  .ksplat (splat-transform)              [web delivery]
   →  SAM3 concept segmentation  →  key-item ranking (min_object_gaussians + keyness)
   →  per-object hull: orbit render → FLUX.2 recovery → TRELLIS.2 / Hunyuan3D-2.1 (textured GLB)
   →  environment mesh (CoMe; MILo / GaussianWrapping / TSDF fallbacks)
   →  USD scene graph (native LichtFeld export; v2g:* metadata + lineage)  +  .ksplat
```

The pipeline is driven by a pre-run **`exhibit.toml` manifest** (exhibit identity, object list, Drive URL, `env:`-indirected secrets) and orchestrated by an in-container agent. A **SOTA idiot-check** (`pipeline/sota_registry.py`, wired into `preflight.py`) validates the host — staged weights, VRAM fit, version pins, licence posture — before any run.

## SOTA stack (research/non-commercial posture)

Web-verified mid-2026; weights staged + verified on the reference host. Run `python -m pipeline.sota_registry check` to validate.

| Element | Model | Licence | VRAM | Notes |
|---|---|---|---|---|
| Inpaint / recovery | **FLUX.2-dev** | non-commercial | ~34 GB fp8 | masked recovery of unseen object faces |
| Local VLM | **gemma-4-26B-A4B-it** (Q8_0) | Apache-2.0 | ~28 GB | artifact analysis + recovery oversight |
| 3D hull (primary) | **TRELLIS.2-4B** | MIT | ~24 GB | single-image → textured PBR mesh |
| 3D hull (fallback) | **Hunyuan3D-2.1** | Tencent-community | ~29 GB | multiview, matches the orbit renderer |
| GS→surface mesh | **CoMe** (default) | CC BY-NC-ND | ~20 GB | best indoor F1; MILo / PGSR / TSDF fallbacks |
| Training | **ImprovedGS+** (native) | — | — | −27% time vs MCMC; MRNF/MCMC also native |
| SfM | **ALIKED+LightGlue** | — | — | via LichtFeld COLMAP plugin; SIFT fallback |
| USD | **native `scene.export_usd`** | — | — | LichtFeld v0.5.1+; custom assembler fallback |

Commercial deployment requires swapping the non-commercial models (CoMe→PGSR, FLUX.2→Qwen-Image-Edit); the idiot-check fails the run under `--commercial` if a non-commercial model is selected. See [research/decisions/work-order-sota-modernisation.md](research/decisions/work-order-sota-modernisation.md).

## Deployment

```bash
# 1. Provision the exhibit manifest (web wizard or hand-edit exhibit.example.toml)
cd onboarding && cargo build --release && ./target/release/vitrine-onboarding   # http://localhost:8088

# 2. Bring up the pipeline containers
docker compose -f docker-compose.consolidated.yml up -d

# 3. Bring up the canonical ComfyUI (owner install + staged model tree, over v2g-net)
scripts/run_comfyui.sh                                                          # http://localhost:8200

# 4. Web UI
# http://localhost:7860
```

| Container | Base | GPU | Purpose |
|---|---|---|---|
| `gaussian-toolkit` | Ubuntu 24.04 / CUDA 12.8 / Py 3.12 | 0 | COLMAP, LichtFeld 3DGS, web UI, Blender, SAM3, pipeline |
| `vitrine-comfyui` | — | 0 | FLUX.2 / TRELLIS.2 / Hunyuan / SAM3D ComfyUI (`scripts/run_comfyui.sh`) |
| `milo` | Ubuntu 22.04 / CUDA 11.8 / Py 3.10 | 1 | MILo (+ optional GaussianWrapping) mesh extraction |
| `come` | Ubuntu 22.04 / CUDA 12.1 / Py 3.10 | 1 | CoMe mesh extraction (gated: `INSTALL_COME=1`) |

Containers share a `v2g-net` bridge; the pipeline reaches ComfyUI as `V2G_COMFYUI_URL=http://vitrine-comfyui:8188`.

| Port | Service |
|---|---|
| 7860 | Web UI (Flask) |
| 8088 | Onboarding wizard (Rust/Axum) |
| 8200 | Canonical ComfyUI |
| 7681 | Web terminal (ttyd / Claude Code orchestrator) |
| 45677 | LichtFeld MCP server (70+ tools) |
| 5901 | VNC (Blender remote desktop) |

## Pipeline modules (`src/pipeline/`)

| Category | Modules |
|---|---|
| Core | `stages.py`, `cli.py`, `config.py`, `preflight.py`, `sota_registry.py` |
| Manifest / infra | `manifest.py` (exhibit.toml), `model_lifecycle.py` (serial VRAM), `endpoints.py` (v2g-net) |
| Ingestion | `drive_ingestor.py`, `frame_selector.py`, `frame_quality.py`, `fibonacci_sampler.py` |
| Reconstruction | `colmap_parser.py`, `coordinate_transform.py`, `mcp_client.py`, `gsplat_trainer.py` |
| Segmentation | `sam2_segmentor.py`, `sam3_segmentor.py`, `sam3d_client.py`, `mask_projector.py` |
| Hull / recovery | `multiview_renderer.py`, `comfyui_inpainter.py`, `comfyui_control.py`, `hunyuan3d_client.py` |
| Mesh | `mesh_extractor.py` (TSDF), `milo_extractor.py`, `come_extractor.py`, `gaussianwrapping_extractor.py`, `mesh_cleaner.py` |
| Delivery / scene | `splat_optimizer.py`, `texture_baker.py`, `material_assigner.py`, `blender_assembler.py`, `usd_assembler.py` |
| Quality | `quality_gates.py`, `person_remover.py` |

Web UI in `src/web/` (Flask :7860). Onboarding wizard in `onboarding/` (Rust/Axum).

## Hardware

Reference host: 2× NVIDIA RTX 6000 Ada (48 GB each), AMD Threadripper PRO, 256 GB RAM, NVMe. The serial model lifecycle keeps peak VRAM at `max(stage)`, not the sum, so a single 48 GB GPU is the practical floor for the full SOTA stack (TSDF-only mesh works on 12 GB).

## Status

**Working end-to-end** (v2 core): video → COLMAP → 3DGS → SAM3 → mesh → Blender texture bake → USD → web viewer.

**Built and host-validated** (v3): exhibit.toml manifest + loader, SOTA registry + idiot-check (wired into preflight), serial model lifecycle, v2g-net endpoints, agent-controlled ComfyUI client, native USD export wiring, key-item ranking, per-image metadata sidecar, secret-stripped config snapshots, the onboarding wizard (build + secret-containment verified), and the full SOTA weight set staged + verified (FLUX.2, gemma-4, TRELLIS.2, Hunyuan-2.1, SAM3D). Test suites for `model_lifecycle` + `comfyui_control` pass (31/31).

**In progress**: hull custom-node dependency builds (TRELLIS.2/Hunyuan ComfyUI nodes), gemma VLM serving (`agent-vlm`), end-to-end validation on real capture data, and the config-correctness items the idiot-check flags (switch training default to `igs+`, verify CoMe CLI flags, probe native-USD `v2g:*` customData parity). See [docs/engineering-log.md](docs/engineering-log.md) and the [report](report/main.pdf).

> Source video quality is the dominant quality bottleneck: lossy/featureless/reflective footage breaks reconstruction regardless of downstream method. See [docs/capture-methodology.md](docs/capture-methodology.md).

## Boundaries & license

Fork of LichtFeld Studio (GPL-3.0). Upstream directories (`src/core/`, `src/app/`, `src/mcp/`, `src/rendering/`, `src/training/`, `src/geometry/`, `src/io/`, …) are not modified; all additions live in `src/pipeline/`, `src/web/`, `onboarding/`, `docker/`, and `scripts/`. Pipeline additions: GPL-3.0 (derivative work). See [BOUNDARIES.md](BOUNDARIES.md).
