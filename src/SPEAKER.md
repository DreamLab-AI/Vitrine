# Vitrine — Speaker Reference (named components + figures)

Per-slide name-drop lists (to show optionality & depth) + guide figures on the proof slide.
These are also embedded as **reveal speaker notes** — press **`s`** in the deck to see them per slide.
Sources: `report/v5/sec_results.tex`, `docs/engineering-log.md`, `output/dreamlab/*` logs, `docs/asset-creation-decision-tree.md`.

> Legend: **[primary]**, *(alt/fallback)*, ⚠ = say carefully (not fully done).

## 1 · From video to a living 3D world (headline)
COLMAP → **LichtFeld Studio** (3DGS) → **SAM 3** → **TRELLIS.2 / Hunyuan3D-2.1** → **NVIDIA ArtiFixer** → **Unreal Engine 5.8** (Nanite + NanoGS). Agent: **Claude Code** + **DiffusionGemma**. ~30 named SOTA components, picked per video.

## 2 · Save real places (capture & ingest)
Capture: smartphone **4K** (Pixel 9 Pro); recommend **4K/60, 1/500 s**; *iPhone LiDAR/ARKit, event cameras*. Ingest: **rclone**, **PyAV + ffmpeg**, **SHA-256 manifest**, **multi-sweep COLMAP**. Partners: DreamLab AI · University of Salford.

## 3 · Why it is hard (quality)
**MUSIQ** (pyiqa) · *Q-Align* · **Variance-of-Laplacian** (full-res) · **structure-tensor λ₂** motion-blur metric → recapture flag. Finding: motion blur is the limit, not the algorithms.

## 4 · What you get (formats)
**FBX** (Nanite, baked) = UE deliverable · **GLB** (PBR objects) · **.ksplat** web (@playcanvas/splat-transform) · SOG/SPZ/PLY · *USD archival only — ParticleField dropped for UE*.

## 5 · Smart by design (router)
Diagnose: MUSIQ + Laplacian + λ₂ + SfM success + coverage % + hardware. Axes: **sharpness · coverage · hardware · deliverable**. Guardrails: quality gates (PASS/WARN/FAIL), SOTA preflight.

## 6 · Step by step (pipeline)
- **SfM:** COLMAP 4.1 + **ALIKED** + **LightGlue** ⚠*(SIFT fallback live)* · *VGGT, DUSt3R, InstantSplat*
- **3DGS:** **LichtFeld v0.5.3** (ImprovedGS+, MRNF, PPISP) · *gsplat, 3DGUT/3DGRUT*
- **Segment:** **SAM 3** (+ SAM 3.1 Object Multiplex), SAM 2
- **Objects:** **TRELLIS.2-4B** / *Hunyuan3D-2.1* / *SAM-3D-Objects*; view-completion **FLUX.2** (+ Mistral-3), *Qwen-Image-Edit, MV-Adapter*; DINOv3; on **ComfyUI**
- **Scene mesh:** **CoMe** / **gsplat-TSDF** (Open3D) / MILo / 2DGS / PGSR / SuGaR
- **Texture:** MeshCleaner → **xatlas** → **Blender Cycles**
- **Engine:** **UE 5.8** + Nanite + NanoGS

## 7 · One AI runs it all (agent + container)
Orchestrator **Claude Code** (in-container: diagnose→select→drive→evaluate→recover; vision triage + discovery). Reasoner **DiffusionGemma-26B-A4B-it** (llama.cpp `:8084`) · *gemma-4 vision, Qwen3-VL*. MCP: **LichtFeld `:45677` (70+ tools)**, ComfyUI `:8188`/`:3001`, UE Web Remote Control `:30010` + UE MCP `:8000` + bridge `:9100`. Harness: serial VRAM `/free` lifecycle, SOTA preflight, quality gates. Containers: gaussian-toolkit (GPU0), vitrine-comfyui (GPU0), milo/come (GPU1), unreal (GPU1); nets v2g-net + visionclaw_network; ~216 GB model store.

## 8 · Proof it works (guide figures)
**INPUT** — 1× **4K/30** handheld (Pixel 9 Pro), multi-pass; **~13.6k frames → ~1,650 kept** (orig 1,200→600); RGB only. **MUSIQ ≈ 31** (abandon < ~19); **0 % under-observed**, ~90 cams/point; blur-select **+20–38 %** p25.
**OUTPUT** — SfM **600/600** (locked 750), **~290k–367k** points, **~1 px** reproj · 3DGS **30k** iters → **4.0 M → 2.48 M clean → 1.55 M** for UE · SAM 3 found **8 objects → 3–4 clean** (TRELLIS.2 won), **4096 px** PBR, real scale (chair 60 cm, ladder 210 cm) · room TSDF **~51k v / 81k f**, **2048²** albedo (planarity 72 % — capture-limited → NanoGS splat hero).
**COMPUTE** — **RTX 6000 Ada, 48 GB**; 3DGS ~**25 min**, CoMe ~**59 min**, ArtiFixer 14B peak **45.5 GB**. *Win = objects; room limited by capture → recapture is the lever.*

## 9 · Many tools, one goal (options)
- **SfM:** COLMAP+ALIKED+LightGlue | SIFT | VGGT | DUSt3R | InstantSplat
- **Scene:** NanoGS splat | TSDF | CoMe | MILo | 2DGS | PGSR | SuGaR
- **Objects:** TRELLIS.2 | Hunyuan3D-2.1 | SAM-3D-Objects
- **Inpaint/view:** FLUX.2 | FLUX.1-Fill | Qwen-Image-Edit | MV-Adapter
- **Enhancers:** NVIDIA ArtiFixer (coverage) | BAD-Gaussians / 3dgs-deblur (blur) | AGS-Mesh (LiDAR) | EBAD-Gaussian (event)
- **Splat-in-UE:** NanoGS | MLSLabs | LiDAR point-cloud | XScene-UE
- **Agent:** Claude Code | DiffusionGemma | gemma-4 / Qwen3-VL

## 10 · Explore your world (result/vision)
**UE 5.8**, Nanite, **NanoGS** (real Gaussians in-engine), Lumen; web via SuperSplat/SplatTransform → **.ksplat** (PlayCanvas/2Xplat); MR/VR headset; Web Remote Control.

### ⚠ Say-it-carefully
ALIKED+LightGlue intended but **SIFT** ran · UE overlay **built, not started** (Blender proves scenes) · SAM-3D-Objects staged but wiring incomplete · DiffusionGemma optional, **Claude Code** is the live orchestrator · ArtiFixer **runs** on the 48 GB card but **not applied** to this room (fixes coverage, not blur).
