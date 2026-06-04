# Video-to-Scene Pipeline -- Claude Code Orchestration

You are running inside the gaussian-toolkit Docker container on a dual RTX 6000 Ada system.
**You are the orchestrator.** There is no state machine. You run each pipeline stage manually,
inspect results between steps, and decide what to do next.

**CRITICAL: You MUST complete ALL stages through to USD assembly and validation.
Do NOT stop after training. The full pipeline is:
ingest -> select_frames -> reconstruct -> train -> segment -> extract_objects -> mesh_objects -> assemble_usd -> validate**

## Standing Directives (READ FIRST — apply to every job and every change)

These supersede convenience shortcuts elsewhere in this file.

1. **Prefer LichtFeld Studio native capabilities + its plugins/tools.** Before
   writing or calling custom Python, check the LichtFeld MCP surface (`AGENTS.md`;
   bridge at `http://localhost:45677/mcp`) and the installed plugins. USD export,
   training, pose-opt, densification and scene export are now **native** — use
   them instead of reimplementing in `src/pipeline`. Fall back to the custom
   Python path only when a native equivalent is missing or lacks a required
   feature (e.g. per-object USD hierarchy + `v2g:*` metadata).
2. **Always run clean SOTA.** Use the newest stable models/tools, never a stale
   default. SOTA weights are already staged — do NOT re-download. They live on
   the **host** (`~/comfyui-models-staging`, `~/comfyui-api-data/ComfyUI`); if
   `/models-staging` in this container is empty, the first task is to get those
   mounted/copied in (bind mount or set `MODELS_STAGING_DIR`).
3. **Always version-check before use, then pin.** For every model/tool, verify
   the current latest (MCP catalog, ComfyUI `/object_info`, HF, upstream repo),
   update if behind, then pin the exact tag/commit/checkpoint. No HEAD clones,
   no floating `latest`.

**Mesh backend = CoMe (project pivot, 2026-06-04).** The config default is now
`come` (~3x faster than MILo at comparable F1). To honour the pivot the image
MUST be built with `INSTALL_COME=1`, the `come` sidecar MUST be running, and you
MUST **verify `come_extractor.py`'s CLI flags against the released CoMe repo**
(they were inferred). Otherwise every run silently degrades to the TSDF fallback.

## SOTA Modernization Mandate (research → implement → validate → pin → report)

You own these upgrades. Work one item at a time on a real test job; gate each
behind a capability probe so an upgrade never hard-breaks a run. Canonical
detail + traceability: `research/decisions/work-order-sota-modernisation.md`
(ADR-012 SOTA tooling + the v3 closure PRD).

| # | Upgrade | Current (file:line) | Target | Notes |
|---|---------|---------------------|--------|-------|
| 1 | Inpaint model | `comfyui_inpainter.py:88` flux1-fill-dev | **FLUX.2-dev** | staged `flux2_dev_fp8mixed.safetensors` (+ `flux2-vae`, `mistral_3_small_flux2_fp8`, Turbo LoRAs); add `workflows/flux2_inpaint.json`; probe→fallback FLUX.1-Fill |
| 2 | Image-to-3D hull | `hunyuan3d_client.py:61,77` Hunyuan3D-2.0 | **Hunyuan3D-2.1** | local `comfy/ldm/hunyuan3dv2_1` + `comfyui-sam3dobjects`; probe→fallback 2.0 |
| 3 | SAM3D fallback | `sam3d_client.py` orphaned (unimported) | **wire as Hunyuan fallback** | `config.hunyuan3d.fallback_sam3d` already True; node + weights present |
| 4 | COLMAP features | SIFT + exhaustive matcher | **ALIKED+LightGlue** | prefer the LichtFeld COLMAP plugin if present; SIFT fallback |
| 5 | USD assembly | subprocess `scripts/assemble_usd_scene.py` | **native LichtFeld USD export** (MCP `scene.export`) | keep custom path only if native lacks per-object hierarchy + `v2g:*` metadata |
| 6 | Plugins | `splat_ready` referenced (`stages.py`) but not installed | **install + enable** splat_ready, PPISP, bilateral-grid, 3DGUT, pose-opt, ImprovedGS+ | wire native flags in `train` |
| 7 | Version pins | Dockerfiles clone ComfyUI/Hunyuan/COLMAP/MILo/CoMe/GW/SAM3/gsplat at HEAD | **pin all** to tag/commit | write a lock manifest |
| 8 | Endpoints | hardcoded `192.168.2.48:8189/:3001` (~11 sites) + in-container :8188 | **single source via config/manifest** | reconcile Salad(:3001) vs local ComfyUI(:8188) |

For each item: (a) RESEARCH the current latest + whether LichtFeld-native/plugin
already covers it; (b) IMPLEMENT against the staged weights / local ComfyUI,
behind a capability probe; (c) VALIDATE on a test job against the Quality
Targets below; (d) PIN the exact version; (e) REPORT via the REST API.

## Available Tools

- LichtFeld Studio: `/opt/gaussian-toolkit/build/LichtFeld-Studio`
- COLMAP: `/usr/local/bin/colmap`
- Blender: `/usr/local/bin/blender` (DISPLAY=:1, VNC on :5901)
- ComfyUI API: `http://localhost:8188`
- Python pipeline stages: `from pipeline.stages import PipelineStages`
- Web API: `http://localhost:7860`

## When a job arrives

Check for new jobs:

```bash
curl -s http://localhost:7860/jobs | python3 -m json.tool
```

For each queued job, run the pipeline stage by stage.

---

## Step 1: Ingest -- extract frames from video

Extract frames at 4fps to oversample, then select the best subset.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.ingest('/data/output/JOB_ID/input.mp4', fps=4.0)
print(result)
"
```

Update the web UI:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "ingest", "progress": 0.05, "message": "Extracting frames at 4fps"}'
```

**Inspect**: `ls /data/output/JOB_ID/frames/ | wc -l`

---

## Step 2: Remove people (if needed)

Look at the frames. If people are visible:

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.remove_people('/data/output/JOB_ID/frames/')
print(result)
"
```

If no people, skip to step 3 using the frames directory directly.

---

## Step 3: Select best frames (IMPORTANT for COLMAP registration)

Select 60-80 diverse, high-quality frames from the oversampled set.
This is critical -- sending all frames to COLMAP causes low registration rates.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.select_frames('/data/output/JOB_ID/frames/', target=80)
print(result)
"
```

**Check**: The selected frame count should be 60-80. If less than 40, re-run with lower blur_threshold.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "select_frames", "progress": 0.1, "message": "Selected N frames from M extracted"}'
```

---

## Step 4: COLMAP reconstruction

Use the **sequential** matcher for video input (NOT exhaustive -- sequential is faster
and produces better registration rates for video where frames are temporally ordered).

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.reconstruct('/data/output/JOB_ID/frames_selected/', matcher='sequential')
print(result)
"
```

**Check**: Look for `sparse/0/cameras.bin` and `images/` in the colmap dir.
**CRITICAL**: Check the registration rate. At least 70% of input frames should register.
If registration is below 50%, re-run with `matcher='exhaustive'` or reduce frame count.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "reconstruct", "progress": 0.25, "message": "COLMAP: N/M frames registered"}'
```

---

## Step 5: Train gaussian splatting

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.train('/data/output/JOB_ID/colmap/undistorted/', iterations=30000)
print(result)
"
```

Update progress during training:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "train", "progress": 0.5, "message": "30k iter, loss 0.02"}'
```

**Quality check**: The PLY should be > 10 MB for a good scene.
If training fails or quality is poor, adjust `iterations` or try `strategy="mcmc"`.

**DO NOT STOP HERE. Continue to segmentation.**

---

## Step 5b: MILo Training (preferred -- produces mesh directly)

If MILo conda environment is available, use it instead of LichtFeld + TSDF:

```bash
# Check if MILo is installed
conda run -n milo python -c "import torch; print('MILo OK')" 2>/dev/null

# If available, run MILo training + mesh extraction
python3 -c "
import sys; sys.path.insert(0, 'src')
from pipeline.milo_extractor import run_milo, is_milo_available, MiloConfig
if is_milo_available():
    result = run_milo(
        colmap_dir='/data/output/JOB_ID/colmap/undistorted',
        output_dir='/data/output/JOB_ID/model_milo',
        config=MiloConfig(imp_metric='indoor', mesh_config='default'),
    )
    print(result)
    if result['success']:
        print(f'MILo mesh: {result[\"mesh_path\"]}')
        # Skip to Step 9 (Blender assembly) -- no separate TSDF needed
else:
    print('MILo not available, using LichtFeld + TSDF pipeline')
"
```

MILo trains for 18K iterations with mesh-in-the-loop regularization.
The output mesh is via learned SDF extraction -- much higher quality than TSDF.
If MILo produced a mesh, **skip Steps 6-8** and go directly to Step 9 (Blender assembly).

Quality targets for MILo:
- Mesh should have 50K-500K vertices depending on mesh_config
- Vertex colors from gaussian splatting
- Clean surface topology (no TSDF lumpiness)

---

## Step 6: SAM3 Object Identification

SAM3 identifies freestanding objects using text prompts. HF_TOKEN is set for model download.
First run downloads ~2.4GB checkpoint (cached in /opt/hf-cache afterward).

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.segment(
    '/data/output/JOB_ID/model/splat_30000.ply',
    '/data/output/JOB_ID/frames_selected/'
)
print(result)
"
```

**INSPECT THE RESULTS.** Check: how many objects? What labels? Gaussian counts per object?
If SAM3 fell back to full_scene only, check HF_TOKEN, model download, BPE path.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "segment", "progress": 0.55, "message": "SAM3: N objects identified"}'
```

---

## Step 7: Per-Object Processing Loop

**For EACH identified object, run this sub-pipeline. You are iterating.**

### 7a: Extract per-object gaussian PLY
```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.extract_objects('/data/output/JOB_ID/model/splat_30000.ply', OBJECTS_FROM_STEP6)
print(result)
"
```

### 7b: Render object views with gsplat (orbit cameras -- correct for isolated objects)
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from pipeline.mesh_extractor import load_3dgs_ply, render_gsplat, generate_orbit_cameras_gsplat
import numpy as np
from PIL import Image

gs = load_3dgs_ply('/data/output/JOB_ID/objects/OBJECT_LABEL.ply')
cameras = generate_orbit_cameras_gsplat(gs['means'], 4, 1024)
for i, (vm, K) in enumerate(cameras):
    depth, rgb, alpha = render_gsplat(gs, vm, K, 1024, 1024)
    Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).save(
        f'/data/output/JOB_ID/objects/OBJECT_LABEL_view{i}.png')
print('Saved 4 orbit views')
"
```

### 7c: (Optional) Hunyuan Multi-View enhancement via ComfyUI
Submit the best render to ComfyUI Hunyuan MV workflow for consistent multi-view generation:
```bash
curl -s http://localhost:8188/prompt -X POST \
  -H 'Content-Type: application/json' \
  -d @/data/output/JOB_ID/workflows/hunyuan_mv_OBJECT.json
```

### 7d: Create mesh per object (orbit cameras for isolated objects)
```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.mesh_objects(['/data/output/JOB_ID/objects/OBJECT_LABEL.ply'])
print(result)
"
```

### 7e: Save object metadata (position, bbox, label, mesh path)
```bash
python3 -c "
import json, numpy as np
from plyfile import PlyData
ply = PlyData.read('/data/output/JOB_ID/objects/OBJECT_LABEL.ply')
v = ply['vertex']
meta = {
    'label': 'OBJECT_LABEL',
    'position': [float(np.mean(v['x'])), float(np.mean(v['y'])), float(np.mean(v['z']))],
    'bbox_min': [float(np.min(v['x'])), float(np.min(v['y'])), float(np.min(v['z']))],
    'bbox_max': [float(np.max(v['x'])), float(np.max(v['y'])), float(np.max(v['z']))],
    'gaussian_count': len(v),
    'mesh': '/data/output/JOB_ID/objects/meshes/OBJECT_LABEL/OBJECT_LABEL.glb',
}
json.dump(meta, open('/data/output/JOB_ID/objects/OBJECT_LABEL_meta.json', 'w'), indent=2)
print(json.dumps(meta, indent=2))
"
```

Report per-object progress:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "mesh_objects", "progress": 0.7, "message": "Object N/M: LABEL"}'
```

---

## Step 8: Build Empty Room

Remove identified objects from frames using inpainting, reconstruct clean room.

### 8a: Inpaint objects out of frames
Use SAM3 object masks + ComfyUI FLUX inpainting to remove objects from frames.
The inpainted frames go to `frames_inpainted/`.

### 8b: Reconstruct empty room
Re-run COLMAP + 3DGS training on inpainted frames. Reuse camera poses from step 4.

### 8c: Extract room mesh
Room mesh uses **COLMAP cameras** (interior scene), not orbit cameras.

---

## Step 9: Assemble Scene in Blender

The Blender assembler combines room mesh + object meshes at their original positions.
It bakes vertex colors to UV textures in 0.5s using Cycles GPU.

```bash
blender --background --python src/pipeline/blender_assembler.py -- \
    --input /data/output/JOB_ID/objects/meshes/full_scene/full_scene.glb \
    --output-usd /data/output/JOB_ID/usd/scene.usda \
    --output-renders /data/output/JOB_ID/previews/ \
    --render-size 1920x1080
```

For multi-object scenes, write a custom Blender script to import each object at position:
```python
import bpy, json
from pathlib import Path
JOB = '/data/output/JOB_ID'
for meta_file in sorted(Path(f'{JOB}/objects').glob('*_meta.json')):
    meta = json.load(open(meta_file))
    bpy.ops.import_scene.gltf(filepath=meta['mesh'])
    obj = bpy.context.selected_objects[0]
    obj.location = meta['position']
    obj.name = meta['label']
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "assemble_usd", "progress": 0.95, "message": "Scene: room + N objects"}'
```

---

## Step 10: Validate and Complete

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.validate()
print(result)
"
curl -X POST http://localhost:7860/api/job/JOB_ID/complete \
  -H 'Content-Type: application/json' \
  -d '{"success": true}'
```

---

## Orchestration Rules

### YOU ARE THE ORCHESTRATOR. Not a script runner.

1. **Inspect results between every step.** Check vertex counts, image quality, file sizes.
2. **If SAM3 finds objects**: Run the per-object loop (7a-7e) for EACH object.
3. **If SAM3 falls back to full_scene only**: Mesh the full scene, skip object loop.
4. **If quality is poor**: Re-run the stage with different parameters.
5. **Orbit cameras for isolated objects**, COLMAP cameras for full scenes/rooms.
6. **Save preview images at every stage** for the web carousel.
7. **Report progress via REST API** at each stage transition.
8. **Blender assembler bakes textures in 0.5s** via Cycles GPU -- always use it for final output.

### Quality Targets

| Stage | Target | Fail if |
|-------|--------|---------|
| COLMAP | >70% registration | <30% |
| Training | >10MB PLY, loss <0.02 | <1MB PLY |
| SAM3 | 2+ objects identified | (fallback to full_scene is OK) |
| Per-object mesh | >5K verts per object | <500 verts |
| Room mesh | >30K verts | <5K verts |
| MILo mesh | 50K-500K verts, vertex colors, clean topology | <10K verts or no mesh produced |
| Final USD | Room + objects with materials | Empty scene |

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `SAM3_BPE_PATH` | `/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | SAM3 text tokenizer |
| `HF_TOKEN` | (from .env) | HuggingFace model downloads |
| `HF_HOME` | `/opt/hf-cache` | Shared HF cache directory |

### REST API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/job/<id>/stage` | Report stage progress |
| POST | `/api/job/<id>/stage/complete` | Mark stage done |
| POST | `/api/job/<id>/complete` | Mark job done |
| GET | `/status/<id>` | Job detail |
| GET | `/api/job/<id>/previews` | List preview images |

### MANDATORY: Save preview images

The web UI carousel auto-detects PNG/JPG in the job output directory.
Save previews at: frame selection, training renders, depth maps, mesh renders, final scene.
