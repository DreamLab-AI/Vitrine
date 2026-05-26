# PRD v2: Upstream Sync & Pipeline Enhancement Upgrade

**Date**: 2026-05-26
**Status**: Draft
**Supersedes**: [prd.md](prd.md) (v1 pipeline architecture)

---

## 1. Executive Summary

LichtFeld Studio upstream has shipped ~410 commits since our fork diverged (2026-03-28), spanning two releases (v0.5.1, v0.5.2) plus 195 unreleased commits toward v0.5.3. Major changes include: native USD import/export, mesh support, MRNF densification, NanoGS, viewport splitting, a plugin marketplace, and — critically — a **complete migration from OpenGL/CUDA rendering to Vulkan** (the CUDA renderer has been removed entirely upstream). The Gaussian splatting field has also advanced with production-ready tools for real-time rendering, compact representations, 4D dynamic scenes, and web-optimized splat delivery.

This PRD defines a phased upgrade that:
1. Syncs the upstream C++ core (deciding between stable v0.5.2 tag or bleeding-edge master with Vulkan)
2. Integrates new tools that directly enhance our video-to-geometry pipeline
3. Extends the pipeline from static-only to dynamic scene support (4D)
4. Adds a web delivery path for splat output alongside USD/mesh

**Critical Decision**: Upstream master has removed the CUDA renderer (#1234) and migrated to Vulkan (#1170). Our pipeline uses LichtFeld via MCP for training (not rendering), so this may not break us, but the Vulkan transition needs careful testing. The safer path is to sync to the **v0.5.2 tag** first, then evaluate the Vulkan migration separately.

---

## 2. Current State Assessment

### 2.1 Our Fork (gaussian-toolkit branch)

| Component | Status | Version/Commit |
|-----------|--------|----------------|
| LichtFeld Studio core | Stale — hundreds of upstream commits behind | Pre-USD-export, pre-mesh-support |
| COLMAP | 4.1.0 (headless, CUDA sm_89) | Built from source |
| gsplat | Latest pip | CUDA 12.8 |
| MILo | Integrated (sidecar container) | SIGGRAPH Asia 2025 |
| CoMe | Research analysis complete, **code released 2026-04-22** | Ready for integration |
| SAM2/SAM3 | Integrated (segmentation) | pip + repo install |
| Blender | 5.0.1 | Scene assembly + Cycles bake |
| Pipeline modules | 28 Python modules in src/pipeline/ | Functional |
| Web UI | Flask on port 7860 | Basic but working |

### 2.2 Upstream LichtFeld Studio (origin/master)

Key features shipped since fork divergence:

| Feature | PR/Commit | Impact |
|---------|-----------|--------|
| **USD import/export** | #1032 | Native USD I/O — eliminates our custom USD assembly workaround |
| **Mesh support** | #876 | Native mesh loading, picking (#889), mesh-to-splat conversion (#879) |
| **MRNF densification** | #1031, rename | New densification strategy replacing LFS |
| **Rendering refactor** | #1000 | Major rendering pipeline overhaul |
| **Depth mode + box selection** | #1037 | Depth-based editing and selection tools |
| **Viewport splitting** | #1043 | Multi-viewport workflow |
| **Plugin marketplace** | #914 | Install/uninstall plugins, styling, git opt-in |
| **NanoGS** | #1014 | Compact Gaussian representation |
| **VRAM improvements** | Multiple | Reduced VRAM spikes during eval, image loading, IGS+ |
| **Scene graph performance** | Recent | Faster scene graph operations |
| **Improved undo/redo** | #990 | Robust undo system |
| **Enhanced MCP** | #984 | Hardened MCP server with more capabilities |
| **Sequencer/animation** | Multiple | Keyframe frustums, scrub controls, panoramic rendering |
| **Equirectangular rendering** | #902 | Seamless 360 rendering |
| **PPISP reuse** | #1015 | Point position interpolation optimization |
| **MCMC defaults fix** | #1046 | Corrected MCMC training defaults |
| **Native file dialog** | #1013 | OS-native file picker |

#### v0.5.2 (Released 2026-04-21)

| Feature | PR/Commit | Impact |
|---------|-----------|--------|
| **GUI Eval Mode** | #1118 | Benchmark evaluation from GUI |
| **USD export crash fix** (Windows) | #1112 | USD stability |
| **Plugin toolbar** | Multiple | Action-only plugins, toolbar support |
| **CLI progress improvements** | Multiple | Better headless progress reporting |
| **Color picker / eyedropper** | Multiple | Image preview tools |

#### v0.5.3-dev (Unreleased — BREAKING CHANGES)

| Feature | PR/Commit | Impact |
|---------|-----------|--------|
| **Vulkan migration** | #1170 | **BREAKING**: Complete OpenGL removal, Vulkan-only rendering |
| **CUDA renderer removed** | #1234 | **BREAKING**: No more CUDA rendering backend |
| **VkSplat (FastGS ported)** | #1162 | Vulkan-based VRAM-efficient splatting |
| **Depth composition (mesh+splat)** | #1197 | Vulkan depth compositing |
| **UI redesign (RmlUi)** | Multiple | ImGui → RmlUi migration |
| **Asset Manager** | #1166, #1200, #1222-#1226 | Thumbnails, list view, URL import |
| **TCP event server** | #1231 | External app connectivity via JSON/TCP |
| **CLI background mode** | #1246 | Headless initialization arguments |
| **Selection tools (Vulkan)** | #1194 | Lasso, brush, frustum, box selection |
| **In-memory masks** | #1236 | Direct-scene plugin data |
| **8K image support** | #1164 | Large format training |
| **Mesh2splat CLI** | Multiple | Mesh-to-splat conversion |
| **COLMAP data export** | Multiple | COLMAP integration |
| **Training speed regression fix** | #1232 | Fixes issue #1169 |
| **Coordinate system cleanup** | #1066 | **CAUTION**: May affect coordinate_transform.py |

### 2.3 What's Missing from v1 Pipeline

| Gap | Impact |
|-----|--------|
| No dynamic scene support | Cannot reconstruct scenes with motion |
| No real-time rendering path | Splats rendered offline only |
| No web-optimized delivery | Large PLY files, no compression |
| CoMe not integrated | Still using slower MILo for mesh extraction |
| No splat editing/manipulation | Can't warp or deform reconstructed scenes |
| Single-camera-model COLMAP | No advanced camera patterns |

---

## 3. Technology Assessment

### 3.1 Must-Integrate (High-value, production-ready)

#### 3.1.1 Upstream LichtFeld Studio Sync

**What**: Rebase/merge ~1,400 upstream commits into our fork.

**Key wins**:
- **Native USD import/export** (#1032) replaces our `usd_assembler.py` with built-in C++ USD I/O
- **Mesh support** (#876, #879, #889) gives us native mesh loading, editing, and mesh↔splat conversion inside the viewer
- **MRNF densification** (#1031) is the new default densification strategy, likely better quality than our current gsplat DefaultStrategy
- **Enhanced MCP** (#984) provides more automation tools for our agentic pipeline
- **VRAM optimizations** reduce GPU memory pressure, critical for our dual-GPU setup
- **Plugin marketplace** (#914) enables distributing pipeline extensions as LichtFeld plugins

**Risk**: BOUNDARIES.md says we don't modify upstream dirs, and conflicts should resolve in favor of upstream. The sync should be clean if we've maintained that boundary.

**Effort**: Medium (1-2 weeks). Mostly a `git merge origin/master` with conflict resolution in build files.

#### 3.1.2 splat-transform (PlayCanvas)

**Repo**: https://github.com/playcanvas/splat-transform
**What**: JavaScript/CLI library for transforming .ply splat files — compress, convert, crop, filter, reorient, sort.

**Pipeline value**:
- **Compress** trained splats before web delivery (significant size reduction)
- **Convert** between PLY and compressed splat formats (.splat, .ksplat)
- **Crop** splats to region of interest (removes sky/ground noise)
- **Sort** Gaussians for optimal rendering order
- **Filter** by opacity/scale to remove floaters post-training

**Integration point**: Post-training stage, before web delivery. Add as a new pipeline module `splat_optimizer.py`.

**Maturity**: Production-ready. PlayCanvas ships this in their engine. Active maintenance, JS/Node toolchain.

**Effort**: Low (2-3 days). npm install, CLI wrapper in pipeline.

#### 3.2.2 RT-Splatting (Reflection & Transmission)

**Repo**: https://github.com/sjj118/RT-Splatting (83 stars, MIT license, CVPR 2026 Highlight)
**What**: Hybrid surface-volume rendering for semi-transparent scenes. "RT" = Reflection & Transmission (not "Real-Time"). Decomposes scenes into independent reflection and transmission layers.

**Pipeline value**:
- Handles glass, water, transparent barriers, reflective surfaces correctly
- Standard 3DGS produces ghosting/floaters on these — RT-Splatting solves this
- Layer decomposition separates reflections during mesh extraction
- Enables material editing post-reconstruction

**Integration point**: Alternative training mode for scenes with transparency. Selected via scene-type flag.

**Maturity**: Research code, CVPR 2026 Highlight. Actively maintained (pushed 2026-05-20). MIT license.

**Effort**: Medium (1 week). Builds on 2DGS, requires nvdiffrast. Niche — only for transparent/reflective scenes.

### 3.2 Should-Integrate (High-value, requires effort)

#### 3.1.4 CoMe — Confidence-Based Mesh Extraction

**Repo**: https://github.com/r4dl/CoMe (61 stars, **code released 2026-04-22**)
**What**: Trains 3DGS with per-Gaussian confidence values, then extracts meshes via marching tetrahedra. 3x faster than MILo with comparable or better quality.

**Pipeline value**:
- **25 min total** (train + extract) vs MILo's **69 min** on RTX 4090
- Confidence mechanism reduces floater artifacts
- F1 scores: 0.521 on Tanks & Temples, 0.662 on ScanNet++
- Accepts standard COLMAP dataset format (images/ + sparse/) — exactly our intermediate format
- Could replace MILo as primary mesh extraction path

**Current status**: **Code now available** (initial release 2026-04-22). Our `research/come/` analysis was written when code was still placeholder and needs updating.

**Action** (immediate):
1. Clone and test in isolated conda env (Python 3.10, CUDA 12.1)
2. Benchmark against MILo on our test scenes
3. Integrate as third mesh extraction backend (alongside TSDF and MILo)
4. Needs its own sidecar container (Python 3.10, CUDA 12.1 — different from both main and MILo)

**Effort**: Medium-High (1-2 weeks). Separate container, new pipeline module `come_extractor.py`.

#### 3.1.5 GaussianWrapping — Thin-Structure Mesh Extraction

**Repo**: https://github.com/diego1401/GaussianWrapping (187 stars, actively developed, pushed 2026-05-19)
**What**: Reinterprets 3D Gaussians as "stochastic oriented surface elements" to extract watertight, textured meshes that capture extremely thin structures (bicycle spokes, wires, fences, railings).

**Pipeline value**:
- Handles thin structures that TSDF and standard marching cubes fail on
- Produces meshes significantly smaller than competing approaches
- Accepts COLMAP-formatted datasets (images + sparse reconstruction)
- Two rasterization backends: RaDeGS (quality) and custom median-depth (speed)
- "Primal Adaptive Meshing" for targeted high-resolution extraction
- **Can share MILo sidecar container** (both support CUDA 11.8)

**Integration point**: Fourth mesh extraction backend. Selected when scene contains fine geometric detail.

**Maturity**: Research code, actively maintained. 187 stars (more than CoMe). No formal license file.

**Effort**: Medium (1 week). Can share MILo sidecar container (CUDA 11.8, Python 3.9).

#### 3.2.2 4C4D — 4D Gaussian Splatting

**Repo**: https://github.com/yangzf-1023/4C4D
**What**: Extends 3DGS to handle dynamic scenes (4D: 3 spatial + 1 temporal). Reconstructs scenes with motion from multi-view video.

**Pipeline value**:
- Enables reconstruction of **scenes with moving objects** (people, vehicles, machinery)
- Currently a v1 non-goal, but high user demand
- Temporal decomposition separates static background from dynamic foreground

**Integration point**: Alternative training path when input video contains motion. Replaces standard 3DGS training.

**Maturity**: Research code. Active development.

**Effort**: High (2-3 weeks). New training module, temporal segmentation logic, different output format.

#### 3.2.3 2Xplat — Cross-Platform Gaussian Splatting

**Repo**: https://github.com/HwasikJeong/2Xplat
**What**: Optimized cross-platform Gaussian splatting renderer/viewer. Runs on CPU+GPU with platform-specific optimizations.

**Pipeline value**:
- Web viewer component for splat delivery
- Mobile-compatible rendering (extends our reach beyond desktop)
- Could replace model-viewer for splat preview in web UI

**Integration point**: Web delivery stage. Replace or supplement the current Flask + model-viewer frontend.

**Maturity**: Working code, actively developed.

**Effort**: Medium (1 week). JavaScript integration, web UI changes.

### 3.3 Nice-to-Have (Research / Niche)

#### 3.3.1 GaussianWrapping

**Repo**: https://github.com/diego1401/GaussianWrapping
**What**: Gaussian-based scene wrapping and deformation. Enables non-rigid transformation of reconstructed splat scenes.

**Pipeline value**:
- Scene editing: warp/deform reconstructed environments
- Object manipulation: resize, reposition, reshape extracted objects
- Relevant to the "roger project" and contract work

**Integration point**: Post-reconstruction editing tool. Adds scene manipulation capability.

**Maturity**: Research code. Novel technique.

**Effort**: Medium (1 week). Python module, integration with scene editor.

#### 3.3.2 GenRecon

**Page**: https://kasothaphie.github.io/GenRecon/
**What**: Generative 3D reconstruction from sparse views using diffusion priors. Hallucinates plausible geometry for unseen regions.

**Pipeline value**:
- Handles **few-view scenarios** where COLMAP struggles (3-10 images instead of 50+)
- Scene completion: fills in areas the camera didn't capture
- Currently a v1 non-goal (scene completion) but useful for imperfect captures

**Maturity**: Research paper. Limited code availability.

**Effort**: High. Diffusion model dependencies, uncertain quality.

#### 3.3.3 COLMAP Bridge (Cinema 4D)

**Source**: https://radiancefields.com/colmap-bridge-adds-fibonacci-camera-jittered-aa-and-octane-redshift-baking
**What**: Cinema 4D plugin adding Fibonacci camera patterns, jittered anti-aliasing, and Octane/Redshift baking for synthetic COLMAP datasets.

**Pipeline value**:
- Fibonacci camera patterns provide mathematically optimal viewpoint distribution
- Jittered AA reduces aliasing in rendered training data
- Relevant if generating synthetic training data from C4D scenes

**Integration point**: Not directly applicable to our pipeline (we process real video, not C4D renders). However, the **Fibonacci camera pattern** concept could improve our frame selection algorithm (`frame_selector.py`) for optimal viewpoint coverage.

**Action**: Extract the Fibonacci sphere sampling algorithm and apply it to our frame selection scoring.

**Effort**: Low (1-2 days). Algorithm port to frame selector.

#### 3.3.4 Moshpit360

**What**: 360-degree multi-view capture and reconstruction workflow. Limited public information available.

**Pipeline value**: Relevant if supporting 360/VR capture workflows.

**Maturity**: Early stage. Insufficient information for integration assessment.

**Action**: Monitor for code release and documentation.

---

## 4. Upgrade Architecture

### 4.1 Phase 1: Upstream Sync (Week 1-2)

```
                    ┌─────────────────────────────────────┐
                    │   git merge origin/master            │
                    │   Resolve conflicts (favor upstream) │
                    │   Rebuild Docker image               │
                    │   Verify MCP server still works      │
                    │   Test pipeline end-to-end           │
                    └─────────────────────────────────────┘
```

**Deliverables**:
- [ ] Merged upstream into gaussian-toolkit branch
- [ ] Dockerfile.consolidated updated for any new build deps
- [ ] All 28 pipeline modules verified working
- [ ] LichtFeld native USD export tested and compared with our usd_assembler.py
- [ ] LichtFeld native mesh support tested with pipeline meshes
- [ ] MRNF densification benchmarked against DefaultStrategy

**Key decision**: After merge, evaluate whether `usd_assembler.py` should be deprecated in favor of LichtFeld's native USD I/O. If native USD I/O covers our needs (hierarchical scene with per-object prims), deprecate our module.

### 4.2 Phase 2: Pipeline Enhancement (Week 3-4)

```
Video → Frames → COLMAP SfM → 3DGS Training (MRNF) →
  → Object Segmentation (SAM3) →
  → Mesh Extraction (MILo | CoMe when available) →
  → Splat Optimization (splat-transform) →          ← NEW
  → Blender Assembly + Texture Bake →
  → USD Scene + Compressed Splat + Web Viewer        ← ENHANCED
```

**New pipeline modules**:

| Module | Purpose | Depends On |
|--------|---------|------------|
| `splat_optimizer.py` | Compress, crop, sort splats via splat-transform CLI | Node.js, splat-transform |
| `fibonacci_sampler.py` | Fibonacci-sphere frame selection scoring | numpy (no new deps) |
| `rt_depth_renderer.py` | RT-Splatting depth maps for TSDF | RT-Splatting CUDA build |

**Modified pipeline modules**:

| Module | Change |
|--------|--------|
| `frame_selector.py` | Add Fibonacci-sphere scoring for viewpoint coverage |
| `config.py` | Add splat-transform, RT-Splatting, training strategy options |
| `orchestrator.py` | Add splat optimization stage, training strategy selection |
| `stages.py` | New stage enum: SPLAT_OPTIMIZE |
| `mesh_extractor.py` | Option to use RT-Splatting depth for higher-quality TSDF |

**Docker changes**:
- Add Node.js splat-transform to Dockerfile.consolidated (npm install)
- Add RT-Splatting CUDA build (optional, build arg)

### 4.3 Phase 3: 4D / Dynamic Scenes (Week 5-8)

```
Video with Motion → Frames → COLMAP SfM →
  → Motion Detection (optical flow / SAM2 tracking) →     ← NEW
  → Static/Dynamic Split →                                ← NEW
  → Static: 3DGS Training → Mesh → USD                    (existing)
  → Dynamic: 4C4D Training → Temporal Splats → Alembic    ← NEW
  → Scene Merge → USD Scene with animated prims            ← NEW
```

**New pipeline modules**:

| Module | Purpose |
|--------|---------|
| `motion_detector.py` | Detect motion in video via optical flow + SAM2 temporal tracking |
| `scene_splitter.py` | Separate static background from dynamic foreground |
| `dynamic_trainer.py` | 4C4D training for dynamic regions |
| `temporal_exporter.py` | Export dynamic splats as Alembic point caches |

**New container**: 4C4D likely needs its own sidecar (different PyTorch/CUDA version), similar to MILo pattern.

### 4.4 Phase 4: Web Delivery (Week 5-6, parallel with Phase 3)

```
Trained Splats → splat-transform compress →
  → 2Xplat viewer embed OR model-viewer + KHR_gaussian_splatting →
  → Web preview with orbit controls, quality toggle, download
```

**Enhancements to web UI**:
- Replace model-viewer mesh preview with splat-native viewer (2Xplat or SuperSplat)
- Add quality toggle (compressed splat vs full PLY)
- Progressive loading for large scenes
- Support KHR_gaussian_splatting glTF extension for interop

---

## 5. Upstream Sync Plan (Detailed)

### 5.0 Critical Decision: v0.5.2 Tag vs. Bleeding-Edge Master

Upstream master has undergone a **complete rendering backend migration**:
- OpenGL renderer → **removed** (#1170)
- CUDA renderer → **removed** (#1234)
- Vulkan renderer → **only option** (VkSplat, #1162)

**Option A: Sync to v0.5.2 tag (Recommended)**
- Stable release (2026-04-21)
- Still has CUDA/OpenGL renderer
- Gets us: USD import/export, mesh support, MRNF, NanoGS, viewport splitting
- Does NOT get: Vulkan backend, asset manager, TCP event server, 8K images
- Lower risk, proven stable

**Option B: Sync to master (Bleeding Edge)**
- Gets everything including Vulkan, asset manager, TCP server
- Risk: Vulkan migration is still in progress (v0.5.3 not released)
- Risk: Coordinate system cleanup (#1066) may break our coordinate_transform.py
- Risk: Python API changes may break our mcp_client.py
- Risk: ImGui → RmlUi migration may affect any UI interactions

**Recommendation**: Start with **v0.5.2 tag sync**, then evaluate the Vulkan migration as a separate task once stable.

### 5.1 Pre-Merge Checklist

```bash
# Add upstream remote (currently missing!)
git remote add upstream https://github.com/MrNeRF/LichtFeld-Studio.git
git fetch upstream

# Check divergence from v0.5.2 tag
git log --oneline v0.5.2..main | wc -l    # our commits ahead
git log --oneline main..v0.5.2 | wc -l    # upstream commits we'd gain

# Identify conflict zones
git diff --name-only main...v0.5.2 | sort

# Later, evaluate master for Vulkan:
git diff --name-only v0.5.2...upstream/master | sort
```

### 5.2 Merge Strategy

1. **Create sync branch**: `git checkout -b sync/upstream-v0.5.2 main`
2. **Merge upstream tag**: `git merge v0.5.2 --no-ff -m "Sync upstream LichtFeld Studio v0.5.2"`
3. **Conflict resolution rules** (per BOUNDARIES.md):
   - Upstream directories (src/core/, src/app/, src/mcp/, etc.) → **accept upstream**
   - Our directories (src/pipeline/, src/web/, docker/) → **keep ours**
   - Build files (CMakeLists.txt, vcpkg.json) → **accept upstream**, re-add our additions
   - README.md → **keep ours**
   - .gitignore → **merge both**
4. **Test build**: Rebuild C++ core, verify MCP server starts
5. **Test MCP client**: Verify `mcp_client.py` still works with updated MCP server (#984)
6. **Test pipeline**: Run full video-to-USD pipeline
7. **PR into main**: Review, merge

### 5.2.1 Known Risk Areas

| File | Risk | Action |
|------|------|--------|
| `src/pipeline/mcp_client.py` | MCP API may have changed (#984) | Test all MCP calls post-merge |
| `src/pipeline/coordinate_transform.py` | Coordinate cleanup (#1066) only in master, not v0.5.2 | Safe for v0.5.2 sync |
| `CMakeLists.txt` | New deps, Vulkan (master only) | Accept upstream for v0.5.2 |
| `docker/Dockerfile` | Upstream updated (#931) | Keep our Dockerfile.consolidated |

### 5.3 Post-Merge Validation

| Test | Pass Criteria |
|------|---------------|
| LichtFeld binary starts | MCP server responds on port 45677 |
| Pipeline end-to-end | Video → USD scene completes without error |
| Native USD export | LichtFeld exports .usda/.usdc with correct hierarchy |
| Mesh import/export | Load mesh, convert to splat, re-export |
| MRNF densification | Training completes, quality ≥ DefaultStrategy |
| Web UI | Upload, process, preview, download all work |
| Docker build | Both containers build and start |

---

## 6. Technology Integration Specs

### 6.1 splat-transform Integration

```python
# src/pipeline/splat_optimizer.py

class SplatOptimizer:
    """Compress and optimize trained Gaussian splats for delivery."""

    def optimize(self, input_ply: Path, output_dir: Path, config: SplatOptConfig) -> SplatOptResult:
        """
        Pipeline: input.ply → crop → filter → sort → compress → output.ksplat

        Steps:
        1. Crop: Remove Gaussians outside scene bounding box
        2. Filter: Remove Gaussians with opacity < threshold or scale > max_scale
        3. Sort: Reorder for front-to-back rendering
        4. Compress: Quantize SH coefficients, half-precision positions
        """
```

**CLI integration**:
```bash
npx @playcanvas/splat-transform compress input.ply -o output.ksplat
npx @playcanvas/splat-transform crop input.ply --box "-10,-10,-10,10,10,10" -o cropped.ply
```

### 6.2 Fibonacci Frame Selection

```python
# Enhancement to src/pipeline/frame_selector.py

def fibonacci_sphere_score(camera_positions: np.ndarray, n_target: int) -> np.ndarray:
    """Score frames by coverage of a Fibonacci sphere distribution.

    The Fibonacci sphere provides near-optimal uniform distribution of
    viewpoints. Frames closest to Fibonacci points score highest.
    """
    # Generate Fibonacci sphere points
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_target)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_target)
    fib_points = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=-1)

    # Normalize camera positions to unit sphere
    center = camera_positions.mean(axis=0)
    dirs = camera_positions - center
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Score: for each frame, minimum angular distance to nearest Fibonacci point
    # Higher score = closer to an under-covered Fibonacci direction
    ...
```

### 6.3 CoMe Integration (Deferred)

When code releases, integrate as:

```yaml
# docker-compose.consolidated.yml addition
  come:
    build:
      context: ./docker
      dockerfile: Dockerfile.come
    image: gaussian-toolkit-come:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/data
    # Python 3.10, CUDA 12.1, SOF-based environment
```

```python
# src/pipeline/come_extractor.py
class CoMeExtractor:
    """Mesh extraction via CoMe confidence-based Gaussian training."""

    def extract(self, colmap_dir: Path, output_dir: Path) -> MeshResult:
        # 1. Train CoMe Gaussians with confidence (~18 min on 4090)
        # 2. Extract mesh via marching tetrahedra (~7 min)
        # 3. Return geometry-only PLY mesh
        # 4. Separate texturing pass needed (xatlas + reprojection)
```

### 6.4 GaussianWrapping Integration

```python
# src/pipeline/gaussianwrapping_extractor.py

class GaussianWrappingExtractor:
    """Mesh extraction via GaussianWrapping — specialized for thin structures."""

    def extract(self, colmap_dir: Path, output_dir: Path, config: GWConfig) -> MeshResult:
        """
        Uses GaussianWrapping's stochastic surface element interpretation.
        Particularly effective for: bicycle spokes, wires, fences, railings,
        thin architectural elements.

        Can share MILo sidecar container (both use CUDA 11.8).
        Two rasterization backends:
        - RaDeGS: Higher quality, slower
        - Median-depth: Faster, good for previews
        """
```

**Docker integration**: Add GaussianWrapping to MILo sidecar Dockerfile:
```dockerfile
# In docker/Dockerfile.milo — append after MILo install
RUN git clone https://github.com/diego1401/GaussianWrapping.git /opt/gaussianwrapping && \
    cd /opt/gaussianwrapping && \
    python install.py --cuda_version 11.8
```

### 6.5 4C4D Dynamic Scene Integration (Phase 3)

```python
# src/pipeline/motion_detector.py
class MotionDetector:
    """Detect and classify motion in input video."""

    def analyze(self, frames_dir: Path) -> MotionAnalysis:
        # Optical flow between consecutive frames
        # SAM2 object tracking for persistent motion regions
        # Classify: static_background, dynamic_foreground, camera_motion_only

# src/pipeline/dynamic_trainer.py
class DynamicTrainer:
    """4C4D training for dynamic scene regions."""

    def train(self, colmap_dir: Path, motion_masks: Dict, output_dir: Path) -> DynamicResult:
        # Train 4C4D on frames with motion annotations
        # Output: temporal Gaussian sequence
        # Export as Alembic point cache for USD integration
```

---

## 7. Priority Matrix

| Priority | Component | Value | Effort | Risk | Timeline |
|----------|-----------|-------|--------|------|----------|
| **P0** | Upstream sync (v0.5.2) | Critical — unlocks USD, mesh, MRNF, VRAM fixes | Medium | Low (BOUNDARIES.md protects us) | Week 1-2 |
| **P1** | splat-transform | High — web delivery, format conversion, LOD | Low | Very low (npm CLI wrapper) | Week 3 |
| **P1** | Fibonacci frame selection | Medium — better viewpoint coverage | Low | Very low (algorithm only) | Week 3 |
| **P1** | MRNF densification in pipeline | High — better training quality | Low | Low (upstream has it) | Week 2 (post-sync) |
| **P1** | CoMe mesh extraction | High — 3x faster than MILo, **code now available** | Medium-High | Medium (sidecar container) | Week 3-4 |
| **P1** | GaussianWrapping | High — thin-structure meshes, can share MILo sidecar | Medium | Medium (CUDA build) | Week 3-4 |
| **P2** | RT-Splatting | Medium — transparent/reflective scenes only | Medium | Medium (niche use case) | Week 5 |
| **P2** | 2Xplat web viewer | Medium — better web preview | Medium | Low | Week 5 |
| **P3** | Vulkan migration (v0.5.3) | High — future-proofing | High | High (breaking change) | Week 6+ |
| **P3** | 4C4D dynamic scenes | High — new capability | High | High (research code) | Week 7-10 |
| **P4** | GenRecon sparse-view | Low-Medium — niche, **no code** | High | High (research) | Watch |
| **P4** | 2Xplat (pose-free) | Low-Medium — **no code** | Unknown | High | Watch |
| **P4** | Moshpit360 | Low — **no public info found** | Unknown | High | Watch |
| **P4** | COLMAP Bridge (C4D) | Low — C4D-specific, Fibonacci concept portable | Low | Low | Extract algorithm only |

---

## 8. Updated Pipeline Architecture (Post-Upgrade)

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                                   │
│  MP4/MOV from phone, drone, camera, 360 rig                    │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME EXTRACTION + QUALITY ASSESSMENT                          │
│  PyAV extraction → Fibonacci-sphere viewpoint scoring     [NEW] │
│  → blur/exposure quality gates                                  │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MOTION ANALYSIS (Phase 3)                               [NEW]  │
│  Optical flow + SAM2 tracking → static/dynamic split            │
│  Skip if no significant motion detected                         │
└─────────────┬──────────────────────┬────────────────────────────┘
              │ Static               │ Dynamic
              ▼                      ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│  COLMAP SfM          │  │  4C4D Training (Phase 3)       [NEW] │
│  Camera poses +      │  │  Temporal Gaussian sequence          │
│  sparse point cloud  │  │  → Alembic point cache               │
└─────────────┬────────┘  └──────────────────┬───────────────────┘
              │                              │
              ▼                              │
┌─────────────────────────────────────────┐  │
│  3DGS TRAINING                          │  │
│  LichtFeld (MRNF densification)   [UPG] │  │
│  OR gsplat (DefaultStrategy)            │  │
│  → Trained Gaussian PLY (~1M splats)    │  │
└─────────────┬───────────────────────────┘  │
              │                              │
              ▼                              │
┌─────────────────────────────────────────┐  │
│  SPLAT OPTIMIZATION                [NEW] │  │
│  splat-transform: crop, filter, sort,   │  │
│  compress → .ksplat for web delivery    │  │
└─────────────┬───────────────────────────┘  │
              │                              │
              ▼                              │
┌─────────────────────────────────────────┐  │
│  OBJECT SEGMENTATION                    │  │
│  SAM3 (text+visual, 4M concepts)        │  │
│  → Per-object 2D masks → 3D labels      │  │
└─────────────┬───────────────────────────┘  │
              │                              │
              ▼                              │
┌─────────────────────────────────────────┐  │
│  MESH EXTRACTION                        │  │
│  TSDF (fast preview)                    │  │
│  | MILo (general high-quality)          │  │
│  | CoMe (3x faster, confidence)   [NEW] │  │
│  | GaussianWrapping (thin struct) [NEW] │  │
│  → Per-object GLB meshes                │  │
└─────────────┬───────────────────────────┘  │
              │                              │
              ▼                              │
┌─────────────────────────────────────────┐  │
│  SCENE ASSEMBLY                         │  │
│  LichtFeld native USD I/O         [UPG] │  │
│  OR Blender (Cycles GPU bake)           │  │
│  → Textured USD scene                   │  │
│  + Compressed splat (.ksplat)     [NEW] │  │
│  + Alembic dynamic prims (Phase 3)     ◄──┘
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  WEB DELIVERY                                                   │
│  Splat viewer (2Xplat / SuperSplat)                       [NEW] │
│  + Mesh viewer (model-viewer)                                   │
│  + Download ZIP (USD, PLY, GLB, .ksplat)                        │
│  + KHR_gaussian_splatting glTF export                     [NEW] │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Updated Docker Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  docker-compose.consolidated.yml                                       │
│                                                                        │
│  ┌─────────────────────────────────────┐  ┌──────────────────────────┐ │
│  │  gaussian-toolkit (main)            │  │  milo (sidecar)          │ │
│  │  Ubuntu 24.04, CUDA 12.8            │  │  Ubuntu 22.04, CUDA 11.8│ │
│  │  Python 3.12                        │  │  Python 3.9              │ │
│  │                                     │  │                          │ │
│  │  + LichtFeld Studio v0.5.2   [UPG] │  │  MILo mesh extraction    │ │
│  │  + COLMAP 4.1.0                     │  │  + GaussianWrapping [NEW]│ │
│  │  + gsplat                           │  │  (shares CUDA 11.8 env)  │ │
│  │  + SAM2/SAM3                        │  └──────────────────────────┘ │
│  │  + Blender 5.0.1                    │                               │
│  │  + ComfyUI                          │  ┌──────────────────────────┐ │
│  │  + Node.js + splat-transform  [NEW] │  │  come (sidecar)    [NEW] │ │
│  │  + Pipeline (30+ modules)           │  │  Ubuntu 22.04, CUDA 12.1│ │
│  │  + Flask web UI                     │  │  Python 3.10             │ │
│  │  + Claude Code                      │  │  CoMe mesh extraction    │ │
│  └─────────────────────────────────────┘  └──────────────────────────┘ │
│                                                                        │
│                                           ┌──────────────────────────┐ │
│                                           │  4c4d (sidecar, Phase 3) │ │
│                                           │  Dynamic scene training  │ │
│                                           └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Success Metrics (v2)

| Metric | v1 Target | v2 Target |
|--------|-----------|-----------|
| Input | Video (MP4/MOV) | Video + 360 video |
| Output | USD scene | USD + compressed splat + KHR glTF |
| Training strategy | gsplat DefaultStrategy | MRNF densification (upstream) |
| Mesh extraction speed | ~69 min (MILo) | ~25 min (CoMe) or thin-struct via GaussianWrapping |
| Mesh extraction backends | 2 (TSDF, MILo) | 4 (TSDF, MILo, CoMe, GaussianWrapping) |
| Web delivery size | Raw PLY (100+ MB) | Compressed .ksplat (<20 MB) |
| Frame selection | Sequential sampling | Fibonacci-sphere optimal coverage |
| Dynamic scenes | Not supported | 4C4D temporal reconstruction (Phase 3) |
| Runtime (60s video) | <2 hours | <1.5 hours (MRNF + splat-transform) |
| USD I/O | Custom Python assembler | Native LichtFeld C++ USD I/O |

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Upstream merge conflicts in build files | Medium | Medium | BOUNDARIES.md separation; test incrementally |
| CoMe code never releases | Medium | Low | MILo remains viable; not blocking |
| 4C4D quality insufficient for production | High | Medium | Phase 3 is optional; static pipeline remains primary |
| RT-Splatting CUDA version incompatibility | Medium | Low | Optional component; TSDF works without it |
| LichtFeld native USD doesn't support our hierarchy | Low | Medium | Fall back to usd_assembler.py; extend upstream |
| splat-transform format not widely supported | Low | Low | PLY remains primary; .ksplat is bonus |

---

## 12. Implementation Schedule

```
Week 1-2:  ████████████████  Phase 1: Upstream Sync (v0.5.2)
                              - Add upstream remote, merge to v0.5.2 tag
                              - Rebuild Docker, verify pipeline
                              - Test native USD I/O, mesh support, MRNF
                              - Test MCP client compatibility

Week 3:    ████████          Phase 2a: Quick Wins
                              - splat-transform integration (npm install)
                              - Fibonacci frame selection algorithm
                              - MRNF as default training strategy

Week 3-4:  ████████████████  Phase 2b: Mesh Extraction Backends
                              - CoMe sidecar container (CUDA 12.1, Python 3.10)
                              - GaussianWrapping in MILo sidecar (CUDA 11.8)
                              - Benchmark all 4 backends: TSDF vs MILo vs CoMe vs GW
                              - Auto-selection logic based on scene type

Week 5:    ████████          Phase 2c: Niche Enhancements
                              - RT-Splatting for transparent/reflective scenes
                              - Pipeline config: scene-type flags

Week 5-6:  ████████████████  Phase 4: Web Delivery (parallel)
                              - 2Xplat or SuperSplat viewer
                              - KHR_gaussian_splatting glTF export
                              - Progressive loading, LOD via splat-transform

Week 6+:   ░░░░░░░░░░░░░░░░  Phase 5: Vulkan Migration (v0.5.3)
                              - Evaluate when v0.5.3 releases
                              - Test Vulkan-only rendering in container
                              - Update coordinate_transform.py if needed

Week 7-10: ████████████████████████████████  Phase 3: 4D Dynamic
                              - Motion detection module
                              - Scene splitter
                              - 4C4D sidecar container
                              - Alembic export + USD merge
```

---

## 13. Open Questions

1. **v0.5.2 vs master**: Do we sync to the stable v0.5.2 tag or wait for v0.5.3 with Vulkan? Recommendation: v0.5.2 first.
2. **Vulkan in Docker**: When we do adopt v0.5.3, does headless Vulkan rendering work in our Docker container? May need vulkan-tools and mesa-vulkan-drivers. Our pipeline uses MCP for training (not rendering), so training may still work, but any rendering call will need Vulkan.
3. **LichtFeld native USD**: Does it support hierarchical scenes with per-object prims, or only flat export? Need to test post-merge.
4. **MRNF vs MCMC**: Upstream renamed LFS to MRNF and fixed MCMC defaults. Which should be our default training strategy? Needs benchmarking.
5. **NanoGS**: Upstream added NanoGS (#1014). Is this useful for our pipeline (compact representation for web delivery)?
6. **Plugin marketplace**: Should we distribute our pipeline as a LichtFeld plugin via the marketplace?
7. **CoMe license**: No LICENSE file in repo (marked NOASSERTION). SOF uses a custom license. Evaluate commercial use implications before shipping.
8. **GaussianWrapping license**: No formal license file. Evaluate before production use.
9. **Mesh backend auto-selection**: With 4 backends (TSDF, MILo, CoMe, GaussianWrapping), how do we auto-select? Proposed heuristic: TSDF for previews, CoMe for speed, GaussianWrapping for thin structures, MILo for general high-quality.
10. **4C4D scope**: Is full 4D reconstruction too ambitious for v2? Consider limiting to "static background extraction from dynamic video" as a simpler first step.
11. **Moshpit360**: No public information found under this name. Need to clarify source/spelling.
12. **Coordinate system**: Upstream PR #1066 (in master, not v0.5.2) caused issues (issue #1104: "ERP + GUT training produces degenerate flat-plane output"). Our `coordinate_transform.py` may need updates when we eventually sync to master.
13. **Python API changes**: Multiple upstream fixes for "stale python api" and "python training interop". Our `mcp_client.py` may need updates.
14. **TCP event server**: Upstream added TCP event broadcasting (#1231). Could we use this instead of MCP for training progress monitoring?

---

## Appendix A: Upstream Commits of Interest

| Commit | Feature | Relevance |
|--------|---------|-----------|
| `08efc08d` | USD import/export (#1032) | **Critical** — native USD I/O |
| `e5cf619f` | Mesh support (#876) | **Critical** — native mesh handling |
| `673d6614` | Mesh-to-splat (#879) | High — bidirectional mesh↔splat |
| `3b974d69` | Mesh picking (#889) | High — interactive mesh selection |
| `b418dc68` | LFS/MRNF densification (#1031) | High — new training strategy |
| `f6f46492` | Rendering refactor (#1000) | High — major rendering overhaul |
| `0cb2f99d` | Depth mode + box selection (#1037) | High — depth-based tools |
| `5067a06b` | Viewport splitting (#1043) | Medium — multi-viewport |
| `250b8337` | Plugin Marketplace (#914) | Medium — plugin ecosystem |
| `120dd6f6` | NanoGS (#1014) | Medium — compact representation |
| `ea31c556` | Enhanced MCP (#984) | Medium — better automation |
| `8e22be84` | Improved undo/redo (#990) | Medium — robustness |
| `c8fec482` | Equirectangular rendering (#902) | Medium — 360 support |
| `2c41d00f` | Fix MCMC defaults (#1046) | Low — correctness fix |

## Appendix B: Technology Repository Links

| Technology | Repository | Status |
|------------|-----------|--------|
| LichtFeld Studio (upstream) | https://github.com/MrNeRF/LichtFeld-Studio | Active, ~1400 commits ahead |
| CoMe | https://github.com/r4dl/CoMe | **Code released 2026-04-22**, 61 stars |
| GenRecon | https://kasothaphie.github.io/GenRecon/ | Research page |
| RT-Splatting | https://github.com/sjj118/RT-Splatting | Research code |
| splat-transform | https://github.com/playcanvas/splat-transform | Production-ready |
| COLMAP Bridge | radiancefields.com (C4D plugin) | C4D-specific |
| GaussianWrapping | https://github.com/diego1401/GaussianWrapping | Active research, 187 stars, pushed 2026-05-19 |
| 4C4D | https://github.com/yangzf-1023/4C4D | Research code |
| 2Xplat | https://github.com/HwasikJeong/2Xplat | Working code |
