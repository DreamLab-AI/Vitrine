# Engineering Log

Development history for Vitrine (a standalone project that vendors LichtFeld Studio as a pinned tool; formerly a fork — see ADR-021).

## 2026-07-10 — R&D frontier: 4 Fable teammates, branch `feat/object-rd-frontier`

Ran a 4-agent Fable R&D fan-out on the open object-arc frontier; each teammate
owned distinct new files and returned integration diffs, consolidated serially
onto the branch. All integrations compile, the container pipeline suite is
**209 passed** (+47), and both new geometry paths are live-verified in the USD
assembler.

- **Native TRELLIS.2 service (R5)** — `scripts/trellis2_native_service.py`
  fleshed out: `/health` (liveness) vs `/ready` (pipeline loaded), a full error
  taxonomy (400/413/503/500, all JSON), param hardening (unparseable→default,
  out-of-range→clamp, `face_count_low ≤ high`), thread-safe lazy load, 64 MB
  upload cap, complete lineage. This is the durable answer to the ComfyUI drtk
  fragility (the native pipeline uses nvdiffrast, not the ABI-brittle drtk
  path). 15 hermetic tests. VERIFY-ON-ENV-BUILD kept on the two upstream-API
  wrappers (the ComfyUI-ported tree lacks the `pipelines/` module).
- **R10 orientation solve** — `src/pipeline/object_orientation.py`: the mesh's
  canonical front (+Z) corresponds to the observing crop camera's ray, so an
  upright-object **yaw** is solved from the COLMAP camera pose (frame math
  verified against `colmap_parser` + the assembler; COLMAP world→cam vs USD
  Y/Z-flip handled). Wired through `object_placement.build_placements` →
  `usd/placements.json` → the assembler's `AddOrientOp` (TRS order). Live USD
  check: yaw-90 → `xformOp:orient=(0.7071,0,0.7071,0)`. Roll/pitch stay pinned
  upright (single-view can't observe them); near-vertical views flagged
  `degenerate`→unsolved. 17 solver tests + 2 integration tests.
- **R7 rung (b) image-edit view** — `src/pipeline/image_edit_view.py` +
  `qwen_image_edit_view.json`: for objects whose best seed scores below
  `escalation_threshold`, synthesize ONE "show the back" alternate view
  (Qwen-Image-Edit-2509, Apache-2.0) and add it as another single-image
  best-of-N candidate (NOT panel-stitching). Self-gates via
  `probe_edit_model()` (the edit UNET isn't staged), flattens input alpha
  (avoids the LoadImage premultiplied-ghost trap), re-mattes the output. A
  winning edit-view rewrites the asset's `surface` to `image-edit-inferred`.
  18 tests.
- **R8 Pixal3D** — `src/pipeline/pixal3d_client.py` mirrors the single-image
  contract as a drop-in generator (opt-in, default-off, eval-gated), + the
  `research/2026-07-pixal3d-eval.md` head-to-head plan. Pixal3D is **MIT** (the
  ADR-025 correction), so its licence gate passes; weights are NOT staged, so
  the client is a contract-validated scaffold (VERIFY-ON-ENV-BUILD), never
  live-run. `eval/objects/run_eval.py --generator pixal3d` added. 13 tests.

Integration hygiene: config.py gained `Pixal3DConfig` + `ImageEditConfig` +
Trellis2Config escalation fields; stages `_generate_object_from_crop` gained a
Pixal3D primary tier + the image-edit rung (best-of-N tuple widened, backward
compatible — the default single-shot path is byte-identical). CI now runs all
four new hermetic suites (+flask dep). One correction applied during
integration: the orientation camera-pose reads from `provenance` (the crops
manifest), not `crop_entry`, which the teammate's diff had assumed.

## 2026-07-10 — R7 best-of-N quality ladder + drtk self-heal hardened

### R7 — best-of-N seed re-rolls with a silhouette-consistency scorer

Single-shot generation was a quality ceiling; backside hallucination is
ADR-025's stated #1 risk. `trellis2.best_of_n > 1` now runs N seed re-rolls
and keeps the best (default 1 = unchanged). New pure module
`object_candidate_score.py` scores each candidate by a deliberately modest,
honest signal — a FRONT-silhouette proxy, not a full 3D metric: proportion
agreement (mesh front aspect vs the crop's observed silhouette aspect, via
`|log(ratio)|`, scale-invariant + symmetric) gated by mesh sanity (face-count
floor + watertight bonus; a degenerate mesh scores 0). It does not directly
detect backside hallucination (true multi-view scoring is future work) — it
picks, among N seeds, the candidate most consistent with the one observation
we have, and records EVERY candidate's score in the winner's lineage for
human / R9 review. 8 scorer unit tests + a best-of-N selection stage test;
validated in a clean CI-deps venv (99 passed) + container subset (162).

### drtk self-heal hardened after a live PBR failure

The R7 live proof surfaced a real infra fragility (NOT in the R7 code, which
degraded correctly): ComfyUI's `drtk` was ABI-broken against torch 2.12
(`Trellis2RasterizePBR` → `undefined symbol` mid-generation), even though an
earlier run that day succeeded. The `comfyui_entrypoint.sh` self-heal rebuilds
drtk from source but was a silent single shot (`|| true`), so a transient
compile failure left PBR broken until the next restart — discovered only after
minutes of wasted shape-stage compute. Hardened to **retry twice** with a
loud, greppable ERROR on final failure. Rebuilt drtk in the live container to
unblock the proof.

## 2026-07-10 — Mega-image rebuilt (gate + SPA baked) + R10 object pose-solve

### Image rebuild closed the last runtime security gap

Rebuilt `gaussian-toolkit:latest` (`ed36b0bd26e6`, `BUILD_SPA=1`) and recreated
the container. The LichtFeld builder + base layers cache-hit (binary intact,
174 MB; no CUDA recompile), so only the tail rebuilt. Runtime-verified in the
fresh container:
- **ttyd gate is now LIVE** (gap #3): with the default `VITRINE_CLAUDE_ENABLED=0`
  there is NO `:7681` listener — it had been an unconditional listener because
  the *baked* `supervisord.conf` predated the gate. Both branches exercised
  (disabled → exits; enabled → loopback ttyd, warns without a credential).
- **ttyd/VNC second-factor paths baked** (gap #14): `VITRINE_TTYD_CREDENTIAL` +
  `VITRINE_VNC_PASSWORD` present and wired.
- **SPA baked** into `static/spa/` — retires the live-`docker cp` injection; a
  fresh clone+build now serves the Vitrine SPA with no manual step.
- All host publishes remain loopback-pinned (7860/7681/8188/8200/45677/5902).

### R10 — object pose-solve at USD assembly (PRD v4)

Generated GLBs are in TRELLIS's own normalized frame, so before this they
assembled at the world origin at unit scale. New pure module
`object_placement.py` solves position + uniform scale from the object's
Gaussian subset (already in COLMAP-world coords): `scale = real_max_extent /
glb_max_extent`, `translate = world_centroid`. `mesh_objects` records the GLB's
own bbox (`glb_extent`); `assemble_usd` writes `usd/placements.json`; the
standalone assembler applies translate + uniform scale (× SCENE_SCALE) when a
placement exists, else the legacy self-centroid path. **Orientation is left
identity and flagged `unsolved`** — the honest ADR-025 D3 posture (a full
orientation solve from the crop pose + silhouette is future work). 8 new
solver unit tests + a placements-emission stage test; pure math, no deps.

## 2026-07-09 (evening) — Security posture APPLIED live + hardening pass + `pipeline objects` CLI

### The live system was still LAN-open — now loopback-everywhere (verified)

The ADR-024 loopback pins existed only in compose; the running `gaussian-toolkit`
predated them and was publishing ttyd `:7681`, ComfyUI `:8188`, LichtFeld MCP
`:45677` and passwordless VNC `:5902` on `0.0.0.0`. Containers recreated from
current config; **`ss -tlnp` now shows every pipeline port
(7860/7681/8188/8200/45677/5902) on `127.0.0.1` only.** Three new findings
fixed in the same pass (gap register #11–#13):

- **#11** `run_comfyui.sh` published `:8200` on `0.0.0.0` (no-auth ComfyUI,
  RCE-class Manager surface) — missed by the register's #9 verdict. Pinned.
- **#12** the "secure by default" `LFS_WEB_HOST=127.0.0.1` compose default was
  a **functional regression, not a hardening**: docker-proxy connects to the
  container's bridge IP, so a container-loopback bind makes the pinned host
  publish — and the documented SSH tunnel — dead. The boundary is the
  `host_ip: 127.0.0.1` publish; compose now binds `0.0.0.0` in-container with
  the rationale documented in compose + `app.py`. Tunnel path live-verified.
- **#13** live probing the file API found `%00` in `?path=` → 500 (unhandled
  past the jail); now 400. Traversal probes (plain/encoded/absolute/run-id)
  all correctly refused; extension allow-list enforced.
- **#14** defence-in-depth credentials plumbed: `VITRINE_TTYD_CREDENTIAL`
  (ttyd basic-auth) + `VITRINE_VNC_PASSWORD` (x11vnc), through compose +
  the mounted `vitrine-terminal.sh`/`entrypoint.sh`. Default = tunnel-only,
  warned at boot. Caveat: the baked `supervisord.conf` predates the ttyd gate,
  so the gate + credential activate at the next image rebuild; the loopback
  publish is the interim control.

### `pipeline objects` — the object arc as a first-class CLI

`python3 -m pipeline.cli objects <job_dir> [--concepts ...] [--skip-segment]`
runs segment → object_crops → extract_objects → mesh_objects → texture_bake
on an existing trained run — the committed form of the flow that produced the
first validated automated objects, without repeating the COLMAP/training
front-end. Auto-detects the newest model PLY + frames dir; loud failures
propagate; generator assets are tagged `[generator, PBR verbatim]`.

## 2026-07-09 — Object pipeline convergence implemented (PRD v4 / ADR-025) + SAM3 root cause FIXED

### R1 SOLVED: "SAM3 returns boxes" was an HWC/CHW bug in OUR wrapper — one-line fix

The audit's blocker defect F1 is closed, and the root cause was ours, not SAM3's:
`Sam3Processor.set_image`'s numpy branch reads `height, width = image.shape[-2:]`
(correct for CHW tensors); our `sam3_segmentor` passed HWC frames, so it read
`height=W, width=3`. Detection still worked (model input conversion handles HWC)
but every output mask was interpolated to a **W×3 grid** and boxes were scaled by
(3, W) — which downstream resizing smeared into the notorious "coarse boxes".
Fix: pass PIL images (`_to_pil` in `sam3_segmentor.py`). Verified on rawcapdev:
pixel-accurate silhouettes for vessel (fill 0.86, score 0.94), ketchup bottle
(0.79/0.57) and wooden block (0.75/0.92) — overlay at
`output/rawcapdev/sam3_fixed_overlay.jpg`.

### First-ever validated automated per-object isolation on real data

With silhouettes fixed + new binary-COLMAP support (`parse_cameras_bin`/
`parse_images_bin` + format-agnostic loaders — the direct-COLMAP path emits
bin-only models, which would otherwise fail loudly), the full Phase-1/2 arc ran
live on rawcapdev: segment → **object_crops** (new stage) → extract_objects
MV-projected **1.09M/4M gaussians (vessel), 246k (bottle), 250k (block)** across
6 views. Zero fallbacks, zero failures. Crops are generator-ready RGBA mattes at
native res (vessel 3175²) with full provenance (frame, bbox, COLMAP pose,
matting method, selection scores) in `object_crops/crops.json`.

### ADR-025 implementation (all PRD v4 phases except live 3D generation)

- **R2** `extract_objects`: full-scene-copy + world-XY fallbacks DELETED; per-object
  failures reported with cause (`object_failures` artifact). `full_scene` label
  (the environment) still copies by design.
- **R3** new `object_crops` stage (`src/pipeline/object_crops.py` + config):
  best-frame selection = silhouette area × log-sharpness × centrality × edge
  clearance; SAM-matte alpha; rembg fallback for box-like masks; ≥1024² square pad.
- **R4** multiview conditioning RETIRED: `trellis2_client.py` rewritten single-image
  (ComfyUI executor now, native-service contract ready); `view_completer.py`,
  `trellis2_multiview_pbr.json`, `flux2_turnaround.json` + all 3 stock-node Hunyuan
  workflow JSONs deleted; `multiview_renderer` kept preview-only and fixed to
  straight alpha (F4 un-premultiply).
- **R5** `scripts/trellis2_native_service.py` scaffold (HTTP contract final; model
  calls marked VERIFY-ON-ENV-BUILD; client flips via `trellis2.native_url`).
- **R6** generator GLB bytes persisted byte-identical (sha256 recorded, lineage
  sidecar per asset); `texture_bake` skips generator meshes (F7 closed).
- **R6a** Hunyuan fallback = `reconstruct_from_image()` submitting the PROVEN
  Hy3D21 graph (hy3d_turnaround.py lineage) in code.
- **R9** eval harness `eval/objects/run_eval.py` (+ blender turntable): mesh
  stats/regression gates; `--stats-only` validated against the 2026-07-02 brass
  vessel GLB (274,588 verts / 483,671 faces / PBRMaterial — exact match).
- **Tests**: 30 new unit tests (crops, single-image client, object-arc
  regressions); pipeline subset **144/144 green** in-container.

### LIVE RUNTIME VERIFICATION: crop → single-image TRELLIS.2 → PBR GLB

The full ADR-025 arc ran end-to-end on GPU 1: the automated vessel crop →
`Trellis2ImageToShape` (1536_cascade, seed 42) → **54.8 MB PBR GLB, 286,009
verts / 491,067 faces in 479 s**, persisted byte-identical (sha256
`496d9a84…`) with lineage sidecar — same output class as the manual
2026-07-02 vessel (274,588 / 483,671). Workbench turntable confirms solid
geometry + patina albedo (`output/object_e2e/`). Smoke script:
`scripts/run_hull_e2e.py` (rewritten crop-based; the splat-render version is
gone).

### Infra root causes fixed along the way

- **ComfyUI has been fighting DiffusionGemma for GPU 0 all along**:
  `comfyui_entrypoint.sh` passed `--cuda-device 0`, and ComfyUI implements
  that flag by UNCONDITIONALLY overwriting `CUDA_VISIBLE_DEVICES` — stomping
  `run_comfyui.sh`'s `COMFYUI_GPU` masking and un-masking the container back
  onto physical GPU 0 (~40 GB held by the resident LLM server). Explains the
  2026-07-04 exit-137 OOM-kill and today's first Trellis2ImageToShape OOM
  (torch saw 19 MiB free on a "47 GiB" device). Fixed: flag dropped, GPU
  selection now genuinely `COMFYUI_GPU` (`COMFYUI_GPU=1 scripts/run_comfyui.sh`
  is the working setup while DiffusionGemma owns GPU 0).
- `set -u` crash in the drtk self-heal (`$PYTHONPATH` unset) — entrypoint died
  before ComfyUI launch. Fixed (`${PYTHONPATH:-}`).
- Blender 5 eval turntables: EEVEE renders TRELLIS.2's atlas-padding alpha as
  transparent patchwork → harness renders with WORKBENCH + texture color.

**Open:** native service env stand-up (R5 acceptance); Pixal3D head-to-head
(R8; NOTE: Pixal3D is MIT — ADR-025 amended 2026-07-09); R10 pose-solve at
assembly (placement hints already recorded per asset); R9 references baseline
committed from the first 3-object eval sweep.

## 2026-07-02 — First LichtFeld-native E2E on real raw capture (rawcapdev) + build, pipeline, and web fixes

### E2E milestone: rawcapdev gallery still-life

First complete run of the LichtFeld-native pipeline on fresh raw capture:
**55 Pixel DNG frames** of a gallery still-life (brass patina vessel + inverted Heinz ketchup bottle)
decoded via rawpy → 50–55 sharpest selected → **COLMAP 100% registration, 10.4k sparse points** →
**LichtFeld igs+ 4M-Gaussian splat** trained GPU-bound → render + inspect.
Keeper renders committed to `docs/renders/rawcapdev-2026-07-02/`
(neutral-WB training run: `e4d0dd17`; hero crop + SAM3 overlay: `a6334fae`).

Object stage: SAM3 concept segmentation runs but currently returns **coarse bounding boxes, not
per-pixel silhouettes** (known issue — `extract_objects` therefore returns the full scene PLY).
The working object path is a clean SAM image crop → TRELLIS.2 image→3D (ComfyUI/TRELLIS.2 live).

**Known open issues:** SAM3 box-mask output blocks per-object isolation; `.ksplat` runtime production
is inert (the viewer falls back to `.ply`); the Rust onboarding wizard still binds `0.0.0.0`.

### LichtFeld v0.5.3 baked into the mega-image (commit `8944cf6e`)

Added a `lichtfeld-builder` multi-stage that compiles the vendored v0.5.3 submodule via the proven
upstream CI recipe (`lf_build2.sh` + `lf_stage.sh`) and `COPY --from=lichtfeld-builder` the binary
+ runtime `.so` into the runtime image. Removes the `./build:ro` bind-mount from compose — that bind
was **shadowing the in-image binary with an empty host dir**, causing the silent gsplat fallback on
every run. Binary is now in-image at `/opt/gaussian-toolkit/build`; no host build required.

Follow-up commit `309e8964` fixed a gap in `lf_stage.sh`: its flat `-type f` copy missed symlinked
sonames (`libusd_*.so`) and external-dep libs (`libnvimgcodec.so.0`, `libOpenMeshCore`). Fix: `COPY`
the whole `build-lf` tree from the cached builder stage into `/opt/lichtfeld-build` and `ldconfig`-index
every `.so` directory — the binary now resolves all libs regardless of RUNPATH or symlinks.

### reconstruct() defaults to direct COLMAP; undistort image cap (commit `77c64b4c`)

Two defects that blocked the first LichtFeld-native E2E on real raw capture:

1. **SplatReady plugin config collision.** The bundled SplatReady plugin derives its progress-file path
   via `str.replace("_run_config.json", "_progress.txt")`. `stages.py` named the config
   `"splatready_config.json"` (no matching substring), so the status path **collided with the config**
   and the runner clobbered its own config mid-run (`JSONDecodeError` on reuse). The runner also exits 0
   on failure, silently swallowing errors. `reconstruct()` now defaults to the transparent
   `_run_colmap_direct` path; SplatReady stays opt-in behind `config.reconstruct.use_splatready` with
   the naming bug fixed and stdout captured in the error.

2. **Undistorted images were 36MP.** `_run_colmap_direct()` omitted `--max_image_size`, so undistorted
   training images came out full-resolution (~7000px). LichtFeld CPU-downscales anything wider than its
   `--max-width` (3840px) **on every load** — ~5 s/image, uncached — starving the GPU at 0% utilisation
   during 30k-iteration training (projected to hours). Now passes `--max_image_size` from
   `config.ingest.max_image_size` (default 2000); intrinsics are baked in by the undistorter. Validated:
   igs+ 30k trains GPU-bound at 99% util, completing in ~15 min on the rawcapdev set.

### Camera white-balance + cameras.bin uint64 parse fix (commit `b1fb6c4a`)

- **`image_decoder._decode_rawpy`: `use_camera_wb=True`.** Without it, libraw applies a fixed daylight
  white balance, leaving a heavy warm/orange cast on indoor DNG captures. The cast propagated into the
  splat and every downstream SAM crop / object texture. With the Pixel's as-shot WB the gallery walls
  render neutral white (verified on rawcapdev — retrained neutral-WB renders in `e4d0dd17`).

- **`stages.render_previews`: `cameras.bin` stores `width`/`height` as `uint64`, not `int32`.** The old
  `'<iiii'` struct read split the 64-bit width field into `(w, high32=0)`, yielding `render_h=0` → SIGFPE
  (core dump) in the gsplat rasterizer. Fixed to `'<iiQQ'` (uint32 id, int32 model, uint64 w, uint64 h).
  Preview render now succeeds from real COLMAP camera poses.

### Object recon routed to TRELLIS.2 (commit `d2ab4641`)

`_mesh_single` Strategy 0 (gsplat→TSDF) is now gated to full-scene PLYs only. Isolated per-object PLYs
go to **TRELLIS.2 first** (ADR-015 `hull_e2e`-validated) instead of a holey orbit-TSDF — this is the
object-quality lever for the e2e goal (gap #8 closed). `Hunyuan3DConfig` endpoint corrected from the
stale `localhost:8189/3001` default to `vitrine-comfyui:8188` (gap #5).

### Build persistence: scipy/numpy reconcile + zipstream-ng + pynvml

`Dockerfile.consolidated` gained two sets of changes this pass:

- **`zipstream-ng` + `pynvml`** added to the pip install layer (commit `25c6ab78`) — required by the
  new web zip-streaming endpoint and the system-stats API.

- **scipy/numpy reconciliation after SAM3** (working-tree change, uncommitted): SAM3 pins `numpy<2`,
  leaving the image with numpy 1.26 alongside a modern scipy whose wheel references `np.long` (removed in
  numpy ≥ 1.24) → `import scipy.spatial` crashed at runtime, breaking `select_frames` and `reconstruct`.
  Upgrading scipy pulls a numpy ≥ 2 wheel; the pairing (`numpy 2.x + scipy≥1.15 + matplotlib`) was
  validated end-to-end on rawcapdev. `matplotlib` is also installed here (needed for
  `render_previews` depth colormaps). SAM3's `numpy<2` pin is advisory and verified-tolerant of numpy 2.x
  in this configuration. Proper long-term fix is per-tool venv isolation (ADR-022 P2).

### Web consolidation: PR #6 ArchiveSpace + loopback-only Flask (commit `25c6ab78`, ADR-023)

The community ArchiveSpace React SPA (PR #6) was absorbed as **patterns, not code** — its backend was
mocked `delay()` stubs — and the existing Flask control surface was extended instead. Four additive
blueprints (`scenes_api`, `files_api`, `zip_api`, `splat_api`) over the existing job store; vendored
offline `@mkkellogg/gaussian-splats-3d` viewer (`src/web/static/vendor/`, no CDN); a Vitrine-owned
Vite+React app at `src/web/frontend/` compiled at image-build time via a throwaway `node:20-alpine`
stage (`BUILD_SPA` gate, default off).

**Security fix (ADR-022):** `app.py` was binding `host="0.0.0.0"` — fixed to `127.0.0.1` by default,
with `LFS_WEB_HOST=0.0.0.0` as an explicit docker-net opt-in. The compose default was also flipped to
`127.0.0.1:7860`. The web UI is now reached via `ssh -N -L 7860:localhost:7860` (IT-signable posture).

QE-fix pass caught one critical route bug: the blueprint loader imported `web.api.<name>` but modules
are flat `web.<name>` → all four blueprints silently 404'd. Fixed; verified 64 routes register.

---

## 2026-06-24 — Scene-mesh refinement + ArtiFixer trial launched

Two parallel levers on the **dreamlab/locked** scene mesh; full consolidated write-up
(approaches, issues, results, figures) in [`docs/scene-mesh-refinement.md`](./scene-mesh-refinement.md)
(per-voxel evidence + QA renders: [`docs/scene-mesh-refinement-evidence.md`](./scene-mesh-refinement-evidence.md)).

- **CoMe finer-voxel TSDF campaign — resolution is NOT the binding constraint.** Four
  backend attempts to push below the 0.015 baseline voxel: (1) Open3D tensor
  `VoxelBlockGrid` CPU res16 and (2) CPU res8 both SEGFAULT mid-`integrate`/`extract`
  (the `block_count` is a hard pre-allocated cap, not grow-on-demand — crash at 165 GB RAM
  free, so not OOM); (3) CUDA res8 `integrate` succeeds but **ABORTs inside
  `extract_triangle_mesh()`** at the same block count on both CUDA and CPU → an **Open3D 0.19
  tensor-extract size/overflow bug** at scale, not a memory/capacity limit.
- **The legacy `ScalableTSDFVolume` (CPU, render on GPU) is the stable extractor.** Cleared
  every failure: **voxel 0.005 → 51.5M faces** (2.37 GB PLY), **voxel 0.007 → 25.7M faces**
  (1.17 GB). Cleaned (≥2 % connected-component) to 30.4M / 15.9M faces.
- **Verdict:** all three voxels (0.015 / 0.007 / 0.005) yield the **same partial, lumpy
  mass**; finer voxel renders the under-observed/motion-blur "wing" noise more **sharply**,
  not cleaner. The binding constraint is the **capture** (motion blur + partial coverage),
  not mesh resolution.
- **ArtiFixer recon-enhancement trial (ADR-020) LAUNCHED** on the Ada **sm_89** fork — the
  gated branch that regenerates *unobserved* regions (does **not** deblur). Sidecar up on
  `v2g-net`; prepare stage running. Details + gate criterion in
  [`research/decisions/adr-020-artifixer-ada-fork-recon-enhancement.md`](../research/decisions/adr-020-artifixer-ada-fork-recon-enhancement.md)
  and [`docs/artifixer-recon-enhancement.md`](./artifixer-recon-enhancement.md). No enhanced
  result yet — enhance / distill / mesh and the PROCEED/FALL-BACK/ABANDON verdict are PENDING.

## 2026-06-22 — Deliverable = textured MESH scene into UE; USD dropped from the critical path

> **End-state (user-set):** the product is a **mesh + organized-textures polygonal scene** — a
> room mesh + per-object textured meshes — imported into **Unreal Engine 5.8 as game-style assets**.
> Gaussian-splat-in-UE is OPTIONAL. **USD is NOT required** (demoted to optional/archival).
> Interactive elements + lighting are LOW-priority stretch goals. See `research/decisions/PRD-mesh-scene-into-ue.md`,
> `adr-019-mesh-game-assets-not-usd-into-ue.md`, `DDD-vitrine-mesh-pipeline.md`, and `docs/ANTIPATTERNS.md`.

- **USD is off the critical path — the killer fact:** LichtFeld's native USD export emits a custom
  `ParticleField`/gaussian prim that UE's USD importer **cannot import as a mesh**, and UE USD import
  also drops `displayColor`/`customData`. So USD can't be the UE contract. The contract is **FBX with
  a baked-texture material**. (ADR-019; amends ADR-016/ADR-011.)
- **UE drops vertex colours (FBX *and* OBJ *and* GLB `COLOR_0`) → flat white.** The only way captured
  colour survives into UE is a **baked UV texture**. This single fact invalidated a long detour
  (vertex-colour FBX/OBJ/GLB import all rendered white); the dead `build_room_ue.py` VertexColor path
  was purged → breadcrumb in `docs/ANTIPATTERNS.md`.
- **Working scene recipe (validated on TSDF):** env mesh (CoMe default / **gsplat-TSDF fallback**) →
  `MeshCleaner.clean(smooth_iterations=0)` (smoothing makes degenerate tris that **segfault xatlas**)
  → `texture_baker.bake_from_vertex_colors` (xatlas UV atlas, baked albedo) → `scene_to_ue.py`
  (gravity/scale to UE cm) → `blender_obj_to_fbx.py` (embed texture) → UE `import_file`
  (`import_materials=True`, Nanite). Produced `output/dreamlab/scene/room_ue.fbx` — colour reaches UE
  (caveat: lit material over-exposes; an unlit/emissive scan material or toned lighting is the fix).
  Open3D `compute_uvatlas`/xatlas need a manifold mesh and crash on raw TSDF — decimate first.
- **Objects = Hunyuan3D-2.1, NOT TRELLIS.2 (audit-backed).** SAM image-crop → Hy3D21 graph →
  textured GLB + OBJ + PBR maps → `blender_obj_to_fbx.py` → UE. **4/8 done** (chair, dartboard,
  vacuum_cleaner, ladder). Failures were **infra, not data** (ComfyUI VRAM crash after ~1-2 objects;
  TRELLIS empty-input crashes) → **restart-ComfyUI-per-object** (`scripts/hy3d_batch.sh`) + poll on a
  long deadline (big-mesh bakes run hours; ladder's was 3.5 h — the old 1500 s `/history` timeout was
  the bug, now 14400 s). Stay on Hunyuan3D: the 4 successes are clean game-asset bundles.
- **CoMe enabled (documented default env-mesh backend).** Built on the **host** (`agentbox:host`,
  `INSTALL_COME=1`; never agent-side — read-only `~/.docker`). CoMe = `r4dl/CoMe`, **non-commercial
  Gaussian-Splatting licence (research/eval only)** — the gsplat-TSDF path is the commercial track.
  Until CoMe lands, the recipe is validated on the TSDF fallback.
- **Dreamlab data verdict (first quality footage; prior "validated" runs were on the abandoned Drive
  data).** COLMAP **strong** (600/600 frames registered, mean reproj 1.03 px, coherent whole-room
  coverage — NOT fragmented). Frame quality **MUSIQ ~31** (usable; above the abandon ~19; motion-blur
  ceiling → residual haze worst on the under-observed floor). Objects: ~4 solid (chair/mitre_saw/
  toolbox/dartboard) + 2 marginal (vacuum/ladder) + 2 SAM-**over-segmented** (table/workbench — a crop
  fix, not a capture fix). No re-capture needed to ship the room + hero objects.

## 2026-06-21 — Source-quality ceiling: SOTA frame-QA gate, appearance-recipe limits, unpruned-Gaussians + Nanite

> **Outcome: the Drive room dataset was ABANDONED** and all generated working data deleted (~18G:
> scene01/scene02/objects_src + scratch). Root cause = a **reflective TV** dominating the room
> (view-dependent reflections defeat SfM + overfit the 3DGS SH into floaters/haze) compounded by
> **MUSIQ-19 motion blur**. Code, docs, and the QA/Nanite tooling persist for the next (fresh) capture.

Follow-on from the multi-capture e2e below. Four decisions/findings:

- **Source quality is the ceiling — quantified.** A SOTA no-reference IQA pass (MUSIQ via `pyiqa`,
  GPU) scores the Drive frames **15–33 (median ~19)** where a *good* photo is 50–75; classical
  sharpness (Laplacian variance) is 5–13 (sharp ≈ 100s). The footage is heavily motion-blurred. No
  training recipe rescues this — confirmed empirically below.
- **Appearance recipe corrects exposure, not blur.** Retraining scene02 with `--ppisp`
  (per-camera appearance) + `--bilateral-grid` reached loss 0.029 and removed some colour drift, but
  splat renders are **still hazy/rainbow-floatered on the wide sweeps** — the haze is baked-in motion
  blur, not exposure. The realistic fix is a fresh **"gaussian-run" capture**, not more training.
- **Frame-QA is now a callable capability** (`src/pipeline/frame_quality.py`): `NeuralIQA` (pyiqa
  MUSIQ, GPU-first, classical fallback) + `assess_video()` → `VideoQualityVerdict` implementing the
  **drop-and-flag policy** — a video that can't supply `min_good_frames` KEEP-grade frames is marked
  `too_low_quality` and skipped. The production overseer agent owns the orchestration; the pipeline
  only exposes the tool. Pinned `pyiqa==0.1.15.post2` (`--no-deps`) in `Dockerfile.consolidated`.
- **Decouple quality from primitive budget.** Do NOT prune Gaussians (`--enable-sparsity`,
  prune=60%) to hit a perf target — it spiked loss to ~68 and over-thins (4M→1.6M) without fixing
  haze. Keep the **unpruned** field for quality; solve performance downstream on the *mesh* via
  **UE Nanite** (virtualized geometry + auto-LOD) and reserve decimation for the web GLB/`.ksplat`.
  Caveat learned: `--resume` is **config-sticky** — it restores the checkpoint's original schedule
  (incl. sparsity) and ignores CLI overrides; an unpruned model needs a *fresh* run.

## 2026-06-21 — Drive multi-capture e2e (scene01/scene02) — COLMAP 4.1.0, MILo flat-images, UE captured-color, multi-sweep, prefer-existing process lesson

First real **Google-Drive multi-capture** end-to-end run: several handheld video sweeps of the
**same room** (at different heights/approaches) ingested, reconstructed (COLMAP 4.1.0 SfM →
3DGS → MILo mesh), and pushed at UE 5.8 for captured-color display. Eight lessons (L1–L8), the
first being a process correction that supersedes the rest in importance.

**L1 — PROCESS (most important): prefer existing pipeline capabilities over custom Python**
(CLAUDE.md standing directive §1). Concrete failure this session: a slow custom PIL per-face
texture-baker was written for UE color when `src/pipeline/blender_assembler.py`
`bake_vertex_colors_to_texture()` (Cycles **GPU**, ~0.5 s/100k faces) **already existed and was
proven**. ALWAYS `grep src/pipeline` + `scripts` for an existing capability before writing new code.

**L2 — UE 5.8 captured color.** UE Interchange GLB import **IGNORES** vertex colors (`COLOR_0`) —
"Mesh has primitives with no materials assigned". UE USD import **DROPS** the `displayColor` primvar
(renders flat white). To get captured color into UE you need either (a) a **baked-texture material**
(UsdUVTexture / textured GLB) or (b) an explicit **VertexColor material**. Proven bake = Blender Smart
UV Project + Cycles DIFFUSE GPU bake (`blender_assembler.py`). **Caveat discovered this run:** Smart
UV Project **collapses on room-scale MILo meshes** (many thin/disconnected components — 876 removed on
scene02 — produce degenerate near-zero-area UV islands → a near-black 2048² atlas). The Blender bake
works on clean watertight **hulls**, not the messy room mesh. For room/scene-scale captured color, the
highest-fidelity visual is a **direct Gaussian-splat (gsplat) render**, not a baked mesh texture.

**L3 — `exhibit_builder` color preservation.** When filtering mesh components, vertex colors MUST be
remapped through the kept-vertex index map; rebuilding a Trimesh without colors yields trimesh's
default gray `[102,102,102]` (this caused a "white/gray in UE" red herring). `trimesh.split()` +
`concatenate()` **hangs at ~1.8M verts** → use `trimesh.graph.connected_components` + an explicit index
remap instead.

**L4 — COLMAP 4.1.0.** Option namespace changed to `--FeatureExtraction.*` / `--FeatureMatching.*`
(NOT `--SiftExtraction.use_gpu`). `FeatureExtraction.type SIFT` (the **ALIKED** enum is invalid in this
build; `ALIKED_LIGHTGLUE` needs `libcudnn.so.9`, which is missing) → GPU SIFT fallback gave **100%
registration**. Use `--ImageReader.single_camera_per_folder` for mixed cameras; **mkdir the colmap dir**
before `feature_extractor` (database_parent_path check).

**L5 — MILo mesh extraction.** Scripts live at `/opt/milo/milo/` (set `MILO_DIR` accordingly); MILo
references images by **basename**, so `undistorted/images/` MUST be flat (flatten any obj/wide subfolders
the undistorter creates) or it `FileNotFound`s.

**L6 — Multi-capture.** scene02 (3 **combined** handheld video sweeps, **146k pts, ~95% reg**) is
measurably better than scene01 (single sweep, **90k pts**). The Drive videos are several different
handheld approaches to the SAME room at different heights → combine them for coverage and infill gaps
minimally. The best video for the **scene** is NOT the best for an **object** — objects need a dedicated
close orbit (e.g. `IMG_0580`) for turnaround hulls.

**L7 — UE 5.8 launch** (cross-reference: already logged below for the persistent editor + RC bridge).
Persistent windowed `UnrealEditor` on an Xvfb display gets the real Vulkan RHI (a `-run=pythonscript`
commandlet loads NullDrv = no GPU render). Flags: `-unattended` (clears the Zen DDC hang) +
`-ExecCmds="py <script>"` (NOT `-ExecutePythonScript`, which makes `-unattended` exit). Run as uid 1000,
engine mount **rw**, `Config/DefaultEngine.ini [HTTPServer.Listeners] DefaultBindAddress=0.0.0.0`, native
MCP via `-ModelContextProtocolStartServer` + `-ModelContextProtocolPort`. (See the two 2026-06-21 Unreal
entries below for the full debugging arc.)

**L8 — GPU-always / deprecate CPU.** Use MILo **radegs** (GPU) not a numpy TSDF; Cycles GPU bake not a
Python PIL loop; GPU SIFT; GPU video decode. **Any CPU path is a defect to replace.**

## 2026-06-21 — Unreal 5.8 persistent editor + RC bridge LIVE (full agentic control surface)

Built on the nullrhi smoke-test below: the Unreal overlay now comes up agent-drivable via
`docker compose -f unreal/docker-compose.unreal.yml up -d unreal unreal-mcp-bridge`. Validated
end-to-end: `vitrine-unreal` reaches **healthy**, the bridge auto-starts, and from agentbox
`GET http://unreal-mcp-bridge:9100/health → unreal:30010/remote/info` returns the full RC route
list. The editor runs on GPU1 (Vulkan, `VULKAN_SM5`), loads `scene.usda` as a live UsdStageActor,
and stamps **16 v2g:\* tags** (2 prims) at startup; x11vnc exposes it (host `:5905`).

New artifacts: `unreal/runtime/run_editor.sh` (persistent editor launcher), `unreal/smoke_editor.sh`
(host launcher), `unreal/runtime/Config/DefaultEngine.ini` (bind config); `entrypoint.sh` gains a
`RENDER_MODE=editor` branch (now the compose default); `Dockerfile.unreal` adds xvfb/x11vnc/fluxbox.

Five gotchas resolved to get from "boots" to "resident + reachable":

1. **Commandlet → NullDrv.** `-run=pythonscript` is a commandlet; commandlets disable rendering
   (NullDrv) even with `-RenderOffscreen`. Fix: launch the windowed **`UnrealEditor`** (not `-Cmd`)
   on an **Xvfb** display → real Vulkan RHI. (Vulkan itself was always available — instance 1.3.204.)
2. **Editor exits / hangs.** `-unattended` + `-ExecutePythonScript` → editor runs the script then
   calls `UUnrealEdEngine::CloseEditor()` (exits ~25 s). But dropping `-unattended` makes the full
   editor **hang in a Zen DDC `zen service status` retry loop**. Winning combo: keep `-unattended`
   (clears Zen fast) + run the startup import via **`-ExecCmds="py …"`** (does NOT trigger the
   post-script exit) → resident editor, RC up in ~30 s.
3. **RC bound to 127.0.0.1.** UE's `FHttpServerModule` defaults to a loopback bind, so the bridge
   (other container) couldn't reach RC. Fix: `Config/DefaultEngine.ini` → `[HTTPServer.Listeners]
   DefaultBindAddress=0.0.0.0` (confirmed against `HttpServerConfig.cpp`). Log now shows
   `Created new HttpListener on 0.0.0.0:30010`; reachable cross-container as `unreal:30010`.
4. **Bind-mounted scripts not executable.** `runtime/` mounts read-only over the image's
   `chmod +x`, so `ENTRYPOINT /vitrine/unreal/entrypoint.sh` failed with `permission denied`. Fix:
   `chmod +x` `entrypoint.sh` + `run_editor.sh` on the host (tracked mode bits).
5. **Compose relative-path base.** The overlay's `./engine` / `../output` / `./runtime` paths only
   resolve when compose's project dir is `unreal/` — i.e. the **single-`-f`** form
   (`-f unreal/docker-compose.unreal.yml`, base dir = `unreal/`). The two-`-f` form
   (`-f docker-compose.consolidated.yml -f unreal/...`) bases paths at the repo root and mis-mounts;
   use the single-file form (or `--project-directory unreal`). Docs updated accordingly.

Open: the first-party **MCP plugin (:8000) reports `down`** through the bridge (likely a separate
loopback bind / enable) — RC is the designated primary control path and works. GPU **offscreen MRQ
render** (now unblocked by the live Vulkan RHI) still needs camera/sequence wiring.

## 2026-06-21 — Unreal 5.8 nullrhi boot smoke-test PASS (+ 4 bugs fixed); furniture hull e2e PASS

**Furniture hull e2e (reproducibility + new docker-network access).** Ran the full SOTA
hull arc on `output/e2e_run1/objects/furniture.ply` from a fresh `gaussian-toolkit` container
on `v2g-net`, reaching `vitrine-comfyui:8188` **by name** (the new shared-network access).
Result: **267,551 verts / 484,353 faces / 47.0 MB valid glTF 2.0 in 481 s**. FLUX.2 view
completion succeeded for BOTH gaps (front 55 s + back 46 s → TRELLIS.2 ran on all 6 panels),
serial VRAM lifecycle held (FLUX.2 unload → TRELLIS.2 → free), polling-timeout fix held
(16 polls at 30 s through 481 s). NOTE: the `e2e_run1` object PLYs are all 1M-Gaussian
full-scene copies (first-pass isolation), so coverage matches the prior `sculptures` run —
this validates the arc + network, not distinct-object isolation. The "exit 1" the harness
reported was a spurious `tee` perm-denied on the root-owned `output/hull_e2e/`; `docker run`
itself returned 0.

**Unreal 5.8 boot smoke-test — foundational arc validated (`nullrhi`).** Built
`unreal/smoke_nullrhi.sh` + `unreal/runtime/_smoke_probe.py` to answer "does UE 5.8 boot in
`vitrine-unreal:5.8` against the in-repo engine, run a python commandlet, read our `v2g:*`
USD, and spawn a UsdStageActor?". Final PASS (exit 0): `pxr OK: 10 prims, 2 carry v2g:*`
(`/World` pipeline lineage + `/World/Objects/sculptures` full per-object lineage),
`UsdStageActor spawned + root_layer set OK`. Four real bugs surfaced and fixed along the way:

1. **`Vitrine.uproject` enabled non-existent plugins → hard abort.** `USDStageEditor` and
   `WebRemoteControl` are not plugins in this build (`PluginManager.cpp:2353` "missing on disk.
   Aborting" → SIGSEGV). Correct names: USD Stage Actor lives in **`USDImporter`** (pulls
   `USDCore`); the RC HTTP server (`:30010`) is the **`RemoteControl`** plugin (WebRemoteControl
   is a *module* of it). uproject now enables USDImporter + PythonScriptPlugin + RemoteControl +
   ModelContextProtocol (all verified present under `unreal/engine/Engine/Plugins`).
2. **UE refuses to run as root** (`libc++abi` recursive-init abort, exit 134). The engine binary
   is owned by uid 1000 → run the container as `user: "1000:1000"` with `HOME=/tmp` (no passwd
   entry for the uid). Propagated to `docker-compose.unreal.yml`.
3. **Engine must be mounted read-WRITE.** On startup UE rewrites `LibraryPath` in its ThirdParty
   USD `plugInfo.json` files and writes DerivedDataCache; a `:ro` engine mount → `errno=13`
   cascade → corrupt pixar USD plugin registry (`pxr.Usd.Stage.Open` returns 0 prims) + non-zero
   editor exit. Changed the probe and `docker-compose.unreal.yml` engine mount `:ro` → `:rw`.
4. **`v2g:*` lineage read bug in BOTH import scripts.** `usd_assembler.py` writes lineage via
   `SetCustomDataByKey("v2g:hull_glb", …)`, which pxr stores as a NESTED dict
   `customData["v2g"]["hull_glb"]` — NOT a flat `"v2g:hull_glb"` key. `import_usd_stage.py` and
   `import_and_render.py` both filtered for flat `k.startswith("v2g:")` keys on the top-level
   dict → matched nothing → would have mirrored ZERO lineage into Unreal. Added a shared
   `_extract_v2g()` (reads the nested dict + folds in any flat keys) to both; `import_usd_stage.py`
   now also stamps `v2g:<prim>:<key>=<val>` onto the Stage Actor's tags. `runtime/` is bind-mounted
   read-only, so `entrypoint.sh` now copies the project to a writable `$HOME/vitrine-proj` (UE
   needs to write Saved/Intermediate).

**Still pending for the overlay:** the entrypoint runs `-run=pythonscript` (a one-shot
commandlet that exits when the script returns) — so the compose `:30010` healthcheck +
`unreal-mcp-bridge` (`depends_on: service_healthy`) model needs a PERSISTENT editor instead;
and the GPU offscreen render (Vulkan `-RenderOffscreen` on GPU1 + MRQ camera/sequence) is
unproven. Next: confirm Vulkan offscreen RHI init, then wire a persistent editor for bridge
control.

## 2026-06-20 — Hull e2e PASS: polling timeout + cv2 shim segfault (two-fix commit)

**Second (clean) e2e pass** on `sculptures.ply`: **280,799 vertices, 461,352 faces, 56 MB PBR GLB**
at `1536_cascade`/4096 — up from the first run's 273K/499K/45MB (where FLUX.2 failed, degrading to
4-panel). This time FLUX.2 synthesised front+back (54s+48s), TRELLIS.2 ran on all 6 panels, and PBR
rasterise completed without segfault. Wall time 452s (7.5 min), down from ~26 min on the first run
(texture diffusion was faster with all 6 panels providing stronger conditioning).

**Fix 1 — ComfyUI /history polling ReadTimeout.** All 5 pipeline clients (`trellis2_client`,
`view_completer`, `hunyuan3d_client`, `sam3d_client`, `person_remover`) polled `/history/{prompt_id}`
with `timeout=10` — too short when ComfyUI's HTTP server thread is blocked by heavy GPU work (FLUX.2
sampling ~50s, TRELLIS.2 shape diffusion ~5 min). After 10s without a response, `requests.ReadTimeout`
crashed the pipeline. Fix: `timeout=60` + `try/except (ReadTimeout, ConnectionError): continue` on all
clients using `requests`; `person_remover` uses urllib via `_comfyui_request()` so only the timeout was
increased (its existing `except ConnectionError` already catches urllib timeouts).

**Fix 2 — cv2 shim segfault (SIGSEGV in PBR rasterise).** `comfyui-sam3dobjects/vendor/cv2/shim.py`
shadows real OpenCV with scipy/skimage fallbacks. The shim's `cv2.inpaint()` dispatches to
`skimage.restoration.inpaint_biharmonic`, which segfaults in `scipy.sparse.linalg.spsolve` on 4096px
textures (sparse matrix too large). Stack: `nodes_unwrap.py:643` → `shim.py:496` →
`inpaint_biharmonic` → `spsolve` → SIGSEGV. Real OpenCV 4.13.0 is installed in the image and handles
4096px textures fine. Fix: `comfyui_entrypoint.sh` step 1d moves `vendor/cv2` → `vendor/cv2.disabled`
before ComfyUI starts. The existing step 1e (now renumbered) patches a *separate* `cv2.inpaint` ndim
bug in the same file.

## 2026-06-20 — First hull from real reconstructed data + FLUX.2 tokenizer fix

**Milestone: first TRELLIS.2 hull from actual reconstructed Gaussian data.** Slice-A e2e test on
`sculptures.ply` (1M Gaussians extracted from a real video capture) produced a **45 MB PBR-textured
GLB: 273,463 vertices, 498,699 faces** at `1536_cascade` resolution / 4096px textures. Wall time
~26 minutes (dominated by texture diffusion + PBR rasterization at high resolution). The full path:
orbit-render 6 panels → coverage-gate → FLUX.2 view completion (graceful degradation) → TRELLIS.2
multiview → shape diffusion → texture diffusion → PBR rasterize → GLB.

**Coverage breakdown:** front 1.6%, left 46%, back 0%, right 27%, top 28%, bottom 23%. The sculpture
was filmed from one side only — a genuine partial-capture scenario. The coverage gate correctly
identified front + back as gaps and attempted FLUX.2 view completion (which failed due to the
tokenizer issue below, triggering correct graceful degradation → TRELLIS.2 ran on the 4 observed
panels only). The resulting hull is partial but correct.

**Root cause of prior OOM failures: DiffusionGemma on GPU0.** The local agent LLM server
(`diffusiongemma:cuda13`) consumed ~40 GB on GPU0, leaving only ~7 GB for TRELLIS.2. ComfyUI's
`POST /free` only unloads its own models, not another process's VRAM. Fix: `docker stop diffusiongemma`
before hull generation. The oversight backend is `claude_code` (Anthropic API, no GPU cost), so DiffusionGemma
is not needed during hull generation. Documented in `docs/agent-roles.md`.

**FLUX.2 Mistral-3 tokenizer fix (RESOLVED).** ComfyUI v0.8.2 bug in `SDTokenizer.__init__`:
`self.start_token = empty[0]` where `empty = tokenizer("")["input_ids"]`. The Mistral tekken
tokenizer returns an empty list (no start/end tokens for empty strings), causing `IndexError`.
Initial diagnosis (missing `tekken_model` in safetensors metadata) was wrong — the tensor IS present
in both BFL and Comfy-Org distributions. The actual fix (backported from upstream post-v0.8.2):
patch `sd1_clip.py` to add `len(empty) > 0` guard + `start_token` fallback parameter, and patch
`flux.py` Mistral3Tokenizer to pass `start_token=1`. Applied idempotently in `comfyui_entrypoint.sh`
step 3. **Verified end-to-end:** `CLIPLoader(type=flux2)` + `CLIPTextEncode` + FLUX.2 KSampler
produced `flux2_tokenizer_test_00001_.png` successfully. FLUX.2 loads at ~44.6 GB VRAM (within 48 GB
single-GPU budget). Remove the patches when upgrading to ComfyUI >= 0.8.3.

**Also discovered:** `/comfyui/models/clip/t5xxl_fp8_e4m3fn.safetensors` is truncated (168 MB, should be
~4.9 GB). The fp16 version at `/comfyui/models/text_encoders/t5xxl_fp16.safetensors` (9.2 GB, 220 tensors)
is valid. FLUX.2 workflows use `CLIPLoader` (single, Mistral-3 only), not `DualCLIPLoader`
(T5+CLIP-L for FLUX.1), so the truncated T5 fp8 doesn't block FLUX.2 view completion.

**Ops:** `CLAUDE_MODEL=claude-opus-4-6` added to `docker-compose.consolidated.yml` for in-container
oversight agents (was defaulting to overloaded opus-4-8).

## 2026-06-20 (cont.) — TRELLIS.2 wired into the pipeline as primary hull (full-360 multiview)

`Trellis2Client` (`trellis2_client.py`) added, mirroring the Hunyuan3D client plumbing, and wired
into `stages.py` as **Strategy 0c** (tried before Hunyuan3D; degrades to Hunyuan3D/TSDF on failure).
`Trellis2Config` added to `config.py` (+ PipelineConfig + `_from_dict`), default `enabled=True`,
`1536_cascade`/`4096`. GLB retrieval: `Trellis2ExportTrimesh` registers nothing in `/history`, so the
workflow terminates in a **Preview3D** node — confirmed to register `{"result": ["<file>.glb", ...]}`,
which the client's history scan + `/view` download retrieves (verified: HTTP 200, 28.9 MB).

**Full-360 coverage (the single-view-back-missing concern):** the earlier 29 MB GLB came from a
*single-image* smoke (`bottle.png` → `Trellis2ImageToShape`) — its back was generatively guessed.
The production path renders panels from the trained splat and feeds `Trellis2MultiViewImageToShape`.
Extended from 4 → **6 panels**: a new renderer preset `trellis_6` (front/left/back/right at el 0 +
**top** at el +85 + **bottom** at el −85) maps to the node's named slots, and the workflow now wires
all six (LoadImage 10–15 → RemoveBackground 20–25 → node 40 front/left/back/right/top/bottom). So the
hull reconstructs the whole surface — top face and underside included — from real splat geometry,
not a single visible face.

**Generative view completion for PARTIAL captures (ADR-017).** Six panels only help if the splat
*observed* those sides; an object filmed only from the front has no back data, so those panels render
empty. Added a coverage-gated FLUX.2 view-completion stage (`view_completer.py` +
`workflows/flux2_turnaround.json`, `ViewCompletionConfig`): per panel, compute splat coverage (alpha
fraction); panels above `keep_threshold` are kept as real renders, panels at/below `gap_threshold` are
**synthesised** by FLUX.2 — VAE-encoding the observed panels as `ReferenceLatent`s and conditioning on a
JSON-structured prompt (object identity + target view + hard consistency constraints; the Mistral-3
encoder follows it — the original FLUX.2 rationale). This is ADR-014 Generative Recovery applied to the
full turnaround, do-no-harm to real geometry. Wired into `Trellis2Client` (attached by `stages.py` when
`view_completion.enabled`); synthesised panels are plausible-not-measured (tag `v2g:view_synth=true`).
Qwen-Image-Edit-2511 is the commercial-safe alternative generator. Generation quality needs prompt/param
tuning on a real run (like the texture stack did).

## 2026-06-20 (cont.) — TRELLIS.2 PBR texture stack repaired + Unreal access blocker confirmed

Full PBR-textured GLB now verified (29MB: decimated 500k-face mesh + baked PBR texture), in
addition to the raw geometry GLB. Two fixes: (1) the texture chain is
`ImageToShape.mesh → Trellis2ProcessMesh → Trellis2RasterizePBR(trimesh, voxelgrid,
**original_mesh=ImageToShape.mesh**)` — taken from the pack's `geometry_texture.json` reference
(NOT a UV-unwrap node, which was my wrong guess). (2) A genuine node bug: `Trellis2RasterizePBR`
does `cv2.inpaint(single_channel, ...)[..., None]` for the metallic/roughness/alpha maps, which on
**OpenCV 4.13** yields a 4-D array → `np.concatenate` with the 3-D `base_color` raises
"all the input arrays must have same number of dimensions". Patched to force
`(texture_size, texture_size, 1)`; made durable + idempotent in `comfyui_entrypoint.sh` (step 1d).
Production `trellis2_multiview_pbr.json` set to **1536_cascade geometry / 4096 PBR** (the 48 GB-card
max-fidelity target).

**Unreal provisioning blocker confirmed:** `ghcr.io/epicgames/unreal-engine:dev-5.8` is **denied**
from the host (no `ghcr.io` login). The container/MCP/USD design (ADR-016) is complete; the gate is
purely the Epic↔GitHub account link + `read:packages` PAT + EpicGames-org membership (a one-time
credential step), plus the usual days–weeks lag before Epic publishes the `dev-5.8` image.

## 2026-06-20 — TRELLIS.2 hull RUNTIME-VERIFIED (produces GLBs) + Unreal/Drive

The designated-primary hull (ADR-015) now runs end-to-end: a single image (`bottle.png`)
→ **122 MB GLB** at `/comfyui/output/trellis2_*.glb`. Full unblock recipe, all folded into
`scripts/comfyui_entrypoint.sh` (durable across container rebuilds):

1. **Node import**: pip `comfy-env` + `comfy-sparse-attn` (prebuilt **cp312 wheel, no compile**) +
   `comfy-3d-viewers` + `trimesh[easy]` (the node's own requirements.txt). The prior session's
   "needs a comfy_sparse_attn build" was wrong — it's a pip install.
2. **DINOv3** (`Trellis2GetConditioning`): `facebook/dinov3-vitl16-pretrain-lvd1689m` is HF-gated
   (403 even with a valid token unless the account is on the allow-list). Sidestepped by pulling the
   **ungated `camenduru/dinov3-vitl16-pretrain-lvd1689m`** mirror (370k downloads, same checkpoint)
   to `models/dinov3/model.safetensors` — the node loads the local file, never touching the gated repo.
3. **CUDA extensions**: the node imports `cumesh_vb` / `o_voxel_vb_ap` / `flex_gemm_*` — the `_vb`/`_ap`
   variants on `pozzettiandrea.github.io/cuda-wheels/v2/`, which (unlike plain `cumesh`/`flex-gemm`)
   have **exact `cu130/torch2.11/cp312`** prebuilt wheels. Installed `--no-deps` (no compile).
4. **Pure-python**: `easydict`, `igraph`, `xatlas`, `zstandard`, `utils3d` (`--no-deps`).

Iterated each missing import against the live `/object_info` + `/prompt` API (vitrine-comfyui on
visionclaw_network). Authored + structurally-validated `src/pipeline/workflows/trellis2_multiview_pbr.json`
(orbit front/left/back/right → DINOv3 cond → multiview shape → textured voxelgrid → RasterizePBR → GLB).
`run_comfyui.sh` now passes `HF_TOKEN` from `~/.hf_token` for faster authenticated model pulls. Registry
marks TRELLIS.2 runtime-verified. Drive ingest (public folder via gdown) and the Unreal 5.8 export
design (ADR-016) landed alongside; see the 2026-06-19 entry.

## 2026-06-19 — Build fix + object-reconstruction SOTA refresh (PRD/ADR-015/DDD)

### 1. Consolidated Docker build unblocked (rclone step — NOT a network failure)

The build had been failing repeatedly at **Step 8/51** (rclone install), diagnosed across retries as a
network/CDN issue ("URL 200 from host but fetch fails in build"). Reproduced the real failure on the host
(tmux window 6 = `john@HP-Desktop`, legacy builder) with a minimal Dockerfile: the `curl`/`unzip` **succeed**;
the build dies on the final `rclone version` check with
`CRITICAL: Invalid value when setting --version from environment variable RCLONE_VERSION="1.69.1" … ParseBool`.
Root cause: rclone maps every `RCLONE_*` env var to a CLI flag, so the build `ARG RCLONE_VERSION=1.69.1`
(which leaks into the RUN env on this daemon) is read as `--version=1.69.1`. Deterministic, not transient.
**Fix:** `Dockerfile.consolidated` verification line → `env -u RCLONE_VERSION rclone version`. Verified on
host (`Successfully tagged rclone-fix:latest`).

### 2. 4-agent SOTA sweep (object turnaround / multi-view / image-to-3D)

Ran four parallel cited research agents → `research/2026-06-object-reconstruction-sota.md`. Headlines:
- **TRELLIS.2-4B (MIT, 24GB, 17–60s, full PBR, multi-view ≤16)** confirmed the right primary hull.
- **Qwen-Image-Edit (Apache-2.0)** is the only edit model benchmarked to do *instruction-driven view
  rotation* and is commercial-safe → new recovery option vs non-commercial FLUX.2.
- **MV-Adapter (Apache-2.0)** best-licensed dedicated camera-conditioned turnaround (P2 option).
- Community levers: git-lfs-before-pull, BiRefNet-HR matting, IC-Light relight, 512² diffuse spec.
- **LichtFeld upstream is still v0.5.2** (no newer tag); the opportunity is the v0.5.0 plugin system +
  community `lichtfeld-360` plugin (mirrors our SfM stack), not a version bump.

### 3. Engineering artifacts (drove the changes)

`research/prd/prd-object-reconstruction-sota-2026-06.md`, `research/decisions/adr-015-object-reconstruction-sota-refresh.md`,
`research/ddd/object-reconstruction-context.md` — PRD→ADR→DDD for the hull/recovery contexts: registry-driven
strategy, honest (logged) fallback chain, posture-aware recovery, ACL-isolated ComfyUI coupling.

### 4. Live-system ground truth (probed `vitrine-comfyui` /object_info, 605 nodes)

Connected `vitrine-comfyui` to `visionclaw_network` (it was only on bridge+v2g-net) so the agent reaches the
ComfyUI API directly. Findings that **change the decisions** vs the paper-level view:
- **TRELLIS.2 weights are present in the model tree** (`/comfyui/models/trellis2/ckpts/*` full set + `TRELLIS.2-4B`;
  `extra_model_paths.yaml` maps `trellis2:`/`hunyuan3d:`/`sam3d:`). The **only** blocker for the designated
  primary is that the **ComfyUI-Trellis2 node pack isn't installed** (no trellis class_types) — it needs a
  compiled `comfy_sparse_attn` ext (consistent with the prior session's note). **No 25GB pull needed.**
- **Hunyuan3D**: only `2.0-mv` weights staged (not 2.1 dit/paintpbr); PBR bake nodes
  (`Hy3DMultiViewsGenerator`→`Hy3DBakeMultiViews`) ARE installed but `hunyuan3d21_multiview.json` never
  calls them; stock `ImageOnlyCheckpointLoader` reads the empty `checkpoints/` dir, not the mapped
  `hunyuan3d/` dir.
- **Qwen recovery**: nodes + VAE + a qwen CLIP staged; **diffusion UNET not staged**.
- Net: *every* ComfyUI hull/recovery backend is currently gated on a node-pack install or a weight pull —
  which is why e2e runs fall back to TSDF. Captured in registry caveats (`sota_registry.py`, corrected
  2026-06-19) incl. the **engine-date bug fix** (v0.5.2 is 2026-04-21, was wrongly 2025).

### 5. Prioritised next actions (per ADR-015 phasing)

- **P1 unblock (highest leverage):** install + pin `ComfyUI-Trellis2` (build `comfy_sparse_attn`) in
  `vitrine-comfyui` → smoke one GLB from the already-staged weights → promote TRELLIS.2 to active primary.
- **P0 quick win:** rework Hunyuan workflow to the Hy3D wrapper loaders + PBR bake (or stage 2.1 dit) for
  textured output now.
- **Recovery:** pull Qwen-Image-Edit diffusion UNET (+ DiffSynth inpaint patch) to activate the
  commercial-safe path.
- Fold the ComfyUI node deps into `Dockerfile.consolidated` (drop per-launch install).

### 6. TRELLIS.2 hull UNBLOCKED (designated primary now loads)

The owner ComfyUI already had `ComfyUI-TRELLIS2` cloned but it failed to import
(`ModuleNotFoundError: comfy_sparse_attn` / `comfy_env`) — the prior session called it
"a separate build." It is **not** a build: its `requirements.txt` (`comfy-env==0.3.89`,
`comfy-sparse-attn==0.1.3`, `comfy-3d-viewers==0.2.44`, `trimesh[easy]`) is pip-installable, and
**comfy-sparse-attn ships a prebuilt cp312 wheel** — no compile. Installed it into the running
container via `docker exec`; the modules import; `docker restart` → ComfyUI now loads
**24 TRELLIS2 nodes** (`/object_info` 629 total). Made durable by adding the install to
`scripts/comfyui_entrypoint.sh` (before the safetensors repair so the repair stays last).
Validated hull chain: `Trellis2RemoveBackground → Trellis2GetConditioning →
Trellis2MultiViewImageToShape → Trellis2ShapeToTexturedMesh → Trellis2RasterizePBR →
Trellis2ExportGLB`. **Remaining runtime gate:** `Trellis2GetConditioning` needs DINOv3
(`facebook/dinov3-vitl16-pretrain-lvd1689m`), which is **HF-gated (401)** and unstaged — needs an
HF token with the licence accepted. Registry updated to this state.

### 7. Unreal Engine 5.8 USD-scenegraph export — designed (ADR-016)

UE 5.8 shipped 2026-06-17 with a **first-party experimental MCP plugin** (:8000 HTTP+SSE).
Designed an `unreal` GPU service: `research/decisions/adr-016-unreal-scenegraph-export.md`,
`unreal/Dockerfile.unreal`, `unreal/docker-compose.unreal.yml`, `unreal/runtime/import_usd_stage.py`.
Highest-fidelity decision: ingest via a **live USD Stage Actor + pxr** (NOT baked import, which
**silently drops `v2g:*` customData** in UE5.8). Lumen offscreen render (`-RenderOffscreen`); Path
Tracer is Windows/DX12-only (out of scope on Linux). Provisioning gated on Epic GitHub-org link +
`read:packages` PAT to pull `ghcr.io/epicgames/unreal-engine:dev-5.8`.

### 8. Newest-methods sweep (3 more agents) + Drive ingest

- **2D turnaround:** #1 = **Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA** (Apache-2.0, trained on
  Gaussian-Splat renders, 96 camera poses, node `jtydhr88/ComfyUI-qwenmultiangle`); Qwen edit nodes
  already installed. #2 = MV-Adapter (Apache-2.0). TAPESTRY (mesh→turntable video) on watch.
- **Textured 3D:** best texture path = TRELLIS.2 + **Pixal3D-T** texture inside `visualbruno/ComfyUI-Trellis2`;
  **CHORD** (Ubisoft, research-only) for PBR-channel expansion; `Trellis2MeshTexturingMultiView` for
  multiview bake (`texture_size=4096, resolution=1536`). Hunyuan3D-Omni new (point/voxel/bbox control).
- **Drive ingest:** the shared folder is public — `gdown` pulled `VERS.02a-TOP_ONLY.MOV` (1.78 GB) to
  `data/raw/` (no creds needed). `vitrine-comfyui` joined `visionclaw_network` for direct
  agent→ComfyUI API access.

---

## Phase 1: Foundation

### LichtFeld Studio Fork

Forked [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio), a native C++23/CUDA workstation for 3D Gaussian Splatting. LichtFeld provides training, visualization, editing, and export with an MCP server exposing 70+ tools. We chose it over alternatives (gsplat standalone, nerfstudio) because:

- MCP integration allows agentic control from Claude Code
- Native C++ performance for training (not Python-bound)
- Built-in scene graph, selection system, and multi-format export (PLY, SOG, SPZ, HTML, USD)

Established [BOUNDARIES.md](../BOUNDARIES.md) to enforce clean separation: upstream code is never modified on our branch.

### Docker Consolidation

Built a single consolidated Dockerfile (`Dockerfile.consolidated`) on `nvidia/cuda:12.8.1-devel-ubuntu24.04` containing:

- COLMAP 4.1.0 (built from source with METIS/GKlib)
- LichtFeld Studio (host-compiled binary, bind-mounted)
- Python 3.12 pipeline modules
- ComfyUI with SAM3D and FLUX nodes
- Claude Code (Node.js 23)
- Blender (headless)
- ttyd web terminal, VNC, supervisord

Single-command deployment: `docker compose -f docker-compose.consolidated.yml up -d`.

### SplatReady Integration

Integrated the SplatReady plugin for automated video-to-COLMAP pipeline: PyAV frame extraction at configurable FPS, automatic COLMAP feature extraction, exhaustive matching, sparse reconstruction, and undistortion.

---

## Phase 2: TSDF Mesh Extraction

### Initial Approach

After 3DGS training produces a gaussian splat model, we need polygonal meshes for downstream use (game engines, USD scenes, web viewers). First approach: render depth maps from the trained gaussians using gsplat, then fuse them into a mesh via Open3D TSDF.

### Implementation

Built `mesh_extractor.py` using:

1. gsplat to render depth + RGB from training viewpoints
2. Open3D `ScalableTSDFVolume` to fuse depth frames
3. Marching cubes to extract a triangle mesh
4. Vertex colour transfer from the gaussian splat

Results: 22K vertices, 49K faces. Geometric accuracy was acceptable for large structures but poor for fine details.

### Vertex Colours vs Texture Baking

TSDF meshes come with vertex colours, not UV-mapped textures. For web delivery this is sufficient (model-viewer handles vertex colours). For production USD scenes, texture baking is needed. Built `texture_baker.py` skeleton using xatlas for UV unwrapping, but deferred full implementation after discovering the quality ceiling.

### Discovery: TSDF Quality Ceiling

TSDF fusion from expected (rendered) depth has a hard quality ceiling. The depth maps from gaussian splatting are noisy at object boundaries and in regions with sparse training views. No amount of TSDF parameter tuning (voxel size, truncation distance, depth scale) fixes this because the problem is in the input signal, not the fusion algorithm.

---

## Phase 3: Mesh Extraction Research

### Methods Evaluated

| Method | Source | Approach | Finding |
|--------|--------|----------|---------|
| SuGaR | Guédon & Lepetit 2024 | Regularise gaussians to lie on surfaces, then Poisson mesh | Good surface alignment but slow (hours). Requires modified training. |
| GOF (Gaussian Opacity Fields) | Yu et al. 2024 | Learn opacity fields, extract level set | Better than TSDF but still limited by training quality |
| MILo | Wewer et al. SIGGRAPH Asia 2025 | Differentiable mesh-in-the-loop: Delaunay triangulation + learned SDF, mesh participates in the gaussian loss | Best quality. Mesh quality is bounded by gaussian quality. |
| CoMe (Compact Mesh) | various | Mesh compression of gaussians | Targets compression, not reconstruction quality |

### Key Insight: Training Quality is the Bottleneck

MILo produces the best meshes among evaluated methods, but all methods share a common ceiling: **the mesh can only be as good as the trained gaussians**. If the gaussians are noisy (floaters, stretched ellipsoids, missing regions), no mesh extraction method recovers lost geometry.

Root causes of poor gaussian quality in our test scenes:

1. **YouTube-compressed video**: H.264 compression artifacts reduce feature matching quality in COLMAP, producing fewer and less accurate camera poses
2. **Featureless walls**: Large uniform surfaces have no visual features for COLMAP to match, creating holes in the sparse reconstruction
3. **Reflective surfaces**: Glass cases, polished floors, and metallic frames violate the Lambertian assumption in both COLMAP and 3DGS
4. **Insufficient view coverage**: Walk-through videos miss ceiling details and behind-object views

The correct fix is better input capture, not better mesh extraction.

---

## Phase 4: MILo Integration

### CUDA Version Conflict

MILo requires:
- CUDA 11.8 (its CUDA extensions fail to compile with 12.x)
- GCC <= 11 (CUDA 11.8 does not support GCC 12+)
- PyTorch 2.3.1 with cu118

Our main container runs CUDA 12.8 + GCC 14 + Python 3.12. These are fundamentally incompatible. Conda environments were attempted but the CUDA toolkit version is a system-level constraint, not a Python-level one.

### Sidecar Container Solution

Built `docker/Dockerfile.milo` on `nvidia/cuda:11.8.0-devel-ubuntu22.04` with:
- Python 3.10 (Ubuntu 22.04 default)
- PyTorch 2.3.1 + cu118
- All 4 MILo rasterizer variants compiled from source
- nvdiffrast, simple-knn, fused-ssim
- tetra-triangulation (Delaunay, CGAL + pybind11)

The sidecar runs on GPU 1, sleeps until called. The main container invokes it via:
```bash
docker exec milo python3 train.py --source_path /data/output/JOB/colmap ...
```

Shared `/data/output` volume allows both containers to read COLMAP data and write mesh results without network transfer.

### MILo Extractor Module

Built `src/pipeline/milo_extractor.py` to:
1. Check if the `milo` container is running
2. Convert pipeline paths to container-relative paths
3. Call MILo training via `docker exec` with appropriate arguments
4. Monitor progress via log file polling
5. Convert MILo's PLY output to GLB for the web viewer
6. Fall back to TSDF if the sidecar is unavailable

---

## Phase 5: Blender Scene Assembly

### Motivation

The pipeline needs a final assembly step that:
- Imports TSDF or MILo meshes
- Cleans debris (small disconnected components)
- Creates proper materials from vertex colours
- Sets up lighting
- Renders preview images
- Exports a USD scene with proper hierarchy

### Implementation

Built `src/pipeline/blender_assembler.py` to run headless:
```bash
blender --background --python blender_assembler.py -- --input mesh.glb --output-usd scene.usda
```

Uses Blender's Cycles renderer with GPU compute for texture baking. Creates a 3-point lighting setup, imports COLMAP camera poses for aligned preview renders.

---

## Phase 6: Web Interface

### Flask App

Built `src/web/` with:
- Video upload (drag-and-drop, file size validation)
- Job management (create, track, cancel, delete)
- SSE log streaming for real-time pipeline progress
- 3D model preview via Google's `<model-viewer>` web component
- Preview image carousel from Blender renders
- ZIP download of all job outputs
- Anthropic API key management (stored on persistent volume, not in container image)

### SAM3 Object Segmentation

SAM3 (Segment Anything Model 3) provides concept-based segmentation with 4M concepts using text + visual prompts. Requires `HF_TOKEN` environment variable for model downloads from HuggingFace. Falls back to SAM2 grid-point prompts if SAM3 is unavailable.

---

## Current State

The end-to-end pipeline works: video upload through web UI, frame extraction, COLMAP SfM, 3DGS training via LichtFeld MCP, object segmentation, TSDF or MILo mesh extraction, Blender assembly, USD export, and web preview/download.

Primary quality limiter remains the input video. High-quality results require:
- 4K or higher resolution source video
- Slow, deliberate camera motion with overlap
- Multiple passes from different heights/angles
- Avoiding reflective and transparent surfaces
- Good, even lighting

The mesh extraction backend (TSDF vs MILo) matters less than the quality of the trained gaussians, which in turn depends almost entirely on the quality of the input video and COLMAP reconstruction.

---

## v2 Upgrade — 2026-05-26

### Overview

This entry records the v2 pipeline upgrade, which was designed by a managed mesh swarm (multi-agent parallel development) and covers the decisions, new modules, and architecture changes introduced on branch `feat/v2-upgrade-swarm`.

### Upstream Sync to v0.5.2 (ADR-002)

The fork had diverged from upstream LichtFeld Studio by approximately 410 commits spanning two stable releases (v0.5.1 and v0.5.2). The high-value features in v0.5.2 were:

- **Native USD import/export** (#1032) — eliminates the need for our custom `usd_assembler.py` as the sole USD path
- **Native mesh support** (#876, #879, #889) — mesh loading, mesh-to-splat conversion, mesh picking inside LichtFeld
- **MRNF densification** (#1031) — new default densification strategy; renamed from LFS
- **Enhanced MCP server** (#984) — additional tools for agentic pipeline control
- **VRAM optimisations** — reduced peak VRAM during evaluation and image loading

We chose to sync to the **v0.5.2 stable tag** (released 2026-04-21) and explicitly deferred the v0.5.3 Vulkan-only rendering migration to ADR-008. The rationale: v0.5.3 is unreleased; the Vulkan migration removed the CUDA and OpenGL renderers entirely (#1170, #1234); and a known coordinate-system regression (issue #1104, from PR #1066) could break `coordinate_transform.py`. The v0.5.2 baseline delivers all capability we need without any of those risks.

**Isolation policy confirmed**: this fork is one-way pull only. We never push to or open PRs against the upstream repository (origin/MrNeRF/LichtFeld-Studio).

### Four Mesh Extraction Backends (ADR-003, ADR-004, ADR-005)

The v1 pipeline supported two mesh extraction backends: TSDF (fast, lower quality) and MILo (high quality, ~69 min). v2 adds two more:

**CoMe** (`come_extractor.py`, ADR-004): Confidence-based Mesh Extraction from github.com/r4dl/CoMe. CoMe trains 3DGS with per-Gaussian confidence values, then extracts a mesh via marching tetrahedra. Benchmarks: ~25 min total on RTX 4090 vs. MILo's ~69 min, at comparable F1 scores (0.521 Tanks & Temples, 0.662 ScanNet++). CoMe requires Python 3.10 and CUDA 12.1, which is incompatible with both the main container (CUDA 12.8, Python 3.12) and the MILo sidecar (CUDA 11.8). It therefore runs in a new dedicated `come` sidecar (`docker/Dockerfile.come`). The sidecar is present in `docker-compose.consolidated.yml` but gated behind `--build-arg INSTALL_COME=1` because CoMe carries no LICENSE file as of 2026-05-26 (SPDX: NOASSERTION). It must not be used in commercial distribution until a permissive licence is published and reviewed.

**GaussianWrapping** (`gaussianwrapping_extractor.py`, ADR-005): From github.com/diego1401/GaussianWrapping. Reinterprets 3D Gaussians as stochastic oriented surface elements and extracts watertight, textured meshes that capture thin structures (bicycle spokes, wires, fences, railings) where TSDF and marching cubes fail. GaussianWrapping requires exactly CUDA 11.8 and Python 3.9 -- matching the MILo sidecar environment -- so it is installed into the existing `milo` container at `/opt/gaussianwrapping` rather than requiring a new container. It is gated behind `--build-arg INSTALL_GAUSSIANWRAPPING=1` for the same licensing reason (no formal LICENSE file).

**Pluggable backend architecture** (ADR-003): All four backends expose the same three-symbol interface (`XConfig` dataclass, `is_X_available() -> bool`, `run_X(colmap_dir, output_dir, config) -> dict`). Backend selection is centralised in `stages._select_mesh_backend()`. When `config.training.mesh_method = "auto"`, the function applies the heuristic: thin-structure hint → GaussianWrapping; CoMe available → CoMe; MILo available → MILo; fallback → TSDF. Explicit values (`"tsdf"`, `"milo"`, `"come"`, `"gaussianwrapping"`) bypass auto-selection for reproducible runs.

**CLI flag notice**: The CoMe and GaussianWrapping CLI flags are inferred from their upstream repositories (the SOF codebase for CoMe; the GaussianWrapping repository structure for GW). They have not been verified against the actual released source. All script names and flag constants are defined as module-level constants in `come_extractor.py` and `gaussianwrapping_extractor.py` so that corrections can be made in one place once the code is reviewed.

### Splat-Transform Delivery Stage (ADR-006)

Added `splat_optimizer.py`, which wraps the PlayCanvas `@playcanvas/splat-transform` npm CLI. The module is invoked as a `SPLAT_OPTIMIZE` stage after 3DGS training and before web delivery. It applies crop, filter, sort, and compress operations to produce a `.ksplat` file targeting under 20 MB from a raw PLY of 100+ MB. Node.js and npx must be present in the main container. The original `.ply` is always kept alongside the compressed form for downstream mesh extraction backends. The stage is opt-in via `config.delivery.enable_splat_optimize = True`.

### Fibonacci-Sphere Frame Selection (ADR-007)

Added `fibonacci_sampler.py`, which provides `fibonacci_sphere()`, `fibonacci_coverage_score()`, and `select_frames_by_coverage()`. The module is imported by `frame_selector.py` when `config.ingest.use_fibonacci_coverage = True`. After COLMAP SfM, camera positions are scored by their coverage of a Fibonacci-sphere distribution (a near-optimal low-discrepancy point set on the unit sphere). The combined frame score is:

```
score = 0.6 * quality_score + 0.4 * fibonacci_coverage_score
```

The weights are configurable via `config.ingest.coverage_weight`. The Fibonacci scoring falls back silently to the v1 quality-only path if COLMAP positions are unavailable (pre-SfM pass or degenerate reconstruction). No new runtime dependencies beyond NumPy.

### Architecture and DDD Model

The v2 upgrade produced eight Architecture Decision Records (`research/decisions/adr-001` through `adr-008`) and a full DDD domain model (`research/ddd/bounded-contexts.md`, `research/ddd/aggregates.md`). The domain model identifies seven bounded contexts (Ingestion, Reconstruction, Training, Segmentation, MeshExtraction, SceneAssembly, Delivery) and four aggregate roots (`ReconstructionJob`, `GaussianModel`, `MeshAsset`, `SceneGraph`, `DeliveryArtifact`).

The MeshExtraction context is architecturally the most complex in v2 because it spans all three containers and four backends. The physical container boundary is treated as infrastructure, not a bounded-context boundary; the sidecar ACL (`milo_extractor.py`, `come_extractor.py`, `gaussianwrapping_extractor.py`) translates domain commands into container-specific CLI invocations.

### What Was Deferred (ADR-008)

The v0.5.3 Vulkan migration is explicitly deferred. Trigger conditions for revisiting:
1. v0.5.3 released as a stable tagged version
2. Coordinate-system regression (issue #1104) resolved upstream
3. Headless Vulkan validated in a Docker container on our GPU model
4. MCP API compatibility verified for `mcp_client.py`
5. Python API audit complete

No v0.5.3-dev commits are merged until all five conditions are met. The `upstream/master-watch` branch tracks upstream progress monthly.

---

## v3 Upgrade — 2026-06-04

### Overview

The v3 increment converts the pipeline from a single-host, hardcoded-IP research
script into a manifest-driven, service-meshed, agent-overseen system targeting
2026 SOTA models. The work was built as a six-agent mesh swarm with disjoint file
ownership and reconciled against ADR-011 through ADR-015 and the v3 work-order
(items 0–9). FR-40 (the `video2gaussian` / `gaussian-toolkit` → `Vitrine` codebase
rename) remains explicitly deferred for blast-radius reasons; GPU-host validation,
live weight staging, and live pin resolution are out of band on the .48 host.

### Single Pre-Run Manifest (ADR-013 / D-013.1)

Replaced ad-hoc CLI flags with one human-authored `exhibit.toml`. `pipeline/manifest.py`
parses it, resolves `env:NAME` secret indirection at load time (a missing referenced
env var is a hard, named failure — exit 2), and materialises a runtime `PipelineConfig`.
Secrets (`hf_token`, `gcloud_credentials`) reject inline literals and are stripped before
the redacted JSON run-record is written. The loader maps objects → `decompose.sam3_concepts`,
`mesh_backend` → `training.mesh_method`, `matcher` → `reconstruct.matcher`, and the
endpoint/oversight overlays onto their config sub-objects. `exhibit.example.toml` documents
the schema. CLI: `python -m pipeline.manifest exhibit.toml [-o run.json]`.

### SOTA Idiot-Check Wired Into Preflight

`pipeline/sota_registry.check_environment()` is now invoked from `preflight.check_all()`
and `print_report()`. It is advisory by default — it logs a registry report (checkpoints
staged, VRAM fit, licence posture, pinning, caveats) and never raises — but escalates a
`FAIL` overall to a hard `RuntimeError` when `SOTA_STRICT` is set. Default posture remains
RESEARCH / non-commercial.

### Serial VRAM Lifecycle + Service-DNS Endpoints (D-013.2, D-013.3)

`pipeline/model_lifecycle.py` introduces `ModelLifecycleManager.stage()`, a context manager
that asserts VRAM headroom before a stage and unloads serially afterwards (soft = POST /free
+ `torch.cuda.empty_cache()`; hard = container stop), so peak VRAM is `max(stage)` rather than
the sum. `pipeline/endpoints.py` replaces hardcoded `localhost` IPs with an `Endpoints`
dataclass reading `V2G_*` env vars over a docker service-DNS mesh (`comfyui:8188`,
control-plane `:3001`, `agent-vlm:8080`, `milo:8090`, `come:8091`); the legacy single-host IPs
are retained only as named fallback constants.

### Agent-Controlled ComfyUI (ADR-014)

`pipeline/comfyui_control.py` gives the oversight agent direct probe/download/run/free control
over the .48 ComfyUI instance and Salad control-plane (health, `probe_models`, `ensure_model`,
`submit_workflow`, `wait`, `download_outputs`, `free_vram`), with a `requests`→`urllib` fallback
so it runs in a dependency-thin container.

### Web Onboarding (ADR-015)

`onboarding/` is a Rust/Axum service (`:8088`) serving a six-step vanilla-JS wizard that
round-trips `exhibit.toml`. `POST /api/manifest` writes the manifest with `env:` references only;
raw tokens are diverted to a `chmod 0600` `.secrets.env` and never echoed back. `cargo check`
clean.

### SOTA Model Modernisation (work-order items 0, 2, 4, 8)

- **Inpainting**: `comfyui_inpainter.py` adds a FLUX.2 path (`flux2_inpaint.json`, 15-node API
  graph) selected when FLUX.2 weights are present, otherwise falling through unchanged to the
  proven FLUX.1-Fill path.
- **3D recovery**: `hunyuan3d_client.py` adds Hunyuan3D-2.1 textured-PBR multiview
  (`hunyuan3d21_multiview.json`, 16-node graph) with graceful degradation 2.1 → 2.0-mv →
  single-view, and a SAM3D fallback when multi/single-view both fail.
- **Defaults**: `config.py` moves the training strategy default to `igs+`, mesh backend to
  `come`, the inpaint model to `flux2`, and Hunyuan to `2.1`; adds `EndpointsConfig` and
  `OversightConfig` with `validate()` coverage. Matching is ready for ALIKED + LightGlue via
  `reconstruct.matcher`.

### Version Pinning (work-order item 7)

`pins.lock.toml` records 12 upstream components (11 git, 1 pip) with repo / kind / ref /
host-path / clone-site, leaving `resolved_commit` empty rather than fabricating SHAs.
`scripts/resolve_pins.sh` performs a read-only `git rev-parse HEAD` per component on the host
and writes `pins.resolved.toml`. Resolution itself is a host-side step.

### What Was Deferred

FR-40 codebase rename (high blast radius, mechanical, scheduled separately); GPU smoke and
weight staging (host-only); live pin resolution (requires the .48 checkouts); pytest execution
(no pytest in the build container — the two new suites, `test_model_lifecycle.py` and
`test_comfyui_control.py`, are AST-clean and run in CI / on the host).

---

## v4 End-to-End Validation — 2026-06-05

### Overview

The pipeline was run **end to end on a real scene** for the first time, taking it from
*designed* to *demonstrated*. A reused 80-frame indoor capture (`output/milo_run`) was trained
to a 4M-gaussian field (LichtFeld `igs+`, `splat_30000.ply`, SH degree 3) and driven through
segmentation → object isolation → meshing → dual-USD assembly. The run produced five isolated
object PLYs, five per-object meshes, a 901 MB native splat USD, and a composed textured
`scene.usda` with four preview renders. Every stage surfaced a real defect; eight were fixed.

### Object resolution now works (SAM3 + ADR-010 D10)

- **SAM3 #507** — the SAM 3.1 fused `addmm_act` casts operands to bfloat16 and never restores,
  crashing segmentation (`mat1 and mat2 must have the same dtype`). Patched at the source binding
  in `sam3_segmentor.py` (monkey-patch, survives container rebuilds). SAM3 now resolves 5 objects
  (sculptures, furniture, walls, floor, ceiling). We segment **stills** per-frame then union by
  concept — no video tracker — so SAM3 (kept after a web-verified SOTA check) remains the right model.
- **Depth-aware multi-view projection (D10)** — replaced the broken world-XY heuristic in
  `extract_objects`. `segment()` now persists per-frame masks; `_extract_with_mask_mv` projects every
  gaussian centre through each registered COLMAP camera that has a per-frame mask, votes inside/outside
  the mask across views (skipping empty-mask frames), and keeps gaussians inside the object in a
  majority of detected views. Isolated: sculptures 1,080,171; furniture 452,401; floor 964,074;
  walls 357,276; ceiling 11,012 gaussians — correctly keyness-ranked (sculptures/furniture above
  structural surfaces). Three sub-defects fixed en route (empty-mask vote inflation, absent SH
  vertex colours, alignment).

### Meshing, native USD, composed USD

- **gsplat SH-degree loader** — `load_3dgs_ply` hardcoded 45 f_rest coefficients but the trained PLY
  is SH degree 1 (9); now reads the actual degree and zero-pads. gsplat-TSDF then produces real
  meshes (was a degenerate fallback).
- **Hunyuan3D kwargs** — `turbo` keyword crashed the client; kwargs filtered to the constructor signature.
- **LichtFeld runtime** — the prebuilt binary needs the host CUDA-13 runtime, vcpkg OpenUSD libs and
  `libz-ng`; resolved via `LD_LIBRARY_PATH` (CUDA + vcpkg dirs) plus staging `libz-ng.so.2` into the
  bind-mounted `build/`. This unblocked **training** in-container.
- **Native USD** — rewired `_export_native_usd` from the never-running MCP server to the headless
  LichtFeld CLI `convert` subcommand (`LichtFeld-Studio convert <ply> <out.usda>`).
- **Blender** — fixed the bake selecting the glTF `world` root (`Object 'world' is not a mesh`) and the
  Blender 5.0 `wm.usd_export` keyword change (`overwrite_existing_textures` removed); both now pass, so
  `blender_assembled=True` with textured `scene.usda` + 4 renders.

### Honest boundary

Validation was on the **reused** `milo_run` scene, not a fresh Drive→ingest→COLMAP capture. Meshes use
the **gsplat-TSDF fallback**, not the SOTA single-image hulls (TRELLIS.2 / Hunyuan node deps unbuilt).
The FLUX.2 recovery loop and local gemma-4 VLM are staged but not wired. Isolation quality is first-pass
(sparse SAM3 detection, no per-view depth occlusion → coherent but over-inclusive). These are tracked as
in-progress/pending in `report/main_v4.tex` (the consolidated current-state report) and the README.

### Docs

ADR catalogue reconciled to current design (ADR-001 rewritten as the live architecture; evolved ADRs
amended in place). The original bid + pitch brief extracted to `docs/brief/`. New consolidated
current-state report `report/main_v4.tex` (v1/v2/v3 left as historical snapshots).

## Agent-LLM rewire to DiffusionGemma + ComfyUI fix — 2026-06-08

### Agent LLM: gemma-4 `agent-vlm` → host-served DiffusionGemma

The pipeline's local agent LLM was rewired from the never-wired, containerised gemma-4 `agent-vlm:8080`
to **DiffusionGemma 26B-A4B** (Gemma-4 MoE, ~4B active, Q8_0), served on the GPU host by the llama.cpp
`llama-diffusion-gemma-visual-server` (PR #24423) behind a thin stdio→HTTP wrapper
(`llm-server/diffusiongemma-lan-server.py`). It is OpenAI-compatible on `:8084`
(`POST /v1/chat/completions`, `GET /health`), model id `diffusiongemma-26B-A4B-it-Q8_0`.

Key property: **the GGUF build is text-only** (multimodal arch, text-only weights). Length is set by
`n_blocks` (256 tok/block), not `max_tokens`; deterministic by `seed`; temperature ignored; 12288-token
context; single-context (serialize calls). Consequence: *visual* per-frame artifact triage (FR-27) now
defaults to the `claude_code` oversight backend; DiffusionGemma fills the **text reasoner/overseer** role.
The staged gemma-4 vision GGUF (+mmproj) is retained in the registry as the vision fallback.

Code:
- New `src/pipeline/agent_llm.py` — OpenAI client (`AgentLLM`, health/chat/ask, reasoning_content split,
  text-only image guard) + advisory `check_agent_llm()`.
- `endpoints.py` — service `agent_vlm_url` → `agent_llm_url` (default `http://localhost:8084`, env
  `V2G_AGENT_LLM_URL`); `agent-llm`/`llm` aliases added; deprecated `agent-vlm`/`vlm` aliases still resolve.
- `config.py` — `EndpointsConfig.agent_llm_url` + `agent_llm_model`; oversight values `gemma_local` →
  `diffusiongemma`; `artifact_vlm` default → `claude_code` (vision-capable). Validation sets updated.
- `sota_registry.py` — `vlm` element → `agent_llm` (DiffusionGemma primary, `requires_staged=False` —
  validated by the preflight connectivity probe, not a staged-weight check; gemma-4 vision + Qwen3-VL fallbacks).
- `preflight.py` — advisory DiffusionGemma connectivity probe (`check_agent_llm_endpoint`) in `check_all`
  + `print_report` (never blocks the core reconstruct→decompose flow).
- `manifest.py` — `agent_llm_url` override (accepts the deprecated `agent_vlm_url` key); `gemma_local` →
  `diffusiongemma` migration in the loader.
- `docker-compose.consolidated.yml` — `V2G_AGENT_LLM_URL` (default `http://host.docker.internal:8084`) +
  `V2G_AGENT_LLM_MODEL`, plus `extra_hosts: host.docker.internal:host-gateway` so the container reaches
  the host endpoint. `exhibit.example.toml` + tests updated.

Validated: 31/31 relevant pytest pass (`test_model_lifecycle.py`, `test_comfyui_control.py`); the
`AgentLLM` client reaches DiffusionGemma **from inside a container** (`host.docker.internal:8084`) — health
ok, a real reasoning call returned a coherent answer; the preflight probe prints `Agent LLM [PASS]`.

### ComfyUI fixed + stabilised

`vitrine-comfyui` was crash-looping (exit 1): installing `diffusers` (for the Hunyuan3D node) exposed a
corrupt `safetensors` dist-info in the image — `importlib.metadata.version()` returned `None`, so
`transformers`' `require_version()` raised and ComfyUI never started. Fixed with a new
`scripts/comfyui_entrypoint.sh` (mounted by `run_comfyui.sh`, which now recreates a fresh container each
launch) that: installs the Hunyuan3D-2.1 node `requirements.txt` + extras (`GitPython`, `toml`, `loguru`,
`rembg`, `onnxruntime`), then **repairs safetensors last** (physically removes the broken dist-info +
clean reinstall — `--force-reinstall` alone can't overwrite the unreadable metadata).

Result: ComfyUI serves on host `:8200` (→ `:8188` on v2g-net). `ComfyUI-Manager`, `comfyui-sam3dobjects`,
`websocket_image_save`, and **`ComfyUI-Hunyuan3d-2-1`** import cleanly. `ComfyUI-TRELLIS2` still needs a
compiled `comfy_sparse_attn` extension (+ a newer `comfy_env`) — a separate build, left as a follow-up;
Hunyuan3D-2.1 is the staged PBR-hull path meanwhile. Folding these deps into `Dockerfile.consolidated`
(to drop the per-launch install) remains a persistence chore.

### End-to-end re-validation (`output/e2e_run3`)

Re-ran preflight + the decompose→mesh→USD path on the reused `e2e_run2` reconstruction (trained splat +
COLMAP; no 30k retrain) on GPU1. Preflight printed `Agent LLM [PASS] diffusiongemma … @
host.docker.internal:8084`. Stages: segment OK (SAM3, 5 objects) → extract_objects OK (5; ranking
sculptures/furniture/floor/walls/ceiling) → mesh_objects OK (5) → texture_bake OK (5) → assemble_usd OK
(composed Blender `scene.usda` + 4 previews). The per-object PLY sizes are **byte-identical to e2e_run2**,
confirming the rewire did not perturb the decompose→USD path. Native USD export was skipped only because
the GPU1 one-off didn't wire the LichtFeld binary's runtime libs (`LD_LIBRARY_PATH` + `libz-ng`) — it fell
back to the composed Blender USD by design; native export worked in e2e_run2 with the full lib setup.

## 2026-06-23 — Dreamlab CoMe scene + full Blender assembly (overnight)

Pushed the user-chosen **CoMe** env-mesh backend through to a textured game-asset scene and a full
room+objects assembly.

- **CoMe mesh rescue.** CoMe trained its own gaussians (30k iters, 59 min) and `extract_mesh_tets.py`
  produced a 552 MB / 29M-face geometry-only mesh. The **raw tets mesh is unusable** — marching
  tetrahedra emit huge triangles bridging empty space + thousands of floater islands (a tiny dense core
  under giant stray sheets; bbox inflated ~2.5×). Rescue, in `scripts/come_color_texture.py`:
  edge-length filter (drop tris > 8×median edge → −776k strays) → keep largest connected component
  (−44,944 islands → 1 comp) → memory-safe decimate (voxel `simplify_vertex_clustering` THEN light
  quadric; quadric-alone OOMs/SIGKILLs on 29M faces) → NN colour-transfer from the colour-bearing TSDF
  mesh (CoMe shares TSDF's COLMAP world frame, so the same `ue_transform.json` M is reusable). Result:
  coherent 434k-face coloured room at **rough TSDF-parity** — not a clear win, and NON-COMMERCIAL licence,
  so **TSDF stays the safer scene mesh**.
- **Texture bake gotchas.** `scene_texture.py` (MeshCleaner → xatlas) **hangs >12 min** inside
  `MeshCleaner.clean()` (its `trimesh.mesh.split()` component filter is pathological on this mesh). The
  CoMe mesh is already clean, so `scripts/scene_texture_fast.py` skips MeshCleaner. Second: open3d
  **quadric decimation preserves open boundaries**, so on a holey room it barely reduces faces (418k) AND
  leaves thousands of tiny UV charts → xatlas grinds 20+ min. **Vertex-clustering** decimation merges
  across the holes → 120k faces, far fewer charts → xatlas finishes in ~2 min → `room_textured.obj` +
  `room_albedo.png` (2048, uv=True) → `room_come_ue.fbx`.
- **Reliable full-scene render (Blender, not UE).** The UE live-editor MCP assembly is **flaky on the
  final step**: `get_actor_bounds` throws (→ objects fall back to scale=1.0, tiny), `find_actors` returns
  0, the room mesh doesn't render, and the script exits mid-`tone_lighting` before capture. So the
  authoritative assembly proof is `scripts/blender_scene_assemble.py` — Blender bbox measurement is
  deterministic, so objects size correctly (chair 95, dartboard 45, vacuum 115, ladder 210 cm), placed
  per `placements.json` with the room transformed COLMAP→UE-cm. (Watch: scene is in **cm** units → set
  camera `clip_end` large or the far wall clips to black.)
- **Object quality (honest).** Per-object Hunyuan3D GLBs carry textures and render cleanly solo. Quality
  varies: **chair excellent** (recognisable, wood-grain + cushion), dartboard OK, **vacuum marginal**,
  **ladder poor** (reconstructs as a flat slab, not an open ladder — and is the one whose FBX UE rejects
  with "produced no assets"). `mitre_saw`/`toolbox` never built (ComfyUI was down). The dominant limiter
  is the capture itself (motion-blur, MUSIQ ~31) for both the room and the weaker objects — a better
  capture is the real fix, not a better extractor.
- **Artifacts:** `output/dreamlab/scene_come/{room_come_colored.ply, room_textured.obj, room_albedo.png,
  room_come_ue.fbx, scene_full*.png, objects_showcase.png}`.
- **UE assembly RESOLVED (same night).** A clean UE 5.8 assembly succeeds via
  `unreal/runtime/ue_place_prescaled.py`: `docker restart vitrine-unreal` (clears flaky editor state) →
  pre-scale object FBXs to real-world cm offline (`blender_prescale_objects.py`, regenerated from the
  **GLB** so the UE-incompatible `ladder.fbx` is replaced) → place at `scale=1.0`, location
  `(x, y, floor_z_cm)` with NO `get_actor_bounds` calls (heights from an offline `object_meta.json`).
  Room (Nanite) + all 4 objects import, place, and render in UE — `output/renders/dreamlab/
  ue_come_final{,_top,_corner}.png`. The brief target (textured mesh scene as UE game assets) is met;
  the render is dominated by capture-driven room noise. Open polish: clamp `floor_z_cm ≥ 0` (vacuum sank
  to z=-48); an interior camera would read more "room-like" than the floating-shell overview.

## ArtiFixer recon-enhancement branch vendored + sidecar stood up (ADR-020) — 2026-06-24

Brought the NVIDIA **ArtiFixer** trial (researched 2026-06-23; ADR-020 + PRD + DDD +
QE audits) into the repo and stood up the sidecar on `v2g-net`.

- **Submodule:** `docker/artifixer/ArtiFixer` = `nv-tlabs/ArtiFixer` pinned to the
  QE-audited `c320752` (Apache-2.0 code; recursive incl. `3DGRUT-ArtiFixer`,
  tiny-cuda-nn, optix-dev). The 67.6 GB `artifixer-14b.pt` weight (NVIDIA
  OneWay Noncommercial) stays **out of git** — downloaded at `~/artifixer-ada/ckpt`,
  bind-mounted via `ARTIFIXER_CKPT_DIR` (gitignored `data/artifixer/ckpt` symlink
  resolves the compose default).
- **Image:** `docker/artifixer/Dockerfile.ada-sm89` (vendored Ada fork of upstream
  `Dockerfile.cuda12`) — `FROM nvcr.io/nvidia/pytorch:25.01-py3`, strips the
  FlashAttention-3/4 block + CI asserts, builds torch 2.11 cu128 + 3DGRUT
  (slangc/JIT) + MoGe. Zero application-code changes (the model falls through to
  cuDNN SDPA on sm_89). Built with the **legacy builder** (`DOCKER_BUILDKIT=0` —
  no buildx on this host).
- **Sidecar:** `docker-compose.artifixer.yml` — optional overlay, `artifixer`
  service on `v2g-net`, GPU 1, `sleep infinity` + `docker exec` (milo/come pattern).
  App code bind-mounted (the image bakes only `thirdparty/` deps); COLMAP inputs +
  enhanced-recon outputs shared via `./output`.
- **Adapter:** `src/pipeline/artifixer_adapter.py` — locates + validates the
  enhanced 3DGRUT recon PLY (3DGRUT exports standard INRIA 3DGS format:
  `f_dc_*/f_rest_*/opacity/scale_*/rot_*`, exactly what CoMe/TSDF ingest) and
  writes the PLY the mesh stage consumes; optional similarity `world_transform`
  for COLMAP/LichtFeld alignment (defaults to identity — alignment to be confirmed
  on the first real scene, per the ADR-020 adapter risk).
- **Docs:** `docs/artifixer-recon-enhancement.md` (build/run/gate/risks) + README.

**Standup verified (2026-06-24).** Image `gaussian-toolkit-artifixer:ada-sm89`
built (54.6 GB; `cv2 ok`, `MoGe ok`) and the sidecar is up on `v2g-net`. Two
runtime findings:
- **GPU compose fix.** The container saw `nvidia-smi` but torch threw "CUDA
  unknown error" when the service used **both** `runtime: nvidia` *and*
  `deploy.resources.devices` (DeviceRequests) — the legacy runtime + the new
  device-request API double-inject and break CUDA init. Dropped the `deploy`
  block; `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=1` (the proven host pattern)
  now works: `cuda avail True`, capability `(8,9)`, bf16 matmul + SDPA both run.
- **arch_list note (QE risk #1).** The torch `2.11.0+cu128` wheel ships
  `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — **no explicit `sm_89`**.
  Ada runs fine via Ampere `sm_86` binary-compat (verified: a real bf16 CUDA
  matmul + SDPA execute on the device). The ArtiFixer app stack
  (`data_processing.run_artifixer3d`, `prepare_colmap_artifixer_inputs`) imports
  clean — no FlashAttention-3 crash on sm_89, confirming the fork ruling.

Still a **trial gated on per-scene evidence** (ADR-020): the first ArtiFixer3D run
on a real dreamlab scene must show a measurably less holey/noisy downstream
CoMe/TSDF mesh before the branch is wired into the gated pipeline. ArtiFixer does
not deblur, so blur-dominated scenes fall back to upstream frame-QA/recapture.

