# Video Ingestion Plan — Google Drive Bulk Capture Processing

**Date**: 2026-05-26
**Status**: Proposed
**Context**: ~70 GB of raw 4K MP4 on Google Drive, multiple "sets" per room, slow orbits at varying heights/angles always facing center. Current ingestion is browser drag-and-drop (2 GB cap) — unsuitable.

---

## 1. Capture Assessment (good news first)

The described capture — **slow orbits around a small room at different heights, always facing the center** — is close to ideal for COLMAP + 3DGS:

- **Inward-facing / object-centric** geometry → bounded-scene reconstruction. Matches the MILo `imp_metric="indoor"` preset and the indoor-reflective training preset already in `config.py`.
- **Multiple orbits at different heights** give the *vertical parallax* that single-height orbits lack — this is exactly what fixes the "ceiling/floor smear" failure mode. Keep doing this.
- **Slow motion** → low motion blur, dense temporal overlap → reliable feature matching.

**The key processing decision this enables**: all sets of the *same room* should be fused into **one combined COLMAP reconstruction**, not processed independently. More registered views = better coverage and a single consistent coordinate frame. This is the single biggest quality lever.

**Watch-outs**:
- Sets must share visual overlap so COLMAP registers them into one connected model (not disconnected components). Slow orbits of the same room normally do.
- 4K is VRAM- and time-heavy and largely wasted for geometry. Downscale for SfM/training; retain full-res only for final texture baking.
- The photographer/rig may appear in frame on a 360 orbit → `person_remover.py` pass.

---

## 2. Ingestion Architecture — Options & Recommendation

Current state: drag-and-drop, `MAX_CONTENT_LENGTH = 2 GB` (`src/web/app.py:136`). No rclone, no Google Drive client, no scratch volume in the container.

| Option | How | Verdict |
|--------|-----|---------|
| **A. Drag-and-drop (current)** | Browser upload | ❌ 2 GB cap, no resume, manual. Dead end for 70 GB. |
| **B1. `rclone mount` (read-only)** | FUSE-mount Drive into container; files stream on demand | ⚠️ Good for *discovery* and the *one-time read* during frame extraction. **Bad as the COLMAP/training working set** — those do heavy random re-reads; over a network mount that's painfully slow and fragile. |
| **B2. `rclone copy` pull-per-set** | Copy one set to local NVMe, process, push results, delete raw | ✅ Best for the heavy compute. Local NVMe random access is what COLMAP/3DGS need. |
| **C. Hybrid (RECOMMENDED)** | `rclone mount` (or `lsjson`) for listing/selection + `rclone copy` the active set to NVMe scratch; pull→process→push→purge | ✅ Best overall. Never holds more than the active session locally; raw stays in Drive as source of truth. |

**Recommendation: Option C — a pull → process → push → purge worker.** Mount only for browsing; copy the active session to fast local scratch for the actual pipeline; upload outputs back to a Drive output folder; delete the local raw + heavy intermediates when done.

---

## 3. Credentials & Security

- Use a **Google service account** (headless, no OAuth browser dance) with **`drive.readonly`** scope on the raw-capture folder. A separate writable folder (or a second account/scope) receives outputs — keep raw read-only.
- Provide the service-account JSON as a **Docker secret / mounted file**, *not* an environment variable. (The v2 security audit, FINDING-006, flags plaintext-env secrets like `HF_TOKEN`/`ANTHROPIC_API_KEY`; don't repeat that pattern for Drive creds.)
- Configure the rclone remote with `service_account_file = /run/secrets/gdrive_sa.json`. Mount read-only.
- Network egress is outbound-only to Google APIs; no inbound exposure added.

---

## 4. Storage Provisioning

4K frame extraction is the space driver, not the MP4s:

| Artifact | Rough size per ~10 min 4K set |
|----------|-------------------------------|
| Raw MP4 | 5–15 GB |
| Extracted frames (JPEG, ~2 fps) | 3–7 GB |
| COLMAP DB + sparse + **undistorted images** | 6–20 GB (undistorted images dominate) |
| Trained splat PLY | 0.03–0.2 GB |
| Meshes + USD + previews | ~0.5 GB |

A multi-set room session can transiently need **30–80 GB** of scratch. Provision a dedicated NVMe scratch volume (the host has NVMe per the hardware table). Recommended: a `/data/scratch` bind mount sized **250–500 GB**, purged per session.

```yaml
# docker-compose.consolidated.yml (add to the main service)
    volumes:
      - /mnt/nvme/scratch:/data/scratch       # session working space (purged)
      - /mnt/nvme/raw-cache:/data/raw          # active set(s) copied here, then deleted
    secrets:
      - gdrive_sa
secrets:
  gdrive_sa:
    file: ./secrets/gdrive_service_account.json
```

---

## 5. Processing Strategy for Orbit Sets

### 5.1 Frame extraction + selection (uses the new v2 modules)
1. Extract frames per set at **1–3 fps** (slow orbit → low fps is enough; PyAV path already exists).
2. Run quality gates (`frame_quality.py`: blur/exposure) to drop bad frames.
3. **Enable the new Fibonacci-coverage selection** (`fibonacci_sampler.py`, wired via `config.ingest.use_fibonacci_coverage = True`) across the *pooled frames of all sets for the room* — this dedups overlapping viewpoints between orbits and yields an even viewpoint distribution. Target **300–1000 selected frames** for a room.
4. Score selection on full-res; **downscale to ≤3200 px for COLMAP**, and use a training downscale factor for 3DGS.

### 5.2 Combined reconstruction
- Feed all selected frames from all sets of one room into **one COLMAP run**.
- Matcher: **vocab-tree (spatial)** for multi-set pooled frames (or exhaustive if total < ~1500). Sequential-only is wrong here because frames come from multiple orbits.
- `single_camera = True` if all sets used the same lens; otherwise one camera model per set.
- Verify a single connected model registers (>70% of frames). If sets split into components, they lacked overlap — flag for re-capture or add bridging frames.

### 5.3 Training + mesh
- Training preset: **indoor / indoor_reflective**. MRNF densification once upstream v0.5.2 sync lands; gsplat `DefaultStrategy` until then.
- Mesh backend (new v2 multi-backend selector, `config.training.mesh_method`):
  - `auto` → picks GaussianWrapping for thin structures (furniture edges, fixtures, railings), else CoMe (fast) / MILo (highest quality) — see ADR-003.
  - For a furnished room, **GaussianWrapping** (thin structures) or **MILo** (general indoor) are the strong picks; **CoMe** if turnaround speed matters.
- Optional **splat optimization** (`delivery.enable_splat_optimize = True`) → compressed `.ksplat` for web preview.
- Final texture bake can pull from the retained full-res frames of one set.

### 5.4 Recommended starting config (room orbit)
```python
ingest.fps = 2.0
ingest.use_fibonacci_coverage = True
ingest.coverage_weight = 0.4
ingest.target_frames = 600          # pooled across all sets of the room
reconstruct.matcher = "vocab_tree"
reconstruct.single_camera = True    # if one lens across sets
training.scene_preset = "indoor_reflective"
training.mesh_method = "auto"       # or "milo" / "gaussianwrapping"
delivery.enable_splat_optimize = True
```

---

## 6. Proposed Component: `drive_ingestor.py` + session worker

New pipeline module (additive, fits BOUNDARIES.md under `src/pipeline/`):

- **Discovery**: `rclone lsjson` the raw folder → enumerate rooms/sessions and their sets (by folder or filename convention).
- **Ledger**: a small SQLite (or a manifest JSON on Drive) tracking `{session_id, sets, status, checksums, output_path}` → enables **skip-completed** and **resume**.
- **Per-session loop** (idempotent):
  1. `rclone copy` all sets for the session → `/data/raw/<session>` (with `--checksum`).
  2. Extract + Fibonacci-select frames across sets → `/data/scratch/<session>`.
  3. Combined COLMAP → train → mesh → USD assemble (existing stages).
  4. `rclone copy` outputs → Drive `outputs/<session>/`.
  5. Verify upload, update ledger, **delete** `/data/raw/<session>` and heavy scratch (keep logs + small previews).
- **Failure handling**: leave raw in place and mark `failed` if any stage fails; never delete unprocessed source.
- **Web UI integration**: add a "Drive source" job type so the existing Flask UI/queue can list discovered sessions and enqueue them, replacing drag-and-drop for bulk work. Raise/remove `MAX_CONTENT_LENGTH` only for the legacy path; bulk goes through Drive, never the browser.

---

## 7. Phased Rollout

| Phase | Deliverable | Goal |
|-------|-------------|------|
| **1. Plumbing** | Add `rclone` to `Dockerfile.consolidated`; service-account secret; `/data/scratch` + `/data/raw` volumes | Can `rclone copy` one set into the container |
| **2. Validate quality** | Manually pull ONE room's sets, run combined COLMAP→train→mesh with §5.4 config | Confirm reconstruction quality on real data before automating |
| **3. `drive_ingestor.py`** | Discovery + ledger + pull→process→push→purge worker | Hands-off bulk processing, resumable |
| **4. Multi-set fusion** | Combined reconstruction + cross-set Fibonacci selection as the default | Best geometry from the orbit-at-different-heights captures |
| **5. UI + scheduling** | "Drive source" job type in web UI; optional cron | Operator-friendly, unattended runs |

Phases 1–2 are low-risk and prove the approach before any automation is built.

---

## 8. Decisions (resolved 2026-05-26)

1. **Set grouping** → **flat folder per capture**. Each folder under the base path *is* one session; every video file inside it is a "set" of that session. Discovery = `rclone lsjson <base> --dirs-only`; no filename-prefix parsing.
2. **Same lens across sets** → **yes**. `reconstruct.single_camera = True` for the combined COLMAP run.
3. **Output destination** → **back to the same capture folder, in an `outputs/` sub-folder**, pushed via the Google Cloud service-account creds. Implication: the service account needs **write** scope on the capture folder (not read-only raw + separate writable folder). Dest = `<remote>:<base>/<session>/outputs/`.
4. **Throughput** → **one-time batch**. Point the worker at the base folder, process every capture sequentially (pull → process → push → purge), then exit. No watcher/daemon; a skip-completed ledger still makes the batch resumable if interrupted.
5. **Retention** → **delete local NVMe copy only**. The raw is already preserved on Drive (source of truth) and outputs are pushed back beside it, so after a verified upload the worker purges `/data/raw/<session>` + heavy scratch and keeps only a small per-session log + the ledger.

Implemented in `src/pipeline/drive_ingestor.py` (one-time batch worker). The heavy COLMAP/CUDA stages still run on the GPU host; the worker only orchestrates pull/extract/stage-run/push/purge around the existing `PipelineStages`.

---

## 9. How to run (GPU host)

**One-time setup**
1. `cp secrets/rclone.conf.example secrets/rclone.conf` and fill in the remote (Drive or GCS) + path to the service-account JSON. Drop the JSON at `secrets/gdrive_sa.json`.
2. In `.env` set `DRIVE_INGEST_REMOTE=captures:` (Drive, with `root_folder_id`) or `captures:<bucket>/captures` (GCS), `RCLONE_CONF_FILE=./secrets/rclone.conf`, and optionally `DRIVE_RAW_HOST_DIR` / `DRIVE_SCRATCH_HOST_DIR` to point at a 250–500 GB NVMe path.
3. `docker compose -f docker-compose.consolidated.yml build gaussian-toolkit` (installs rclone), then `up -d`.

**Dry run (discovery only — no copy, no compute)**
```bash
docker exec gaussian-toolkit python -m pipeline.drive_ingestor list --remote "$DRIVE_INGEST_REMOTE" --rclone-config /run/secrets/rclone_conf
docker exec gaussian-toolkit python -m pipeline.drive_ingestor run  --remote "$DRIVE_INGEST_REMOTE" --rclone-config /run/secrets/rclone_conf --dry-run
```

**Full batch**
```bash
docker exec gaussian-toolkit python -m pipeline.drive_ingestor run \
  --remote "$DRIVE_INGEST_REMOTE" --rclone-config /run/secrets/rclone_conf \
  --scratch /data/scratch --raw /data/raw \
  --ledger /data/output/drive_ingest_ledger.sqlite \
  --mesh-method auto --scene-preset indoor_reflective --target-frames 600
```
Or from the web UI: `GET /api/ingest/drive/sessions` to preview, `POST /api/ingest/drive` (optional `dry_run=true`) to launch detached. Re-running skips sessions already marked `done` in the ledger.

**Caveats**: validate quality on ONE session first (plan Phase 2) before trusting the batch. CoMe/GaussianWrapping CLI flags are still inferred (ADR-004/005) — `--mesh-method milo` is the safest verified backend until those sidecars are built and checked. rclone version (`RCLONE_VERSION` build arg) and the service-account write scope must be confirmed on the host.
