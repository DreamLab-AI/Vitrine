# ADR-009: Per-Video Ingest Loop, Per-Video Retention, and Per-Image Metadata Sidecar

## Status

Proposed

## Context

The aspirational workflow (`research/pipelines/aspirational-e2e-flowchart.md` §2) states the
operator's intended order explicitly: **extract → quality-check → delete the local video → tag
the images → next video**. The current ingestor does none of this per-video. Three concrete gaps
follow, registered as D1, D2, D3 (and the root of D14) in
`research/decisions/gap-analysis-e2e-aspiration.md` and commissioned as FR-1..FR-5 in
`research/decisions/prd-v3-e2e-closure.md`.

1. **Granularity (D1).** `drive_ingestor.py:655-674` iterates over **session folders**, not
   videos. Each session is handed to `process_session`, and `_extract_pooled_frames` pools **all**
   videos carrying the `set00_/set01_` prefixes into one combined frame set. There is no unit of
   work smaller than a session folder, so there is no per-video resume, no per-video deletion, and
   no per-video provenance.

2. **Retention (D2).** The local raw purge `_purge(local_raw)` runs at `drive_ingestor.py:583-587`,
   **after** the whole reconstruct + train + mesh pipeline for the session has completed. A video
   copied to NVMe at the start of an 8-hour session occupies local storage for the entire session,
   even though it is needed only during its own frame extraction. On a multi-session overnight
   batch this is the dominant cause of NVMe pressure.

3. **Provenance (D3, D14-root).** There is **no per-image metadata at all**. The only persisted
   record is the session-level `manifest.json` (`drive_ingestor.py:434-441`) plus the session-level
   SQLite ledger whose schema is a single `sessions` table keyed on `session_id`
   (`drive_ingestor.py:137-200`). The richest per-frame signal — `FrameQuality`
   (`frame_quality.py:38-64`, populated by the assessor at `frame_quality.py:79-92`) — is computed
   **in memory and discarded**. Nothing records which source video a frame came from, its timestamp,
   its blur/exposure/sharpness scores, or (later) its recovered pose. The image→video lineage chain
   that FR-17/FR-19 must thread into the USD therefore has no root to grow from.

The crux is a real tension: the operator wants per-video ingest/tag/delete, but reconstruction
quality demands that all videos of a room be **pooled into one combined COLMAP run** (a single
SfM model with shared scale, the input every downstream stage already assumes —
`fibonacci_sampler.py`, `gsplat_trainer.py`, `usd_assembler.py`). Per-video reconstruction would
fragment the scene into N un-registerable sub-models. This ADR must reconcile the two: make the
**ingest unit** per-video while keeping the **reconstruction unit** per-room.

## Decision

**Split the ingest unit from the reconstruction unit. Ingest, verify, tag and delete each video
individually; pool the resulting tagged frames into one combined COLMAP run per room.** Concretely:

### (a) Per-video unit of work, pooled reconstruction (resolves the crux)

Replace the folder-pooled iteration with a two-level loop. The **outer** loop is per room/session
(the reconstruction boundary, unchanged). The **inner** loop is per video and owns
extract → quality-gate → verify → delete → tag, writing each video's retained frames into the
**shared** room frame directory that COLMAP later consumes. Pooling is preserved because every
inner iteration appends to the same `frames/` pool; only the *act of producing* those frames
becomes per-video. SfM, training, segmentation and mesh extraction continue to operate on the
combined pool exactly as today — no downstream stage changes its input contract.

```
for room in sessions:                      # reconstruction unit (per-room COLMAP)
    for video V in videos(room):           # ingest unit (D1)
        copy V → NVMe scratch
        frames = extract(V)
        kept   = quality_gate(frames)       # per-video distribution (FR-4)
        verify(V, kept)                      # frame_count + ≥1 passing frame (D2)
        delete_local(V)                      # immediately on verify (D2)
        write_sidecars(kept, source_video=V) # per-image metadata (D3)
        ledger.upsert_video(room, V, "done")
    pool = frames_dir(room)                  # all videos' kept frames, combined
    reconstruct(pool); train; segment; mesh; assemble_usd   # per-room, unchanged
```

### (b) Delete-on-verified-extraction

A video's local copy is deleted **immediately after its extraction is verified**, not after
reconstruction. "Verified" is defined precisely (matching FR-2): `frame_count ≥
expected_from_duration AND at least one frame passed the quality gate`. The raw video remains on
Drive as the single source of truth (`drive_ingestor.py:19-20`, outputs pushed back to
`<remote>/outputs/`). The purge call currently at `drive_ingestor.py:583-587` is **narrowed** to
the per-video raw file and moved to fire on the post-extraction verification of `V`; the
end-of-session purge of heavy scratch (`frames`, `frames_cleaned`, `frames_selected`, `colmap`)
is retained as-is, because those are per-room artefacts not per-video.

### (c) Per-image metadata sidecar schema

Each retained frame `<frame>.jpg` gets a co-located sidecar `<frame>.json`. The schema
(`schema_version: "v2g-frame-1"`) persists the in-memory `FrameQuality` plus lineage:

```json
{
  "schema_version": "v2g-frame-1",
  "source_video": "set01_room_a.mp4",
  "capture_session": "2026-06-04_room_a",
  "frame_index": 142,
  "source_timestamp_pts": 4.733,
  "blur_score": 184.2,
  "exposure_mean": 0.51,
  "exposure_std": 0.18,
  "sharpness": 184.2,
  "phash": "c3a1f0e8d2b49157",
  "kept": true,
  "selection_reason": "passed:blur,exposure;fibonacci_selected",
  "pose_hint": null,
  "schema_notes": "pose_hint backfilled post-COLMAP"
}
```

`source_timestamp_pts` is the PyAV/ffmpeg presentation timestamp of the frame within its source
video. `selection_reason` records why the frame was kept (and later, whether Fibonacci selection
chose it — ADR-007). `pose_hint` is reserved (`null` at ingest) and **backfilled after COLMAP**
(see Rationale). The same fields are mirrored into a new `frames` table in the ledger (below) so
they are queryable in bulk without a directory walk; the JSON sidecar is the portable, per-frame
copy that survives a move off the host.

### (d) Per-video ledger rows

The ledger (`drive_ingestor.py:137-200`) is extended with a second table at video granularity,
preserving the existing `sessions` table:

```sql
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,   -- session_id + '/' + filename
    session_id    TEXT NOT NULL,
    remote_path   TEXT,
    status        TEXT NOT NULL,      -- pending|extracting|extracted|deleted|tagged|done|failed
    n_frames_kept INTEGER,
    checksum      TEXT,               -- of the raw video, verified before delete
    error         TEXT,
    started_at    REAL, finished_at REAL, updated_at REAL
);
```

A video is resumable at its exact status: a kill between `extracted` and `tagged` resumes at the
sidecar write, not at re-download. This is the per-video ledger FR-1 requires and the resume key
FR-7 (the optional DAG) consumes.

## Rationale

- **Sidecar JSON over EXIF.** Frames are extracted as JPEG/PNG; EXIF would carry `phash`,
  `selection_reason` and `pose_hint` only as non-standard maker-notes that most DCC tools strip on
  re-encode. A sibling `<frame>.json` is tool-agnostic, human-readable, diffable, and survives any
  image transcode. It is also the natural carrier for the `pose_hint` slot, which does not exist
  yet at extraction time.
- **Sidecar JSON *and* a ledger `frames` table, not one or the other.** The sidecar is the
  portable per-frame record (moves with the image); the ledger table is the queryable index (bulk
  lineage joins for FR-19 without walking thousands of files). They carry the same fields and are
  written in the same transaction, so they cannot drift (NFR-2 idempotency).
- **`pose_hint` is backfilled, not computed at ingest.** Camera pose is only known after COLMAP
  SfM produces `images.bin`. At ingest, `pose_hint` is `null`. After the per-room SfM completes, a
  backfill pass joins each registered COLMAP image name back to its `frame_index`/`source_video`
  (the sidecar key) and writes the camera centre + quaternion into both the sidecar and the ledger
  `frames` row. This is the same join ADR-007 uses to map selected frames to camera positions, so
  no new parsing is introduced.
- **Per-video quality gate (FR-4).** Running the gate per video stops one bad video's exposure
  distribution from skewing another's thresholds, which the current pooled aggregate
  (`quality_gates.py` `FrameStats`) cannot avoid. Bad frames are dropped *before* the sidecar write,
  so no sidecar is ever written for a discarded frame (keeps G4/G5 coverage exact).
- **Pooling is preserved by construction.** Because each inner iteration appends to the shared
  room frame pool, the combined COLMAP run is byte-for-byte the input it is today; only the
  producer changed. This is the cheapest possible reconciliation of per-video ingest with
  per-room reconstruction.

## Consequences

### Positive

- Local NVMe holds **at most one raw video at a time** (NFR-3); a processed video is gone within
  its extraction window, not held through reconstruction. Closes the dominant overnight storage
  risk.
- Every retained frame is self-describing and traceable to its source video (D3, D14-root),
  giving FR-17/FR-19 a real lineage root to thread into the USD.
- Per-video resume (D1) means an 8-hour batch killed mid-way restarts at the failed video, not
  from zero (NFR-1, satisfies U2).
- No downstream stage changes: COLMAP, training, segmentation, mesh and USD all keep consuming the
  same pooled frame directory. The blast radius is confined to `drive_ingestor.py`,
  `frame_quality.py`, and a new sidecar writer.

### Negative

- The ledger schema migration (adding the `videos` and `frames` tables) requires a one-time
  upgrade of any in-flight ledger; old session-only ledgers must be readable for resume.
- Sidecar files multiply small-file count on disk (one JSON per retained frame). On scenes with
  thousands of frames this is a non-trivial inode count, though sidecars are tiny and are purged
  with the room scratch.
- The `pose_hint` backfill adds a second pass over the sidecars after SfM; it is cheap but is a
  new ordering dependency (sidecars must be written before backfill runs).

### Risks

- **Deletion races verification (PRD §7).** If the delete fires before verification completes,
  the raw is lost locally. Mitigation: delete is gated strictly on the `verify()` return, and the
  raw is *always* retained on Drive as source of truth, so a mis-fire is recoverable by re-download.
- **Sidecar/ledger drift.** If the sidecar write and the ledger `frames` upsert are not atomic,
  the two indices disagree. Mitigation: both are written inside one ledger transaction per frame
  batch; a partial write leaves the video in `extracted` (not `tagged`), so resume rewrites both.
- **Backfill join failure.** A frame that COLMAP fails to register has no pose; `pose_hint` stays
  `null`. This is correct (unregistered frames have no pose) and must not be treated as an error
  by the lineage gate (G14 resolves *objects* to frames, not the reverse).

## Alternatives Considered

- **Per-video COLMAP reconstruction.** Reconstruct each video independently, then merge models.
  Rejected: fragments a room into N sub-models with incompatible scale and no shared registration;
  defeats the whole-room SfM every downstream stage assumes. The split-unit design (ingest
  per-video, reconstruct per-room) achieves the operator's intent without this cost.
- **EXIF/maker-note metadata only.** Rejected: lossy across transcodes, no slot for `pose_hint`,
  poor tooling for the non-standard fields (`phash`, `selection_reason`).
- **Database-only metadata (no sidecar).** Rejected: the per-frame record must move with the image
  off the host for archival lineage (U7); a DB row does not travel with a copied frame. The ledger
  table is retained *in addition to* the sidecar purely as a bulk query index.
- **Keep session-level deletion, just copy videos lazily.** Rejected: lazy copy reduces but does
  not bound peak retention; only delete-on-verified-extraction guarantees the ≤1-video ceiling
  (G2).

## Related Decisions

- `research/decisions/prd-v3-e2e-closure.md` — commissions this ADR; realises FR-1..FR-5,
  closes D1, D2, D3 and the root of D14.
- `research/decisions/gap-analysis-e2e-aspiration.md` — delta register D1, D2, D3, D14 (§1, §5).
- `research/pipelines/aspirational-e2e-flowchart.md` — Phase 1 per-video loop (§1, §2).
- `adr-007-fibonacci-sphere-frame-selection.md` — frame selection consumes the per-image sidecars
  and writes its `fibonacci_selected` flag into `selection_reason`; shares the COLMAP image→frame
  join used for `pose_hint` backfill.
- `adr-010-key-item-hull-recon.md` — consumes the `pose_hint` lineage when persisting per-object
  pose.
- `adr-011-usd-metadata-enrichment.md` — threads the sidecar `source_video`/`frame_index` lineage
  into the USD `v2g:*` metadata (D14).
- `adr-001-pipeline-architecture.md` — this ADR modifies the ingestion stage defined there.
