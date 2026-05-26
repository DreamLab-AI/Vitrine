# V2 Upgrade Code Review Report

**Date:** 2026-05-26  
**Branch:** feat/v2-upgrade-swarm  
**Reviewer:** Code Review Agent (Ruflo swarm)  
**Scope:** Correctness, quality, and integration consistency — NOT security (audited separately)

---

## Executive Summary

The v2 upgrade is **conditionally ready to commit to the feature branch** with two blockers that must be fixed before any merge to `main`. The new modules (`come_extractor.py`, `gaussianwrapping_extractor.py`, `splat_optimizer.py`, `fibonacci_sampler.py`) are individually well-written. The critical issues are integration gaps and a precedence deviation from the ADR.

**Verdict: NO-GO for main merge; YES-GO for feature branch commit after fixing the two Blockers.**

---

## Findings Table

| # | Severity | File : Line(s) | Issue | Recommendation |
|---|----------|----------------|-------|----------------|
| 1 | **Blocker** | `src/pipeline/come_extractor.py:305,345` | `subprocess.run(..., cwd=str(COME_DIR))` passes the **container-side path** (`/opt/come`) as the HOST-side working directory. When the pipeline runs on a host where `/opt/come` does not exist, `subprocess.run` raises `FileNotFoundError`, violating the "never raises to caller" contract. Only `subprocess.TimeoutExpired` is caught. | Change `cwd=str(COME_DIR)` to `cwd=None` when `exec_prefix[0] == "docker"` (docker's cwd is the host process's cwd, irrelevant to the container). Pass `-w /opt/come` as an argument to `docker exec` instead. Same fix needed in `gaussianwrapping_extractor.py:331,369` and in the pre-existing `milo_extractor.py:208,246`. |
| 2 | **Blocker** | `src/pipeline/stages.py:583-808` | `ingest.use_fibonacci_coverage` and `ingest.coverage_weight` are never propagated from `PipelineConfig` into `SelectionConfig` in `select_frames()`, and `selector.select()` is called without `camera_positions`. This makes the Fibonacci coverage feature **permanently inert** regardless of config — the config field documented in ADR-007 has no code path that activates it. | In `stages.py` `select_frames()` (around line 356): forward `use_fibonacci_coverage=self.config.ingest.use_fibonacci_coverage` and `coverage_weight=self.config.ingest.coverage_weight` to `SelectionConfig`. After COLMAP completes, pass the parsed camera centres to `selector.select(scores, camera_positions=...)`. Camera positions can be read from `colmap_dir/sparse/0/images.bin` using the existing binary parser already present in `render_previews()`. |
| 3 | **Major** | `src/pipeline/stages.py:798-810` | `_select_mesh_backend()` selects CoMe **unconditionally** whenever it is available (step 2), but ADR-003 specifies that CoMe should be preferred only when **speed is a priority**; the default high-quality path should be MILo. The current implementation demotes MILo below CoMe in all auto-selection scenarios, inverting the quality/speed trade-off documented in the ADR's priority matrix. | Add a `speed_priority` parameter or read it from a new `training.speed_priority: bool` config field. Apply the ADR-003 table: skip CoMe when quality is the goal and proceed to MILo. Alternatively, update ADR-003 to document the changed precedence if the team has deliberately decided CoMe is now the default over MILo. |
| 4 | **Major** | `src/pipeline/gaussianwrapping_extractor.py:331,369` | Same `cwd=` container-path-on-host bug as finding #1. The variable `gw_dir_in_ctx` is correctly computed per-context, but it is then passed as `cwd=` to `subprocess.run` on the **host**. When using docker exec, the value is `/opt/gaussianwrapping`, which does not exist on the host. | Same fix as #1: omit `cwd=` for docker exec invocations; use `docker exec -w /opt/gaussianwrapping` instead. |
| 5 | **Minor** | `src/pipeline/splat_optimizer.py:214-219` | Both `"ply"` and `"compressed-ply"` map to `.ply` in `ext_map`. If `@playcanvas/splat-transform` determines output format from the file extension (inferred behaviour — see Verification Checklist item 5), the tool cannot distinguish between the two, and the compression pass may be skipped or applied incorrectly. | Pass an explicit `--format` flag to `splat-transform` if the CLI supports it (verify per checklist item 5). If no flag exists, rename `"compressed-ply"` output to `.cply` or another unique extension, and document the choice. |
| 6 | **Minor** | `src/pipeline/come_extractor.py:177-199` | `_find_sparse_dir` and `_find_dataset_root` are copy-pasted verbatim across `come_extractor.py`, `gaussianwrapping_extractor.py`, and `milo_extractor.py` — three identical implementations. Any fix (e.g., adding a new COLMAP layout variant) must be made in three places. | Extract these two functions into `src/pipeline/_colmap_utils.py` (or similar) and import from there in all three extractors. |
| 7 | **Minor** | `src/pipeline/come_extractor.py:423`, `src/pipeline/gaussianwrapping_extractor.py:444` | `load_come_mesh` and `load_gaussianwrapping_mesh` are annotated `-> Any`. The actual return type is `trimesh.Trimesh`. Using `Any` suppresses type checker warnings on callers. | Change return annotation to `"trimesh.Trimesh"` (quoted string avoids import-time cost). Add `TYPE_CHECKING` guard for the import. |
| 8 | **Minor** | `src/pipeline/stages.py` (whole file) | `stages.py` is 2077 lines — more than 4× the project's 500-line style guideline. | Split out the `_train_*` backend methods and `_mesh_single` strategy chain into a `src/pipeline/train_backends.py` module; move COLMAP binary parsing into `colmap_parser.py`. This is non-blocking for the feature branch but should be tracked. |
| 9 | **Minor** | `src/pipeline/come_extractor.py:160-168` vs `src/pipeline/gaussianwrapping_extractor.py:204-211` | Licensing warning detection uses **two different env vars**: `COME_DEV_ENVIRONMENT` in `come_extractor.py` and `LICHTFELD_ENV` in `gaussianwrapping_extractor.py`. Operators configuring deployment environments must set two separate variables. | Standardise on `LICHTFELD_ENV` (already used by the GW extractor). Update `come_extractor.py` to read `LICHTFELD_ENV` with a fallback to `COME_DEV_ENVIRONMENT` for backward compatibility. |
| 10 | **Nit** | `src/pipeline/fibonacci_sampler.py:319` | `if __name__ == "__main__":` block uses a triple-quoted string as a "docstring" (`"""Self-test..."""`) at module scope inside the block — it is not actually attached to any function and is a dead string literal. | Replace with a comment (`# Self-test with synthetic camera positions.`). |
| 11 | **Nit** | `src/pipeline/come_extractor.py:925` / `src/pipeline/stages.py:925,1008` | `Path(come_result.get("glb_path") or come_result.get("mesh_path", ""))` — when both values are `None`, this evaluates to `Path("")` which resolves to `Path(".")` (current directory), and `.exists()` returns `True`. The subsequent `shutil.copy2(".", dest)` would raise an `IsADirectoryError`. This cannot happen on a successful run (the function returns early if no mesh is found), but the guard is fragile. | Use `come_result.get("glb_path") or come_result.get("mesh_path")` and skip the copy when the value is `None`. |

---

## Config Round-Trip Verification

All new config fields round-trip correctly through `save()` / `load()` / `_from_dict()`:

- `DeliveryConfig` fields (`enable_splat_optimize`, `output_format`, `opacity_min_threshold`, `max_scale`, `sort`, `generate_html_viewer`) serialise and deserialise faithfully, including `max_scale=None`.
- `IngestConfig.use_fibonacci_coverage` and `coverage_weight` round-trip correctly.
- `TrainingConfig.mesh_method` and `mesh_backend_auto` round-trip correctly.
- **Backward compatibility confirmed**: old configs that omit `delivery` and/or `fibonacci` keys load without error; new fields receive their declared defaults.
- `validate()` correctly catches: bad `mesh_method`, bad `delivery.output_format`, `opacity_min_threshold` out of `[0, 1]`, `coverage_weight` out of `[0, 1]`. No gaps found in validation coverage.

---

## `frame_selector.py` Backward Compatibility

- `FrameSelector.select()` now accepts an optional `camera_positions` parameter (default `None`). All existing callers that pass only `scores` continue to work identically.
- `stages.py` calls `selector.select(scores)` (no `camera_positions`) — the existing quality-only path is unchanged.
- The `use_fibonacci_coverage` guard and `camera_positions is None` short-circuit path function correctly when invoked directly.

**The backward compatibility is intact. The integration gap (finding #2) is that the feature can never be activated from the pipeline's main flow.**

---

## Fallback Recursion Analysis

The fallback pattern in `_train_come`, `_train_gaussianwrapping`, and `_train_milo` is:

```python
self.config.training.mesh_method = "tsdf"
result = self.train(colmap_dir, self.config.training.iterations)
self.config.training.mesh_method = original_method
```

**No infinite recursion** is possible because:

1. The fallback path is only reached when `is_X_available()` returns `False`.
2. In the recursive `train()` call, `mesh_method="tsdf"`, so `_select_mesh_backend()` is only re-entered if `mesh_backend_auto=True`. In that case, `_select_mesh_backend()` calls `is_come_available()` again — which returns `False` (same environment) — so `come` is not selected again.
3. The chain terminates at `tsdf` which does not call `_select_mesh_backend`.

**Efficiency note**: when `mesh_backend_auto=True`, the recursive call triggers a second round of backend availability probes (up to 3 × 10-second docker timeouts). This is not a correctness issue but adds latency on degraded infrastructure.

---

## ADR-003 Precedence Cross-Check

| ADR-003 Condition | ADR-003 Backend | Implementation Backend |
|---|---|---|
| Thin-structure hint AND GW available | gaussianwrapping | gaussianwrapping ✓ |
| Speed priority AND CoMe available | come | **(CoMe if available — no speed check)** ✗ |
| Default high-quality (no speed priority) | milo | come (if available) ✗ |
| MILo unavailable, CoMe unavailable | tsdf | tsdf ✓ |

The implementation deviates from the ADR in the second and third rows. CoMe is selected unconditionally before MILo regardless of whether speed is a priority.

---

## CLI Verification Checklist

The following constants are **inferred** and have not been verified against the released source code of CoMe and GaussianWrapping. They must be checked once the sidecars are built:

### CoMe (`come_extractor.py`)

| Constant | Value | Must Verify |
|---|---|---|
| `COME_TRAIN_SCRIPT` | `"train.py"` | Script name at repository root |
| `COME_EXTRACT_TETS_SCRIPT` | `"extract_mesh_tets.py"` | Script name for unbounded (marching-tetrahedra) extraction |
| `COME_EXTRACT_TSDF_SCRIPT` | `"extract_mesh_tsdf.py"` | Script name for bounded (TSDF) extraction |
| `COME_TRAIN_FLAG_CONFIG` | `"--splatting_config"` | Flag name for JSON config path |
| `COME_TRAIN_FLAG_SOURCE` | `"-s"` | Flag for dataset source root |
| `COME_TRAIN_FLAG_MODEL` | `"-m"` | Flag for model/output directory |
| `COME_EXTRACT_FLAG_MODEL` | `"-m"` | Flag for model directory passed to extraction scripts |
| `CoMeConfig.splatting_config` default | `"configs/come_unbounded.json"` | Actual path of the default JSON config inside the CoMe repo |
| Output mesh glob | `"mesh_*.ply"` | Whether CoMe names its mesh PLY output with the `mesh_` prefix |
| Gaussian PLY location | `"point_cloud/*/point_cloud.ply"` | Whether CoMe follows the 3DGS checkpoint path convention |

### GaussianWrapping (`gaussianwrapping_extractor.py`)

| Constant | Value | Must Verify |
|---|---|---|
| `GW_TRAIN_SCRIPT` | `"train.py"` | Script name at repository root |
| `GW_EXTRACT_SCRIPT` | `"extract_mesh.py"` | Script name for mesh extraction |
| `GW_FLAG_SOURCE` | `"-s"` | Dataset root flag |
| `GW_FLAG_MODEL` | `"-m"` | Output model flag |
| `GW_FLAG_RASTERIZER` | `"--rasterizer"` | Rasterizer selection flag |
| `GW_FLAG_ITERATIONS` | `"--iterations"` | Iteration count flag |
| `GW_FLAG_ADAPTIVE_MESHING` | `"--adaptive_meshing"` | Primal Adaptive Meshing flag (boolean, no value) |
| `GW_FLAG_MESH_PATH` | `"--mesh_path"` | Output mesh PLY path flag |
| Rasterizer values | `"radegs"`, `"median_depth"` | Accepted values for `--rasterizer` |
| GW install path in milo sidecar | `/opt/gaussianwrapping` | Actual install path post Dockerfile.milo rebuild |

### PlayCanvas splat-transform (`splat_optimizer.py`)

| Assumption | Must Verify |
|---|---|
| `compress` is the correct subcommand for format conversion + filtering | Check `npx @playcanvas/splat-transform --help` |
| `--alpha-min` is the opacity threshold flag | Verify flag name |
| `--scale-max` is the maximum Gaussian scale flag | Verify flag name |
| `--sort` enables Morton-order sort | Verify flag name |
| `--html` generates an HTML viewer | Verify flag name |
| Format is inferred from output file extension (not `--format` flag) | Critical: if a `--format` flag exists and is required, `_build_cli_args` is missing it |
| `"compressed-ply"` → `.ply` extension produces different output than `"ply"` → `.ply` | If both produce identical output, remove `"compressed-ply"` from `_VALID_FORMATS` |

---

## Code Quality Summary

| Aspect | Assessment |
|---|---|
| Type hints | Good in new modules. `load_come_mesh` / `load_gaussianwrapping_mesh` return `Any` instead of `trimesh.Trimesh` (finding #7). `from __future__ import annotations` is present everywhere. |
| Docstrings | Thorough — all public functions have Args/Returns documentation. CoMe module docstring clearly flags inferred CLI constants. |
| Error handling | Extractor run functions never raise (conforming to spec). However, uncaught `OSError` from `cwd=<container-path>` on the host breaks this contract for docker exec (Blocker #1). |
| Logging consistency | Consistent with `milo_extractor.py` pattern. INFO for milestones, WARNING for non-fatal issues, ERROR for subprocess failures. |
| Dead code / unused imports | None identified. `field` is imported in `splat_optimizer.py` but used (`dataclass` + `field`). |
| File lengths | New modules are all under 500 lines. `stages.py` is 2077 lines (finding #8). |
| Function lengths | All functions reviewed are under 40 lines except the large `run_come` / `run_gaussianwrapping` orchestration functions (~100 lines each due to two sequential subprocess + artifact-location blocks). These are acceptable given the sequential subprocess pattern. |

---

## Go / No-Go Decision

| Destination | Decision | Rationale |
|---|---|---|
| `feat/v2-upgrade-swarm` (feature branch, no push) | **GO** | No external impact; Blockers can be fixed iteratively on the branch. |
| Merge to `main` | **NO-GO** | Blockers #1 and #2 must be resolved first. Blocker #1 (`cwd` bug) will crash any real deployment using docker exec. Blocker #2 (dead fibonacci wiring) renders a documented feature permanently non-functional. |

### Required before `main` merge

1. Fix `cwd=` for docker exec in `come_extractor.py`, `gaussianwrapping_extractor.py` (and the pre-existing same bug in `milo_extractor.py`).
2. Wire `ingest.use_fibonacci_coverage` and camera positions into `stages.py` `select_frames()`.
3. Reconcile `_select_mesh_backend()` precedence with ADR-003 (or update the ADR to document the changed decision).
4. Verify CLI constants against the built sidecars and update any that are wrong (see Verification Checklist).
