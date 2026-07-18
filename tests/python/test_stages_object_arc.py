# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Object-arc regression tests for pipeline.stages (ADR-025 / PRD v4).

These pin the audit fixes:
  R2 — extract_objects fails LOUDLY per object; no full-scene-copy fakes.
  R3 — object_crops reports per-object failures with cause.
  R4 — mesh_objects requires a crop for objects (no splat-render conditioning).
  R6 — generator GLB bytes persisted byte-identical; texture_bake skips them.

No GPU, no network: generator clients are stubbed at the module boundary.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os  # noqa: E402

# These tests exercise the pure-logic stage methods (crops/isolation/placement
# bookkeeping) with the GPU generators stubbed, so skip the torch/GPU preflight
# in PipelineStages.__init__ — lets the suite run on a CPU CI runner.
os.environ.setdefault("LFS_SKIP_PREFLIGHT", "1")

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.stages import PipelineStages, STAGE_NAMES  # noqa: E402

GLB_BYTES = b"glTF-generator-payload" * 8


@pytest.fixture()
def stages(tmp_path) -> PipelineStages:
    return PipelineStages(str(tmp_path / "job"), config=PipelineConfig())


@pytest.fixture()
def trained_ply(tmp_path) -> Path:
    p = tmp_path / "trained.ply"
    p.write_bytes(b"ply-fake")
    return p


def test_gpu_env_picks_freest_gpu(monkeypatch):
    import subprocess as _sp
    import types as _t
    # nvidia-smi: GPU 0 has 5000 MiB free, GPU 1 has 42000 -> pick 1.
    monkeypatch.setattr(_sp, "run", lambda *a, **k: _t.SimpleNamespace(
        stdout="0, 5000\n1, 42000\n"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = PipelineStages._gpu_env({})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_gpu_env_respects_existing_pin(monkeypatch):
    import subprocess as _sp
    called = {"n": 0}

    def _fake_run(*a, **k):
        called["n"] += 1
        raise AssertionError("nvidia-smi should not run when pinned")
    monkeypatch.setattr(_sp, "run", _fake_run)
    env = PipelineStages._gpu_env({"CUDA_VISIBLE_DEVICES": "0"})
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert called["n"] == 0


def test_gpu_env_falls_back_when_nvidia_smi_absent(monkeypatch):
    import subprocess as _sp
    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(_sp, "run", _boom)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env = PipelineStages._gpu_env({"FOO": "bar"})
    assert env == {"FOO": "bar"}          # inherited, no CVD added


def test_object_crops_is_a_registered_stage():
    assert "object_crops" in STAGE_NAMES
    assert STAGE_NAMES.index("object_crops") == STAGE_NAMES.index("segment") + 1


# ---------------------------------------------------------------------------
# R2 — extract_objects: loud failure, no fakes
# ---------------------------------------------------------------------------

def test_extract_objects_full_scene_still_copies(stages, trained_ply):
    result = stages.extract_objects(str(trained_ply),
                                    labels=[{"label": "full_scene", "count": -1}])
    assert result.success
    plys = json.loads(result.artifacts["object_plys"])
    assert len(plys) == 1
    assert Path(plys[0]).read_bytes() == b"ply-fake"


def test_extract_objects_without_masks_fails_loudly(stages, trained_ply):
    """Pre-ADR-025 this copied the WHOLE SCENE as 'vase' — the audit's F1."""
    result = stages.extract_objects(
        str(trained_ply), labels=[{"label": "vase", "object_id": 1}])
    assert not result.success
    failures = json.loads(result.artifacts["object_failures"])
    assert failures[0]["label"] == "vase"
    assert "no SAM3 masks" in failures[0]["error"]
    # And crucially: no fake per-object PLY was written.
    assert not (stages.job_dir / "objects" / "vase.ply").exists()


def test_extract_objects_mixed_reports_both(stages, trained_ply):
    result = stages.extract_objects(str(trained_ply), labels=[
        {"label": "full_scene", "count": -1},
        {"label": "vase", "object_id": 1},
    ])
    assert result.success                        # the environment extracted
    assert result.metrics["failed"] == 1         # the object did not — reported
    failures = json.loads(result.artifacts["object_failures"])
    assert failures[0]["label"] == "vase"


def test_extract_objects_no_colmap_is_a_cause(stages, trained_ply):
    masks_dir = stages.job_dir / "sam3_masks"
    masks_dir.mkdir(parents=True)
    (masks_dir / "mask_0001.npy").write_bytes(b"npy-fake")
    result = stages.extract_objects(
        str(trained_ply), labels=[{"label": "vase", "object_id": 1}])
    assert not result.success
    failures = json.loads(result.artifacts["object_failures"])
    assert "COLMAP" in failures[0]["error"]


# ---------------------------------------------------------------------------
# R3 — object_crops stage failure reporting
# ---------------------------------------------------------------------------

def test_object_crops_skips_cleanly_with_no_objects(stages, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    result = stages.object_crops(str(frames),
                                 objects=[{"label": "full_scene", "count": -1}])
    assert result.success
    assert result.metrics.get("skipped")


def test_object_crops_reports_missing_perframe_masks(stages, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    result = stages.object_crops(
        str(frames), objects=json.dumps([{"label": "vase", "object_id": 1}]))
    assert not result.success
    assert "no per-frame masks" in result.error
    manifest = json.loads(
        (stages.job_dir / "object_crops" / "crops.json").read_text())
    assert manifest["failures"][0]["label"] == "vase"


# ---------------------------------------------------------------------------
# R4/R6 — mesh_objects: crop-conditioned generator, verbatim GLB persistence
# ---------------------------------------------------------------------------

def _object_ply(stages: PipelineStages, name: str) -> Path:
    d = stages.job_dir / "objects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.ply"
    p.write_bytes(b"ply-fake")
    return p


def _stub_trellis2(monkeypatch, glb: bytes = GLB_BYTES):
    """Stub pipeline.trellis2_client at the module boundary."""
    res = types.SimpleNamespace(
        glb_data=glb, glb_low_data=b"low", mesh=None, vertex_count=0,
        face_count=200_000, duration_seconds=1.0, backend="stub", error=None,
        lineage={"conditioning": "single-image", "generator": "TRELLIS.2-4B"},
    )
    client = MagicMock()
    client.reconstruct_from_image.return_value = res
    mod = types.ModuleType("pipeline.trellis2_client")
    mod.Trellis2Client = MagicMock()
    mod.Trellis2Client.from_config.return_value = client
    monkeypatch.setitem(sys.modules, "pipeline.trellis2_client", mod)
    return client


def test_mesh_objects_without_crop_fails_loudly(stages, monkeypatch):
    _object_ply(stages, "vase")
    monkeypatch.setattr(stages.config.hunyuan3d, "enabled", False)
    result = stages.mesh_objects([str(stages.job_dir / "objects" / "vase.ply")],
                                 crops=[])
    assert not result.success
    failures = json.loads(result.artifacts["mesh_failures"])
    assert failures[0]["label"] == "vase"
    assert "crop" in failures[0]["error"]


def test_best_of_n_runs_n_seeds_and_keeps_best(stages, tmp_path, monkeypatch):
    ply = _object_ply(stages, "vase")
    crop = tmp_path / "0001_vase.png"
    crop.write_bytes(b"\x89PNG-fake")
    monkeypatch.setattr(stages.config.hunyuan3d, "enabled", False)
    monkeypatch.setattr(stages.config.trellis2, "best_of_n", 3)

    # Crops manifest with a SQUARE observed silhouette (bbox aspect 1.0), so the
    # square seed-43 mesh matches the observation better than the squashed 42.
    crops_dir = stages.job_dir / "object_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "crops.json").write_text(json.dumps({
        "version": "object_crops.1",
        "crops": [{"label": "vase", "object_id": 1, "crop": str(crop),
                   "bbox": [0, 0, 100, 100]}],   # w==h -> aspect 1.0
        "failures": [],
    }))

    # Per-seed fake meshes: seed 43 has the "best" proportions + faces.
    class _FakeMesh:
        def __init__(self, extents, faces, wt):
            self.extents = extents
            self.faces = list(range(faces))
            self.vertices = list(range(faces))
            self.is_watertight = wt

    profiles = {
        42: ([0.3, 1.0, 1.0], 200_000, True),    # squashed
        43: ([1.0, 1.0, 1.0], 300_000, True),    # square, healthy -> best
        44: ([1.0, 1.0, 1.0], 100, False),       # degenerate
    }
    calls = []

    def fake_recon(crop_path, seed=42, label="object", provenance=None):
        ext, faces, wt = profiles[seed]
        calls.append(seed)
        return types.SimpleNamespace(
            glb_data=GLB_BYTES + bytes([seed]), glb_low_data=None,
            mesh=_FakeMesh(ext, faces, wt), vertex_count=faces, face_count=faces,
            duration_seconds=1.0, backend="stub", error=None,
            lineage={"conditioning": "single-image", "seed": seed})

    client = MagicMock()
    client.reconstruct_from_image.side_effect = fake_recon
    mod = types.ModuleType("pipeline.trellis2_client")
    mod.Trellis2Client = MagicMock()
    mod.Trellis2Client.from_config.return_value = client
    monkeypatch.setitem(sys.modules, "pipeline.trellis2_client", mod)

    result = stages.mesh_objects(
        [str(ply)], crops=[{"label": "vase", "object_id": 1, "crop": str(crop)}])
    assert result.success
    assert sorted(calls) == [42, 43, 44]          # all three seeds generated
    mesh_info = json.loads(result.artifacts["meshes"])[0]
    lineage = json.loads(Path(mesh_info["lineage"]).read_text())
    assert lineage["best_of_n"] == 3
    assert lineage["selected_seed"] == 43         # highest-scoring kept
    assert len(lineage["candidates"]) == 3
    # The persisted GLB is seed 43's bytes.
    assert Path(mesh_info["mesh"]).read_bytes() == GLB_BYTES + bytes([43])


def test_mesh_objects_persists_generator_glb_verbatim(stages, tmp_path, monkeypatch):
    ply = _object_ply(stages, "vase")
    crop = tmp_path / "0001_vase.png"
    crop.write_bytes(b"\x89PNG-fake")
    client = _stub_trellis2(monkeypatch)

    result = stages.mesh_objects(
        [str(ply)], crops=[{"label": "vase", "object_id": 1, "crop": str(crop)}])
    assert result.success
    mesh_info = json.loads(result.artifacts["meshes"])[0]

    # Byte-identical persistence + recorded hash (R6).
    written = Path(mesh_info["mesh"]).read_bytes()
    assert written == GLB_BYTES
    assert mesh_info["glb_sha256"] == hashlib.sha256(GLB_BYTES).hexdigest()
    assert mesh_info["generator"] is True
    assert mesh_info["method"] == "trellis2"
    # Low-poly pair + lineage sidecar.
    assert Path(mesh_info["mesh_low"]).read_bytes() == b"low"
    lineage = json.loads(Path(mesh_info["lineage"]).read_text())
    assert lineage["conditioning"] == "single-image"
    assert lineage["glb_sha256"] == mesh_info["glb_sha256"]
    # The generator saw the crop, not the PLY.
    called_with = client.reconstruct_from_image.call_args
    assert str(called_with.args[0]).endswith("0001_vase.png")


# ---------------------------------------------------------------------------
# R6 — texture_bake skips generator meshes
# ---------------------------------------------------------------------------

def test_assemble_usd_emits_r10_placements(stages, tmp_path, monkeypatch):
    # Generator meshes with placement hints -> usd/placements.json (R10).
    glb = tmp_path / "vase.glb"
    glb.write_bytes(GLB_BYTES)
    meshes = [
        {"label": "full_scene", "mesh": str(glb)},              # no placement
        {"label": "vase", "mesh": str(glb), "generator": True,
         "placement": {"centroid": [1.0, 0.0, 2.0], "extent": [0.5, 0.5, 0.5]},
         "glb_extent": [1.0, 1.0, 1.0]},
    ]
    # Stub the standalone assembler + Blender helper so the stage runs offline.
    monkeypatch.setattr(stages, "_run_blender_assembler", lambda p: {"success": False})
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""))

    result = stages.assemble_usd(meshes)
    assert result.success
    placements_file = stages.job_dir / "usd" / "placements.json"
    assert placements_file.exists()
    data = json.loads(placements_file.read_text())
    assert set(data) == {"vase"}                                # full_scene excluded
    assert data["vase"]["world_centroid"] == [1.0, 0.0, 2.0]
    assert data["vase"]["scale_ratio"] == 0.5
    assert data["vase"]["orientation"] == "unsolved"


def test_texture_bake_skips_generator_meshes(stages, tmp_path, monkeypatch):
    # Stub the baker module so this test needs no xatlas/trimesh.
    baker_mod = types.ModuleType("pipeline.texture_baker")
    baker_instance = MagicMock()
    baker_mod.TextureBaker = MagicMock(return_value=baker_instance)
    baker_mod.BakeConfig = MagicMock()
    monkeypatch.setitem(sys.modules, "pipeline.texture_baker", baker_mod)

    glb = tmp_path / "vase.glb"
    glb.write_bytes(GLB_BYTES)
    meshes = [{"label": "vase", "mesh": str(glb),
               "method": "trellis2", "generator": True}]
    result = stages.texture_bake(meshes)
    assert result.success
    assert result.metrics["skipped_generator_meshes"] == 1
    assert result.metrics["baked_count"] == 0
    baker_instance.bake.assert_not_called()
    baker_instance.bake_from_vertex_colors.assert_not_called()
    # The generator GLB is untouched.
    assert glb.read_bytes() == GLB_BYTES
