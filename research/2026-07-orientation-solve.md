# R10 orientation solve — upright yaw from the crop camera pose

**Status:** R&D complete — pure module + tests landed (`src/pipeline/object_orientation.py`,
`tests/python/test_object_orientation.py`); integration diff below, not yet applied.
**Extends:** ADR-025 D3 (placement from splat + crop camera pose), PRD v4 R10
(position + uniform scale shipped; orientation was left `"unsolved"`).
**Date:** 2026-07-10

---

## 1. Problem

`object_placement.solve_placement` puts a generated object at the right place
(Gaussian-subset centroid) at the right size (extent ratio), but leaves the
rotation identity and honestly flags `orientation: "unsolved"`. TRELLIS.2
emits every mesh in its own canonical frame, so identity rotation means every
object in the assembled scene faces the same arbitrary direction — chairs
facing the wall, a bust facing away from the room.

## 2. The assumption: canonical front ↔ observing-camera ray

Single-image 3D generators canonicalize their output **relative to the
conditioning image**: the surface visible in the input photo becomes the
mesh's canonical *front*, exported front-facing **+Z, Y-up** (glTF
convention). ADR-025 conditions the generator on exactly one crop from one
known source frame, and `object_crops` records that frame's COLMAP pose in
the crop provenance (`camera_pose.quaternion_wxyz`, `camera_pose.translation`).

So the imaged side of the object — the mesh's +Z — was, in reality, the side
facing the crop camera. **Rotate the mesh so its canonical +Z points from the
object's position back toward the crop camera, and the generated front lines
up with the photographed front.** One view cannot pin all three rotational
DoF (see §5), but objects in captured scenes overwhelmingly rest upright, so
we pin pitch/roll with an upright prior and solve **yaw only**.

This is principled, not heuristic: the crop selector already biases toward
frontal, centred, un-clipped observations (centrality + edge-clearance
scores), i.e. toward exactly the views where this assumption is strongest.

## 3. Derivation

### 3.1 Conventions (verified against the pipeline sources)

**COLMAP** (`colmap_parser.ColmapImage`, images.txt/bin fields) stores
world→camera:

```
p_cam = R(q) · p_world + t        camera frame: +X right, +Y down, +Z forward (RDF)
```

with `q = (qw, qx, qy, qz)`. Two standard consequences:

```
camera center (world):   C   = −Rᵀ · t                     (p_cam = 0)
optical axis  (world):   fwd = Rᵀ · (0,0,1)ᵀ = third ROW of R
```

(`assemble_usd_scene.colmap_camera_world_position` already computes exactly
`−Rᵀt`; the module mirrors it.)

**COLMAP → USD** (`assemble_usd_scene.py`, Y-up stage): points map through

```
colmap_to_usd_position: (x, y, z) → (x·s, −y·s, −z·s),  s = SCENE_SCALE = 0.5
```

The linear part is a 180° rotation about X (the assembler's
`_RDF_TO_YUP_QUAT = (0,1,0,0)`). **Directions** transform with the linear
part only — scale never applies to a direction:

```
d_usd = (d_x, −d_y, −d_z)
```

Therefore scene **up** is +Y in USD ⇔ **−Y in COLMAP world**. This is the
same gravity assumption the whole assembler already bakes in; the yaw solve
adds no new one.

**Mesh local frame:** the generator GLB is glTF Y-up/front +Z; the assembler
inlines it via trimesh into the Y-up stage without an up-axis conversion, so
mesh-local up already equals stage up. The upright prior is a *no-op
constraint* on pitch/roll — we simply never rotate about anything but +Y.

### 3.2 The desired front direction

The front axis must point from the object toward the observing camera, in
COLMAP world:

* **Exact (method `camera-ray`)** — we know where the object is (the
  Gaussian-subset centroid `p_obj` the position solve already uses):

  ```
  f_colmap = normalize(C − p_obj) = normalize(−Rᵀt − p_obj)
  ```

* **Proxy (method `optical-axis`)** — no centroid available; the object was
  approximately centred in the crop frame (the selector's centrality bias),
  so the viewing ray ≈ the optical axis:

  ```
  f_colmap = −fwd = −(third row of R)
  ```

  (minus: the camera looks *at* the object; the object's front points *back*.)

`camera-ray` is strictly better whenever the centroid exists — the object
need not be centred in frame — and the integration below always passes it.

### 3.3 Map to USD and apply the upright constraint

```
f_usd = (f_x, −f_y, −f_z)                      (direction map, §3.1)
f_h   = (f_usd.x, 0, f_usd.z)                  (project onto horizontal XZ plane)
```

If `‖f_h‖ ≈ 0` the camera looked straight down (or up) at the object: yaw is
unobservable from this view → return identity, `method: "degenerate"` (the
placement stays honestly `"unsolved"`).

### 3.4 Yaw and quaternion

A yaw of θ about +Y takes the canonical front +Z to `(sin θ, 0, cos θ)`:

```
R_y(θ)·(0,0,1)ᵀ = (sin θ, 0, cos θ)ᵀ    ⇒    θ = atan2(f_usd.x, f_usd.z)
```

As a wxyz quaternion (axis +Y):

```
q_obj = (cos(θ/2), 0, sin(θ/2), 0)         — expressed in the USD stage frame
```

Because the rotation axis is the stage/mesh up-axis and the placement scale
is uniform, the orient op commutes with the scale op; order in the Xform
stack is translate → orient → scale (standard TRS).

### 3.5 Worked check (also a unit test)

Level camera on the +X axis in USD, looking at the origin: COLMAP
`R = R_y(90°)` ⇒ `q_cam = (√½, 0, √½, 0)`, `t = (0,0,2)` (center `C=(2,0,0)`).
Optical axis (row 3) = `(−1,0,0)`; front `= (1,0,0)`; USD `= (1,0,0)`;
`θ = atan2(1,0) = 90°`; `q_obj = (√½, 0, √½, 0)`. Rotating +Z by `q_obj`
gives +X — the object faces the camera. ✔ (16 more cases in the test file,
including elevated, off-centre, behind, top-down-degenerate, malformed.)

## 4. The module

`src/pipeline/object_orientation.py` — pure, deterministic, stdlib-`math`
only (no numpy, no pxr, no I/O), GPL-3.0, mirroring `object_placement`'s
testability contract:

```python
solve_yaw(camera_quaternion_wxyz, camera_translation,
          object_centroid=None) -> {
    "quat_wxyz":     [w, x, y, z],   # USD-frame rotation, pure yaw about +Y
    "yaw_deg":       float,
    "method":        "camera-ray" | "optical-axis" | "degenerate",
    "elevation_deg": float,          # confidence signal (see §5)
    "front_usd":     [x, y, z],      # diagnostic, pre-projection
}
```

Helpers exposed for reuse/tests: `camera_center_colmap`, `camera_forward_colmap`,
`colmap_dir_to_usd`, `yaw_from_front_usd`, `quat_wxyz_from_yaw`,
`quat_multiply`, `rotate_vec_by_quat`, `normalize_quat_wxyz`,
`quat_to_rotation_matrix`, plus `FRONT_AXIS_USD` / `UP_AXIS_USD` constants.

Failure posture: malformed input (zero quaternion, wrong arity) raises
`ValueError`; *geometric* degeneracy (top-down view, centroid == camera
center) degrades gracefully with an explicit `method` flag — a wrong
orientation is worse than an unsolved one, same ethos as the scale fallback.

## 5. Limitations (stated, not hidden)

1. **Roll is unobservable, pitch is confounded** — one view fixes one axis
   correspondence only. The upright prior resolves both; it is wrong for
   wall-mounted, leaning, or toppled objects. `method` + `elevation_deg`
   in the lineage let downstream consumers judge.
2. **Gravity assumption** — "up" = USD +Y = COLMAP −Y is inherited from the
   assembler, not established by it. If a capture's COLMAP frame is not
   gravity-aligned, *everything* in the scene (env mesh, cameras, objects)
   is tilted by the same global rotation, so the yaw solve remains
   self-consistent within the scene; it does not fix the global tilt.
3. **Canonical-front assumption needs one empirical check** — if TRELLIS.2's
   export fronts −Z rather than +Z (a fixed 180° offset), every solved yaw
   is off by exactly 180°. That is a one-constant fix (`FRONT_AXIS_USD`).
   Validation: assemble a run with a hero object whose front is obvious
   (bust, chair), render the crop camera's view of the USD scene, compare.
4. **High-elevation crops are weak evidence** — a crop shot from 70° above
   mostly shows the top; the generator's "front" choice gets noisy.
   `elevation_deg` is reported precisely so an escalation rung (e.g. R7
   best-of-N with a silhouette scorer) can gate on it. A future refinement:
   score N yaw hypotheses against the SAM silhouette from a second observing
   frame (the "+ silhouette" half of the R&D ticket; out of scope here).
5. **Rotation about the local origin** — the GLB is only *near*-centred;
   rotating a slightly off-centre mesh about its origin displaces it
   slightly. The same is already true of the scale op; magnitude is bounded
   by the centring error × extent, negligible next to centroid noise.

## 6. Integration (exact diff — NOT applied; shared files untouched)

### 6.1 placements.json schema additions

New per-object fields (absent ⇒ legacy placement, assembler behaves as today):

```json
{
  "vase": {
    "label": "vase",
    "world_centroid": [...],
    "scale_ratio": 1.23,
    "orientation": "yaw-solved",          // was always "unsolved"
    "orientation_quat": [0.707, 0.0, 0.707, 0.0],   // wxyz, USD frame
    "orientation_yaw_deg": 90.0,
    "orientation_method": "camera-ray",   // camera-ray | optical-axis
    "orientation_elevation_deg": 12.4,
    "world_extent": [...], "glb_extent": [...]
  }
}
```

`orientation` stays `"unsolved"` (and no quat is written) when the pose is
missing/degenerate — consumers keep a single honest flag to key off.

### 6.2 `src/pipeline/stages.py` — thread the crop camera pose to the mesh info

The crops manifest already carries `camera_pose` per crop entry;
`_generate_object_from_crop._persist` just needs to forward it:

```diff
@@ def _persist(glb_data, method, lineage, glb_low=None, mesh=None):  # stages.py ~L2628
             info: dict[str, Any] = {
                 "label": label,
                 "mesh": str(mesh_glb_path),
                 "ply": ply_path,
                 "vertex_count": 0 if mesh is None else len(mesh.vertices),
                 "method": method,
                 "generator": True,
                 "glb_sha256": sha,
                 "crop": str(crop_path),
                 "placement": placement,
                 "glb_extent": glb_extent,
+                # R10 orientation solve input: the crop frame's COLMAP pose.
+                "camera_pose": (crop_entry or {}).get("camera_pose"),
             }
```

### 6.3 `src/pipeline/object_placement.py` — solve yaw in `build_placements`

```diff
@@ def build_placements(meshes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
     out: dict[str, dict[str, Any]] = {}
     for m in meshes:
         placement = m.get("placement")
         if not placement:
             continue
         label = m.get("label", "object")
         p = solve_placement(
             label,
             placement.get("centroid", [0, 0, 0]),
             placement.get("extent", [0, 0, 0]),
             m.get("glb_extent"),
         )
-        out[label] = p.to_dict()
+        d = p.to_dict()
+        # R10 orientation: upright yaw from the crop camera pose (ADR-025 D3).
+        pose = m.get("camera_pose") or {}
+        if pose.get("quaternion_wxyz") and pose.get("translation"):
+            from pipeline.object_orientation import solve_yaw
+            try:
+                o = solve_yaw(pose["quaternion_wxyz"], pose["translation"],
+                              object_centroid=placement.get("centroid"))
+            except ValueError:
+                o = None    # malformed pose provenance -> stay "unsolved"
+            if o is not None and o["method"] != "degenerate":
+                d["orientation"] = "yaw-solved"
+                d["orientation_quat"] = o["quat_wxyz"]
+                d["orientation_yaw_deg"] = o["yaw_deg"]
+                d["orientation_method"] = o["method"]
+                d["orientation_elevation_deg"] = o["elevation_deg"]
+        out[label] = d
     return out
```

(If keeping `object_placement` import-free of siblings is preferred, hoist
the import to module top — `object_orientation` is equally pure.)

### 6.4 `scripts/assemble_usd_scene.py` — apply the quat between translate and scale

```diff
@@ def assemble_scene(...):  # object placement branch, ~L648
         placement = placements.get(label)
         if placement is not None:
             wc = placement.get("world_centroid", [0.0, 0.0, 0.0])
             ux, uy, uz = colmap_to_usd_position(wc[0], wc[1], wc[2])
             xform.AddTranslateOp().Set(Gf.Vec3d(ux, uy, uz))
+            # R10 orientation: upright yaw solved from the crop camera pose.
+            # quat is authored in the USD stage frame; TRS order (orient
+            # between translate and scale; commutes with the uniform scale).
+            oq = placement.get("orientation_quat")
+            if placement.get("orientation") == "yaw-solved" and oq and len(oq) == 4:
+                xform.AddOrientOp().Set(Gf.Quatf(
+                    float(oq[0]), float(oq[1]), float(oq[2]), float(oq[3])))
             # scale_ratio is in raw world units; USD positions carry SCENE_SCALE,
             # so the object's size must be scaled by the same factor to match.
             s = float(placement.get("scale_ratio", 1.0)) * SCENE_SCALE
             xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))
             prim.SetCustomDataByKey("v2g:placement_orientation",
                                     placement.get("orientation", "unsolved"))
+            if placement.get("orientation_method"):
+                prim.SetCustomDataByKey("v2g:placement_orientation_method",
+                                        placement.get("orientation_method"))
+                prim.SetCustomDataByKey("v2g:placement_yaw_deg",
+                                        float(placement.get("orientation_yaw_deg", 0.0)))
             prim.SetCustomDataByKey("v2g:placement_scale_ratio",
                                     float(placement.get("scale_ratio", 1.0)))
```

(`Gf.Quatf(w, x, y, z)` — real part first, matching our wxyz.)

## 7. Validation plan (runtime, next e2e run)

1. Unit: `pytest tests/python/test_object_orientation.py` — 17 tests, green
   (known poses → exact yaw/quaternion, upright constraint, ray-vs-axis,
   degenerate + malformed inputs).
2. Hermetic: extend the object-arc suite so `build_placements` on a mesh
   entry carrying `camera_pose` emits `orientation: "yaw-solved"` + a unit
   pure-yaw quat.
3. Empirical (the §5.3 flip check): dreamlab hero object (bust) — render the
   assembled USD from the crop camera's USD pose; the generated front must
   face the viewport. If it shows its back, flip `FRONT_AXIS_USD` once.
4. Regression guard: legacy placements.json (no `orientation_quat`) must
   assemble identically to today — the diff is additive-only.
