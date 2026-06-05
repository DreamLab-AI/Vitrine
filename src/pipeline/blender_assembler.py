# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Blender-based scene assembler for the Gaussian Toolkit.

Imports a TSDF mesh GLB, cleans debris, creates proper vertex-color materials,
sets up 3-point lighting, renders 4 preview images, and exports a USD scene.

Intended to run headless::

    blender --background --python src/pipeline/blender_assembler.py -- \\
        --input /data/output/JOB_ID/objects/meshes/full_scene/full_scene.glb \\
        --output-usd /data/output/JOB_ID/usd/scene.usda \\
        --output-renders /data/output/JOB_ID/previews/ \\
        --render-size 1920x1080 \\
        --colmap-dir /data/output/JOB_ID/colmap/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
#  Blender API — only available when running inside Blender
# ---------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from mathutils import Matrix, Vector, Quaternion
except ImportError:
    print(json.dumps({
        "success": False,
        "error": "This script must be run inside Blender: blender --background --python <script>",
    }))
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Argument parsing (args after "--")
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments supplied after Blender's ``--`` separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender scene assembler")
    parser.add_argument("--input", required=True, help="Path to input GLB mesh")
    parser.add_argument("--output-usd", required=True, help="Output USD (.usda) path")
    parser.add_argument("--output-renders", required=True, help="Directory for preview renders")
    parser.add_argument("--render-size", default="1920x1080", help="WxH render resolution")
    parser.add_argument("--colmap-dir", default=None, help="Optional COLMAP sparse dir for camera data")
    parser.add_argument("--min-faces", type=int, default=100, help="Remove components with fewer faces")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
#  Scene cleanup
# ---------------------------------------------------------------------------

def clear_scene() -> None:
    """Remove all default objects from the Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Purge orphaned data
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights]:
        for item in block:
            block.remove(item)


# ---------------------------------------------------------------------------
#  GLB import
# ---------------------------------------------------------------------------

def import_glb(filepath: str) -> list:
    """Import a GLB file and return the imported mesh objects."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = set(bpy.data.objects)
    new_objects = list(after - before)
    return new_objects


# ---------------------------------------------------------------------------
#  Debris removal
# ---------------------------------------------------------------------------

def remove_small_components(obj, min_faces: int) -> int:
    """Remove disconnected mesh components with fewer than *min_faces* faces.

    Returns the number of components removed.
    """
    if obj.type != "MESH":
        return 0

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    visited = set()
    components: list[set[int]] = []

    for face in bm.faces:
        if face.index in visited:
            continue
        component: set[int] = set()
        stack = [face]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            component.add(f.index)
            for edge in f.edges:
                for linked_face in edge.link_faces:
                    if linked_face.index not in visited:
                        stack.append(linked_face)
        components.append(component)

    removed_count = 0
    faces_to_delete = []
    for comp in components:
        if len(comp) < min_faces:
            removed_count += 1
            for fi in comp:
                faces_to_delete.append(bm.faces[fi])

    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return removed_count


# ---------------------------------------------------------------------------
#  Material from vertex colors
# ---------------------------------------------------------------------------

def create_vertex_color_material(obj) -> None:
    """Create and assign a material that uses vertex colors as base color."""
    if obj.type != "MESH":
        return

    mat = bpy.data.materials.new(name=f"{obj.name}_VertexColor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.3

    # Vertex color node
    vcol_node = nodes.new(type="ShaderNodeVertexColor")
    vcol_node.location = (-300, 0)

    # Pick the first available color attribute
    if obj.data.color_attributes:
        vcol_node.layer_name = obj.data.color_attributes[0].name
    elif obj.data.vertex_colors:
        vcol_node.layer_name = obj.data.vertex_colors[0].name

    links.new(vcol_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
#  Lighting setup (3-point)
# ---------------------------------------------------------------------------

def setup_three_point_lighting(target_location: Vector = Vector((0, 0, 0))) -> None:
    """Add key, fill, and back lights aimed at the scene center."""

    def _add_light(name: str, light_type: str, energy: float, location: tuple, color: tuple = (1, 1, 1)):
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_data.color = color
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = Vector(location)
        direction = target_location - light_obj.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        light_obj.rotation_euler = rot_quat.to_euler()
        return light_obj

    _add_light("KeyLight", "SUN", energy=5.0, location=(4, -4, 6))
    _add_light("FillLight", "AREA", energy=300.0, location=(-4, -2, 4), color=(0.9, 0.95, 1.0))
    _add_light("BackLight", "SPOT", energy=500.0, location=(0, 5, 4))


# ---------------------------------------------------------------------------
#  Camera helpers
# ---------------------------------------------------------------------------

CAMERA_POSITIONS = {
    "front": (0, -6, 2),
    "back": (0, 6, 2),
    "left": (-6, 0, 2),
    "right": (6, 0, 2),
}


def create_camera(name: str, location: tuple, target: Vector = Vector((0, 0, 0))) -> bpy.types.Object:
    """Create a camera aimed at *target*."""
    cam_data = bpy.data.cameras.new(name=name)
    cam_data.lens = 50
    cam_data.clip_start = 0.1
    cam_data.clip_end = 1000.0
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = Vector(location)

    direction = target - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    return cam_obj


def compute_scene_center(objects: list) -> Vector:
    """Compute the bounding-box center of all mesh objects."""
    min_co = Vector((float("inf"), float("inf"), float("inf")))
    max_co = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in objects:
        if obj.type != "MESH":
            continue
        found = True
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for co in bbox_corners:
            min_co.x = min(min_co.x, co.x)
            min_co.y = min(min_co.y, co.y)
            min_co.z = min(min_co.z, co.z)
            max_co.x = max(max_co.x, co.x)
            max_co.y = max(max_co.y, co.y)
            max_co.z = max(max_co.z, co.z)
    if not found:
        return Vector((0, 0, 0))
    return (min_co + max_co) / 2


def compute_camera_distance(objects: list, padding: float = 1.5) -> float:
    """Estimate a good camera distance based on scene bounding box."""
    max_dim = 0.0
    for obj in objects:
        if obj.type != "MESH":
            continue
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        for co in bbox_corners:
            dist = co.length
            if dist > max_dim:
                max_dim = dist
    return max(max_dim * padding, 3.0)


# ---------------------------------------------------------------------------
#  COLMAP camera loading (optional)
# ---------------------------------------------------------------------------

def load_colmap_cameras(colmap_dir: str):
    """Attempt to parse COLMAP images.txt for camera extrinsics.

    Returns a list of dicts with position/rotation or empty list on failure.
    """
    images_txt = Path(colmap_dir) / "sparse" / "0" / "images.txt"
    if not images_txt.exists():
        images_txt = Path(colmap_dir) / "images.txt"
    if not images_txt.exists():
        return []

    cameras = []
    try:
        with open(images_txt, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        # images.txt has pairs of lines: header then points2D
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9] if len(parts) > 9 else f"cam_{i // 2}"
            cameras.append({
                "name": name,
                "quat": (qw, qx, qy, qz),
                "translation": (tx, ty, tz),
            })
    except Exception:
        return []
    return cameras


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

def configure_renderer(width: int, height: int) -> None:
    """Configure Cycles renderer for preview quality."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # Try GPU if available
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        cprefs = prefs.preferences
        for compute_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cprefs.compute_device_type = compute_type
                cprefs.get_devices()
                for device in cprefs.devices:
                    device.use = True
                scene.cycles.device = "GPU"
                break
            except Exception:
                continue


def render_preview(camera_obj, output_path: str) -> None:
    """Render the scene from *camera_obj* to *output_path*."""
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
#  Texture baking (Cycles GPU, ~0.5s for 100K faces)
# ---------------------------------------------------------------------------

def bake_vertex_colors_to_texture(
    obj: 'bpy.types.Object',
    output_path: str,
    texture_size: int = 2048,
) -> None:
    """Bake vertex colors to a UV-mapped texture using Cycles.

    Uses Smart UV Project for fast UV unwrapping, then Cycles DIFFUSE
    bake at 1 sample (color only, no lighting) to transfer vertex colors
    to a texture atlas. Total time ~0.5s for 100K faces on GPU.
    """
    import bpy

    if getattr(obj, "type", None) != 'MESH':
        return  # only mesh objects can be baked

    # Ensure object mode and that ONLY this mesh is selected. The glTF import
    # leaves a non-mesh 'world' root selected, and bpy.ops.object.bake operates
    # on every selected object -> "Object 'world' is not a mesh".
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    me = obj.data

    # Smart UV Project (fast, no xatlas needed)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=1.15,  # ~66 degrees
        island_margin=0.02,
        area_weight=0.0,
        correct_aspect=True,
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create bake target image
    img = bpy.data.images.new(
        name=f"{obj.name}_diffuse",
        width=texture_size,
        height=texture_size,
        alpha=False,
    )

    # Set up material for baking: need an Image Texture node selected
    mat = obj.data.materials[0] if obj.data.materials else None
    if mat is None:
        return

    tree = mat.node_tree
    # Add image texture node for bake target
    tex_node = tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    tex_node.select = True
    tree.nodes.active = tex_node

    # Configure Cycles for baking
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Try GPU, fall back to CPU
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        prefs.preferences.compute_device_type = 'CUDA'
        prefs.preferences.get_devices()
        for device in prefs.preferences.devices:
            device.use = True
    scene.cycles.device = 'GPU'

    scene.cycles.samples = 1  # Color only, no noise
    scene.cycles.bake_type = 'DIFFUSE'

    # Bake
    bpy.ops.object.bake(
        type='DIFFUSE',
        pass_filter={'COLOR'},  # No direct/indirect lighting, just color
        use_clear=True,
        margin=4,
        margin_type='EXTEND',
    )

    # Save baked texture
    img.filepath_raw = output_path
    img.file_format = 'PNG'
    img.save()

    # Wire the baked texture into the material (replace vertex color input)
    bsdf = None
    for node in tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf = node
            break

    if bsdf:
        # Remove vertex color link, connect image texture instead
        base_color_input = bsdf.inputs['Base Color']
        for link in list(tree.links):
            if link.to_socket == base_color_input:
                tree.links.remove(link)
        tree.links.new(tex_node.outputs['Color'], base_color_input)

    obj.select_set(False)


# ---------------------------------------------------------------------------
#  USD export
# ---------------------------------------------------------------------------

def export_usd(output_path: str) -> None:
    """Export the scene as USD (.usda or .usdc based on extension)."""
    bpy.ops.object.select_all(action="SELECT")
    desired = dict(
        filepath=output_path,
        selected_objects_only=False,
        export_materials=True,
        export_meshes=True,
        export_lights=True,
        export_cameras=True,
        overwrite_existing_textures=True,
    )
    # wm.usd_export's keyword set varies across Blender versions (5.0 dropped
    # 'overwrite_existing_textures'). Pass only properties this build accepts.
    try:
        valid = set(bpy.ops.wm.usd_export.get_rna_type().properties.keys())
        kwargs = {k: v for k, v in desired.items() if k in valid}
    except Exception:
        kwargs = {k: v for k, v in desired.items() if k != "overwrite_existing_textures"}
    bpy.ops.wm.usd_export(**kwargs)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    result = {
        "success": False,
        "input": args.input,
        "output_usd": args.output_usd,
        "output_renders": args.output_renders,
        "renders": [],
        "components_removed": 0,
        "mesh_objects": 0,
        "error": None,
    }

    try:
        width, height = (int(x) for x in args.render_size.split("x"))
    except ValueError:
        result["error"] = f"Invalid render size: {args.render_size!r}. Expected WxH (e.g. 1920x1080)."
        print(json.dumps(result))
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        result["error"] = f"Input file not found: {args.input}"
        print(json.dumps(result))
        sys.exit(1)

    # Prepare output directories
    usd_dir = Path(args.output_usd).parent
    usd_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = Path(args.output_renders)
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clean scene and import GLB
    clear_scene()
    imported = import_glb(str(input_path))
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    result["mesh_objects"] = len(mesh_objects)

    if not mesh_objects:
        result["error"] = "No mesh objects found in GLB"
        print(json.dumps(result))
        sys.exit(1)

    # Step 2: Remove small debris components
    total_removed = 0
    for obj in mesh_objects:
        total_removed += remove_small_components(obj, min_faces=args.min_faces)
    result["components_removed"] = total_removed

    # Step 3: Create vertex-color materials
    for obj in mesh_objects:
        create_vertex_color_material(obj)

    # Step 3b: Bake vertex colors to UV texture via Cycles (~0.5s for 100K faces)
    for obj in mesh_objects:
        try:
            tex_path = str(renders_dir / f"{obj.name}_diffuse.png")
            bake_vertex_colors_to_texture(obj, tex_path, texture_size=2048)
            result["texture_baked"] = True
            result["texture_path"] = tex_path
        except Exception as bake_exc:
            print(f"Texture bake failed for {obj.name}: {bake_exc}", file=sys.stderr)
            result["texture_baked"] = False

    # Step 4: Compute scene center and camera distance
    center = compute_scene_center(mesh_objects)
    cam_dist = compute_camera_distance(mesh_objects)

    # Scale camera positions by computed distance
    scaled_positions = {}
    for view_name, base_pos in CAMERA_POSITIONS.items():
        direction = Vector(base_pos).normalized()
        scaled_positions[view_name] = tuple(center + direction * cam_dist)

    # Step 5: Setup lighting
    setup_three_point_lighting(target_location=center)

    # Step 6: Configure renderer
    configure_renderer(width, height)

    # Step 7: Render from 4 views
    for view_name, pos in scaled_positions.items():
        cam = create_camera(f"Camera_{view_name}", pos, target=center)
        render_path = str(renders_dir / f"{view_name}.png")
        render_preview(cam, render_path)
        result["renders"].append(render_path)
        # Remove camera after render to keep USD clean (re-add for export if needed)
        bpy.data.objects.remove(cam, do_unlink=True)

    # Step 8: Add COLMAP cameras if available
    if args.colmap_dir:
        colmap_cams = load_colmap_cameras(args.colmap_dir)
        for cam_info in colmap_cams:
            qw, qx, qy, qz = cam_info["quat"]
            tx, ty, tz = cam_info["translation"]
            # COLMAP stores world-to-camera; invert for camera-in-world
            rot = Quaternion((qw, qx, qy, qz))
            rot_mat = rot.to_matrix().to_4x4()
            trans_vec = Vector((tx, ty, tz))
            # Camera world position = -R^T * t
            cam_pos = -(rot_mat.to_3x3().transposed() @ trans_vec)
            cam_name = Path(cam_info["name"]).stem
            create_camera(f"COLMAP_{cam_name}", tuple(cam_pos), target=center)

    # Add a default camera for USD export
    default_cam = create_camera("DefaultCamera", scaled_positions["front"], target=center)
    bpy.context.scene.camera = default_cam

    # Step 9: Export USD
    try:
        export_usd(args.output_usd)
    except Exception as exc:
        result["error"] = f"USD export failed: {exc}"
        print(json.dumps(result))
        sys.exit(1)

    result["success"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
