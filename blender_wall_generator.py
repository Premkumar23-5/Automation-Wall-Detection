"""Generate floor-plan wall meshes from JSON and export them as a binary GLB.

Run from Blender, for example:
blender --background --python blender_wall_generator.py -- --walls_json walls.json --output_glb DetectedWalls.glb
"""

import json
import math
import os
import sys
import traceback

import bpy


DEFAULT_WALLS_JSON = r"E:\Humcode\Test Images\Python wall\walls.json"
DEFAULT_GLB_OUTPUT = r"E:\Humcode\Test Images\Python wall\DetectedWalls.glb"
WALL_HEIGHT_UU = 300.0
DEFAULT_THICKNESS_UU = 15.0
JOIN_INTO_SINGLE_MESH = True
CLEAR_EXISTING_OBJECTS = True
# The wall JSON is in Unreal units (centimeters); GLB coordinates are meters.
GLTF_METERS_PER_UNREAL_UNIT = 0.01


def get_command_line_arguments():
    """Read only the arguments passed after Blender's `--` separator."""
    walls_json = DEFAULT_WALLS_JSON
    output_glb = DEFAULT_GLB_OUTPUT
    user_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    index = 0
    while index < len(user_args):
        argument = user_args[index]
        if argument in {"--walls_json", "--output_glb"}:
            if index + 1 >= len(user_args):
                raise ValueError(f"Missing value for {argument}")
            if argument == "--walls_json":
                walls_json = user_args[index + 1]
            else:
                output_glb = user_args[index + 1]
            index += 2
        else:
            raise ValueError(f"Unknown argument: {argument}")

    return os.path.abspath(walls_json), os.path.abspath(output_glb)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_unreal_scale():
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.01  # 1 Blender unit equals 1 cm / Unreal unit.


def load_walls(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Support either a bare JSON list or an object that contains a `walls` list.
    walls = data.get("walls") if isinstance(data, dict) else data
    if not isinstance(walls, list):
        raise ValueError("Wall JSON must be a list or an object with a 'walls' list.")
    return walls


def create_wall(wall, index):
    start = wall["start"]
    end = wall["end"]
    thickness = float(wall.get("thickness", DEFAULT_THICKNESS_UU))
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    length = math.hypot(x2 - x1, y2 - y1)

    if length < 0.001:
        print(f"Skipping degenerate wall {index} (zero length)")
        return None
    if thickness <= 0:
        raise ValueError(f"Wall {index} has non-positive thickness: {thickness}")

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=((x1 + x2) / 2.0, (y1 + y2) / 2.0, WALL_HEIGHT_UU / 2.0),
    )
    wall_object = bpy.context.active_object
    wall_object.name = f"Wall_{index:03d}"
    wall_object.scale = (length, thickness, WALL_HEIGHT_UU)
    wall_object.rotation_euler = (0.0, 0.0, math.atan2(y2 - y1, x2 - x1))

    bpy.ops.object.select_all(action="DESELECT")
    wall_object.select_set(True)
    bpy.context.view_layer.objects.active = wall_object
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return wall_object


def join_all_walls(wall_objects):
    bpy.ops.object.select_all(action="DESELECT")
    for wall_object in wall_objects:
        wall_object.select_set(True)
    bpy.context.view_layer.objects.active = wall_objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = "HallWalls_Combined"
    return joined


def export_glb(output_glb):
    output_directory = os.path.dirname(output_glb)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    # The FBX exporter honored the scene's 0.01 unit scale.  glTF writes its
    # coordinate values in meters, so bake the equivalent conversion into the
    # mesh and its location before exporting.  This preserves FBX-scale assets
    # in UE5 while leaving walls.json in Unreal-centimeter units.
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    for mesh_object in mesh_objects:
        mesh_object.location *= GLTF_METERS_PER_UNREAL_UNIT
        mesh_object.scale *= GLTF_METERS_PER_UNREAL_UNIT
        mesh_object.select_set(True)
        bpy.context.view_layer.objects.active = mesh_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format="GLB",
        use_selection=True,
        # glTF is Y-up.  Blender's exporter applies the required Z-up to
        # Y-up conversion, which UE5's Interchange importer preserves.
        export_yup=True,
        export_apply=True,
    )
    if not os.path.isfile(output_glb) or os.path.getsize(output_glb) == 0:
        raise RuntimeError(f"GLB export did not create a usable file: {output_glb}")
    print(f"GLB saved to: {output_glb} ({os.path.getsize(output_glb)} bytes)")


def main():
    walls_json, output_glb = get_command_line_arguments()
    print(f"Loading walls from: {walls_json}")
    print(f"Exporting GLB to: {output_glb}")
    if not os.path.isfile(walls_json):
        raise FileNotFoundError(f"Walls JSON not found: {walls_json}")

    walls = load_walls(walls_json)
    print(f"Found {len(walls)} wall segments")
    if not walls:
        raise ValueError("No wall segments were supplied; refusing to export an empty GLB.")

    if CLEAR_EXISTING_OBJECTS:
        clear_scene()
    setup_unreal_scale()

    wall_objects = [obj for i, wall in enumerate(walls) if (obj := create_wall(wall, i))]
    if not wall_objects:
        raise ValueError("All wall segments were degenerate; refusing to export an empty GLB.")
    print(f"Generated {len(wall_objects)} wall meshes")

    if JOIN_INTO_SINGLE_MESH:
        joined = join_all_walls(wall_objects)
        print(f"Joined into single mesh: {joined.name}")
    export_glb(output_glb)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
