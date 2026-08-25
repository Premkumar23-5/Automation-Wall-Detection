"""Generate floor-plan wall meshes from JSON and export them as an FBX.

Run from Blender, for example:
blender --background --python blender_wall_generator.py -- --walls_json walls.json --output_fbx DetectedWalls.fbx
"""

import json
import math
import os
import sys
import traceback

import bpy


DEFAULT_WALLS_JSON = r"E:\Humcode\Test Images\Python wall\walls.json"
DEFAULT_FBX_OUTPUT = r"E:\Humcode\Test Images\Python wall\DetectedWalls.fbx"
WALL_HEIGHT_UU = 300.0
DEFAULT_THICKNESS_UU = 15.0
JOIN_INTO_SINGLE_MESH = True
CLEAR_EXISTING_OBJECTS = True


def get_command_line_arguments():
    """Read only the arguments passed after Blender's `--` separator."""
    walls_json = DEFAULT_WALLS_JSON
    output_fbx = DEFAULT_FBX_OUTPUT
    user_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    index = 0
    while index < len(user_args):
        argument = user_args[index]
        if argument in {"--walls_json", "--output_fbx"}:
            if index + 1 >= len(user_args):
                raise ValueError(f"Missing value for {argument}")
            if argument == "--walls_json":
                walls_json = user_args[index + 1]
            else:
                output_fbx = user_args[index + 1]
            index += 2
        else:
            raise ValueError(f"Unknown argument: {argument}")

    return os.path.abspath(walls_json), os.path.abspath(output_fbx)


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


def export_fbx(output_fbx):
    output_directory = os.path.dirname(output_fbx)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        use_selection=True,
        apply_unit_scale=True,
        axis_forward="-Y",
        axis_up="Z",
    )
    if not os.path.isfile(output_fbx) or os.path.getsize(output_fbx) == 0:
        raise RuntimeError(f"FBX export did not create a usable file: {output_fbx}")
    print(f"FBX saved to: {output_fbx} ({os.path.getsize(output_fbx)} bytes)")


def main():
    walls_json, output_fbx = get_command_line_arguments()
    print(f"Loading walls from: {walls_json}")
    print(f"Exporting FBX to: {output_fbx}")
    if not os.path.isfile(walls_json):
        raise FileNotFoundError(f"Walls JSON not found: {walls_json}")

    walls = load_walls(walls_json)
    print(f"Found {len(walls)} wall segments")
    if not walls:
        raise ValueError("No wall segments were supplied; refusing to export an empty FBX.")

    if CLEAR_EXISTING_OBJECTS:
        clear_scene()
    setup_unreal_scale()

    wall_objects = [obj for i, wall in enumerate(walls) if (obj := create_wall(wall, i))]
    if not wall_objects:
        raise ValueError("All wall segments were degenerate; refusing to export an empty FBX.")
    print(f"Generated {len(wall_objects)} wall meshes")

    if JOIN_INTO_SINGLE_MESH:
        joined = join_all_walls(wall_objects)
        print(f"Joined into single mesh: {joined.name}")
    export_fbx(output_fbx)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
