import cv2
import json
import math
import os
import subprocess
import sys
import tkinter as tk

from tkinter import filedialog, messagebox

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Blender configuration
# ------------------------------------------------------------

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"

BLENDER_SCRIPT = r"E:\Humcode\Test Images\Python wall\blender_wall_generator.py"

# FBX will be created beside the selected image unless changed
FBX_OUTPUT_NAME = "DetectedWalls.fbx"


# ------------------------------------------------------------
# Image processing
# ------------------------------------------------------------

USE_ADAPTIVE_THRESHOLD = False

THRESHOLD_VALUE = 200

ADAPTIVE_BLOCK_SIZE = 21
ADAPTIVE_C = 5

CANNY_LOW = 50
CANNY_HIGH = 150


# ------------------------------------------------------------
# Hough detection
# ------------------------------------------------------------

HOUGH_THRESHOLD = 50

MIN_LINE_LENGTH = 100

MAX_LINE_GAP = 25


# ------------------------------------------------------------
# Wall filtering
# ------------------------------------------------------------

ANGLE_TOLERANCE = 8

MIN_WALL_LENGTH = 100

# Two parallel lines must be within this distance
MIN_WALL_DISTANCE = 5
MAX_WALL_DISTANCE = 80

# How much the two wall lines need to overlap
MIN_OVERLAP_RATIO = 0.40


# ------------------------------------------------------------
# Line merging
# ------------------------------------------------------------

MERGE_DISTANCE = 15

MERGE_GAP = 30


# ------------------------------------------------------------
# Unreal scale
# ------------------------------------------------------------

# Example:
#
# If 500 pixels on the floor plan represents 10 feet:
#
# KNOWN_DISTANCE_PIXELS = 500
# KNOWN_DISTANCE_FEET = 10
#
# Then:
#
# 1 pixel = 0.02 feet
#
# Unreal uses centimeters.
# 1 foot = 30.48 cm

KNOWN_DISTANCE_PIXELS = 500.0

KNOWN_DISTANCE_FEET = 10.0


# ------------------------------------------------------------
# Wall thickness
# ------------------------------------------------------------

DEFAULT_WALL_THICKNESS_UU = 15.0


# ============================================================
# FILE PICKER
# ============================================================

def select_floorplan():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Floor Plan Image",
        filetypes=[
            ("Floor Plan Images", "*.png *.jpg *.jpeg"),
            ("PNG Files", "*.png"),
            ("JPEG Files", "*.jpg *.jpeg"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


# ============================================================
# SCALE
# ============================================================

def calculate_scale():

    if KNOWN_DISTANCE_PIXELS <= 0:
        raise ValueError(
            "KNOWN_DISTANCE_PIXELS must be greater than zero."
        )

    if KNOWN_DISTANCE_FEET <= 0:
        raise ValueError(
            "KNOWN_DISTANCE_FEET must be greater than zero."
        )

    feet_per_pixel = (
        KNOWN_DISTANCE_FEET /
        KNOWN_DISTANCE_PIXELS
    )

    cm_per_pixel = feet_per_pixel * 30.48

    return cm_per_pixel


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):

    print("\nLoading image:")
    print(image_path)

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            "OpenCV could not load the selected image."
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    return image


# ============================================================
# THRESHOLD
# ============================================================

def create_threshold(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Small blur helps reduce tiny text/noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    if USE_ADAPTIVE_THRESHOLD:

        if ADAPTIVE_BLOCK_SIZE % 2 == 0:
            raise ValueError(
                "ADAPTIVE_BLOCK_SIZE must be odd."
            )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C
        )

    else:

        _, binary = cv2.threshold(
            gray,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY_INV
        )

    return gray, binary


# ============================================================
# MORPHOLOGICAL CLEANING
# ============================================================

def clean_binary(binary):

    # --------------------------------------------------------
    # Remove tiny isolated noise
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Connect small gaps in structural lines
    # --------------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    return cleaned


# ============================================================
# CANNY
# ============================================================

def create_edges(binary):

    edges = cv2.Canny(
        binary,
        CANNY_LOW,
        CANNY_HIGH
    )

    return edges


# ============================================================
# HOUGH LINES
# ============================================================

def detect_hough_lines(edges):

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    if lines is None:
        return []

    result = []

    for line in lines:

        # ----------------------------------------------------
        # Make the Hough result independent of OpenCV's
        # returned array shape.
        #
        # It may be:
        #   [[x1, y1, x2, y2]]
        #
        # or:
        #   [x1, y1, x2, y2]
        # ----------------------------------------------------

        values = np.asarray(line).reshape(-1)

        if len(values) < 4:
            continue

        x1, y1, x2, y2 = map(
            int,
            values[:4]
        )

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(
            dx,
            dy
        )

        if length < MIN_WALL_LENGTH:
            continue

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        # Normalize angle to -90 ... +90
        while angle > 90:
            angle -= 180

        while angle < -90:
            angle += 180

        # ----------------------------------------------------
        # Near-horizontal
        # ----------------------------------------------------

        if abs(angle) <= ANGLE_TOLERANCE:

            y = (y1 + y2) / 2.0

            result.append({
                "x1": float(x1),
                "y1": float(y),
                "x2": float(x2),
                "y2": float(y),
                "length": float(length),
                "orientation": "horizontal"
            })

        # ----------------------------------------------------
        # Near-vertical
        # ----------------------------------------------------

        elif abs(abs(angle) - 90) <= ANGLE_TOLERANCE:

            x = (x1 + x2) / 2.0

            result.append({
                "x1": float(x),
                "y1": float(y1),
                "x2": float(x),
                "y2": float(y2),
                "length": float(length),
                "orientation": "vertical"
            })

    return result


# ============================================================
# LINE DISTANCE
# ============================================================

def line_distance(a, b):

    if a["orientation"] != b["orientation"]:
        return float("inf")

    if a["orientation"] == "horizontal":

        ay = (a["y1"] + a["y2"]) / 2
        by = (b["y1"] + b["y2"]) / 2

        return abs(ay - by)

    else:

        ax = (a["x1"] + a["x2"]) / 2
        bx = (b["x1"] + b["x2"]) / 2

        return abs(ax - bx)


# ============================================================
# LINE OVERLAP
# ============================================================

def line_overlap(a, b):

    if a["orientation"] == "horizontal":

        a_min = min(a["x1"], a["x2"])
        a_max = max(a["x1"], a["x2"])

        b_min = min(b["x1"], b["x2"])
        b_max = max(b["x1"], b["x2"])

    else:

        a_min = min(a["y1"], a["y2"])
        a_max = max(a["y1"], a["y2"])

        b_min = min(b["y1"], b["y2"])
        b_max = max(b["y1"], b["y2"])

    overlap = max(
        0,
        min(a_max, b_max) -
        max(a_min, b_min)
    )

    return overlap


# ============================================================
# FIND WALL PAIRS
# ============================================================

def find_wall_candidates(lines):

    candidates = []

    for i, line_a in enumerate(lines):

        if line_a["length"] < MIN_WALL_LENGTH:
            continue

        best_pair = None
        best_score = 0

        for j, line_b in enumerate(lines):

            if i == j:
                continue

            if (
                line_a["orientation"]
                !=
                line_b["orientation"]
            ):
                continue

            distance = line_distance(
                line_a,
                line_b
            )

            if distance < MIN_WALL_DISTANCE:
                continue

            if distance > MAX_WALL_DISTANCE:
                continue

            overlap = line_overlap(
                line_a,
                line_b
            )

            shorter = min(
                line_a["length"],
                line_b["length"]
            )

            if shorter <= 0:
                continue

            overlap_ratio = (
                overlap /
                shorter
            )

            if overlap_ratio < MIN_OVERLAP_RATIO:
                continue

            score = (
                overlap_ratio *
                shorter
            )

            if score > best_score:

                best_score = score
                best_pair = line_b

        # ----------------------------------------------------
        # A line is considered wall-like only when it has
        # another nearby parallel supporting line.
        # ----------------------------------------------------

        if best_pair is not None:

            candidates.append(
                line_a
            )

    return candidates


# ============================================================
# MERGE HORIZONTAL LINES
# ============================================================

def merge_horizontal(lines):

    lines = sorted(
        lines,
        key=lambda x: (
            x["y1"],
            min(x["x1"], x["x2"])
        )
    )

    merged = []

    for line in lines:

        x1 = min(
            line["x1"],
            line["x2"]
        )

        x2 = max(
            line["x1"],
            line["x2"]
        )

        y = (
            line["y1"] +
            line["y2"]
        ) / 2

        merged_into_existing = False

        for existing in merged:

            ey = existing["y"]

            if abs(y - ey) > MERGE_DISTANCE:
                continue

            if (
                x1 >
                existing["x2"] +
                MERGE_GAP
            ):
                continue

            if (
                x2 <
                existing["x1"] -
                MERGE_GAP
            ):
                continue

            existing["x1"] = min(
                existing["x1"],
                x1
            )

            existing["x2"] = max(
                existing["x2"],
                x2
            )

            existing["y"] = (
                existing["y"] + y
            ) / 2

            merged_into_existing = True

            break

        if not merged_into_existing:

            merged.append({
                "x1": x1,
                "x2": x2,
                "y": y
            })

    return merged


# ============================================================
# MERGE VERTICAL LINES
# ============================================================

def merge_vertical(lines):

    lines = sorted(
        lines,
        key=lambda x: (
            x["x1"],
            min(x["y1"], x["y2"])
        )
    )

    merged = []

    for line in lines:

        y1 = min(
            line["y1"],
            line["y2"]
        )

        y2 = max(
            line["y1"],
            line["y2"]
        )

        x = (
            line["x1"] +
            line["x2"]
        ) / 2

        merged_into_existing = False

        for existing in merged:

            ex = existing["x"]

            if abs(x - ex) > MERGE_DISTANCE:
                continue

            if (
                y1 >
                existing["y2"] +
                MERGE_GAP
            ):
                continue

            if (
                y2 <
                existing["y1"] -
                MERGE_GAP
            ):
                continue

            existing["y1"] = min(
                existing["y1"],
                y1
            )

            existing["y2"] = max(
                existing["y2"],
                y2
            )

            existing["x"] = (
                existing["x"] + x
            ) / 2

            merged_into_existing = True

            break

        if not merged_into_existing:

            merged.append({
                "y1": y1,
                "y2": y2,
                "x": x
            })

    return merged


# ============================================================
# CREATE WALL SEGMENTS
# ============================================================

def create_wall_segments(lines):

    horizontal = [
        x for x in lines
        if x["orientation"] == "horizontal"
    ]

    vertical = [
        x for x in lines
        if x["orientation"] == "vertical"
    ]

    horizontal = merge_horizontal(
        horizontal
    )

    vertical = merge_vertical(
        vertical
    )

    walls = []

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------

    for line in horizontal:

        length = (
            line["x2"] -
            line["x1"]
        )

        if length < MIN_WALL_LENGTH:
            continue

        walls.append({
            "start": [
                line["x1"],
                line["y"]
            ],
            "end": [
                line["x2"],
                line["y"]
            ]
        })

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    for line in vertical:

        length = (
            line["y2"] -
            line["y1"]
        )

        if length < MIN_WALL_LENGTH:
            continue

        walls.append({
            "start": [
                line["x"],
                line["y1"]
            ],
            "end": [
                line["x"],
                line["y2"]
            ]
        })

    return walls


# ============================================================
# PIXELS → UNREAL UNITS
# ============================================================

def convert_to_unreal_units(walls):

    cm_per_pixel = calculate_scale()

    print(
        f"\nScale: "
        f"{cm_per_pixel:.4f} cm/pixel"
    )

    converted = []

    for wall in walls:

        x1, y1 = wall["start"]
        x2, y2 = wall["end"]

        x1 *= cm_per_pixel
        y1 *= cm_per_pixel

        x2 *= cm_per_pixel
        y2 *= cm_per_pixel

        converted.append({
            "start": [
                round(x1, 2),
                round(y1, 2)
            ],
            "end": [
                round(x2, 2),
                round(y2, 2)
            ],
            "thickness": DEFAULT_WALL_THICKNESS_UU
        })

    return converted


# ============================================================
# SAVE WALLS.JSON
# ============================================================

def save_walls_json(
    walls,
    output_path
):

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            walls,
            f,
            indent=2
        )

    print(
        f"\nSaved walls.json:"
    )

    print(output_path)

    print(
        f"Walls exported: {len(walls)}"
    )


# ============================================================
# DEBUG IMAGE
# ============================================================

def draw_debug_lines(
    image,
    walls,
    output_path
):

    debug = image.copy()

    for wall in walls:

        x1, y1 = map(
            int,
            wall["start"]
        )

        x2, y2 = map(
            int,
            wall["end"]
        )

        cv2.line(
            debug,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

    cv2.imwrite(
        output_path,
        debug
    )

    print(
        f"Debug image:"
    )

    print(output_path)


# ============================================================
# SAVE DEBUG STAGES
# ============================================================

def save_debug_images(
    image,
    binary,
    edges,
    walls,
    output_directory
):

    threshold_path = os.path.join(
        output_directory,
        "01_threshold.png"
    )

    edges_path = os.path.join(
        output_directory,
        "02_edges.png"
    )

    final_path = os.path.join(
        output_directory,
        "03_detected_walls.png"
    )

    cv2.imwrite(
        threshold_path,
        binary
    )

    cv2.imwrite(
        edges_path,
        edges
    )

    draw_debug_lines(
        image,
        walls,
        final_path
    )

    print("\nDebug files created:")
    print(threshold_path)
    print(edges_path)
    print(final_path)


# ============================================================
# BLENDER HANDOFF
# ============================================================

def run_blender(
    walls_json_path,
    image_directory
):

    if not os.path.isfile(
        BLENDER_EXE
    ):

        raise FileNotFoundError(
            "Blender executable not found:\n"
            + BLENDER_EXE
        )

    if not os.path.isfile(
        BLENDER_SCRIPT
    ):

        raise FileNotFoundError(
            "Blender Python script not found:\n"
            + BLENDER_SCRIPT
        )

    fbx_output = os.path.join(
        image_directory,
        FBX_OUTPUT_NAME
    )

    print("\n")
    print("=" * 60)
    print("STARTING BLENDER")
    print("=" * 60)

    command = [
        BLENDER_EXE,

        "--background",

        "--python",
        BLENDER_SCRIPT,

        "--",

        "--walls_json",
        walls_json_path,

        "--output_fbx",
        fbx_output
    ]

    print(
        "\nBlender command:"
    )

    print(
        " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # --------------------------------------------------------
    # Stream Blender output
    # --------------------------------------------------------

    for line in process.stdout:

        print(
            "[BLENDER]",
            line,
            end=""
        )

    process.wait()

    print("\n")

    if process.returncode != 0:

        print(
            "ERROR: Blender failed."
        )

        print(
            f"Blender return code: "
            f"{process.returncode}"
        )

        return False

    print(
        "Blender completed successfully."
    )

    if os.path.isfile(
        fbx_output
    ):

        print(
            "\nFINAL FBX:"
        )

        print(
            fbx_output
        )

        return True

    print(
        "\nWARNING:"
    )

    print(
        "Blender returned success, "
        "but the FBX file was not found."
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("      FLOOR PLAN WALL DETECTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\nSTEP 1: Select floor plan"
    )

    image_path = select_floorplan()

    if not image_path:

        print(
            "No image selected."
        )

        return

    print(
        f"Selected:\n{image_path}"
    )

    image_directory = os.path.dirname(
        image_path
    )

    walls_json_path = os.path.join(
        image_directory,
        "walls.json"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\nSTEP 2: Load image"
    )

    image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\nSTEP 3: Threshold"
    )

    gray, binary = create_threshold(
        image
    )

    binary = clean_binary(
        binary
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\nSTEP 4: Canny edges"
    )

    edges = create_edges(
        binary
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "\nSTEP 5: Hough line detection"
    )

    lines = detect_hough_lines(
        edges
    )

    print(
        f"Candidate lines: "
        f"{len(lines)}"
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print(
        "\nSTEP 6: Wall filtering"
    )

    wall_candidates = find_wall_candidates(
        lines
    )

    print(
        f"Wall-supported lines: "
        f"{len(wall_candidates)}"
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    print(
        "\nSTEP 7: Merge lines"
    )

    walls_pixels = create_wall_segments(
        wall_candidates
    )

    print(
        f"Merged wall segments: "
        f"{len(walls_pixels)}"
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    print(
        "\nSTEP 8: Convert to Unreal Units"
    )

    walls_uu = convert_to_unreal_units(
        walls_pixels
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    print(
        "\nSTEP 9: Save walls.json"
    )

    save_walls_json(
        walls_uu,
        walls_json_path
    )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print(
        "\nSTEP 10: Save debug images"
    )

    save_debug_images(
        image,
        binary,
        edges,
        walls_pixels,
        image_directory
    )

    # --------------------------------------------------------
    # BLENDER
    # --------------------------------------------------------

    print(
        "\nSTEP 11: Blender generation"
    )

    success = run_blender(
        walls_json_path,
        image_directory
    )

    if success:

        print()
        print("=" * 60)
        print("SUCCESS")
        print("=" * 60)

    else:

        print()
        print("=" * 60)
        print("FAILED")
        print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            str(e)
        )

    finally:

        print()
        print("=" * 60)
        print("PROCESS FINISHED")
        print("=" * 60)

        input(
            "\nPress ENTER to exit..."
        )
