#!/usr/bin/env python
#not tested yet
import cv2 as cv
import numpy as np
from cv2 import aruco
from collections import deque
import sys, os

sys.path.append(os.path.dirname(__file__))
from path_planning import greedy_path, tsp_bruteforce

# --- Configuration ---
NUM_CELLS = 10
MARKER_SIZE_CM = 10.0
REFERENCE_ID = 0
font = cv.FONT_HERSHEY_SIMPLEX

# --- Define ArUco dictionary and parameters ---
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters()

# --- Color assignments ---
color_assignments = {
    1: ("red", (0, 0, 255)),
    2: ("green", (0, 255, 0)),
    3: ("blue", (255, 0, 0))
}

# --- Initialize camera ---
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

# --- Draw grid ---
def draw_grid(frame):
    h, w, _ = frame.shape
    cell_w, cell_h = w // NUM_CELLS, h // NUM_CELLS
    for i in range(NUM_CELLS + 1):
        cv.line(frame, (i * cell_w, 0), (i * cell_w, h), (255, 255, 255), 1)
        cv.line(frame, (0, i * cell_h), (w, i * cell_h), (255, 255, 255), 1)
    return frame

# --- Convert grid coordinates to pixel center ---
def grid_to_pixel(cell_x, cell_y, frame_shape):
    h, w, _ = frame_shape
    cell_w, cell_h = w // NUM_CELLS, h // NUM_CELLS
    px = int(cell_x * cell_w + cell_w / 2)
    py = int(cell_y * cell_h + cell_h / 2)
    return (px, py)

# --- Draw text overlay ---
def draw_overlay(frame, lines):
    y_offset = 30
    for line in lines:
        cv.putText(frame, line, (10, y_offset), font, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        y_offset += 30

# --- For stable text display ---
stable_text = []
text_history = deque(maxlen=10)

# --- Path storage ---
visual_paths = []  # list of (path_points, color)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, marker_dict, parameters=param_markers)
    frame = draw_grid(frame)

    current_text = []

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
        for i, corner in enumerate(corners):
            id_val = int(ids[i][0])
            c = corner[0]
            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
            cell_x, cell_y = int((cx / frame.shape[1]) * NUM_CELLS), int((cy / frame.shape[0]) * NUM_CELLS)

            # Reference marker
            if id_val == REFERENCE_ID:
                color = (0, 255, 0)
                cv.putText(frame, "Reference Marker", (cx - 40, cy - 20), font, 0.6, color, 2)
                cv.rectangle(frame, (cx - 15, cy - 15), (cx + 15, cy + 15), color, 2)
                cv.circle(frame, (cx, cy), 5, color, -1)
                current_text.append(f"ID {id_val}: Reference ({cell_x}, {cell_y})")

            # Color-assigned cubes
            elif id_val in color_assignments:
                color_name, color = color_assignments[id_val]
                cv.polylines(frame, [np.int32(c)], True, color, 2, cv.LINE_AA)
                cv.circle(frame, (cx, cy), 5, color, -1)
                current_text.append(f"ID {id_val}: {color_name} ({cell_x}, {cell_y})")

    # --- Flicker suppression ---
    if current_text:
        text_history.append(tuple(current_text))
    if len(text_history) > 0:
        stable_text = max(set(text_history), key=text_history.count)
    draw_overlay(frame, stable_text)

    # --- Path Planning Trigger ---
    if len(text_history) == text_history.maxlen:
        red_cubes, green_cubes, blue_cubes = [], [], []
        reference_cell = None

        for t in stable_text:
            if "Reference" in t:
                gx, gy = map(int, t.split("(")[-1].strip(")").split(","))
                reference_cell = (gx, gy)
            elif "red" in t:
                gx, gy = map(int, t.split("(")[-1].strip(")").split(","))
                red_cubes.append((gx, gy))
            elif "green" in t:
                gx, gy = map(int, t.split("(")[-1].strip(")").split(","))
                green_cubes.append((gx, gy))
            elif "blue" in t:
                gx, gy = map(int, t.split("(")[-1].strip(")").split(","))
                blue_cubes.append((gx, gy))

        if reference_cell:
            print("\n================ PATH PLANNING RESULTS ================")
            print(f"Reference Marker (Bin) at {reference_cell}")
            visual_paths.clear()

            for color_label, cubes, color_rgb in [
                ("🟥 RED", red_cubes, (0, 0, 255)),
                ("🟩 GREEN", green_cubes, (0, 255, 0)),
                ("🟦 BLUE", blue_cubes, (255, 0, 0))
            ]:
                if cubes:
                    path_tsp, dist_tsp = tsp_bruteforce(reference_cell, cubes)
                    print(f"\n{color_label} cubes: {cubes}")
                    print(f"TSP Path: {path_tsp} | Distance: {dist_tsp}")

                    # Store for visualization
                    visual_paths.append((path_tsp, color_rgb))
            text_history.clear()

    # --- Draw stored paths ---
    for path, color in visual_paths:
        for i in range(len(path) - 1):
            p1 = grid_to_pixel(path[i][0], path[i][1], frame.shape)
            p2 = grid_to_pixel(path[i + 1][0], path[i + 1][1], frame.shape)
            cv.line(frame, p1, p2, color, 2, cv.LINE_AA)

    cv.imshow("Aruco Grid Tracker", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
