#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from cv2 import aruco
from collections import deque
import sys, os

sys.path.append(os.path.dirname(__file__))
from path_planning import astar, greedy_path, tsp_bruteforce

# =============================
# CONFIGURATION
# =============================
NUM_CELLS = 10
REFERENCE_ID = 0
font = cv.FONT_HERSHEY_SIMPLEX

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters()

# Cube color mapping
color_assignments = {
    1: ("red", (0, 0, 255)),
    2: ("green", (0, 255, 0)),
    3: ("blue", (255, 0, 0))
}

# Obstacle markers (IDs 10–19)
OBSTACLE_IDS = set(range(10, 20))

# =============================
# GRID HELPERS
# =============================

def draw_grid(frame):
    h, w, _ = frame.shape
    cell_w, cell_h = w // NUM_CELLS, h // NUM_CELLS
    for i in range(NUM_CELLS + 1):
        cv.line(frame, (i * cell_w, 0), (i * cell_w, h), (255, 255, 255), 1)
        cv.line(frame, (0, i * cell_h), (w, i * cell_h), (255, 255, 255), 1)
    return frame

def get_cell_coords(cx, cy, frame_shape):
    h, w, _ = frame_shape
    cell_x, cell_y = int((cx / w) * NUM_CELLS), int((cy / h) * NUM_CELLS)
    return (cell_x, cell_y)

def grid_to_pixel(cell_x, cell_y, frame_shape):
    h, w, _ = frame_shape
    cell_w, cell_h = w // NUM_CELLS, h // NUM_CELLS
    px = int(cell_x * cell_w + cell_w / 2)
    py = int(cell_y * cell_h + cell_h / 2)
    return (px, py)

# =============================
# TEXT & VISUALIZATION HELPERS
# =============================

def draw_overlay(frame, lines):
    y_offset = 30
    for line in lines:
        cv.putText(frame, line, (10, y_offset), font, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        y_offset += 30

# =============================
# MAIN
# =============================

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

text_history = deque(maxlen=10)
stable_text = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, marker_dict, parameters=param_markers)
    frame = draw_grid(frame)

    current_text = []
    detections = {}
    obstacles = set()

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)

        for i, corner in enumerate(corners):
            id_val = int(ids[i][0])
            c = corner[0]
            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
            cell_x, cell_y = get_cell_coords(cx, cy, frame.shape)

            # ------------- Reference Marker -------------
            if id_val == REFERENCE_ID:
                color = (0, 255, 0)
                cv.putText(frame, "Reference Marker", (cx - 40, cy - 20),
                           font, 0.6, color, 2)
                cv.rectangle(frame, (cx - 15, cy - 15), (cx + 15, cy + 15), color, 2)
                cv.circle(frame, (cx, cy), 5, color, -1)
                detections["reference"] = (cell_x, cell_y)
                current_text.append(f"ID {id_val}: Reference ({cell_x}, {cell_y})")

            # ------------- Color Cubes -------------
            elif id_val in color_assignments:
                color_name, color = color_assignments[id_val]
                cv.polylines(frame, [np.int32(c)], True, color, 2, cv.LINE_AA)
                cv.circle(frame, (cx, cy), 5, color, -1)
                detections[color_name] = (cell_x, cell_y)
                current_text.append(f"ID {id_val}: {color_name} ({cell_x}, {cell_y})")

            # ------------- Obstacles -------------
            elif id_val in OBSTACLE_IDS:
                obstacles.add((cell_x, cell_y))
                cv.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 255, 255), 2)
                cv.putText(frame, "OBST", (cx - 15, cy - 25), font, 0.5, (0, 255, 255), 2)
                current_text.append(f"ID {id_val}: obstacle ({cell_x}, {cell_y})")

    # Stable text filtering
    if current_text:
        text_history.append(tuple(current_text))
    if len(text_history) > 0:
        stable_text = max(set(text_history), key=text_history.count)
    draw_overlay(frame, stable_text)

    # =============================
    # PATH COMPUTATION AND DRAWING
    # =============================

    if "reference" in detections:
        ref_cell = detections["reference"]

        for color_name, (gx, gy) in detections.items():
            if color_name == "reference":
                continue

            if color_name in ["red", "green", "blue"]:
                path, _ = astar(ref_cell, (gx, gy), grid_size=(NUM_CELLS, NUM_CELLS), obstacles=obstacles)
                if path:
                    col = color_assignments[[k for k, v in color_assignments.items() if v[0] == color_name][0]][1]
                    for i in range(len(path) - 1):
                        p1 = grid_to_pixel(path[i][0], path[i][1], frame.shape)
                        p2 = grid_to_pixel(path[i + 1][0], path[i + 1][1], frame.shape)
                        cv.line(frame, p1, p2, col, 2)

    # =============================
    # SHOW FRAME
    # =============================
    cv.imshow("Aruco Path Planning with Obstacles", frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv.destroyAllWindows()
