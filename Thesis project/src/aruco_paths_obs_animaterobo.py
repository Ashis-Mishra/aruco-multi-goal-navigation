#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from cv2 import aruco
from collections import deque
import sys, os
import time

sys.path.append(os.path.dirname(__file__))
from path_planning import astar, tsp_bruteforce

# ================= CONFIG =================
NUM_CELLS = 10
REFERENCE_ID = 0
OBSTACLE_IDS = set(range(10, 20))
font = cv.FONT_HERSHEY_SIMPLEX

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
param_markers = aruco.DetectorParameters()

color_assignments = {
    1: ("red", (0, 0, 255)),
    2: ("green", (0, 255, 0)),
    3: ("blue", (255, 0, 0))
}

# ================= HELPERS =================
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

# ================= MAIN =================
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible.")
    exit()

robot_pos = None
robot_path = []
robot_index = 0
mode = "IDLE"
move_speed = 0.25
obstacles = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, marker_dict, parameters=param_markers)
    frame = draw_grid(frame)

    detections = {}
    obstacles.clear()

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners)
        for i, corner in enumerate(corners):
            id_val = int(ids[i][0])
            c = corner[0]
            cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
            cell_x, cell_y = get_cell_coords(cx, cy, frame.shape)

            if id_val == REFERENCE_ID:
                detections["reference"] = (cell_x, cell_y)
                cv.putText(frame, "Reference", (cx - 40, cy - 25), font, 0.6, (0,255,0), 2)
                cv.rectangle(frame, (cx-15, cy-15), (cx+15, cy+15), (0,255,0), 2)

            elif id_val in color_assignments:
                cname, color = color_assignments[id_val]
                detections[cname] = (cell_x, cell_y)
                cv.polylines(frame, [np.int32(c)], True, color, 2)
                cv.circle(frame, (cx, cy), 5, color, -1)

            elif id_val in OBSTACLE_IDS:
                obstacles.add((cell_x, cell_y))
                cv.rectangle(frame, (cx-20,cy-20),(cx+20,cy+20),(0,255,255),2)
                cv.putText(frame, "OBST", (cx-15,cy-25), font, 0.5, (0,255,255), 2)

    # ===== Once stable detection, plan route =====
    if mode == "IDLE" and "reference" in detections and len(detections) > 1:
        ref = detections["reference"]

        # Collect all cube coordinates
        cubes = [pos for cname, pos in detections.items() if cname != "reference"]
        if cubes:
            tsp_order, _ = tsp_bruteforce(ref, cubes)
            print(f"📍 Visiting order: {tsp_order}")

            # Build full A* route joining each consecutive point
            full_path = []
            for i in range(len(tsp_order) - 1):
                leg_path, _ = astar(tsp_order[i], tsp_order[i + 1],
                                    grid_size=(NUM_CELLS, NUM_CELLS),
                                    obstacles=obstacles)
                if leg_path:
                    if full_path:
                        full_path.extend(leg_path[1:])
                    else:
                        full_path.extend(leg_path)

            robot_path = full_path
            robot_index = 0
            robot_pos = np.array(grid_to_pixel(ref[0], ref[1], frame.shape), dtype=float)
            mode = "MOVING"
            print(f"🚗 Total path: {robot_path}")

    # ===== Robot motion =====
    if mode == "MOVING" and robot_path:
        next_cell = robot_path[robot_index]
        next_px = np.array(grid_to_pixel(next_cell[0], next_cell[1], frame.shape), dtype=float)
        robot_pos += (next_px - robot_pos) * move_speed

        if np.linalg.norm(robot_pos - next_px) < 5:
            robot_index += 1
            if robot_index >= len(robot_path):
                mode = "IDLE"
                print("✅ Robot completed full pickup-return cycle.")

    # ===== Draw robot =====
    if robot_pos is not None:
        cv.circle(frame, tuple(robot_pos.astype(int)), 10, (0,255,255), -1)
        cv.putText(frame, "BOT", (int(robot_pos[0])-15, int(robot_pos[1])-15), font, 0.5, (0,255,255), 2)

    cv.imshow("Robot Path Simulation - Multi Cube", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
