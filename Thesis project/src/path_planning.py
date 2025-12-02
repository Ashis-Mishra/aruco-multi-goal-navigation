#!/usr/bin/env python3
"""
path_planning.py
Implements A*, Greedy, and TSP algorithms for path optimization
on a 10x10 grid environment with obstacle support.
"""

import heapq
import itertools
import math

# ============================================================
# 🔹 Utility Functions
# ============================================================

def heuristic(a, b):
    """Manhattan distance heuristic (used by A*)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_distance(path):
    """Compute total Manhattan distance for a given path."""
    if not path or len(path) < 2:
        return 0
    return sum(heuristic(path[i], path[i + 1]) for i in range(len(path) - 1))


# ============================================================
# 🔹 A* PATH PLANNING
# ============================================================

def astar(start, goal, grid_size=(10, 10), obstacles=None):
    """
    Compute shortest path using A* algorithm.
    - start: (x, y)
    - goal: (x, y)
    - grid_size: (width, height)
    - obstacles: set of (x, y) tuples to avoid
    Returns: (path, distance)
    """
    if obstacles is None:
        obstacles = set()

    width, height = grid_size
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, cost

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            # Skip invalid or blocked nodes
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if neighbor in obstacles:
                continue

            tentative_g = cost + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], float('inf')  # No valid path found


# ============================================================
# 🔹 GREEDY NEAREST-NEIGHBOR PATH
# ============================================================

def greedy_path(start, targets):
    """
    Compute path visiting all targets by always moving to nearest next cube.
    Returns visiting order and total distance.
    """
    if not targets:
        return [start], 0

    unvisited = set(targets)
    current = start
    path_order = [start]
    total_dist = 0

    while unvisited:
        next_target = min(unvisited, key=lambda t: heuristic(current, t))
        total_dist += heuristic(current, next_target)
        path_order.append(next_target)
        current = next_target
        unvisited.remove(next_target)

    # return to base
    total_dist += heuristic(current, start)
    path_order.append(start)
    return path_order, total_dist


# ============================================================
# 🔹 TRAVELLING SALESMAN PROBLEM (TSP - Brute Force)
# ============================================================

def tsp_bruteforce(start, targets):
    """
    Brute-force TSP solver for small number of cubes.
    Returns optimal visiting order and minimum total distance.
    """
    if not targets:
        return [start], 0

    best_order = None
    min_distance = float('inf')

    for perm in itertools.permutations(targets):
        distance = 0
        current = start
        for p in perm:
            distance += heuristic(current, p)
            current = p
        distance += heuristic(current, start)  # return to start

        if distance < min_distance:
            min_distance = distance
            best_order = [start] + list(perm) + [start]

    return best_order, min_distance


# ============================================================
# 🔹 DEMO / TEST
# ============================================================

if __name__ == "__main__":
    start = (2, 2)
    cubes = [(7, 1), (5, 6), (9, 4)]
    obstacles = {(3, 2), (3, 3), (4, 3), (5, 3)}  # Example walls

    print("🟢 Greedy Path:")
    greedy_order, greedy_dist = greedy_path(start, cubes)
    print(f"Order: {greedy_order}, Distance: {greedy_dist}")

    print("\n🔵 TSP (Brute Force):")
    tsp_order, tsp_dist = tsp_bruteforce(start, cubes)
    print(f"Order: {tsp_order}, Distance: {tsp_dist}")

    print("\n🔴 A* Path (start -> first cube, avoiding obstacles):")
    path, dist = astar(start, cubes[0], grid_size=(10, 10), obstacles=obstacles)
    print(f"Path: {path}, Distance: {dist}")
