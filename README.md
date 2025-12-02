# ArUco Multi-Goal Navigation
![Project Banner](assets/banner.png)

**Vision-guided multi-goal navigation on a 10×10 grid using ArUco markers — detection → mapping → TSP/Greedy ordering → A* path planning → simulation.**

---

## Table of contents
- [Demo](#demo)
  - [Images](#images)
  - [Video demos](#video-demos)
- [Features](#features)
- [Quick start](#quick-start)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Run (simulation)](#run-simulation)
- [Repository structure](#repository-structure)
- [How it works (high level)](#how-it-works-high-level)
- [Results & Logs](#results--logs)
- [Adding your own videos / images](#adding-your-own-videos--images)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Demo

### Images
> Click any image to view full size.

| Grid detection | Obstacle-aware planning | Dynamic replanning |
|---:|:---:|---:|
| ![Grid detection](assets/grid_detection.png) | ![Obstacles](assets/obstacles_path.png) | ![Replan GIF](assets/replan_preview.gif) |

**Figure 1.** Detection, grid overlay and example paths.  
**Figure 2.** Obstacles (yellow boxes) and A* paths (polylines).  
**Figure 3.** Animated GIF showing live replanning when an obstacle is added.

---

### Video demos
> For reproducible demonstration, keep videos in `/videos` or host them (YouTube unlisted / Google Drive) and link here.

**Simulation demo (full run)**  
[Watch on YouTube (unlisted)](https://youtu.be/YOUR_VIDEO_LINK) • or use local file `videos/sim_demo.mp4`

**Dynamic replanning (short clip)**  
[Watch dynamic replanning](https://drive.google.com/your-demo-link) • or `videos/replan_short.mp4`

> Embedding in GitHub README is limited — link to hosted video or show a GIF preview (recommended).

---

## Features
- Real-time ArUco detection (OpenCV) with robust centroid extraction.  
- Pixel-to-10×10 grid mapping for interpretable planning.  
- Multi-goal ordering: brute-force **TSP** for small N, **Greedy** fallback for speed.  
- Obstacle-aware local planning using **A\*** on a 4-connected grid.  
- Virtual robot simulation with smooth waypoint interpolation and visual overlays.  
- Lightweight, modular code meant for quick porting to physical robots.

---

## Quick start

### Requirements
- Python 3.8+  
- OpenCV (`opencv-python`)  
- NumPy  
- Optional (for video playback / recording): `ffmpeg` or `opencv-python`'s video writer

```bash
# example
python --version
pip install -r requirements.txt
