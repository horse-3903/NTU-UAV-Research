# Technical Report: Real-Time Drone Obstacle Avoidance via Monocular Depth Estimation and UWB Localisation

**Project:** NTU UAV Research  
**Platform:** DJI Tello + Nooploop LinkTrack UWB + Workstation GPU  
**Stack:** Python 3.8 · ROS Noetic · ZoeDepth · PyTorch · tellopy · PyBullet

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work and Design Rationale](#2-related-work-and-design-rationale)
3. [System Architecture](#3-system-architecture)
4. [Hardware Setup](#4-hardware-setup)
5. [Sensor Preprocessing](#5-sensor-preprocessing)
6. [Monocular Depth Estimation](#6-monocular-depth-estimation)
7. [Obstacle Segmentation and 3D Reconstruction](#7-obstacle-segmentation-and-3d-reconstruction)
8. [UWB Localisation](#8-uwb-localisation)
9. [Artificial Potential Field Controller](#9-artificial-potential-field-controller)
10. [Software Architecture](#10-software-architecture)
11. [Simulation Environment](#11-simulation-environment)
12. [Coordinate Frame Conventions](#12-coordinate-frame-conventions)
13. [Safety Architecture](#13-safety-architecture)
14. [Known Limitations and Failure Modes](#14-known-limitations-and-failure-modes)
15. [Discussion](#15-discussion)
16. [References](#16-references)

---

## 1. Introduction

Autonomous indoor navigation for small unmanned aerial vehicles (UAVs) remains a challenging open problem. Unlike outdoor flight where GPS provides reliable absolute localisation and obstacle density is low, indoor environments impose a distinct set of constraints: reflective and textureless surfaces degrade vision-based depth estimation, GPS is unavailable, ceiling and wall clutter confounds obstacle detectors, and limited flight volume demands precise localisation and tight control margins.

Commercial micro-UAVs such as the DJI Tello compound these difficulties further. The Tello carries a single forward-facing monocular camera, no rangefinders, and no onboard compute beyond basic flight stabilisation. All perception and control must run on a connected laptop over a Wi-Fi link that is shared with the video stream, introducing non-trivial latency and the possibility of frame loss.

This project develops and evaluates a lightweight perception-control pipeline that addresses these constraints directly. The approach combines:

- **Monocular depth estimation** (ZoeDepth) to produce per-frame depth maps from the Tello's camera without stereo hardware,
- **UWB absolute localisation** (Nooploop LinkTrack) to provide a stable, GPS-independent position estimate,
- **Artificial Potential Fields** (APF) as the control policy, chosen for its simplicity, real-time computational cost, and natural handling of dynamic obstacle avoidance in continuous space.

The result is an end-to-end system that takes live sensor inputs and produces drone velocity commands at approximately 5 Hz, capable of navigating from a start position to a goal while avoiding detected obstacles within a bounded indoor flight volume.

---

## 2. Related Work and Design Rationale

### 2.1 Monocular Depth for UAV Obstacle Avoidance

Monocular depth estimation from deep neural networks has matured significantly since the seminal work of Eigen et al. (2014). Recent models such as MiDaS (Ranftl et al., 2020), DPT (Ranftl et al., 2021), and ZoeDepth (Bhat et al., 2023) achieve competitive metric depth accuracy on indoor benchmarks without stereo calibration.

For UAV applications, monocular depth has been explored primarily in simulation or with post-processed video. Deploying it in a real-time control loop introduces a fundamental tension: inference latency versus control frequency. On a mid-range GPU, ZoeDepth processes a 960×720 frame in approximately 0.5–2 seconds, making per-frame inference impractical for 30 Hz control. This project resolves the tension by decoupling the depth update rate from the control loop rate: depth estimation runs every 250 frames (~10 s at 24 fps) while the APF controller runs at the UWB callback rate (~5–10 Hz).

### 2.2 UWB Localisation

Ultra-Wideband (UWB) time-difference-of-arrival (TDoA) and two-way ranging (TWR) systems have emerged as a practical alternative to motion capture for indoor localisation. The Nooploop LinkTrack system achieves sub-decimetre position accuracy at ~100 Hz update rates, well within the requirements of a 5 Hz control loop. Unlike visual-inertial odometry, UWB is robust to lighting changes and motion blur — precisely the conditions that degrade monocular depth estimation.

The combination of UWB (for position) and monocular depth (for obstacle detection) is therefore complementary: UWB provides a stable world-frame anchor while depth provides local obstacle geometry.

### 2.3 Artificial Potential Fields

APF, introduced by Khatib (1986), represents the environment as a scalar potential function whose negative gradient gives the robot's desired velocity. The potential has an attractive component toward the goal and repulsive components around obstacles. APF has well-known limitations — most notably local minima and oscillation in narrow corridors — but it is computationally trivial (linear in the number of obstacles), inherently continuous, and easy to extend with boundary walls. For a prototype system with a small, bounded flight space and a single primary goal, these properties are well matched to the task.

---

## 3. System Architecture

The system is structured as a reactive control loop driven by two asynchronous event streams:

```
UWB callback (ROS, ~100 Hz)
    └─ update cur_pos
    └─ check bounds (emergency land if violated)
    └─ run active_task()
           └─ follow_path()
                  └─ [if depth_model_run] compute APF forces
                  └─ send velocity commands to drone

Video thread (PyAV, ~24 fps)
    └─ decode H.264 frame
    └─ write to raw AVI file
    └─ [every 250 frames] spawn depth thread:
           └─ ZoeDepth inference
           └─ K-means obstacle segmentation
           └─ 3D reconstruction
           └─ update obstacle list
           └─ save annotated images
```

The UWB callback is the primary control tick. The video thread runs independently and updates the obstacle map at a lower rate. This separation ensures that video decode errors or slow depth inference do not stall the control loop.

All state — current position, obstacle list, target position, model handles — is held on the `TelloDrone` object. Thread safety is achieved implicitly: the obstacle list is only written by the depth thread and only read by the control thread, with no locking. This is acceptable given the low write frequency (once per 250 frames) and the tolerance of the APF controller to stale obstacle data.

---

## 4. Hardware Setup

### 4.1 DJI Tello

| Parameter | Value |
|---|---|
| Camera | 5 MP, fixed focus, 82.6° FOV |
| Video output | 960 × 720, H.264, ~24 fps over Wi-Fi |
| SDK | tellopy (unofficial, lower-latency than DJI official SDK) |
| Battery life | ~13 min (reduced under load) |
| Max speed | ~8 m/s (constrained to ~0.3 m/s in this system) |
| Communication | 802.11n Wi-Fi, 2.4 GHz |

The Tello is controlled via the `tellopy` library, which provides an event-driven interface for flight data (altitude, speed, battery, Wi-Fi signal) and direct velocity commands. Commands take the form `drone.forward(speed)`, `drone.right(speed)`, etc., where `speed` is an integer in [0, 100] representing a percentage of the maximum speed for that axis.

### 4.2 Nooploop LinkTrack UWB

The Nooploop LinkTrack system uses two-way ranging between anchors and the tag mounted on the drone. Three or more anchors are required for full 3D localisation; the experimental setup used four anchors placed at known positions in the flight space.

Position data is published to the ROS topic `/nlink_linktrack_nodeframe1` as a `LinktrackNodeframe1` message. The relevant field is `nodes[0].pos_3d`, a three-element float array in metres.

| Parameter | Value |
|---|---|
| Ranging method | Two-way ranging (TWR) |
| Reported accuracy | ~10 cm (manufacturer specification) |
| Update rate | ~100 Hz |
| Communication | USB serial to laptop → `nlink_parser` ROS node |

### 4.3 Workstation

All perception and control runs on a connected laptop. A discrete GPU is strongly recommended for ZoeDepth inference speed, though CPU-only operation is functional at reduced inference rate.

---

## 5. Sensor Preprocessing

### 5.1 Camera Calibration

The Tello camera is calibrated using a 9×7 chessboard pattern (square size 20 mm). Calibration images are collected from the live drone feed via `task/test_video.py` and stored in `calibrate/`. The calibration routine in `task/camera_calibrate.py` uses OpenCV's `calibrateCamera`, producing the camera matrix $\mathbf{K}$ and distortion coefficients $\mathbf{d}$.

$$
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

where $f_x, f_y$ are the focal lengths in pixels and $(c_x, c_y)$ is the principal point. These are saved to `calibration_data.npz` and loaded at runtime by `map_obstacle.py`.

### 5.2 Image Undistortion

Point-level undistortion is applied in `undistort_point()` using `cv2.initUndistortRectifyMap` before projecting obstacle centroids to 3D. This corrects for the Tello's notable barrel distortion at the edges of the 82.6° FOV.

### 5.3 Video Decoding

The Tello's H.264 stream is accessed via tellopy's `get_video_stream()` method and decoded with PyAV (`av.open`). Frames are decoded in a dedicated video thread to avoid blocking the control loop. Each decoded frame is written to a raw AVI file (XVID codec, 24 fps) for post-flight analysis.

---

## 6. Monocular Depth Estimation

### 6.1 Model: ZoeDepth

ZoeDepth (Zero-shot transfer of metric depth) is a transformer-based monocular depth estimator that extends MiDaS-style relative depth to produce metric (absolute) depth estimates. The `ZoeDepthNK` variant is trained jointly on NYU Depth v2 (indoor) and KITTI (outdoor), making it suitable for indoor drone footage without domain-specific fine-tuning.

The model is loaded from a local directory via the HuggingFace Transformers API:

```python
image_processor = ZoeDepthImageProcessor.from_pretrained("model/zoedepth-nyu-kitti")
depth_model = ZoeDepthForDepthEstimation.from_pretrained("model/zoedepth-nyu-kitti")
```

Model weights are approximately 1.5 GB and are excluded from the repository.

### 6.2 Inference Pipeline

For each depth cycle:

1. The current frame (`ndarray`, BGR) is converted to a PIL RGB image.
2. `ZoeDepthImageProcessor.preprocess` resizes and normalises the image to the model's expected input format.
3. `ZoeDepthForDepthEstimation.forward` is called under `torch.no_grad()`.
4. `post_process_depth_estimation` upsamples the output depth map to the original frame resolution.

The result is a float32 tensor of shape $(H, W)$ representing metric depth in metres per pixel. A normalised uint8 version is also computed for visualisation:

$$
D_{\text{rel}}[i,j] = \frac{D[i,j] - \min(D)}{\max(D) - \min(D)} \times 255
$$

### 6.3 Depth Update Rate

Depth inference runs every 250 video frames. At 24 fps, this is approximately one depth update every 10.4 seconds. The design trade-off is:

- **Too frequent:** ZoeDepth inference on CPU can take 2–10 s per frame, blocking the depth thread and starving the video writer.
- **Too infrequent:** The obstacle map becomes stale as the drone moves.

At the commanded speeds (maximum 30/100 scale ≈ ~1–2 m/s), the drone travels at most 20 m between depth updates, which would be catastrophic in a 7.5 × 5 m space. In practice, the approach speed toward obstacles is slow enough (and APF forces are reactive) that the 10 s update rate is tolerable for the prototype, but is a clear area for improvement (see §15.1).

### 6.4 Position Anchoring

Each depth snapshot is anchored to the drone's current world-frame position. Rather than using the instantaneous UWB reading (which may be noisy), a running average over the last 100 position readings is computed:

$$
\bar{\mathbf{p}} = \frac{1}{\min(N, 100)} \sum_{k=\max(0,N-100)}^{N} \mathbf{p}_k
$$

where $\mathbf{p}_k$ is the $k$-th UWB position reading and $N$ is the current reading count. This averaged position is used to offset obstacle coordinates from the camera frame to the world frame (§7.4).

---

## 7. Obstacle Segmentation and 3D Reconstruction

### 7.1 Overview

The obstacle detection pipeline in `task/tellodrone/map_obstacle.py` converts a ZoeDepth absolute depth map into a list of 3D obstacles, each represented as a sphere $({\bf c}, r)$ in world coordinates. The pipeline proceeds in five stages: depth clustering, segment extraction, morphological cleaning, contour detection, and 3D projection.

### 7.2 Depth Clustering via K-Means

The depth map $D \in \mathbb{R}^{H \times W}$ is flattened to a vector of $H \cdot W$ scalar depth values. $k$-means clustering (OpenCV `cv2.kmeans`) partitions these values into $k = 5$ clusters with centres $\{c_1, c_2, c_3, c_4, c_5\}$, sorted by depth value. The output is a label map $L \in \{0, 1, 2, 3, 4\}^{H \times W}$ assigning each pixel to its nearest cluster.

The rationale for depth-space clustering (rather than spatial clustering) is that nearby objects form coherent regions in depth — a wall at 2 m occupies a different cluster from a chair at 0.8 m — making the clusters a natural proxy for distinct surfaces in the scene.

### 7.3 Near-Cluster Extraction and Filtering

The $d = 3$ clusters with the smallest centre values (i.e., the nearest surfaces) are extracted as binary maps:

$$
S_j[i,k] = \begin{cases} 255 & \text{if } L[i,k] \in \text{top-}d\text{ nearest clusters} \\ 0 & \text{otherwise} \end{cases}
$$

Two filtering steps then remove scene structure that is not an obstacle:

**Row density filter.** Rows in which more than 85% of pixels are marked as foreground are zeroed. This removes the floor, ceiling, and back walls, which appear as near-continuous horizontal bands when the drone is flying parallel to a surface.

Formally, row $i$ is zeroed if:

$$
\frac{\sum_{k} \mathbf{1}[S_j[i,k] = 255]}{W} \geq \theta_{\text{row}} = 0.85
$$

**Morphological opening and component filtering.** An 11×11 rectangular structuring element is used for morphological opening to remove thin noise filaments. Connected components with area below $A_{\min} = 20{,}000$ px² are then discarded:

$$
\text{keep component } c \iff \text{area}(c) > 20{,}000 \text{ px}^2
$$

At 960×720 resolution, this minimum area corresponds to roughly a 141×141 px region — approximately an object subtending 15% of the frame width, which at 1 m depth corresponds to a real-world object of ~0.3 m width.

### 7.4 Contour Detection and Minimum Enclosing Circle

Obstacle regions are detected using `cv2.findContours` on each cleaned binary segment. Each contour $\mathcal{C}$ yields:

- **Centroid:** $(\hat{x}, \hat{y}) = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right)$ from image moments $m_{pq} = \sum_{x,y} x^p y^q \, I(x,y)$
- **Pixel radius:** $r_{\text{px}}$ from `cv2.minEnclosingCircle`

### 7.5 3D Reconstruction via Camera Intrinsics

Each obstacle centroid is first undistorted using the calibration map:

$$
(\tilde{x}, \tilde{y}) = \text{undistort}(\hat{x}, \hat{y}; \mathbf{K}, \mathbf{d})
$$

The depth at the centroid is estimated as the mean over a 5×5 pixel neighbourhood to reduce noise:

$$
d_{\text{obs}} = \frac{1}{25} \sum_{u=\hat{y}-2}^{\hat{y}+2} \sum_{v=\hat{x}-2}^{\hat{x}+2} D[u, v]
$$

The 3D position in the camera frame is then:

$$
x_{\text{cam}} = -d_{\text{obs}} - 0.4
$$
$$
y_{\text{cam}} = \frac{(\tilde{x} - c_x) \cdot d_{\text{obs}}}{f_x} - 0.4
$$
$$
z_{\text{cam}} = \frac{(\tilde{y} - c_y) \cdot d_{\text{obs}}}{f_y} - 0.4
$$

The $-0.4$ offset on each axis accounts for the approximate displacement between the camera optical centre and the drone's UWB tag mounting point. This is a fixed calibration offset and not dynamically estimated.

The physical obstacle radius is computed from the pixel radius via the pinhole model:

$$
r_{\text{m}} = \frac{r_{\text{px}} \cdot D[\hat{y}, \hat{x}]}{f_x}
$$

Finally, the camera-frame position is translated to world frame by adding the drone's averaged UWB position $\bar{\mathbf{p}}$:

$$
\mathbf{p}_{\text{world}} = \mathbf{p}_{\text{cam}} + \bar{\mathbf{p}}
$$

### 7.6 Obstacle List Maintenance

The obstacle list is maintained as a set of spheres $\{(\mathbf{c}_i, r_i)\}$. When new obstacles are detected, they are merged with the existing list using a distance threshold $\tau = 0.5$ m: if a new obstacle's centre lies within $\tau$ of an existing obstacle's centre, the existing obstacle is replaced. This prevents unbounded list growth during repeated depth cycles.

Two validity filters are applied before adding new obstacles:

1. **Bounds check:** $\mathbf{p}_{\text{world}}$ must lie within the configured flight envelope.
2. **Radius sanity:** $r_{\text{m}} \leq 1.0$ m. Larger radii typically correspond to walls or floors that escaped the row filter.

---

## 8. UWB Localisation

### 8.1 Nooploop LinkTrack Integration

Position data flows from the LinkTrack hardware via USB serial to the `nlink_parser` ROS node, which publishes `LinktrackNodeframe1` messages on `/nlink_linktrack_nodeframe1` at approximately 100 Hz. The `main.py` ROS subscriber extracts `nodes[0].pos_3d` and passes it directly to `task_handler()` on the `TelloDrone` object.

The position is stored as a `Vector3D` and appended to a line-delimited log file (`logs/log-*/log-pos.log`) on every callback:

```
<timestamp> <elapsed_seconds> <x> <y> <z>
```

### 8.2 Coordinate Frame

The UWB coordinate frame is determined by anchor placement. In the experimental setup:

| Axis | Direction | Range |
|---|---|---|
| $x$ | Forward (into room) | $[-0.50, 7.00]$ m |
| $y$ | Lateral (rightward) | $[-0.50, 4.50]$ m |
| $z$ | Vertical (negative = up) | $[-3.75, -0.50]$ m |

The drone faces in the $-x$ direction (toward the room entrance), which is why the `follow_path` controller maps negative $x$-force to `drone.forward()`.

### 8.3 Position Noise and Averaging

UWB systems exhibit multipath noise, particularly near metal surfaces and in corners. No explicit filtering is applied to the raw position readings; the 100-sample running average used in depth anchoring (§6.4) provides partial noise reduction for that specific use case. The control loop uses the raw, unfiltered current position.

---

## 9. Artificial Potential Field Controller

### 9.1 Potential Function

The total scalar potential field is:

$$
U(\mathbf{q}) = U_{\text{att}}(\mathbf{q}) + U_{\text{rep}}(\mathbf{q}) + U_{\text{bound}}(\mathbf{q})
$$

**Attractive potential:**

$$
U_{\text{att}}(\mathbf{q}) = \frac{1}{2} k_a \| \mathbf{q} - \mathbf{q}_{\text{goal}} \|^2
$$

**Repulsive potential (per obstacle $i$):**

$$
U_{\text{rep},i}(\mathbf{q}) = \begin{cases}
\frac{1}{2} k_r \left( \frac{1}{\rho_i} - \frac{1}{\rho_0} \right)^2 & \text{if } \rho_i \leq \rho_0 \\
0 & \text{otherwise}
\end{cases}
$$

where $\rho_i = \| \mathbf{q} - \mathbf{c}_i \| - r_i$ is the surface-to-surface distance from the drone to obstacle $i$, and $\rho_0 = 0.5$ m is the influence radius.

**Boundary repulsive potential (per axis, per wall):**

The same formula is applied to the signed distance from each of the six boundary planes, yielding six additional repulsive components.

### 9.2 Force Computation

The control force is the negative gradient of the potential:

$$
\mathbf{F}(\mathbf{q}) = -\nabla U(\mathbf{q}) = \mathbf{F}_{\text{att}} + \mathbf{F}_{\text{rep}} + \mathbf{F}_{\text{bound}}
$$

**Attractive force:**

$$
\mathbf{F}_{\text{att}} = k_a \cdot \| \mathbf{q}_{\text{goal}} - \mathbf{q} \| \cdot \frac{\mathbf{q}_{\text{goal}} - \mathbf{q}}{\| \mathbf{q}_{\text{goal}} - \mathbf{q} \|}
= k_a \cdot (\mathbf{q}_{\text{goal}} - \mathbf{q})
$$

This is linear distance attraction (gradient of a quadratic potential).

**Repulsive force (per obstacle $i$):**

$$
\mathbf{F}_{\text{rep},i} = k_r \left( \frac{1}{\rho_i} - \frac{1}{\rho_0} \right) \frac{1}{\rho_i^2} \cdot \frac{\mathbf{q} - \mathbf{c}_i}{\| \mathbf{q} - \mathbf{c}_i \|}, \quad \rho_i \leq \rho_0
$$

**Boundary repulsive force (min $x$-wall example):**

$$
F_{\text{bound},x}^- = k_r \left( \frac{1}{x_{\min} + \rho_b - q_x} - \frac{1}{\rho_b} \right) \frac{1}{(x_{\min} + \rho_b - q_x)^2} \cdot \hat{e}_x
$$

where $\rho_b = 0.5$ m is the boundary influence distance and $\hat{e}_x$ is the unit vector in the $+x$ direction. Analogous formulas apply to the other five walls.

### 9.3 Velocity Mapping

The computed force vector $\mathbf{F} = (F_x, F_y, F_z)$ is converted to drone velocity commands:

$$
v_i = \text{round}\!\left( \frac{F_i}{d_{\text{goal}}} \right), \quad i \in \{x, y, z\}
$$

where $d_{\text{goal}} = \| \mathbf{q} - \mathbf{q}_{\text{goal}} \|$ is the current distance to the goal. Dividing by $d_{\text{goal}}$ normalises the command magnitude relative to distance, preventing very large forces at long range from sending maximum-speed commands. Each component is then clamped:

$$
v_i^{\text{cmd}} = \text{clamp}(|v_i|,\ 0,\ 30)
$$

The sign of $v_i$ determines the axis direction (forward/backward, left/right, up/down).

### 9.4 Close-Range Gain Reduction

When $d_{\text{goal}} < 1.0$ m, the attractive gain is reduced to prevent overshoot:

$$
k_a' = \max\left(10,\ k_a \cdot d_{\text{goal}}\right)
$$

This linearly scales the attractive coefficient from its nominal value $k_a = 30$ down to 10 as the drone approaches within 1 m of the goal.

### 9.5 Parameter Summary

| Parameter | Symbol | Value |
|---|---|---|
| Attractive gain | $k_a$ | 30 (10 near goal) |
| Repulsive gain | $k_r$ | 10 |
| Obstacle influence radius | $\rho_0$ | 0.50 m |
| Boundary influence distance | $\rho_b$ | 0.50 m |
| Max velocity command | $v_{\max}$ | 30 (of 100 scale) |
| Arrival threshold | $\epsilon$ | 0.30 m |
| Control loop sleep | $\Delta t$ | 0.20 s |

### 9.6 Local Minima

APF controllers are susceptible to local minima where $\mathbf{F}_{\text{att}} + \mathbf{F}_{\text{rep}} = \mathbf{0}$ but $\mathbf{q} \neq \mathbf{q}_{\text{goal}}$. This occurs most commonly when the goal lies directly behind a large obstacle. The current implementation does not include any escape mechanism; the drone will stall at such a point. Future work should add a stall detector and perturbation strategy (§15).

---

## 10. Software Architecture

### 10.1 TelloDrone Class Design

The `TelloDrone` class in `task/tellodrone/core.py` aggregates all drone state and imports methods from separate module files via class-body `from ... import ...` statements:

```python
class TelloDrone:
    from tellodrone.log import setup_logging, save_log_config
    from tellodrone.flight_control import flight_data_callback, check_bounds
    from tellodrone.video import setup_display, process_image, process_frame, ...
    from tellodrone.task import task_handler
    from tellodrone.follow_path import set_target_pos, add_obstacle, follow_path
    from tellodrone.depth_model import load_depth_model, run_depth_model, estimate_depth
```

This pattern allows the class to be split across multiple files without inheritance, keeping each file focused on a single concern while sharing `self` state naturally.

### 10.2 Threading Model

| Thread | Purpose | Lifecycle |
|---|---|---|
| Main thread | ROS spin (UWB callbacks → control) | Runs until shutdown |
| Video thread | PyAV frame decode + write | Started at `startup()` |
| Depth thread | ZoeDepth inference (spawned per-cycle) | Short-lived, per inference |
| Image task thread | Manual depth trigger from UI | Short-lived, on button press |
| Display thread | Pygame event loop + video display | Started at `setup_display()` |

A `threading.Event` (`stop_video_thread_event`) coordinates clean shutdown of the video thread. The display thread terminates the process via `shutdown()` when the Pygame window is closed.

### 10.3 Logging

Three log artefacts are written per flight:

- `logs/log-<run>/log-info.log` — human-readable INFO/DEBUG/ERROR log with timestamps
- `logs/log-<run>/log-pos.log` — line-delimited UWB position readings (timestamp, elapsed seconds, x, y, z)
- `logs/log-<run>/log-config.json` — JSON snapshot of takeoff position, start position, end position, target, and full obstacle list at shutdown

Image artefacts:

- `img/original/<run>/frame-<N>.png` — raw camera frame at depth cycle $N$
- `img/depth/<run>/frame-<N>.png` — normalised depth map
- `img/annotated/<run>/frame-<N>.png` — frame with obstacle bounding circles and 3D coordinates overlaid

---

## 11. Simulation Environment

### 11.1 PyBullet Simulator

`task/sim.py` and `task/tellodrone_sim/core.py` implement a physics simulation using PyBullet. The simulation uses the same `apf_with_bounds` function as the live drone, so controller parameters can be tuned offline before flight.

The environment is populated with:

- **Drone model:** `duck_vhacd.urdf` (PyBullet built-in), scaled ×5
- **Target:** `soccerball.urdf`, fixed, scaled ×0.5
- **Obstacles:** `sphere2.urdf` instances at randomised positions
- **Boundary walls:** Translucent box visual shapes (no collision) marking the flight envelope

Obstacle positions are generated by `generate_obstacles(num, radius)` using uniform random sampling within the flight bounds.

### 11.2 A* Path Planner

The simulation additionally implements a 3D grid-based A* planner (`a_star_waypoints`) for comparison against APF. The planner discretises the flight volume at a configurable resolution (default 0.5 m) and finds a collision-free waypoint sequence, which the APF controller then follows as local goals.

### 11.3 Path Planning Modes

The simulator supports three navigation modes (configured by commenting/uncommenting in `sim.py`):

1. **Direct APF:** Single waypoint = target; APF navigates directly.
2. **A\* + APF:** A* generates waypoints; APF follows each in sequence.
3. **Sine path + APF:** Sinusoidal waypoints for testing oscillation response.

---

## 12. Coordinate Frame Conventions

Three coordinate frames are relevant:

### 12.1 Camera Frame

Standard pinhole camera convention: $+x$ rightward, $+y$ downward, $+z$ forward (into scene). Depth $D[u,v]$ is the $z$-component of the 3D point at pixel $(u,v)$.

### 12.2 World Frame (UWB)

Determined by anchor placement. In the experimental setup:

$$
\hat{x}_W = \text{forward (into room)}, \quad \hat{y}_W = \text{right}, \quad \hat{z}_W = \text{down (negative = up)}
$$

The drone's hover altitude of approximately $-2.0$ to $-2.5$ m in $z_W$ reflects the negative-up convention.

### 12.3 Camera-to-World Transform

The camera-to-world transform is implemented as a fixed offset rather than a full rigid-body transform:

$$
\mathbf{p}_W = \mathbf{p}_C + \bar{\mathbf{p}}_{\text{UWB}} + \mathbf{t}_{\text{mount}}
$$

where $\mathbf{t}_{\text{mount}} = (-0.4, -0.4, -0.4)$ m is the hardcoded camera-to-tag offset. A full extrinsic calibration (rotation + translation between camera and UWB tag) would improve obstacle localisation accuracy.

### 12.4 Drone Command Frame

The Tello faces in the $-x_W$ direction. The command mapping is:

| Force direction | tellopy command |
|---|---|
| $F_x < 0$ | `drone.forward()` |
| $F_x > 0$ | `drone.backward()` |
| $F_y > 0$ | `drone.right()` |
| $F_y < 0$ | `drone.left()` |
| $F_z < 0$ | `drone.down()` |
| $F_z > 0$ | `drone.up()` |

---

## 13. Safety Architecture

Safety is implemented as a layered set of shutdown triggers, all of which call `TelloDrone.shutdown(error=True)`, which in turn calls `drone.land()` before exiting:

| Trigger | Threshold | Source |
|---|---|---|
| Battery critical | $\leq 5\%$ | Flight data callback |
| Battery warning | $\leq 10\%$ | Flight data callback (logs only) |
| Out of bounds | Outside configured flight envelope | UWB callback, every step |
| Keyboard interrupt | Ctrl+C | `signal.SIGINT` handler |
| Drone not connected | Connection timeout | Startup check |
| Pygame window closed | — | Display thread exit |

The `shutdown()` method attempts a clean land regardless of error state:

```python
def shutdown(self, error=False, reason=None):
    self.drone.backward(0)   # cancel any active velocity command
    self.drone.land()
    self.stop_video_thread()
    pygame.quit()
    self.drone.quit()
    rospy.signal_shutdown(...)
    sys.exit(0)
```

---

## 14. Known Limitations and Failure Modes

### 14.1 Depth Estimation Latency

A ZoeDepth inference pass on CPU takes approximately 2–10 seconds per frame depending on hardware. During this time, the depth thread blocks the image write pipeline. The obstacle map is not updated, meaning the drone may move significantly between depth cycles. On GPU (e.g. NVIDIA RTX 3060), inference takes approximately 0.3–0.8 s, reducing this window substantially.

**Consequence:** The drone may navigate through a region before its obstacles are detected.

**Mitigation (partial):** The 250-frame interval ensures the depth thread finishes before the next cycle begins. The APF controller is conservative (max speed 30/100) to limit exposure during stale periods.

### 14.2 K-Means Non-Determinism

`cv2.kmeans` with `KMEANS_RANDOM_CENTERS` is non-deterministic. On frames with similar depth distributions (e.g. nearly empty corridor), cluster boundaries may shift significantly between runs, causing obstacle detections to appear and disappear erratically.

**Consequence:** Obstacle list instability between depth cycles.

**Mitigation (partial):** The merge threshold of 0.5 m and the minimum area filter of 20,000 px² reduce spurious detections, but do not eliminate them.

### 14.3 UWB Multipath and Dropout

In environments with metal surfaces or dynamic occlusion of UWB anchors, position jumps of 0.5–1 m can occur. The current system applies no filtering to the raw UWB stream for control purposes.

**Consequence:** A position jump can trigger a false out-of-bounds detection and an immediate emergency landing.

**Mitigation:** The flight bounds are set conservatively (0.5 m inset from physical walls), but this does not prevent brief out-of-bounds readings.

### 14.4 APF Local Minima

As noted in §9.6, the APF controller has no escape mechanism. In environments with obstacles between the start and goal positions, the drone may become trapped at a saddle point in the potential field.

**Consequence:** The drone hovers motionless until the battery depletes, or a user intervention is needed.

### 14.5 Fixed Camera-to-Tag Offset

The $(-0.4, -0.4, -0.4)$ m camera-to-tag offset in `compute_3d()` is hardcoded and does not account for drone attitude (pitch/roll), which changes during flight. Obstacle positions will be incorrectly anchored when the drone is not level.

### 14.6 No Temporal Fusion of Depth

Each ZoeDepth output is processed independently. There is no tracking of obstacles across frames — an obstacle detected in frame 250 is simply merged (by proximity) with the list from the prior cycle. This means:

- Moving obstacles are not tracked (only their last detected position is stored).
- A stationary obstacle that the drone has passed will remain in the list until its position falls outside the flight bounds.

---

## 15. Discussion

### 15.1 Inference Speed vs. Update Rate

The central design tension in this system is that ZoeDepth inference is slow relative to the UAV's control bandwidth. The gap between obstacle map update rate (~0.1 Hz) and control rate (~5 Hz) means the drone is operating primarily on a frozen obstacle map. For low-speed navigation in structured environments (slow-moving drone, static obstacles), this is acceptable. For higher-speed or dynamic scenarios, it is not.

Lighter models (MiDaS small, Depth Anything small) achieve inference rates of 10–30 Hz on a mid-range GPU, which would enable near-per-frame depth updates. The trade-off is reduced metric accuracy on indoor scenes.

### 15.2 UWB vs. Visual Odometry

UWB provides accurate absolute position but requires physical infrastructure (anchor installation and surveying). Visual-inertial odometry (VIO) can achieve similar accuracy in textured environments without fixed infrastructure, at the cost of drift over time. A hybrid approach — VIO for relative motion, UWB for periodic drift correction — would be more deployable across environments.

### 15.3 APF vs. Sampling-Based Planning

APF is computationally trivial and well-suited to real-time reactive control. Its failure cases (local minima, oscillation in narrow passages) are well-understood. Sampling-based planners (RRT*, informed RRT*) are more complete but require a more accurate environment model and are computationally heavier. The A* planner in the simulation is an intermediate option when a grid-resolution environment map is available.

### 15.4 Prototype Status

This system is explicitly a research prototype. It demonstrates the viability of the perception-control architecture and provides a foundation for more capable systems, but several components would need to be hardened before deployment outside a controlled lab:

- Extrinsic calibration between camera and UWB tag
- Position filtering (Kalman filter or moving average on raw UWB)
- Temporal depth fusion (EKF on obstacle positions)
- APF local minima escape
- Real-time depth model (GPU or lighter model)
- Formal safety certification for any human-occupied space

---

## 16. References

- Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Müller, M. (2023). *ZoeDepth: Zero-shot transfer by combining relative and metric depth.* arXiv:2302.12288.
- Eigen, D., Puhrsch, C., & Fergus, R. (2014). *Depth map prediction from a single image using a multi-scale deep network.* NeurIPS 2014.
- Khatib, O. (1986). *Real-time obstacle avoidance for manipulators and mobile robots.* The International Journal of Robotics Research, 5(1), 90–98.
- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). *Vision Transformers for Dense Prediction.* ICCV 2021. (DPT/MiDaS v3)
- Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer.* TPAMI.
- Nooploop. (2024). *LinkTrack User Manual.* https://www.nooploop.com/en/linktrack/
- DJI. (2020). *Tello SDK 2.0 User Guide.*
- Quigley, M., et al. (2009). *ROS: an open-source Robot Operating System.* ICRA Workshop on Open Source Software.
