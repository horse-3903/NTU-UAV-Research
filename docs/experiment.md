# Experiment Notes: Setup, Calibration, Tuning, and Observations

**Project:** NTU UAV Research  
**Environment:** Indoor lab, Nanyang Technological University  
**Hardware:** DJI Tello · Nooploop LinkTrack · Ubuntu 20.04 workstation

---

## Table of Contents

1. [Physical Environment](#1-physical-environment)
2. [UWB Anchor Placement](#2-uwb-anchor-placement)
3. [Camera Calibration](#3-camera-calibration)
4. [Depth Model Evaluation](#4-depth-model-evaluation)
5. [APF Parameter Tuning](#5-apf-parameter-tuning)
6. [PyBullet Simulation Experiments](#6-pybullet-simulation-experiments)
7. [Live Flight Protocol](#7-live-flight-protocol)
8. [Observed Failure Modes](#8-observed-failure-modes)
9. [Configuration Reference](#9-configuration-reference)
10. [Checklist: Pre-Flight](#10-checklist-pre-flight)

---

## 1. Physical Environment

### 1.1 Flight Space

The system was designed and tested in a single indoor lab space. The configured flight envelope reflects the usable portion of this space:

| Axis | Min | Max | Span |
|---|---|---|---|
| $x$ (forward) | $-0.50$ m | $7.00$ m | $7.50$ m |
| $y$ (lateral) | $-0.50$ m | $4.50$ m | $5.00$ m |
| $z$ (vertical, negative = up) | $-3.75$ m | $-0.50$ m | $3.25$ m |

Total flight volume: approximately $7.50 \times 5.00 \times 3.25 \approx 121$ m³.

The 0.50 m inset from physical walls on each axis provides a safety buffer for UWB measurement noise and control lag. In practice, the drone operates primarily in the $z \in [-2.5, -1.5]$ m band (approximately 1.5–2.5 m altitude).

### 1.2 Reference Positions

These positions appear in `task/main.py` and `task/sim.py` and correspond to specific physical locations in the lab:

| Label | $x$ | $y$ | $z$ | Notes |
|---|---|---|---|---|
| Default start | 5.95 | 2.30 | −0.85 | Commented out in `main.py` |
| Sim start | 6.168 | 2.187 | −2.148 | Loaded from a real flight log |
| Default target | −0.30 | 1.75 | −2.00 | The primary navigation goal |
| Sim target | −0.20 | 2.15 | −4.00 | Alternative target in simulation |

The start-to-target distance at the nominal positions is approximately:

$$
d = \sqrt{(6.168 - (-0.30))^2 + (2.187 - 1.75)^2 + (-2.148 - (-2.00))^2} \approx 6.49 \text{ m}
$$

This is a traverse of approximately 6.5 m across the long axis of the room, passing through whatever obstacles are present in the space.

### 1.3 Environment Characteristics

The lab contains:
- White-painted concrete walls (low texture, challenging for depth estimation)
- Fluorescent ceiling lighting (relatively uniform, acceptable for depth)
- Metal shelving and equipment (multipath source for UWB)
- Controllable clutter (additional obstacles placed for testing)

---

## 2. UWB Anchor Placement

### 2.1 Requirements

Nooploop LinkTrack requires at least three anchors for 3D localisation. Four anchors were used in the experimental setup to improve geometric dilution of precision (GDOP), particularly in the vertical axis.

### 2.2 Recommended Placement

Anchors should be:
- Placed at distinct heights to ensure full 3D coverage
- Positioned to maximise the volume of the convex hull they form
- Mounted with a clear line-of-sight to the tag's expected flight volume
- Secured firmly (any movement requires re-surveying)

Anchor coordinates are surveyed relative to the UWB system origin and entered into the LinkTrack configuration tool. The origin determines the zero-point of the coordinate frame used in `task/main.py`.

### 2.3 Initialisation Procedure

1. Power on all anchors and allow them to stabilise for 60 s.
2. Power on the tag (on the drone).
3. Verify position output with `rostopic echo /nlink_linktrack_nodeframe1`.
4. Confirm that reported $x, y, z$ match the drone's known physical position (within ~0.1 m).
5. If values are incorrect, check anchor configuration in the Nooploop software.

### 2.4 Observed Noise Characteristics

In the experimental setup, raw UWB position noise was approximately:

- Horizontal ($x, y$): ±3–8 cm RMS under good conditions
- Vertical ($z$): ±5–15 cm RMS (vertical is generally noisier due to shallow anchor elevation angles)
- Occasional jump errors of 0.2–0.5 m in the presence of dynamic occlusion

No filtering is applied to the raw position in the current implementation. The 100-sample running average used for depth anchoring (§6.4 of the technical report) reduces effective noise for that specific purpose to approximately ±1–2 cm, at the cost of a positional lag of approximately 1 second at 100 Hz.

---

## 3. Camera Calibration

### 3.1 Setup

- **Pattern:** 9×7 internal corner chessboard (10×8 squares)
- **Square size:** 20 mm
- **Images:** 19 frames in `calibrate/`, captured at varied angles and distances

The calibration was performed using `task/camera_calibrate.py`, which calls `cv2.calibrateCamera` on detected chessboard corners.

### 3.2 Calibration Quality

The `calibrate/` directory contains 19 images. A good calibration for a fisheye-adjacent lens like the Tello's should include:

- Frames at multiple angles (>30° tilt in both axes)
- Frames where the board fills different quadrants of the image
- Frames at different distances (0.5 m to 2 m)

Reprojection error below 1.0 pixel is generally acceptable; the Tello lens typically achieves 0.5–1.5 px with sufficient image diversity.

### 3.3 Calibration Data

The resulting intrinsics are stored in `calibration_data.npz`:

```python
import numpy as np
data = np.load("calibration_data.npz")
K = data["camera_matrix"]    # shape (3, 3)
d = data["dist_coeffs"]      # shape (1, 5) or (5,)
```

This file is committed to the repository and is the sole camera calibration artefact used at runtime.

### 3.4 Recalibration Procedure

If the camera is recalibrated (e.g. after firmware update or lens damage):

1. Fly the drone and capture calibration frames via `task/test_video.py`:
   ```python
   tello.active_img_task = partial(tello.run_depth_model, manual=True)
   tello.run_calibration()
   ```
   Manual captures are saved to `calibrate/frame-<idx>.jpg`.

2. Run calibration:
   ```bash
   cd task && python camera_calibrate.py
   ```

3. Verify the output camera matrix looks reasonable for a ~82° FOV:
   - $f_x, f_y$ should be approximately 900–1100 px (for 960×720 resolution)
   - $c_x \approx 480$, $c_y \approx 360$
   - Distortion coefficients $k_1$ should be negative (barrel distortion)

---

## 4. Depth Model Evaluation

### 4.1 Model Selection

Three models were evaluated:

| Model | Type | Size | Notes |
|---|---|---|---|
| ZoeDepth NK | Metric depth (NYU+KITTI) | ~1.5 GB | **Selected.** Best indoor metric accuracy |
| MiDaS Large | Relative depth | ~400 MB | No metric scale; good relative structure |
| MiDaS Small | Relative depth | ~80 MB | Faster; acceptable relative depth |

ZoeDepth NK was selected because it produces metric (absolute) depth estimates directly, which are required for 3D obstacle reconstruction in physical units (metres). Relative depth models would require a separate scale estimation step.

### 4.2 Qualitative Observations on the Tello Feed

ZoeDepth performance on the Tello camera feed showed:

**Well-estimated regions:**
- Near-field objects (0.5–3 m) with clear texture
- Furniture with visible edges
- People and large cylindrical objects

**Poorly estimated regions:**
- White walls at distances > 3 m (near-uniform depth, low gradient)
- Regions with motion blur from aggressive drone manoeuvres
- Overexposed areas near the ceiling lights
- Floor regions at shallow angles (the row filter is primarily designed to handle these)

### 4.3 Depth Scale Consistency

ZoeDepth is calibrated for the NYU and KITTI depth distributions. The Tello's lens geometry and sensor size differ from those datasets. In practice, the absolute depth values have approximately the right scale for indoor objects at 0.5–3 m, but may underestimate depth at longer ranges (>4 m) where the Tello footage approaches the training distribution boundary.

### 4.4 Inference Timing

Approximate per-frame inference times (single forward pass, batch size 1, 960×720 input):

| Hardware | Approximate Time |
|---|---|
| CPU (Intel Core i7, 8-core) | 3–8 s |
| GPU (NVIDIA RTX 3060) | 0.3–0.8 s |
| GPU (NVIDIA RTX 4090) | 0.1–0.2 s |

These are rough estimates; actual times vary with CPU/GPU load and memory bandwidth.

The 250-frame inference interval at 24 fps equals approximately 10.4 s. This is comfortably above even the slowest CPU inference time, ensuring no depth cycle overlap. With a GPU, the interval could be reduced to 25–50 frames (1–2 s) without overlap risk.

---

## 5. APF Parameter Tuning

### 5.1 Methodology

APF parameters were tuned using the PyBullet simulation before live flight. The simulator uses the same `apf_with_bounds` function as the live controller, so behaviour in simulation transfers directly.

### 5.2 Parameter Sensitivity

**Attractive gain $k_a$:**

Too low ($k_a < 5$): Drone barely moves toward the goal; small repulsive forces dominate and the drone drifts.  
Too high ($k_a > 50$): Drone rushes toward the goal with insufficient braking near obstacles; overshoots.  
Selected: $k_a = 30$, reduced to $k_a' = \max(10, 30d)$ within 1 m of the goal.

**Repulsive gain $k_r$:**

Too low ($k_r < 3$): Obstacles insufficiently repelled; drone passes through or collides.  
Too high ($k_r > 30$): Drone is pushed aggressively away from any nearby object; unstable oscillation when navigating near walls.  
Selected: $k_r = 10$.

**Influence radius $\rho_0$:**

Too small ($\rho_0 < 0.2$ m): Repulsion activates too late; drone is already close to the obstacle before being deflected.  
Too large ($\rho_0 > 1.0$ m): Large portions of the flight volume become repulsively active; navigation is sluggish.  
Selected: $\rho_0 = 0.5$ m.

**Boundary influence distance $\rho_b$:**

Same sensitivity as $\rho_0$. In the experimental space, 0.5 m gives approximately 4–5 cell-widths of active boundary near walls, which is sufficient to prevent boundary violations at $v_{\max} = 30$.

Selected: $\rho_b = 0.5$ m.

### 5.3 Velocity Clamping

The maximum velocity command `max_val = 30` (of 100 scale) corresponds to approximately 0.5–1.0 m/s actual drone speed based on the Tello specification. This was chosen conservatively for a prototype system; higher values would reduce flight time but risk insufficient braking near obstacles.

### 5.4 Control Loop Rate

The `time.sleep(0.2)` at the end of `follow_path()` limits the effective control rate to 5 Hz (plus APF computation time, which is negligible). The UWB callback may fire at 100 Hz, but the control task only runs every 200 ms. This prevents command flooding (tellopy rejects commands too close together) and gives the drone time to respond to each command before the next is sent.

---

## 6. PyBullet Simulation Experiments

### 6.1 Purpose

The PyBullet simulation serves three purposes:

1. **Parameter tuning:** APF gains can be explored rapidly without hardware risk.
2. **Controller verification:** A new path or obstacle configuration can be tested before flight.
3. **Debugging:** Post-flight, `TelloDroneSim.load_config()` can replay a real flight log to visualise the drone's path and obstacle positions.

### 6.2 Running the Simulation

```bash
cd task
python sim.py
```

The PyBullet GUI opens. The duck model represents the drone; the soccer ball represents the target; spheres represent obstacles.

Key simulation state (in `sim.py`, modify before running):
```python
t.attract_coeff = 20          # APF gains
t.repel_coeff = 10
t.generate_obstacles(5, 0.75) # 5 random spheres, 0.75 m radius
t.start_pos = Vector3D(6.168, 2.187, -2.148)
t.target_pos = Vector3D(-0.30, 1.75, -2.00)
t.path = [t.start_pos, t.target_pos]
t.active_task = t.run_apf
```

### 6.3 Replaying a Real Flight Log

```python
from tellodrone_sim import TelloDroneSim

t = TelloDroneSim()
t.load_config("logs/log-<run-name>")  # loads positions and obstacle list
t.active_task = t.run_log            # replay recorded positions
t.start_sim()
t.run_sim()
```

This visualises the drone's actual trajectory alongside the obstacle map that was generated during the flight.

### 6.4 A* vs APF Comparison

To compare A* waypoints with direct APF:

```python
# A* mode
t.a_star_waypoints(grid_resolution=0.5)
t.active_task = t.run_apf

# Direct APF mode
t.path = [t.start_pos, t.target_pos]
t.active_task = t.run_apf
```

In simulation, A* + APF typically produces smoother paths around complex obstacle configurations. Direct APF is sufficient when obstacles are sparse and well-separated from the direct path.

### 6.5 Simulation Fidelity

The PyBullet simulation does not model:
- Aerodynamic forces (drag, ground effect, prop wash)
- Wi-Fi latency or video frame drops
- UWB measurement noise
- ZoeDepth inference latency or errors

It is used solely for APF control logic validation. Real-flight behaviour will differ, particularly in tight obstacle corridors where the timing of depth updates matters.

---

## 7. Live Flight Protocol

### 7.1 Pre-Flight Setup

Follow the checklist in §10. Key steps:

1. Ensure battery > 50% (ideally > 80% for a full run).
2. Start UWB system and verify position in `rostopic echo`.
3. Confirm target position is set correctly in `task/main.py`:
   ```python
   tello.set_target_pos(Vector3D(-0.30, 1.75, -2.00))
   ```
4. Confirm flight bounds match the physical space.
5. Clear the flight path of people and loose objects.

### 7.2 Launch Sequence

**Terminal 1 — UWB:**
```bash
bash cmd/uwb.sh
```
Wait for `[nlink_parser]` topics to appear in `rostopic list`.

**Terminal 2 — Verify UWB:**
```bash
rostopic echo /nlink_linktrack_nodeframe1
```
Confirm $x, y, z$ values are near the drone's known physical position. Ctrl+C when confirmed.

**Terminal 3 — Main:**
```bash
cd task && python main.py
```
The drone will:
1. Load ZoeDepth model (~20–60 s depending on hardware)
2. Connect to drone
3. Start video
4. Take off
5. Wait for first depth cycle (frame 250, ~10 s after takeoff)
6. Begin APF navigation

### 7.3 During Flight

- Maintain visual line of sight.
- Monitor terminal for WARNING/ERROR/CRITICAL messages.
- Watch for the Pygame display window showing the live camera feed.
- Be ready to press Ctrl+C for emergency landing.
- Note: battery warnings appear at ≤10%, emergency landing triggers at ≤5%.

### 7.4 Post-Flight

Log files are in `logs/log-<timestamp>/`:
- `log-info.log` — full timestamped event log
- `log-pos.log` — position trace (can be imported to Python for trajectory analysis)
- `log-config.json` — flight configuration snapshot

Images are in `img/<original|depth|annotated>/<timestamp>/`.
Raw video is in `vid/raw/vid-<timestamp>.avi`.

### 7.5 Trajectory Analysis

Quick position trace plot from log:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("logs/log-<run>/log-pos.log", usecols=(2, 3, 4))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.8)
ax.scatter(*data[0], c="green", s=50, label="start")
ax.scatter(*data[-1], c="red", s=50, label="end")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
plt.legend(); plt.show()
```

---

## 8. Observed Failure Modes

The following failure modes were observed or anticipated during development. Each entry includes the symptom, probable cause, and recommended action.

---

**Symptom:** Drone does not move after takeoff and depth model run.  
**Cause:** `depth_model_run = False` — depth thread has not completed yet (likely still running ZoeDepth inference on CPU).  
**Action:** Wait. The `follow_path()` function returns immediately if `depth_model_run is False`. Check terminal for "Depth Model Running" message. If it never appears, check that `cur_frame_idx` is counting up (video stream is working).

---

**Symptom:** Drone immediately lands with "Out of X bounds" message.  
**Cause:** UWB position reading is incorrect at startup (multipath, anchor misconfiguration, or tag not fully acquired).  
**Action:** Verify UWB anchor power and configuration. Run `rostopic echo` and confirm position is near the drone's physical location before launching `main.py`. The takeoff position is set to `cur_pos` at startup — if that position is wrong, bounds checks will trigger false positives.

---

**Symptom:** Drone circles around the target without landing.  
**Cause:** APF oscillation near the goal. The attraction force overshoots, the drone passes the goal, and is pulled back repeatedly.  
**Action:** Reduce `attract_coeff` for the close-range regime. The current implementation reduces $k_a$ to $\max(10, 30d)$ within 1 m, which may be insufficient. Try $k_a' = \max(5, 15d)$.

---

**Symptom:** Drone stops moving partway to the goal with no error.  
**Cause:** APF local minimum. An obstacle (or the combination of boundary and obstacle repulsion) is exactly cancelling the attractive force.  
**Action:** This is a known APF limitation. In simulation, verify the obstacle positions using `load_config`. For live flight, Ctrl+C and land. Then either remove the obstacle or adjust the target position to avoid the minimum.

---

**Symptom:** ZoeDepth detects phantom obstacles (walls, floor, ceiling).  
**Cause:** Row filter did not fully remove horizontal surfaces; or a large wall is within the 3 dark cluster threshold.  
**Action:** Adjust `dark_count` in `extract_segments` (default 3) down to 2. Increase `min_area` in `clean_binary` (default 20,000 px²) to require larger detections. Check that the row filter threshold (0.85) is appropriate for the specific flight direction.

---

**Symptom:** Video stream stalls (all frames identical).  
**Cause:** Wi-Fi interference or range. The Tello operates on 2.4 GHz; conflicting networks or distance from the drone causes packet loss.  
**Action:** Reduce interference sources. Ensure the control laptop is the only device connected to the Tello's AP. Move the laptop closer to the drone.

---

**Symptom:** `av.open()` raises an exception at startup.  
**Cause:** The Tello video stream is not yet active, or the H.264 container is malformed.  
**Action:** The `startup()` loop retries `av.open()` until successful. If it loops indefinitely, restart the drone and try again. Ensure `drone.start_video()` is called before opening the container.

---

**Symptom:** Pygame window is black (no video).  
**Cause:** Video thread has not yet produced a frame, or `cur_frame` is still `None`.  
**Action:** Wait 2–5 seconds after takeoff for the first frame. If it remains black, check that the video thread is running (no exception in the terminal).

---

## 9. Configuration Reference

### 9.1 Parameters Currently Hardcoded in Source

These parameters are not yet read from config files. Their locations for manual adjustment:

| Parameter | Location | Default | Notes |
|---|---|---|---|
| Model path | `task/tellodrone/core.py:98` | `"model/zoedepth-nyu-kitti"` | |
| Flight bounds x | `task/tellodrone/core.py:44` | `(-0.50, 7.00)` | |
| Flight bounds y | `task/tellodrone/core.py:45` | `(-0.50, 4.50)` | |
| Flight bounds z | `task/tellodrone/core.py:46` | `(-3.75, -0.50)` | |
| Frame inference interval | `task/tellodrone/depth_model.py:35` | `250` | |
| K-means clusters | `task/tellodrone/map_obstacle.py:31` | `5` | |
| Dark cluster count | `task/tellodrone/map_obstacle.py:80` | `3` | |
| Row filter ratio | `task/tellodrone/map_obstacle.py:47` | `0.85` | |
| Morph kernel size | `task/tellodrone/map_obstacle.py:64` | `11` | |
| Min obstacle area | `task/tellodrone/map_obstacle.py:64` | `20000` px² | |
| Max obstacle radius | `task/tellodrone/map_obstacle.py:162` | `1.0` m | |
| Obstacle merge threshold | `task/tellodrone/depth_model.py:52` | `0.5` m | |
| Camera-to-tag offset | `task/tellodrone/map_obstacle.py:121–123` | `(-0.4, -0.4, -0.4)` m | |
| APF $k_a$ | `task/tellodrone/follow_path.py:49` | `30` | |
| APF $k_r$ | `task/tellodrone/follow_path.py:50` | `10` | |
| APF $\rho_0$ | `task/tellodrone/follow_path.py:51` | `0.5` m | |
| APF $\rho_b$ | `task/tellodrone/follow_path.py:52` | `0.5` m | |
| Max velocity command | `task/tellodrone/follow_path.py:61` | `30` | |
| Arrival threshold | `task/tellodrone/follow_path.py:34` | `0.30` m | |
| Control loop sleep | `task/tellodrone/follow_path.py:83` | `0.20` s | |
| Target position | `task/main.py:33` | `(-0.30, 1.75, -2.00)` | Change per run |
| Catkin workspace | `cmd/uwb.sh:4` | `/home/horse3903/catkin_ws` | Update for your machine |

### 9.2 Config File Templates

Reference config files with full documentation are provided in `configs/`:
- `configs/drone.yaml` — connection and flight envelope
- `configs/depth.yaml` — model and inference parameters
- `configs/uwb.yaml` — ROS topic and coordinate frame
- `configs/control.yaml` — APF and velocity parameters

These are not currently loaded at runtime; they serve as documentation of the parameter space and templates for a future config-driven architecture.

---

## 10. Checklist: Pre-Flight

### Hardware

- [ ] Tello battery charged (≥80% recommended)
- [ ] Propeller guards attached
- [ ] All UWB anchors powered on, LEDs indicating lock
- [ ] UWB tag mounted securely on drone
- [ ] Flight space cleared of people and hazards
- [ ] Minimum 2 m clearance in all directions from launch point

### Software

- [ ] ROS sourced: `source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash`
- [ ] UWB launched and publishing: `bash cmd/uwb.sh`
- [ ] UWB position verified near drone's physical location: `rostopic echo /nlink_linktrack_nodeframe1`
- [ ] Target position set in `task/main.py`
- [ ] Flight bounds verified against physical space in `task/tellodrone/core.py`
- [ ] `model/zoedepth-nyu-kitti` exists and is complete
- [ ] `calibration_data.npz` present in project root
- [ ] Output directories do not have stale data from a prior run (or you're OK overwriting)
- [ ] Laptop Wi-Fi connected to Tello's AP (192.168.10.x network)

### Simulation Verification (recommended before first live flight with new parameters)

- [ ] `cd task && python sim.py` runs without error
- [ ] Drone reaches target in simulation with current APF parameters
- [ ] No boundary violations observed in simulation
