<div align="center">

# NTU UAV Research

Real-time obstacle avoidance and navigation for autonomous UAVs using deep learning depth estimation

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-22314E?style=flat-square&logo=ros&logoColor=white)](https://www.ros.org/)
[![License](https://img.shields.io/github/license/horse-3903/NTU-UAV-Research?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/horse-3903/NTU-UAV-Research?style=flat-square)](../../commits)

</div>

---

## Overview

This project implements a real-time obstacle avoidance and autonomous navigation system for DJI Tello drones. It integrates Ultra-Wideband (UWB) positioning via the Nooploop Linktrack system with deep learning-based monocular depth estimation (ZoeDepth) to detect and avoid obstacles dynamically. The system is designed and tested as part of research at Nanyang Technological University (NTU).

---

## Features

- **Real-Time Depth Estimation** — Uses ZoeDepth to generate accurate depth maps from the drone's monocular camera feed
- **Obstacle Detection and Avoidance** — Clusters and segments depth maps to identify obstacles and compute avoidance trajectories
- **UWB Position Tracking** — Tracks the drone's precise position in real-time via Nooploop Linktrack and ROS
- **Live Visualisation** — Displays annotated video streams with obstacle bounding information and drone state
- **Task Logging** — Records position logs, takeoff/target positions, detected obstacles, and drone configuration at each step

---

## Tech Stack

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![ROS](https://img.shields.io/badge/ROS-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://www.ros.org/)

---

## Getting Started

### Prerequisites

- **Hardware**: DJI Tello drone, Nooploop Linktrack UWB system
- **OS**: Ubuntu 20.04
- **Software**: Python 3.8, ROS1 Noetic
- Camera calibration data (`calibration_data.npz`) — contains camera matrix and distortion coefficients

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/horse-3903/NTU-UAV-Research.git
   cd NTU-UAV-Research
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure ROS and ensure compatibility with the Nooploop Linktrack and Tello SDKs.

4. Place `calibration_data.npz` in the project root.

### Usage

**1. Start the UWB positioning system:**
```bash
bash cmd/uwb.sh
```

**2. Verify UWB data is streaming via ROS:**
```bash
rostopic echo /nlink_linktrack_nodeframe1
```

**3. Launch the main navigation task:**
```bash
python task/main.py
```

---

## Methodology

The system operates as a real-time pipeline:

```
DJI Tello Camera Feed
        ↓
  ZoeDepth Model (Monocular Depth Estimation)
        ↓
  Depth Map Clustering & Segmentation
        ↓
  Obstacle Detection (position + dimensions)
        ↓
  UWB Fusion (absolute position via Nooploop Linktrack + ROS)
        ↓
  Avoidance Algorithm → Drone Control Commands
```

1. **Depth Estimation** — ZoeDepth processes each camera frame to produce a per-pixel depth map without requiring stereo hardware.
2. **Obstacle Segmentation** — Depth clusters are analysed to identify obstacle boundaries and estimate their real-world dimensions.
3. **Pose Fusion** — UWB positioning provides absolute coordinates fused with depth data to plan safe trajectories.
4. **Control** — Computed avoidance commands are sent to the Tello drone via its SDK.

---

## Project Structure

```
NTU-UAV-Research/
├── cmd/                    # Shell scripts (e.g., UWB initialisation)
├── src/
│   ├── control/            # Drone control logic
│   ├── depth/              # Depth estimation pipeline
│   ├── track/              # Position tracking utilities
│   └── util/               # Shared helpers
├── task/
│   └── main.py             # Main entry point
├── calibrate/              # Camera calibration tools
├── calibration_data.npz    # Camera intrinsics
└── requirements.txt
```

---

## Future Enhancements

- **Dynamic Re-Routing** — Advanced path planning for complex multi-obstacle environments
- **SLAM Integration** — Combining UWB with visual SLAM for improved localisation
- **Predictive Avoidance** — ML-based models for anticipating obstacle trajectories

---

## License

This project is licensed under the [MIT License](LICENSE).
