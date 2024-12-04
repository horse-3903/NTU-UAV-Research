# **Real-Time Obstacle Avoidance and Navigation Using Depth Estimation for Autonomous UAVs**

A Python-based implementation for enhanced Tello drone control, integrating UWB (Ultra-Wideband) positioning and depth estimation. This project is designed to enable precise navigation, real-time obstacle detection, and visualisation of the drone's environment.

---

## **Features**

- **UWB Position Tracking**: 
  - Real-time drone position tracking using the Nooploop Linktrack system.
  - ROS-based streaming and integration for seamless communication.

- **Depth Estimation**:
  - Utilises Zoe depth mapping to generate accurate depth information from stereo images.
  - Supports clustering and segmentation for obstacle detection.

- **Obstacle Avoidance**:
  - Processes depth maps to dynamically map obstacles and ensure safe navigation.

- **Real-Time Visualisation**:
  - Displays drone position and detected obstacles in a graphical interface.

- **Task Logging**:
  - Logs drone configurations and position data for debugging and analysis.

---

## **System Requirements**

- **Hardware**:
  - DJI Tello drone.
  - Nooploop Linktrack UWB system.
  
- **Software**:
  - Ubuntu 20.04
  - Python 3.8
  - ROS1-Noetic
  - Required Python libraries:
    - `numpy`
    - `opencv-python`
    - `matplotlib`
    - `torch`
    - `PyAV`
    - Additional dependencies listed in `requirements.txt`.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/horse-3903/NTU-UAV-Research.git
   cd tellodrone-project
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure ROS:
   - Install and set up ROS on your system.
   - Ensure compatibility with Nooploop Linktrack and Tello drone SDKs.

4. Add calibration data:
   - Place `calibration_data.nps` in the project root, containing the camera matrix and distortion coefficients.

---

## **Usage**

### **1. Start the UWB System**
Run the UWB initialisation script to begin position tracking:
```bash
bash cmd/uwb.sh
```

### **2. Verify UWB Data**
Ensure UWB data is streaming correctly via ROS:
```bash
rostopic echo <nlink_linktrack_nodeframe1>
```

### **3. Launch the Main Task**
Run the core script to execute tasks:
```bash
python task/main.py
```

---
## **Key Features in Detail**

### **Position Tracking**
- **Accuracy**: Tracks the drone's position in real-time using UWB and ROS.
- **Integration**: Dynamically updates positions for effective visualisation and control.

### **Depth Mapping**
- Generates high-accuracy depth maps using Zoe depth estimation.
- Segments depth clusters to identify obstacles in the environment.

### **Visualisation**
- Displays annotated video streams with obstacle information and dimensions.
- Offers real-time updates on drone actions and environment mapping.

### **Logging**
- **Position Logs**: Records the drone’s position at each step.
- **Configuration Logs**: Captures:
  - Takeoff and target positions.
  - Detected obstacles.
  - Current drone configurations.

---

## **Future Enhancements**

1. **Dynamic Re-Routing**:
   - Advanced algorithms for re-routing in complex environments.

2. **SLAM Integration**:
   - Combining UWB and visual SLAM for improved localisation.

3. **Machine Learning**:
   - Predictive obstacle avoidance using ML models.

---

## **Contributors**
- **Your Name**: [GitHub Profile](https://github.com/horse-3903)

Feel free to contribute by submitting pull requests or opening issues for bugs or feature suggestions.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

--- 