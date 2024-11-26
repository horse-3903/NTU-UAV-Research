import cv2
import numpy as np
from vector import Vector3D

# Depth Estimation and 3D Mapping
def pixel_to_3d(pixel: Vector3D, depth: float, camera_matrix):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Compute normalized camera coordinates
    x = (pixel.x - cx) / fx
    y = (pixel.y - cy) / fy

    # Scale by depth
    return Vector3D(x * depth, y * depth, depth)


def project_points(points: list[Vector3D], camera_matrix, dist_coeffs, rvec, tvec):
    # Convert points to numpy arrays
    np_points = np.array([p.to_ndarr() for p in points], dtype=np.float32)

    # Use OpenCV's projectPoints
    projected_points, _ = cv2.projectPoints(np_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert back to Vector3D
    return [Vector3D.from_arr(p[0]) for p in projected_points]


# Tello Integration
def run_tello_depth_model_with_calibration(tello, frame):
    # Load calibration data
    data = np.load('calibration_data.npz')
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # Get depth map
    depth_map = tello.depth_model(frame)

    # Example: Map the center pixel to 3D
    height, width = frame.shape[:2]
    center_pixel = Vector3D(width // 2, height // 2, 0)
    depth = depth_map[height // 2, width // 2]

    # Convert pixel to 3D
    center_3d = pixel_to_3d(center_pixel, depth, camera_matrix)
    print(f"Center pixel mapped to 3D: {center_3d}")

    return center_3d