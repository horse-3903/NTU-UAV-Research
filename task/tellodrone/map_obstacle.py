import cv2
import numpy as np

from typing import List, Tuple, Dict

from vector import Vector3D

import traceback

# Load calibration data
calibration_data = np.load("calibration_data.npz")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# Convert camera intrinsics for easier use
intrinsics = {
    "f_x": camera_matrix[0, 0],  # Focal length in x-direction
    "f_y": camera_matrix[1, 1],  # Focal length in y-direction
    "c_x": camera_matrix[0, 2],  # Principal point x-coordinate (image center)
    "c_y": camera_matrix[1, 2],  # Principal point y-coordinate (image center)
}

def find_obstacles(depth_frame: np.ndarray, threshold_value: int = None, percentage_threshold: float = 0.85, min_area: int = 20000) -> List[Tuple[Vector3D, float]]:
    if not threshold_value:
        threshold_value = max(70, np.mean(depth_frame))
    
    _, thresholded_image = cv2.threshold(depth_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    for row_idx in range(thresholded_image.shape[0]):
        black_pixels = np.sum(thresholded_image[row_idx, :] == 255)
        total_pixels = thresholded_image.shape[1]
        percentage = black_pixels / total_pixels
        if percentage >= percentage_threshold:
            thresholded_image[row_idx, :] = 0

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    results = []
    for contour in filtered_contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            _, radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            results.append([(centroid_x, centroid_y), radius])
    return results

def undistort_coordinates(x: int, y: int, image: np.ndarray) -> Tuple[int, int]:
    height, width = image.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (width, height), 5)
    undistorted_x = mapx[y, x]
    undistorted_y = mapy[y, x]
    return int(undistorted_x), int(undistorted_y)

def get_3d_position(centroid: Tuple[int, int], absolute_depth: np.ndarray, intrinsics: dict) -> Tuple[float, float, float]:
    x, y = centroid
    depth = absolute_depth[y, x]
    X = -depth
    Y = (x - intrinsics["c_x"]) * depth / intrinsics["f_x"]
    Z = (y - intrinsics["c_y"]) * depth / intrinsics["f_y"]
    return X, Y, Z

def process_obstacles(image: np.ndarray, absolute_depth: np.ndarray, relative_depth: np.ndarray, intrinsics: dict) -> Tuple[List[Tuple[Vector3D, float]], List[Tuple[Tuple[float, float], float]]]:
    try:
        obstacles = find_obstacles(relative_depth)
        real_res = []
        pixel_res = []

        for obstacle in obstacles:
            centroid, radius_pixels = obstacle
            undistorted_x, undistorted_y = undistort_coordinates(*centroid, image)

            # Get depth at the centroid
            depth = absolute_depth[centroid[1], centroid[0]]

            # Convert radius to global scale (meters)
            radius_meters = (radius_pixels * depth) / intrinsics["f_x"]

            # Get 3D position of the obstacle
            X, Y, Z = get_3d_position((undistorted_x, undistorted_y), absolute_depth, intrinsics)

            real_res.append((Vector3D(X, Y, Z), radius_meters))
            pixel_res.append((centroid, radius_pixels))
    except:
        traceback.print_exc()

    return real_res, pixel_res

def update_obstacles(cur_obs: List[Tuple[Vector3D, float]], new_obs: List[Tuple[Vector3D, float]]) -> List[Tuple[Vector3D, float]]:
    updated_obs = cur_obs.copy() 

    for new_center, new_radius in new_obs:
        intersected = False

        for idx, (cur_center, cur_radius) in enumerate(cur_obs):
            # Calculate the distance between centers of the two spheres
            distance = (new_center - cur_center).magnitude()

            # Check if the spheres intersect
            if distance <= (new_radius + cur_radius):
                # If they intersect, take the newer sphere
                updated_obs[idx] = (new_center, new_radius)
                intersected = True
                break

        if not intersected:
            # If no intersection, add the new obstacle to the list
            updated_obs.append((new_center, new_radius))

    return updated_obs

def draw_obstacles(image: np.ndarray, real_obstacles: List[Tuple[Vector3D, float]], pixel_obstacles: List[Tuple[Tuple[float, float], float]]):
    for idx in range(len(real_obstacles)):
        centroid, radius_pixels = pixel_obstacles[idx]
        position, radius_meters = real_obstacles[idx]
        
        cv2.circle(image, centroid, radius_pixels, (255, 0, 0), 2)
        
        cv2.circle(image, centroid, 5, (255, 0, 0), -1)
        
        cv2.putText(image, f"({position.x:.2f}, {position.y:.2f}, {position.z:.2f})", (centroid[0] + 5, centroid[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(image, f"Radius: {radius_meters:.2f} m", (centroid[0] + 5, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return image