import cv2
import numpy as np

from typing import List, Tuple, Dict

from vector import Vector3D

import traceback

# Load calibration data
calibration_data = np.load("calibration_data.npz")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# Extract camera intrinsics for easier use
intrinsics = {
    "f_x": camera_matrix[0, 0],  # Focal length x
    "f_y": camera_matrix[1, 1],  # Focal length y
    "c_x": camera_matrix[0, 2],  # Principal point x
    "c_y": camera_matrix[1, 2],  # Principal point y
}

# Segment depth map into clusters
def segment_depth(depth_map: np.ndarray, cluster_count: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    depth_values = depth_map.flatten().astype(np.float32)  # Flatten depth values
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # Clustering criteria
    _, labels, centers = cv2.kmeans(depth_values, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # Apply k-means
    clustered_map = labels.reshape(depth_map.shape)  # Reshape labels to original map shape
    return clustered_map, centers

# Visualize depth clusters
def draw_clusters(clustered_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
    vis_map = np.zeros_like(clustered_map, dtype=np.uint8)  # Initialize visualization map
    for idx, center in enumerate(centers):
        vis_map[clustered_map == idx] = int((center - centers.min()) / (centers.max() - centers.min()) * 255)  # Normalize and map
    return vis_map

# Segment rows based on white pixel density
def filter_rows(binary_map: np.ndarray, threshold_ratio: float = 0.95) -> np.ndarray:
    row_counts = np.count_nonzero(binary_map, axis=1)  # Count non-zero pixels per row
    max_count = max(row_counts)  # Get the max count
    
    if max_count <= len(binary_map[0]) // 2:  # Skip if max is small
        return binary_map
    
    for row_idx in range(binary_map.shape[0]):  # Iterate through rows
        white_count = np.sum(binary_map[row_idx, :] == 255)  # Count white pixels
        if white_count >= max_count * threshold_ratio:  # Check threshold
            binary_map[row_idx, :] = 0  # Clear row if above threshold
    
    return binary_map

# Remove small strips in the binary map
def clean_binary(binary_map: np.ndarray, min_area: int = 400) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Morphological kernel
    opened_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)  # Apply morphological opening
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened_map)  # Get connected components
    clean_map = np.zeros_like(binary_map)  # Initialize output map
    
    for i in range(1, num_labels):  # Iterate through labels (skip background)
        x, y, w, h, area = stats[i]  # Extract stats
        if area > min_area:  # Check area threshold
            clean_map[labels == i] = 255  # Retain large components
    
    return clean_map

# Separate clusters into dark segments
def extract_segments(clustered_map: np.ndarray, centers: np.ndarray, dark_count: int = 2) -> List[np.ndarray]:
    sorted_idxs = np.argsort(centers.flatten())  # Sort centers by depth
    dark_idxs = sorted_idxs[:dark_count]  # Select darkest cluster indices
    segments = []
    
    for cluster_idx in dark_idxs:  # Iterate through dark clusters
        segment = np.zeros_like(clustered_map, dtype=np.uint8)  # Initialize segment map
        segment[clustered_map == cluster_idx] = 255  # Extract cluster
        filtered = filter_rows(segment)  # Filter rows
        clean = clean_binary(filtered)  # Clean map
        segments.append(clean)
    
    return segments

# Detect obstacles in binary maps
def detect_obstacles(binary_map: np.ndarray, min_area: float = 1000) -> List[Tuple[Tuple[int, int], float]]:
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Get contours
    obstacles = []
    
    for contour in contours:  # Iterate through contours
        if cv2.contourArea(contour) > min_area:  # Check area threshold
            moments = cv2.moments(contour)  # Get moments
            if moments["m00"] != 0:  # Avoid division by zero
                centroid_x = int(moments["m10"] / moments["m00"])  # Compute x centroid
                centroid_y = int(moments["m01"] / moments["m00"])  # Compute y centroid
                _, radius = cv2.minEnclosingCircle(contour)  # Get enclosing circle radius
                obstacles.append(((centroid_x, centroid_y), radius))  # Add to obstacles list
    
    return obstacles

# Undistort coordinates based on camera calibration
def undistort_point(x: int, y: int, img: np.ndarray) -> Tuple[int, int]:
    h, w = img.shape[:2]  # Get image dimensions
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), 5)  # Undistort map
    return int(map_x[y, x]), int(map_y[y, x])  # Return undistorted coordinates

# Compute 3D position from 2D pixel and depth
def compute_3d(pos: Tuple[int, int], abs_depth: np.ndarray) -> Tuple[float, float, float]:
    x, y = pos  # Extract pixel coordinates
    depth = np.mean(abs_depth[y - 2 : y + 3, x - 2 : x + 3])  # Compute mean depth around pixel
    x_world = -depth  # Compute x world
    y_world = (x - intrinsics["c_x"]) * depth / intrinsics["f_x"]  # Compute y world
    z_world = (y - intrinsics["c_y"]) * depth / intrinsics["f_y"]  # Compute z world
    return x_world, y_world, z_world

# Process image to detect obstacles
def process_image(image: np.ndarray, abs_depth: np.ndarray) -> Tuple[List[Tuple[Vector3D, float]], List[Tuple[Tuple[int, int], float]]]:
    clustered_map, centers = segment_depth(abs_depth, cluster_count=4)  # Segment depth map
    cluster_segments = extract_segments(clustered_map, centers, dark_count=2)  # Extract segments
    detected_obstacles = []
    
    for segment in cluster_segments:  # Process each segment
        detected_obstacles.extend(detect_obstacles(segment))  # Detect obstacles
    
    real_obstacles = []
    pixel_obstacles = []

    for obstacle in detected_obstacles:  # Process each detected obstacle
        centroid, radius_px = obstacle  # Get centroid and radius
        undist_x, undist_y = undistort_point(*centroid, image)  # Undistort centroid
        radius_m = (radius_px * abs_depth[centroid[1], centroid[0]]) / intrinsics["f_x"]  # Convert to meters
        pos_3d = compute_3d((undist_x, undist_y), abs_depth)  # Compute 3D position
        real_obstacles.append((Vector3D(*pos_3d), radius_m))  # Append real obstacle
        pixel_obstacles.append((centroid, int(radius_px)))  # Append pixel obstacle
    
    return real_obstacles, pixel_obstacles  # Return obstacles


def update_obstacles(cur_obs: List[Tuple[Vector3D, float]], new_obs: List[Tuple[Vector3D, float]], threshold: float, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], z_bounds: Tuple[float, float]) -> List[Tuple[Vector3D, float]]:
    updated_obs = cur_obs.copy()

    for new_center, new_radius in new_obs:
        # Check bounds for the new obstacle
        if not (x_bounds[0] <= new_center.x <= x_bounds[1] and
                y_bounds[0] <= new_center.y <= y_bounds[1] and
                z_bounds[0] <= new_center.z <= z_bounds[1]):
            continue

        intersected = False

        for idx, (cur_center, cur_radius) in enumerate(cur_obs):
            # Calculate the distance between centers of the two spheres
            distance = (new_center - cur_center).magnitude()

            # Check if the spheres intersect and the intersection depth exceeds the threshold
            if distance < (new_radius + cur_radius):
                intersection_depth = (new_radius + cur_radius) - distance
                if intersection_depth > threshold:
                    # Update with the newer sphere if the intersection threshold is exceeded
                    updated_obs[idx] = (new_center, new_radius)
                    intersected = True
                    break

        if not intersected:
            # If no significant intersection, add the new obstacle to the list
            updated_obs.append((new_center, new_radius))

    return updated_obs


def draw_obstacles(image: np.ndarray, real_obstacles: List[Tuple[Vector3D, float]], pixel_obstacles: List[Tuple[Tuple[int, int], float]]) -> np.ndarray:
    output = image.copy()
    for idx, real_obstacle in enumerate(real_obstacles):
        position, radius_meters = real_obstacle
        centroid, radius_pixels = pixel_obstacles[idx]
        cv2.circle(output, centroid, radius_pixels, (255, 0, 0), 2)
        cv2.circle(output, centroid, 5, (255, 0, 0), -1)
        
        cv2.putText(output, f"({position.x:.2f}, {position.y:.2f}, {position.z:.2f})", (centroid[0] + 5, centroid[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(output, f"Radius: {radius_meters:.2f} m", (centroid[0] + 5, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
    return output