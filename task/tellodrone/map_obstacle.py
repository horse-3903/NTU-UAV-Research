import cv2
import numpy as np

from typing import List, Tuple

from vector import Vector3D

from PIL import Image

import torch
from transformers import ZoeDepthImageProcessor, ZoeDepthForDepthEstimation

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
    depth_values = depth_map.flatten().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(depth_values, cluster_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    clustered_map = labels.reshape(depth_map.shape)
    return clustered_map, centers


# Visualize depth clusters
def draw_clusters(clustered_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
    vis_map = np.zeros_like(clustered_map, dtype=np.uint8)
    for idx, center in enumerate(centers):
        vis_map[clustered_map == idx] = int((center - centers.min()) / (centers.max() - centers.min()) * 255)
    return vis_map


# Segment rows based on white pixel density
def filter_rows(binary_map: np.ndarray, threshold_ratio: float = 0.95) -> np.ndarray:
    row_counts = np.count_nonzero(binary_map, axis=1)
    max_count = max(row_counts)
    
    if max_count <= len(binary_map[0]) // 2:
        return binary_map
    
    for row_idx in range(binary_map.shape[0]):
        white_count = np.sum(binary_map[row_idx, :] == 255)
        if white_count >= max_count * threshold_ratio:
            binary_map[row_idx, :] = 0
    
    return binary_map


# Remove small strips in the binary map
def clean_binary(binary_map: np.ndarray, min_area: int = 10000) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    opened_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened_map)
    clean_map = np.zeros_like(binary_map)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > min_area:
            clean_map[labels == i] = 255
    
    return clean_map


# Separate clusters into dark segments
def extract_segments(clustered_map: np.ndarray, centers: np.ndarray, dark_count: int = 2) -> List[np.ndarray]:
    sorted_idxs = np.argsort(centers.flatten())
    dark_idxs = sorted_idxs[:dark_count]
    segments = []
    
    for cluster_idx in dark_idxs:
        segment = np.zeros_like(clustered_map, dtype=np.uint8)
        segment[clustered_map == cluster_idx] = 255
        filtered = filter_rows(segment)
        clean = clean_binary(filtered, min_area=10000)
        segments.append(clean)
    
    return segments


# Detect obstacles in binary maps
def detect_obstacles(binary_map: np.ndarray, min_area: float = 2500) -> List[Tuple[Tuple[int, int], float]]:
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Get contours
    obstacles = []
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                _, radius = cv2.minEnclosingCircle(contour)
                obstacles.append(((centroid_x, centroid_y), radius))
    
    return obstacles


# Undistort coordinates based on camera calibration
def undistort_point(x: int, y: int, img: np.ndarray) -> Tuple[int, int]:
    h, w = img.shape[:2]  # Get image dimensions
    map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), 5)
    return int(map_x[y, x]), int(map_y[y, x])


# Compute 3D position from 2D pixel and depth
def compute_3d(pos: Tuple[int, int], abs_depth: np.ndarray) -> Tuple[float, float, float]:
    x, y = pos  # Extract pixel coordinates
    depth = np.mean(abs_depth[y - 2 : y + 3, x - 2 : x + 3])
    x_world = -depth
    y_world = (x - intrinsics["c_x"]) * depth / intrinsics["f_x"] - 0.6
    z_world = (y - intrinsics["c_y"]) * depth / intrinsics["f_y"] - 0.6
    return x_world, y_world, z_world


# Process image to detect obstacles
def process_image(image: np.ndarray, abs_depth: np.ndarray) -> Tuple[List[Tuple[Vector3D, float]], List[Tuple[Tuple[int, int], float]]]:
    clustered_map, centers = segment_depth(abs_depth, cluster_count=5)
    cluster_segments = extract_segments(clustered_map, centers, dark_count=2)
    detected_obstacles = []
    
    for segment in cluster_segments:
        detected_obstacles.extend(detect_obstacles(segment))
    
    real_obstacles = []
    pixel_obstacles = []

    for obstacle in detected_obstacles:
        centroid, radius_px = obstacle
        undist_x, undist_y = undistort_point(*centroid, image)
        
        radius_m = (radius_px * abs_depth[centroid[1], centroid[0]]) / intrinsics["f_x"]
        pos_3d = compute_3d((undist_x, undist_y), abs_depth)
        
        real_obstacles.append((Vector3D(*pos_3d), radius_m))
        pixel_obstacles.append((centroid, int(radius_px)))
    
    return real_obstacles, pixel_obstacles


def update_obstacles(
    cur_obs: List[Tuple[Vector3D, float]],
    new_obs: List[Tuple[Vector3D, float]],
    threshold: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    z_bounds: Tuple[float, float],
) -> List[Tuple[Vector3D, float]]:
    """
    Updates the current list of obstacles by considering new obstacles.
    An intersection occurs if the distance between two obstacles is less than or equal to the threshold.

    Parameters:
        cur_obs (List[Tuple[Vector3D, float]]): Current list of obstacles (position, radius).
        new_obs (List[Tuple[Vector3D, float]]): New obstacles to evaluate (position, radius).
        threshold (float): Maximum distance between obstacles to count as an intersection.
        x_bounds (Tuple[float, float]): X-axis bounds (min, max).
        y_bounds (Tuple[float, float]): Y-axis bounds (min, max).
        z_bounds (Tuple[float, float]): Z-axis bounds (min, max).

    Returns:
        List[Tuple[Vector3D, float]]: Updated list of obstacles.
    """
    updated_obs = cur_obs.copy()

    for new_center, new_radius in new_obs:
        # Check if the new obstacle is within the specified bounds
        if not (x_bounds[0] <= new_center.x <= x_bounds[1] and
                y_bounds[0] <= new_center.y <= y_bounds[1] and
                z_bounds[0] <= new_center.z <= z_bounds[1]):
            continue

        intersected = False

        for idx, (cur_center, cur_radius) in enumerate(updated_obs):
            # Calculate the distance between centers of the two spheres
            distance = (new_center - cur_center).magnitude()

            # Check if the distance is less than or equal to the threshold
            if distance <= threshold:
                # Replace the current obstacle with the new one
                updated_obs[idx] = (new_center, new_radius)
                intersected = True
                break

        if not intersected:
            # Add the new obstacle if it doesn't intersect with any existing ones
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

if __name__ == "__main__":
    model_name = "model/zoedepth-nyu-kitti"
    image_name = "img/original/2024-11-29 10:50:56.252071/frame-200.png"
    
    print(f"Loading model from {model_name}")
    image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
    depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)
    
    print(f"Loading image from {image_name}")
    image = np.array(Image.open(image_name))
    
    print("Estimating Depth for Image")
    pil_image = Image.fromarray(image)
    inputs = image_processor.preprocess(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = depth_model.forward(inputs["pixel_values"])
    
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, source_sizes=[(pil_image.height, pil_image.width)]
    )
    
    absolute_depth = post_processed_output[0]["predicted_depth"]
    relative_depth = (absolute_depth - absolute_depth.min()) / (absolute_depth.max() - absolute_depth.min())
    
    absolute_depth = absolute_depth.numpy()
    relative_depth = (relative_depth.numpy() * 255).astype("uint8")
    
    real_obstacles, pixel_obstacles = process_image(image, absolute_depth)
    
    annotated = draw_obstacles(image, real_obstacles, pixel_obstacles)
    
    cv2.imshow("annotated", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows