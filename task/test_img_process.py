import cv2
import numpy as np

from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import List, Tuple

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

model_name = "model/zoedepth-nyu-kitti"
image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)


def estimate_depth(img: np.ndarray):    
    pil_image = Image.fromarray(img)
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
    
    return absolute_depth, relative_depth


def segment_depth_values(depth_map: np.ndarray, num_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    # Flatten the depth map and normalize values
    depth_values = depth_map.flatten().astype(np.float32)

    # Apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(depth_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels back to the shape of the depth map
    clustered_depth_map = labels.reshape(depth_map.shape)

    return clustered_depth_map, centers


def visualise_clusters(clustered_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Map each cluster ID to a grayscale value
    depth_visualisation = np.zeros_like(clustered_map, dtype=np.uint8)

    for idx, center in enumerate(centers):
        depth_visualisation[clustered_map == idx] = int((center - centers.min()) / (centers.max() - centers.min()) * 255)

    return depth_visualisation


def threshold_segment_rows(image: np.ndarray, percentage_threshold: float) -> np.ndarray:    
    for row_idx in range(image.shape[0]):
        black_pixels = np.sum(image[row_idx, :] == 0)
        total_pixels = image.shape[1]
        percentage = black_pixels / total_pixels
        if percentage >= percentage_threshold:
            image[row_idx, :] = 255
            
    _, image = cv2.threshold(image, np.min(image), 255, cv2.THRESH_BINARY_INV)
    
    return image


def find_obstacles(image: np.ndarray, min_area: float = 1000):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def draw_obstacles(image: np.ndarray, obstacles: List) -> np.ndarray:
    # Create a copy of the image to draw contours
    output_image = image.copy()

    # Iterate over each contour
    for ob in obstacles:
        centroid, radius_pixels = ob
        cv2.circle(output_image, centroid, radius_pixels, (255, 0, 0), 2)
        
        cv2.circle(output_image, centroid, 5, (255, 0, 0), -1)
        
    return output_image


if __name__ == "__main__":
    frame = "img/manual/2024-11-28 14:45:12.458643/frame-200.png"
    
    image = cv2.imread(frame)
    print("Estimating Depth...")
    absolute_depth, relative_depth = estimate_depth(image)

    print("Processing Obstacles...")
    clustered_map, centers = segment_depth_values(absolute_depth, num_clusters=6)
    depth_map = visualise_clusters(clustered_map, centers=centers)
    depth_map = threshold_segment_rows(depth_map, percentage_threshold=0.7)
    
    obstacles = find_obstacles(depth_map)
    image = draw_obstacles(depth_map, obstacles=obstacles)
    
    # print("Drawing Obstacles...")
    # output_image = draw_obstacles(image, real_res, pixel_res)
    
    cv2.imshow("", depth_map)
    cv2.imshow("", image)
    
    while True:
        try:
            if cv2.waitKey(0) == 27:
                break
        except:
            break
    
    cv2.destroyAllWindows()