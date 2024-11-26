import cv2
import glob

import numpy as np
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from tellodrone.map_obstacle import process_obstacles, draw_obstacles

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

if __name__ == "__main__":
    # png = glob.glob("img/original/**/*.png", recursive=True) + glob.glob("img/manual/**/*.png", recursive=True)
    
    # frame = "img/original/2024-11-25 17:29:25.794466/frame-1120.png"
    png = glob.glob("img/manual/2024-11-26 15:37:21.802924/*.jpg")
    
    for frame in png:
        print(f"Processing image : {frame}")
        # Read image and convert to RGB format
        image = cv2.imread(frame)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("Estimating Depth")
        absolute_depth, relative_depth = estimate_depth(image)

        print("Processing Obstacles")
        obstacles_3d = process_obstacles(relative_depth, absolute_depth, relative_depth, intrinsics)

        for obstacle in obstacles_3d:
            print(f"Obstacle 3D Position: {obstacle['3D_position']}") 
            print(f"Radius (pixels): {obstacle['radius_pixels']}")
            print(f"Radius (meters): {obstacle['radius_meters']:.2f}")

        print("Drawing Obstacles")
        output_image = draw_obstacles(image, obstacles_3d)
        print()
        
        cv2.imshow("Obstacles", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()