import os
import numpy as np

import cv2
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def load_depth_model(self: "TelloDrone") -> None:
    self.logger.info(f"Loading model from {self.model_name}")
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(self.model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)
    
def run_depth_model(self: "TelloDrone", frame_img: np.ndarray) -> None:
    if self.frame_idx % 20 == 0:
        self.logger.info("Video frame captured")
        self.logger.info(f"Estimating depth of frame {self.frame_idx}")
        depth_image = self.estimate_depth(img=frame_img)
        
        orig_output_path = os.path.join(f"img/original/{self.init_time}", f"frame-{self.frame_idx}.png")
        depth_output_path = os.path.join(f"img/depth/{self.init_time}", f"frame-{self.frame_idx}.png")
        
        Image.fromarray(frame_img).save(orig_output_path)
        Image.fromarray(depth_image).save(depth_output_path)

def estimate_depth(self: "TelloDrone", img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    self.logger.info("Estimating Depth for Image")
    
    pil_image = Image.fromarray(img)
    inputs = self.image_processor.preprocess(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = self.depth_model.forward(inputs["pixel_values"])
    
    post_processed_output = self.image_processor.post_process_depth_estimation(
        outputs, source_sizes=[(pil_image.height, pil_image.width)]
    )
    
    absolute_depth = post_processed_output[0]["predicted_depth"]
    relative_depth = (absolute_depth - absolute_depth.min()) / (absolute_depth.max() - absolute_depth.min())
    
    absolute_depth = absolute_depth.numpy()
    relative_depth = (relative_depth.numpy() * 255).astype("uint8")
    
    return absolute_depth, relative_depth

def process_frame(frame: np.ndarray, depth_frame: np.ndarray, threshold_value: int = 185, black_percentage_threshold: float = 0.7, min_area: int = 10000) -> Dict:
    # Thresholding
    _, thresholded_image = cv2.threshold(depth_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Noise reduction based on black pixel thresholding
    for row_idx in range(thresholded_image.shape[0]):
        black_pixels = np.sum(thresholded_image[row_idx, :] == 255)  # Count white pixels
        total_pixels = thresholded_image.shape[1]
        black_percentage = black_pixels / total_pixels
        if black_percentage >= black_percentage_threshold:
            thresholded_image[row_idx, :] = 0  # Set the entire row to black

    # Contour detection
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # Initialize results and prepare the original image for visualization
    results = []

    for contour in filtered_contours:
        # Calculate moments
        moments = cv2.moments(contour)

        # Avoid division by zero for centroid calculation
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            # Minimum enclosing circle
            _, radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)

            # Append the results
            results.append({"centroid": (centroid_x, centroid_y), "radius": radius})

    return results