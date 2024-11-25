import os
import numpy as np

import cv2
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def load_depth_model(self: "TelloDrone") -> None:
    self.logger.info(f"Loading model from {self.model_name}")
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(self.model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)

    
def run_depth_model(self: "TelloDrone", frame_img: np.ndarray) -> None:
    if self.frame_idx % 20 == 0:
        print("Depth Model Running...")
        cur_frame_idx = self.frame_idx
        self.logger.info("Video frame captured")
        self.logger.info(f"Estimating depth of frame {self.frame_idx}")
        absolute_depth, relative_depth = self.estimate_depth(img=frame_img)
        
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        
        orig_output_path = os.path.join(f"img/original/{self.init_time}", f"frame-{cur_frame_idx}.png")
        depth_output_path = os.path.join(f"img/depth/{self.init_time}", f"frame-{cur_frame_idx}.png")
        
        Image.fromarray(frame_img_rgb).save(orig_output_path)
        
        relative_depth_rgb = cv2.cvtColor(relative_depth, cv2.COLOR_GRAY2RGB)
        Image.fromarray(relative_depth_rgb).save(depth_output_path)
        
        results = process_depth_frame(relative_depth)
        
        for result in results:
            centroid = result["centroid"]
            radius = result["radius"]
            
            cv2.circle(frame_img, centroid, radius, (0, 255, 0), 2)
            
            cv2.circle(frame_img, centroid, 5, (0, 0, 255), -1)
        
        annotated_frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        
        annotated_output_path = os.path.join(f"img/annotated/{self.init_time}", f"frame-{cur_frame_idx}.png")
        Image.fromarray(annotated_frame_rgb).save(annotated_output_path)


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


def process_depth_frame(depth_frame: np.ndarray, threshold_value: int = 85, black_percentage_threshold: float = 0.95, min_area: int = 20000) -> List:
    _, thresholded_image = cv2.threshold(depth_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    for row_idx in range(thresholded_image.shape[0]):
        black_pixels = np.sum(thresholded_image[row_idx, :] == 255)
        total_pixels = thresholded_image.shape[1]
        black_percentage = black_pixels / total_pixels
        if black_percentage >= black_percentage_threshold:
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

            results.append({"centroid": (centroid_x, centroid_y), "radius": radius})

    return results