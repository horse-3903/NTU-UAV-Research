import numpy as np

import cv2
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple
from typing import TYPE_CHECKING

from vector import Vector3D
from tellodrone.map_obstacle import process_image, update_obstacles, draw_obstacles

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def load_depth_model(self: "TelloDrone") -> None:
    self.logger.info(f"Loading model from {self.model_name}")
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(self.model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)


def run_depth_model(self: "TelloDrone", manual: bool = False) -> None:
    if manual or self.cur_frame_idx % 200 == 0:
        self.logger.critical("Depth Model Running")
        cur_frame_idx = self.cur_frame_idx
        cur_frame = self.cur_frame
        
        with open(self.log_pos_file, "r") as f:
            data = f.read().splitlines()
            data = [Vector3D(*map(float, line.split()[3:])) for line in data]
            data = data[-100:]
            
        avg_x = sum(d.x for d in data) / len(data)
        avg_y = sum(d.y for d in data) / len(data)
        avg_z = sum(d.z for d in data) / len(data)
        
        cur_pos = Vector3D(avg_x, avg_y, avg_z)
        
        self.logger.info("Video frame captured")
        self.logger.info(f"Estimating depth of frame {self.cur_frame_idx}")
        
        absolute_depth, relative_depth = self.estimate_depth(img=cur_frame)
        
        self.logger.info("Processing Obstacles")
        real_obstacles, pixel_obstacles = process_image(cur_frame, absolute_depth, relative_depth)
        
        real_obstacles = [(obs + cur_pos, radius) for obs, radius in real_obstacles]
        
        self.logger.info("Updating Obstacles")
        self.obstacles = update_obstacles(cur_obs=self.obstacles, new_obs=real_obstacles, threshold=1.0, x_bounds=self.x_bounds, y_bounds=self.y_bounds, z_bounds=self.z_bounds)

        self.logger.info("Saving images")
        
        cv2.imwrite(f"img/original/{self.init_time}/frame-{cur_frame_idx}.png", cur_frame)
        
        annotated = draw_obstacles(cur_frame, real_obstacles, pixel_obstacles)
        
        cv2.imwrite(f"img/depth/{self.init_time}/frame-{cur_frame_idx}.png", relative_depth)
        cv2.imwrite(f"img/annotated/{self.init_time}/frame-{cur_frame_idx}.png", annotated)
        
        self.depth_model_run = True
        
        self.logger.info(f"Done with Depth Processing of Frame {cur_frame_idx}")


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