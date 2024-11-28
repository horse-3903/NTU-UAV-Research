import os
import numpy as np

import cv2
from PIL import Image

from tellodrone.map_obstacle import find_obstacles

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple
from typing import TYPE_CHECKING

from tellodrone.map_obstacle import process_obstacles, update_obstacles, draw_obstacles, find_checkerboard_position

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def load_depth_model(self: "TelloDrone") -> None:
    self.logger.info(f"Loading model from {self.model_name}")
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(self.model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)

# vid_task
def run_depth_model(self: "TelloDrone", manual: bool = False) -> None:
    # (not manual and self.frame_idx % 150 == 0)
    if manual:
        self.logger.info("Depth Model Running")
        cur_frame_idx = self.frame_idx
        cur_frame = self.cur_frame
        
        self.logger.info("Video frame captured")
        self.logger.info(f"Estimating depth of frame {self.frame_idx}")
        
        absolute_depth, relative_depth = self.estimate_depth(img=cur_frame)
        
        self.logger.info("Processing Obstacles")
        real_obstacles, pixel_obstacles = process_obstacles(cur_frame, absolute_depth, relative_depth)
        real_obstacles = [(obs + self.cur_pos, radius) for obs, radius in real_obstacles]
        
        self.logger.info("Updating Obstacles")
        # self.obstacles =
        test_obstacles = update_obstacles(self.obstacles, real_obstacles)

        # self.logger.info("Finding checkerboard location")
        # annotated, pos_3d = find_checkerboard_position(image=cur_frame, absolute_depth=absolute_depth, checkerboard_size=(9, 7))
        # self.logger.info(f"Checkerboard at position : {pos_3d}")

        self.logger.info("Saving images")
        
        cv2.imwrite(f"img/manual/{self.init_time}/frame-{cur_frame_idx}.png", cur_frame)
        
        annotated = draw_obstacles(cur_frame, real_obstacles, pixel_obstacles)
        cv2.imwrite(f"img/depth/{self.init_time}/frame-{cur_frame_idx}.png", relative_depth)
        cv2.imwrite(f"img/annotated/{self.init_time}/frame-{cur_frame_idx}.png", annotated)
        
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