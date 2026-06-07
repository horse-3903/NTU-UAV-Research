"""ZoeDepth model loading and per-frame depth estimation.

Runs inference every `frame_interval` frames (default 250) to stay within the
real-time control budget. Each depth snapshot is anchored to the running average
of the last 100 UWB position readings so the 3D obstacle map is correctly placed
in world space.
"""
import numpy as np
import cv2
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple, TYPE_CHECKING

from vector import Vector3D
from tellodrone.map_obstacle import process_image, update_obstacles, draw_obstacles

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone


def load_depth_model(self: "TelloDrone") -> None:
    """Load ZoeDepth weights and processor from the local model directory."""
    self.logger.info(f"Loading model from {self.model_name}")
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(self.model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(self.model_name)


def run_depth_model(self: "TelloDrone", manual: bool = False) -> None:
    """Run a full depth-estimation and obstacle-update cycle.

    Called from the video processing thread every `frame_interval` frames, or
    immediately when `manual=True` (triggered via the Pygame UI button).

    The current UWB position is averaged over the last 100 log entries to reduce
    noise before being used to anchor obstacle positions in world space.
    """
    if not (manual or self.cur_frame_idx % 250 == 0):
        return

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

    self.logger.info(f"Estimating depth of frame {cur_frame_idx}")
    absolute_depth, relative_depth = self.estimate_depth(img=cur_frame)

    self.logger.info("Processing obstacles from depth map")
    real_obstacles, pixel_obstacles = process_image(cur_frame, absolute_depth)
    real_obstacles = [(obs + cur_pos, radius) for obs, radius in real_obstacles]

    self.logger.info("Updating obstacle map")
    self.obstacles = update_obstacles(
        cur_obs=self.obstacles,
        new_obs=real_obstacles,
        threshold=0.5,
        x_bounds=self.x_bounds,
        y_bounds=self.y_bounds,
        z_bounds=self.z_bounds,
    )

    cv2.imwrite(f"img/original/{self.init_time}/frame-{cur_frame_idx}.png", cur_frame)
    annotated = draw_obstacles(cur_frame, real_obstacles, pixel_obstacles)
    cv2.imwrite(f"img/depth/{self.init_time}/frame-{cur_frame_idx}.png", relative_depth)
    cv2.imwrite(f"img/annotated/{self.init_time}/frame-{cur_frame_idx}.png", annotated)

    self.depth_model_run = True
    self.logger.info(f"Done with depth processing of frame {cur_frame_idx}")


def estimate_depth(self: "TelloDrone", img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run ZoeDepth on a single BGR frame.

    Args:
        img: BGR frame as a numpy array (H x W x 3).

    Returns:
        absolute_depth: Float32 depth map in metres (H x W).
        relative_depth: Uint8 normalised depth for visualisation (H x W, 0–255).
    """
    self.logger.info("Estimating depth for frame")

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
