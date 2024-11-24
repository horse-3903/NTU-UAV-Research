import os
import numpy as np

from PIL import Image

import torch
from torch import Tensor
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def run_depth_model(self: "TelloDrone", frame_img: np.ndarray) -> None:
    if self.frame_idx % 20 == 0:
        self.logger.info("Video frame captured")
        self.logger.info(f"Estimating depth of frame {self.frame_idx}")
        depth_image = estimate_depth(frame_img)
        
        orig_output_path = os.path.join(f"img/original/{self.init_time}", f"frame-{self.frame_idx}.png")
        depth_output_path = os.path.join(f"img/depth/{self.init_time}", f"frame-{self.frame_idx}.png")
        
        Image.fromarray(frame_img).save(orig_output_path)
        Image.fromarray(depth_image).save(depth_output_path)

def load_depth_model(self: "TelloDrone") -> None:
    model_name = "model/zoedepth-nyu-kitti"
    self.image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
    self.depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)

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