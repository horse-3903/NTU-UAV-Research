import numpy as np

from PIL import Image
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

import torch

model_name = "model/"
image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)

def estimate_depth(img: np.ndarray):
    """Runs depth estimation on the input image and returns a depth map as a numpy array."""
    pil_image = Image.fromarray(img)
    inputs = image_processor.preprocess(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = depth_model(inputs["pixel_values"])
    
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, source_sizes=[(pil_image.height, pil_image.width)]
    )
    
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth_image = (depth.numpy() * 255).astype("uint8")
    
    return depth_image

if __name__ == "__main__":
    original_img = Image.open("img/im0.png")
    depth_img = estimate_depth(np.array(original_img))
    img = Image.fromarray(depth_img)
    img.show("")