import numpy as np
from PIL import Image

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from matplotlib import pyplot as plt

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
    img = Image.open("img/original/2024-11-25 17:18:12.654954/frame-1111.png")
    absolute_depth, relative_depth = estimate_depth(np.array(img))
    plt.imshow(absolute_depth, cmap="viridis")
    plt.colorbar(label="Depth (arbitrary units)")
    plt.title("Absolute Depth Map")
    plt.axis("off")
    plt.show()