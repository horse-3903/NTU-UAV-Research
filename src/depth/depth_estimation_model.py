import logging
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set up logging with timestamp
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

model = "model/"

# Log the model and processor loading
logger.info("Loading model and processor...")
image_processor = ZoeDepthImageProcessor.from_pretrained(model)
model = ZoeDepthForDepthEstimation.from_pretrained(model)
logger.info("Model and processor loaded successfully.")

# Log the start of the image loading
logger.info("Loading image...")
image = Image.open("img/im0.png")
logger.info("Image loaded successfully.")

# Prepare the image for the model
logger.info(f"Preparing image for model...")
inputs = image_processor.preprocess(images=image, return_tensors="pt")
logger.info(f"Input tensor shape: {inputs['pixel_values'].shape}")

# Perform inference without gradient calculation
logger.info("Performing inference...")
with torch.no_grad():
    outputs = model.forward(inputs["pixel_values"])
logger.info("Model inference completed.")

# Post-process the depth estimation output
logger.info("Post-processing depth estimation output...")
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)],
)
logger.info("Post-processing completed.")

# Extract the predicted depth
predicted_depth = post_processed_output[0]["predicted_depth"]
logger.info(f"Predicted depth shape: {predicted_depth.shape}")

# Normalize and convert depth to image format
logger.info("Normalizing depth map...")
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

with torch.no_grad():
    depth = depth.numpy()
    
depth *= 25
depth = Image.fromarray(depth.astype("uint8"))

# Display the original and depth images
logger.info("Displaying original and depth images.")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')  # Hide axis for better visualization

axes[1].imshow(depth, cmap="plasma")
axes[1].set_title("Depth Estimation")
axes[1].axis('off')  # Hide axis for better visualization

plt.show()