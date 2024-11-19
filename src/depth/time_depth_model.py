import os
import logging
import time

import torch
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from PIL import Image

# Set up logging with timestamps
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

model_path = "model/zoedepth-nyu-kitti"

# Log the model and processor loading
logger.info("Loading model and processor...")
image_processor = ZoeDepthImageProcessor.from_pretrained(model_path)
model = ZoeDepthForDepthEstimation.from_pretrained(model_path)
logger.info("Model and processor loaded successfully.")

# Function to process a single image and measure inference time
def process_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Prepare the image for the model
    inputs = image_processor.preprocess(images=image, return_tensors="pt")
    
    # Perform inference and time it
    start_time = time.time()
    with torch.no_grad():
        outputs = model.forward(inputs["pixel_values"])
    duration = time.time() - start_time
    
    # Post-process the depth estimation output
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        source_sizes=[(image.height, image.width)]
    )
    
    # Extract the predicted depth
    predicted_depth = post_processed_output[0]["predicted_depth"]
    
    # Normalize and convert depth to image format
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    with torch.no_grad():
        depth = depth.numpy()
    
    depth *= 255  # Scale to 0-255 range for visualization
    depth_image = Image.fromarray(depth.astype("uint8"))

    return duration

# Directory containing images
image_dir = "img/"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Measure the total duration for processing all images
durations = []
for image_path in image_files:
    logger.info(f"Processing image: {image_path}")
    duration = process_image(image_path)
    logger.info(f"Processing time: {duration:.4f} seconds")
    durations.append(duration)

# Calculate and print the average duration
average_duration = sum(durations) / len(durations) if durations else 0
logger.info(f"Average processing time: {average_duration:.4f} seconds")