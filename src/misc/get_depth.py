import av
import tellopy
import pygame
import numpy as np
import cv2
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
import torch
from PIL import Image

# Initialize depth model and processor
model_name = "model/"
image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)

def estimate_depth(img):
    """Runs depth estimation on the input image and returns a depth map as a numpy array."""
    pil_image = Image.fromarray(img)  # Convert to PIL Image format
    inputs = image_processor.preprocess(images=pil_image, return_tensors="pt")
    
    # Perform depth estimation
    with torch.no_grad():
        outputs = depth_model(inputs["pixel_values"])
    
    # Post-process depth estimation
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, source_sizes=[(pil_image.height, pil_image.width)]
    )
    
    # Extract and normalize predicted depth
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth_image = (depth.numpy() * 255).astype("uint8")  # Convert to 8-bit grayscale for display
    
    return depth_image

def main():
    # Initialize Tello drone
    drone = tellopy.Tello()
    drone.connect()
    drone.wait_for_connection(60.0)
    drone.start_video()
    
    container = av.open(drone.get_video_stream())

    frame_count = 0
    running = True
    depth_image = None

    for frame in container.decode(video=0):
        frame_count += 1
        
        img = frame.to_ndarray(format='bgr24')
        
        if frame_count % 100 == 0:
            depth_image = estimate_depth(img)
            
            depth_pil = Image.fromarray(depth_image)
            depth_pil.save(f"test/{frame_count // 100}.png")
            print("Saved :", frame_count // 100)

        if not running:
            break

    # Clean up resources
    drone.land()
    drone.quit()
    # pygame.quit()

if __name__ == "__main__":
    main()