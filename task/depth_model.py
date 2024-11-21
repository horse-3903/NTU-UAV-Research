import os
import numpy as np
from PIL import Image
import av
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
import torch

# Initialize depth model
model_name = "model/zoedepth-nyu-kitti"
image_processor = ZoeDepthImageProcessor.from_pretrained(model_name)
depth_model = ZoeDepthForDepthEstimation.from_pretrained(model_name)

# Depth estimation function
def estimate_depth(img: np.ndarray):
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

def main():
    # Directory paths
    video_dir = "vid"
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    
    # List and sort videos in the directory
    videos = sorted(os.listdir(video_dir))
    if not videos:
        print("No videos found in the directory.")
        return
    
    # Open the last video in the sorted list
    last_video_path = os.path.join(video_dir, videos[-1])
    print(f"Processing video: {last_video_path}")
    
    container = av.open(last_video_path)
    stream = container.streams.video[0]
    
    frame_count = 0
    n = 30  # Process every nth frame
    
    for frame in container.decode(stream):
        if frame_count % n == 0:
            # Convert frame to numpy array (RGB format)
            frame_rgb = frame.to_image().convert("RGB")
            frame_np = np.array(frame_rgb)
            
            # Estimate depth
            depth_image = estimate_depth(frame_np)
            
            # Save depth image
            output_path = os.path.join(output_dir, f"frame-{frame_count}.png")
            Image.fromarray(depth_image).save(output_path)
            print(f"Saved depth image: {output_path}")
        
        frame_count += 1

if __name__ == "__main__":
    main()
