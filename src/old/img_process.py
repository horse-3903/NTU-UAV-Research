import cv2
import numpy as np

log = "2024-11-21 17:17:27.547719"
original_img_dir = f"img/original/{log}"
depth_img_dir = f"img/depth/{log}"
frame = "frame-120.png"

# Load the images
original_image = cv2.imread(f"{original_img_dir}/{frame}", cv2.IMREAD_GRAYSCALE)
depth_image = cv2.imread(f"{depth_img_dir}/{frame}", cv2.IMREAD_GRAYSCALE)

# Thresholding and Noise Reduction
threshold_value = 185
_, thresh_image = cv2.threshold(depth_image, threshold_value, 255, cv2.THRESH_BINARY_INV)  # Invert thresholding

# Black Pixel Thresholding (Adjust as needed)
black_percentage_threshold = 0.7
for i in range(thresh_image.shape[0]):
    black_pixels = np.sum(thresh_image[i, :] == 255)  # Count white pixels
    total_pixels = thresh_image.shape[1]
    black_percentage = black_pixels / total_pixels
    if black_percentage >= black_percentage_threshold:
        thresh_image[i, :] = 0  # Set the entire row to black

# Contour Detection and Filtering
contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area (adjust min_area as needed)
min_area = 10000
filtered_contours = []
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        filtered_contours.append(contour)

# Create a copy of the original image to draw contours, centroids, and circles
original_image_with_contours = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

# Loop through each filtered contour
for contour in filtered_contours:
    # Calculate the moments of the contour
    moment = cv2.moments(contour)
    
    # Calculate the centroid (center of mass)
    if moment["m00"] != 0:  # Avoid division by zero
        centroid_x = int(moment["m10"] / moment["m00"])
        centroid_y = int(moment["m01"] / moment["m00"])
        
        # Draw the centroid on the image (Red dot)
        cv2.circle(original_image_with_contours, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red color

        # Calculate the minimum enclosing circle's radius
        _, radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        
        # Draw the enclosing circle using the centroid as its center (Blue color)
        cv2.circle(original_image_with_contours, (centroid_x, centroid_y), radius, (255, 0, 0), 2)  # Blue circle

# Display the images
cv2.imshow('Depth Image', depth_image)
cv2.imshow('Thresholded Image', thresh_image)
cv2.imshow('Original Image with Contours and Centroids', original_image_with_contours)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()