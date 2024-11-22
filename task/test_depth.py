import numpy as np
from PIL import Image
import cv2

depth_image = np.array(Image.open("img/depth/2024-11-21 17:26:15.508154/frame-140.png"))

# Apply Canny edge detection
edges = cv2.Canny(depth_image, threshold1=5, threshold2=15)

cv2.imshow('Canny', edges)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours and approximate to rectangles
rectangles = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
    rectangles.append(approx)
        
print(rectangles)

# Draw rectangles on the original image
for rect in rectangles:
    cv2.drawContours(depth_image, [rect], 0, (0, 255, 0), 2)

# Display the result
cv2.imshow('Rectangles on Depth Image', depth_image)
try:
    cv2.waitKey(0)
except:
    print("Ctrl+C")
finally:
    cv2.destroyAllWindows()