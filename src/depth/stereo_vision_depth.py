import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Log image loading
logger.info("Loading the left and right images in grayscale...")
imgL = cv2.imread('img/im0.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('img/im1.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if imgL is None or imgR is None:
    logger.error("One or both of the images did not load. Check the file paths.")
else:
    logger.info("Images loaded successfully.")

# Log creation of StereoBM object
logger.info("Creating the StereoBM object...")
stereo = cv2.StereoBM_create(numDisparities=16*17, blockSize=19)

# Log computation of disparity map
logger.info("Computing the disparity map...")
disparity = stereo.compute(imgL, imgR)

# Log the shape of the disparity map
logger.info(f"Disparity map computed. Shape: {disparity.shape}")

# Set up a single plot for the disparity map
logger.info("Setting up the plot for the disparity map...")
plt.figure(figsize=(12, 8))
plt.imshow(disparity, cmap='coolwarm')

# Log before showing the plot
logger.info("Displaying the disparity map...")

# Show the plot
plt.show()

# Log completion
logger.info("Disparity map displayed successfully.")