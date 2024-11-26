import cv2
import numpy as np
import os

def process_depth_frame(depth_frame: np.ndarray, threshold_value: int = 120, percentage_threshold: float = 1.00, min_area: int = 20000):
    _, thresholded_image = cv2.threshold(depth_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    for row_idx in range(thresholded_image.shape[0]):
        black_pixels = np.sum(thresholded_image[row_idx, :] == 255)
        total_pixels = thresholded_image.shape[1]
        percentage = black_pixels / total_pixels
        if percentage >= percentage_threshold:
            thresholded_image[row_idx, :] = 0
            
    cv2.imshow("", thresholded_image)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    results = []

    for contour in filtered_contours:
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            _, radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)

            results.append({"centroid": (centroid_x, centroid_y), "radius": radius})

    return results

if __name__ == "__main__":
    logs = ["2024-11-25 17:18:12.654954", "2024-11-25 17:29:25.794466"]
    
    for log in logs:
        original_img_dir = f"img/original/{log}"
        depth_img_dir = f"img/depth/{log}"
        # frame = "frame-1120.png"

        for frame in os.listdir(original_img_dir):
    
    # log = logs[0]
    # original_img_dir = f"img/original/{log}"
    # depth_img_dir = f"img/depth/{log}"
    # frame = "frame-295.png"
    
            print(f"{original_img_dir}/{frame}")
            # Load the images
            original_image = cv2.imread(f"{original_img_dir}/{frame}", cv2.IMREAD_GRAYSCALE)
            depth_image = cv2.imread(f"{depth_img_dir}/{frame}", cv2.IMREAD_GRAYSCALE)

            # Process the depth image to get results
            depth_results = process_depth_frame(depth_image)

            # Map the results onto the original image
            original_image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

            for result in depth_results:
                centroid = result['centroid']
                radius = result['radius']

                # Draw a circle for each result
                cv2.circle(original_image_color, centroid, radius, (0, 255, 0), 2)  # Green circle
                cv2.circle(original_image_color, centroid, 5, (0, 0, 255), -1)  # Red centroid

            # Display the result
            cv2.imshow("Mapped Image", original_image_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
