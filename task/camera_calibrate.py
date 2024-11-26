from __future__ import annotations
import cv2
import numpy as np
from vector import Vector3D
import glob

def calibrate_camera(image_dir: str, pattern_size: tuple[int, int], square_size: float = 20.0):
    # Prepare object points with real-world scaling
    objp = [Vector3D(x * square_size, y * square_size, 0) for y in range(pattern_size[1]) for x in range(pattern_size[0])]

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Load images
    images = glob.glob(f"{image_dir}/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)

            # Use only x and y for 2D points
            imgpoints.append([corner.ravel() for corner in corners])

            # Draw and display corners
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Convert object and image points to numpy arrays for OpenCV
    objpoints_np = [np.array([p.to_ndarr() for p in objp], dtype=np.float32) for objp in objpoints]
    imgpoints_np = [np.array(imgp, dtype=np.float32) for imgp in imgpoints]

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_np, imgpoints_np, gray.shape[::-1], None, None
    )

    # Save calibration data
    np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    return camera_matrix, dist_coeffs


# Undistort images
def undistort_images(image_dir: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    images = glob.glob(f"{image_dir}/*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # Get the optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        # Undistort the image
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Crop the image to the valid region of interest
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        # Display and save the result
        cv2.imshow('Original Image', img)
        cv2.imshow('Undistorted Image', undistorted)
        cv2.waitKey(0)  # Display for a short duration

    cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    # Calibrate the camera
    camera_matrix, dist_coeffs = calibrate_camera("calibrate/img/2024-11-26_10-38-17", (9, 7))

    # Undistort the calibration images
    _ = undistort_images("calibrate/img/2024-11-26_10-38-17", camera_matrix, dist_coeffs)