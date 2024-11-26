from __future__ import annotations
import cv2
import numpy as np
from vector import Vector3D
import glob

# Camera Calibration
def calibrate_camera(image_dir: str, pattern_size: tuple[int, int]):
    """
    Calibrate the camera using chessboard images.
    
    Args:
        image_dir: Directory containing chessboard images.
        pattern_size: The number of inner corners per a chessboard row and column (e.g., (7, 6)).

    Returns:
        camera_matrix, dist_coeffs: The intrinsic camera matrix and distortion coefficients.
    """
    # Prepare object points
    objp = [Vector3D(x, y, 0) for y in range(pattern_size[1]) for x in range(pattern_size[0])]

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
            imgpoints.append([Vector3D(c[0][0], c[0][1], 0) for c in corners])

            # Draw and display corners
            cv2.drawChessboardCorners(img, pattern_size, np.array([c.to_ndarr() for c in corners]), ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Convert object and image points to numpy arrays for OpenCV
    objpoints_np = [np.array([p.to_ndarr() for p in objp], dtype=np.float32) for objp in objpoints]
    imgpoints_np = [np.array([p.to_ndarr() for p in imgp], dtype=np.float32) for imgp in imgpoints]

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints_np, imgpoints_np, gray.shape[::-1], None, None
    )

    # Save calibration data
    np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    return camera_matrix, dist_coeffs


# Depth Estimation and 3D Mapping
def pixel_to_3d(pixel: Vector3D, depth: float, camera_matrix):
    """
    Convert a pixel position and depth to a 3D point in camera space.
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Compute normalized camera coordinates
    x = (pixel.x - cx) / fx
    y = (pixel.y - cy) / fy

    # Scale by depth
    return Vector3D(x * depth, y * depth, depth)


def project_points(points: list[Vector3D], camera_matrix, dist_coeffs, rvec, tvec):
    """
    Project 3D points into 2D image space using the camera model.
    """
    # Convert points to numpy arrays
    np_points = np.array([p.to_ndarr() for p in points], dtype=np.float32)

    # Use OpenCV's projectPoints
    projected_points, _ = cv2.projectPoints(np_points, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert back to Vector3D
    return [Vector3D.from_arr(p[0]) for p in projected_points]


# Tello Integration
def run_tello_depth_model_with_calibration(tello, frame):
    """
    Example function for Tello depth estimation integrated with calibration.
    """
    # Load calibration data
    data = np.load('calibration_data.npz')
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # Get depth map
    depth_map = tello.depth_model(frame)

    # Example: Map the center pixel to 3D
    height, width = frame.shape[:2]
    center_pixel = Vector3D(width // 2, height // 2, 0)
    depth = depth_map[height // 2, width // 2]

    # Convert pixel to 3D
    center_3d = pixel_to_3d(center_pixel, depth, camera_matrix)
    print(f"Center pixel mapped to 3D: {center_3d}")

    return center_3d


# Main Function
if __name__ == "__main__":
    from tellodrone import TelloDrone

    tello = TelloDrone()

    try:
        # Calibrate the camera (once, before running the model)
        camera_matrix, dist_coeffs = calibrate_camera("calibrate/img", (7, 6))

        # Start the Tello video stream and run the depth model
        tello.active_vid_task = lambda frame: run_tello_depth_model_with_calibration(tello, frame)
        tello.startup_video()

    except Exception as e:
        tello.shutdown(error=True, reason=e)
    else:
        tello.shutdown(error=False)
    finally:
        tello.shutdown(error=False)