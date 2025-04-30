import os
import cv2
import numpy as np
import argparse
import torch
import time

from src.depth_scaler import CRA2MDepthScaler
from src.utils import invert_depth_map, normalize_depth_map, on_mouse_click, clicked_points
from src.oak_camera_pipeline import OakCamera
from src.webcam_pipeline import WebcamCamera

# Default resolution settings
width_res = 3840
height_res = 2160

def main(args):
    # Step 1: Load Camera Calibration Parameters (Camera Matrix and Distortion Coefficients)
    camera_matrix = np.load(args.camera_matrix)  # Load the camera matrix (intrinsics)
    dist_coeffs = np.load(args.dist_coeffs)  # Load distortion coefficients (camera lens distortion)

    # Step 2: Initialize the CRA2MDepthScaler for depth scaling using the loaded calibration parameters
    scaler = CRA2MDepthScaler(
        marker_size=args.marker_size,  # Size of the ArUco marker in meters
        camera_matrix=camera_matrix,   # Camera intrinsics
        dist_coeffs=dist_coeffs,       # Distortion coefficients
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),  # Use GPU if available
        use_local_model=args.use_local_model  # Option to use a local MiDaS model instead of downloading it
    )

    # Step 3: Initialize the camera based on user input (either OakCamera or WebcamCamera)
    if args.camera_type == "oak":
        camera = OakCamera()  # Initialize DepthAI OAK-1 camera
    elif args.camera_type == "webcam":
        camera = WebcamCamera()  # Initialize standard webcam camera
    else:
        raise ValueError("Unsupported camera type. Please choose either 'oak' or 'webcam'.")

    # Step 4: Configure the display window for depth map visualization
    cv2.namedWindow("Live Depth Estimation", cv2.WINDOW_NORMAL)
    window_width = int(width_res * 0.25)  # Set window size as 25% of the original image resolution
    window_height = int(height_res * 0.25)
    cv2.resizeWindow("Live Depth Estimation", window_width, window_height)

    try:
        # Step 5: Main loop to continuously capture frames from the camera
        while True:
            # Capture a single frame from the camera
            frame = camera.get_frame()
            if frame is None:  # Skip if no frame is captured
                continue

            # Optional step to flip the camera image if it's mounted upside-down
            if args.flip:
                frame = cv2.flip(frame, -1)  # Flip vertically

            # Convert the frame from BGR to RGB (needed for MiDaS depth estimation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 6: Run CRA2M Depth Estimator to get depth map from RGB image
            depth_map_abs, scale, shift, distance_real, center_img, tvec = scaler.run(frame_rgb, mode=args.mode)
            depth_map_vis = normalize_depth_map(depth_map_abs)  # Normalize depth map for visualization

            # Step 7: Set mouse click callback for inspecting depth at clicked points
            cv2.setMouseCallback("Live Depth Estimation", on_mouse_click, param=depth_map_abs)

            # Step 8: Handle mouse clicks to display the depth value at clicked points
            for x, y, value in clicked_points:  # Iterate through all clicked points
                if 0 <= x < depth_map_abs.shape[1] and 0 <= y < depth_map_abs.shape[0]:
                    # Draw red circle at the clicked point and display depth value
                    cv2.circle(depth_map_vis, (x, y), 20, (0, 0, 255), -1)  # Red circle with larger size
                    cv2.putText(depth_map_vis, f"{value:.2f} m", (x + 20, y - 20),  # Larger font size and offset
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

            # Step 9: Display the depth map with the overlay of clicked points
            cv2.imshow("Live Depth Estimation", depth_map_vis)
            # Print current scale, shift, and real-world distance
            print(f"Scale: {scale:.6f} | Shift: {shift if shift is not None else 'N/A'} | Distance Real: {distance_real:.4f} m")

            # Step 10: Check for user input (press 'q' to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break  # Exit the loop if 'q' is pressed
            camera.cam_params_control(key)  # Control camera parameters (focal length, exposure, etc.)

    finally:
        # Cleanup: Stop the camera and close any open windows
        camera.stop()

if __name__ == "__main__":
    # Step 1: Set up argument parser to accept user input from command line
    parser = argparse.ArgumentParser(description="Run CRA2M live demo with OAK-1 or Webcam.")
    
    # Define command line arguments for camera type, calibration files, and other settings
    parser.add_argument("--camera_type", type=str, choices=["oak", "webcam"], required=True, help="Type of camera to use.")
    parser.add_argument("--camera_matrix", type=str, required=True, help="Path to camera matrix .npy file.")
    parser.add_argument("--dist_coeffs", type=str, required=True, help="Path to distortion coefficients .npy file.")
    parser.add_argument("--marker_size", type=float, default=0.05, help="Marker size in meters (default: 5cm).")
    parser.add_argument("--mode", type=str, choices=["scale", "scale-shift"], default="scale", help="Scaling mode to use.")
    parser.add_argument("--use_local_model", action="store_true", help="Use MiDaS model from local ./models folder instead of downloading.")
    parser.add_argument("--flip", action="store_true", help="Flip the camera vertically if it's mounted upside-down.")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
