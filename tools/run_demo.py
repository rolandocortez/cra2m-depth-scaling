import os
import cv2
import numpy as np
import argparse
import torch
import time

# Import necessary modules for CRA2M and utilities
from src.depth_scaler import CRA2MDepthScaler
from src.utils import invert_depth_map, normalize_depth_map, on_mouse_click, clicked_points

def main(args):
    # Step 1: Load camera calibration parameters (camera matrix and distortion coefficients)
    # These parameters are necessary and for solving the depth (can be used for correcting the camera image).
    camera_matrix = np.load(args.camera_matrix)  # Load camera matrix (intrinsics)
    dist_coeffs = np.load(args.dist_coeffs)  # Load distortion coefficients (corrects lens distortion)

    # Step 2: Initialize CRA2M Depth Scaler
    # CRA2MDepthScaler is the core logic to estimate depth from relative depth (MiDaS) and ArUco marker scaling.
    scaler = CRA2MDepthScaler(
        marker_size=args.marker_size,  # Size of the ArUco marker in meters
        camera_matrix=camera_matrix,   # Camera intrinsics for depth correction
        dist_coeffs=dist_coeffs,       # Camera distortion coefficients
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),  # Use GPU if available
        use_local_model=args.use_local_model  # Option to use a local MiDaS model if already downloaded
    )

    # Step 3: Read input image
    # Load the input image for processing
    image = cv2.imread(args.image_path)  # Read the image from the provided file path
    if image is None:  # Check if the image was loaded successfully
        raise FileNotFoundError(f"Image not found at {args.image_path}")
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR (OpenCV default) to RGB

    # Step 4: Run CRA2M Depth Estimator
    # This will generate the absolute depth map based on the input image
    depth_map_abs, scale, shift, distance_real, center_img, tvec = scaler.run(frame_rgb, mode=args.mode)
    depth_map_vis = normalize_depth_map(depth_map_abs)  # Normalize depth map for better visualization

    # Step 5: Save the output files
    os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    output_depth_path = os.path.join(args.output_dir, "depth_map_abs.npy")  # Path for absolute depth map output
    output_image_path = os.path.join(args.output_dir, "depth_map_vis.png")  # Path for the visualized depth map output

    # Save depth map as a numpy file and the visualized depth map as an image
    np.save(output_depth_path, depth_map_abs)
    cv2.imwrite(output_image_path, depth_map_vis)

    # Step 6: Set up display window for depth map visualization
    cv2.namedWindow("Depth Map Viewer", cv2.WINDOW_NORMAL)  # Create a resizable window
    window_width = int(frame_rgb.shape[1] * 0.25)  # Set window width as 25% of the image width
    window_height = int(frame_rgb.shape[0] * 0.25)  # Set window height as 25% of the image height
    cv2.resizeWindow("Depth Map Viewer", window_width, window_height)  # Resize the window

    # Step 7: Set the mouse callback function to handle clicks on the displayed image
    # When the user clicks on the depth map, it will display the depth at that point
    cv2.setMouseCallback("Depth Map Viewer", on_mouse_click, param=depth_map_abs)  # Register mouse click callback

    # Inform the user about the saved outputs
    print(f"Saved absolute depth map to {output_depth_path}")
    print(f"Saved visualized depth map to {output_image_path}")

    # Step 8: Handle mouse clicks and display the updated depth map with overlays
    while True:
        display_map = depth_map_vis.copy()  # Copy the normalized depth map for display
        print("Valores actualizados")  # Print depth in terminal for the clicked points
        if clicked_points:  # If there are any points clicked by the user
            for x, y, value in clicked_points:  # Iterate over all clicked points
                if 0 <= x < depth_map_abs.shape[1] and 0 <= y < depth_map_abs.shape[0]:
                    # Draw a red circle at the clicked point and display the depth value
                    cv2.circle(display_map, (x, y), 5, (0, 0, 255), -1)  # Red circle with a larger radius
                    cv2.putText(display_map, f"{value:.2f} m", (x + 10, y - 10),  # Display depth value next to the point
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)  # Larger font size and white color
                    print(f"[INFO] Depth at ({x}, {y}) = {value:.3f} m")  # Print the depth in the terminal

        # Step 9: Display the updated depth map with the clicked points and depth values
        cv2.imshow("Depth Map Viewer", display_map)  # Show the depth map with annotations
        time.sleep(3)  # Delay between frames for better visualization

        # Step 10: Exit the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop and stop the program if 'q' is pressed

    # Step 11: Clean up by closing all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Step 1: Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Run CRA2M Depth Scaling on an image.")

    # Step 2: Define all the arguments to be passed from the command line
    parser.add_argument("--image_path", type=str, required=True, help="Path to input RGB image.")
    parser.add_argument("--camera_matrix", type=str, required=True, help="Path to camera matrix .npy file.")
    parser.add_argument("--dist_coeffs", type=str, required=True, help="Path to distortion coefficients .npy file.")
    parser.add_argument("--marker_size", type=float, default=0.05, help="Marker size in meters (default: 5cm).")
    parser.add_argument("--mode", type=str, choices=["scale", "scale-shift"], default="scale", help="Scaling mode to use.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to average (default: 1).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--use_local_model", action="store_true", help="Use MiDaS model from local ./models folder instead of downloading.")

    # Step 3: Parse the command-line arguments
    args = parser.parse_args()

    # Step 4: Run the main function with the parsed arguments
    main(args)
