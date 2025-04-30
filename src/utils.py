# src/utils.py

import numpy as np
import cv2

# --- Mouse click handler for depth value inspection ---
clicked_points = []  # List to store multiple clicked points

def on_mouse_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map_abs = param  # Get the depth map from 'param' (userdata)
        if depth_map_abs is not None:
            if 0 <= x < depth_map_abs.shape[1] and 0 <= y < depth_map_abs.shape[0]:
                value = depth_map_abs[y, x]
                print(f"[INFO] Clicked at ({x}, {y}) → Depth: {value:.3f} m")
                # Store the clicked point and its depth value
                clicked_points.append((x, y, value))  # Append to the list of clicked points




def calculate_bbox_center(x_min, y_min, x_max, y_max):
    """Calculate the center point of a bounding box."""
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    return center_x, center_y


def depth_to_xyz(center_x, center_y, depth_value, camera_matrix):
    """Convert pixel coordinates and depth to 3D coordinates in camera space."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (center_x - cx) * depth_value / fx
    y = (center_y - cy) * depth_value / fy
    z = depth_value

    return x, y, z


def invert_depth_map(depth_map):
    """Invert a depth map safely to avoid division by zero."""
    return 1 / (depth_map + 1e-8)


def normalize_depth_map(depth_map):
    """Normalize a depth map to 0-255 range for visualization."""
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return depth_map_normalized.astype(np.uint8)
