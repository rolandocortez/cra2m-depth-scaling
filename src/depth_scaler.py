# src/depth_scaler.py

import cv2
import numpy as np
import torch
import os

class CRA2MDepthScaler:
    def __init__(self, marker_size, camera_matrix, dist_coeffs, device=None, use_local_model=False):
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        if use_local_model:
            expected_path = os.path.join("models", "hub", "checkpoints", "dpt_large_384.pt")
            if not os.path.exists(expected_path):
                raise FileNotFoundError(
                    f"Local model not found at {expected_path}. Please download it manually or run without --use_local_model to download automatically."
                )
            os.environ["TORCH_HOME"] = os.path.abspath("models")
        # Load MiDaS model
        self.model_type = "DPT_Large"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # Load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform if self.model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

        # Set up ArUco dictionary and detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def estimate_depth_map(self, frame_rgb):
        input_batch = self.transform(frame_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def detect_aruco_marker(self, frame_gray):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame_gray)
        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            center_img = np.mean(marker_corners, axis=0)
            return marker_corners, ids[0], center_img
        else:
            return None, None, None

    def solvepnp_pose(self, marker_corners):
        object_points = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ], dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners,
            self.camera_matrix,
            self.dist_coeffs
        )
        if ret:
            return rvec, tvec
        else:
            return None, None

    def compute_scale_factor(self, distance_real, depth_map, center_img):
        x, y = int(center_img[0]), int(center_img[1])
        depth_rel = depth_map[y, x]
        return distance_real / (depth_rel + 1e-8)

    def compute_scale_and_shift(self, depth_map, pixel_pts, real_dists):
        (x1, y1), (x2, y2) = pixel_pts
        d1 = depth_map[y1, x1]
        d2 = depth_map[y2, x2]
        z1 = real_dists[0]
        z2 = real_dists[1]

        A = np.array([[d1, 1], [d2, 1]], dtype=np.float64)
        b = np.array([z1, z2], dtype=np.float64)

        scale, shift = np.linalg.lstsq(A, b, rcond=None)[0]
        return scale, shift

    def apply_scale(self, depth_map, scale_factor):
        return depth_map * scale_factor

    def apply_scale_and_shift(self, depth_map, scale, shift):
        return depth_map * scale + shift

    def run(self, frame_rgb, mode="scale"):
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        depth_map_rel = self.estimate_depth_map(frame_rgb)

        marker_corners, marker_id, center_img = self.detect_aruco_marker(frame_gray)
        if marker_corners is None:
            raise ValueError("No ArUco marker detected!")

        rvec, tvec = self.solvepnp_pose(marker_corners)
        if rvec is None or tvec is None:
            raise ValueError("solvePnP failed to compute pose!")

        depth_map_inv = 1 / (depth_map_rel + 1e-8)

        if mode == "scale":
            distance_real = np.linalg.norm(tvec)
            scale_factor = self.compute_scale_factor(distance_real, depth_map_inv, center_img)
            depth_map_abs = self.apply_scale(depth_map_inv, scale_factor)
            return depth_map_abs, scale_factor, None, distance_real, center_img, tvec

        elif mode == "scale-shift":
            R, _ = cv2.Rodrigues(rvec)
            object_points = np.array([
                [-self.marker_size / 2,  self.marker_size / 2, 0],
                [ self.marker_size / 2,  self.marker_size / 2, 0],
                [ self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0]
            ], dtype=np.float32)

            corner1_local = object_points[0]
            corner3_local = object_points[2]

            corner1_camera = np.dot(R, corner1_local.reshape(3,1)) + tvec
            corner3_camera = np.dot(R, corner3_local.reshape(3,1)) + tvec

            z1 = corner1_camera[2][0]
            z3 = corner3_camera[2][0]

            p1 = tuple(map(int, marker_corners[0]))
            p2 = tuple(map(int, marker_corners[2]))

            scale, shift = self.compute_scale_and_shift(depth_map_inv, [p1, p2], [z1, z3])
            depth_map_abs = self.apply_scale_and_shift(depth_map_inv, scale, shift)
            return depth_map_abs, scale, shift, np.linalg.norm(tvec), center_img, tvec

        else:
            raise ValueError("Invalid mode. Use 'scale' or 'scale-shift'.")
