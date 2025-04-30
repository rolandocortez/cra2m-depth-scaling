# tools/calibrate_camera.py

import os
import cv2
import numpy as np
import argparse

from src.oak_camera_pipeline import OakCamera
from src.webcam_pipeline import WebcamCamera


CHECKERBOARD = (11, 8)  # default chessboard size


def capture_calibration_images(camera, output_dir, num_images=20):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    while count < num_images:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_display = cv2.resize(frame, (960, 540))
        cv2.imshow("Calibration Capture", frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_path = os.path.join(output_dir, f"calib_{count}.png")
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1
        elif key == ord('q'):
            break
        camera.cam_params_control(key)

    camera.stop()
    cv2.destroyAllWindows()


def calibrate_camera(calib_images_dir, cam_width, cam_height):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = [os.path.join(calib_images_dir, f) for f in os.listdir(calib_images_dir) if f.endswith(".png") or f.endswith(".jpg")]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    ret, cam_mtx, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (cam_width, cam_height), None, None)

    print("Camera Matrix:")
    print(cam_mtx)
    print("Distortion Coefficients:")
    print(dist_coeffs)

    return cam_mtx, dist_coeffs


def main(args):
    # Select camera
    if args.camera_type == "oak":
        camera = OakCamera()
    elif args.camera_type == "webcam":
        camera = WebcamCamera()
    else:
        raise ValueError("Unsupported camera type.")

    # Step 1: Capture images
    capture_calibration_images(camera, args.output_dir, num_images=args.num_images)

    # Step 2: Calibrate
    cam_mtx, dist_coeffs = calibrate_camera(args.output_dir, args.cam_width, args.cam_height)

    os.makedirs(args.result_dir, exist_ok=True)
    np.save(os.path.join(args.result_dir, "cam_mtx.npy"), cam_mtx)
    np.save(os.path.join(args.result_dir, "cam_dist.npy"), dist_coeffs)
    print(f"Calibration saved in {args.result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture calibration images and calibrate a camera.")

    parser.add_argument("--camera_type", type=str, choices=["oak", "webcam"], required=True, help="Type of camera to use.")
    parser.add_argument("--output_dir", type=str, default="data/calib_imgs", help="Folder to save calibration images.")
    parser.add_argument("--result_dir", type=str, default="data/calib_results", help="Folder to save calibration results.")
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to capture.")
    parser.add_argument("--cam_width", type=int, default=3840, help="Camera image width.")
    parser.add_argument("--cam_height", type=int, default=2160, help="Camera image height.")

    args = parser.parse_args()

    main(args)
