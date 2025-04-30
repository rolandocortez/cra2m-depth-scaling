# tools/capture_image.py

import os
import cv2
import argparse
from src.oak_camera_pipeline import OakCamera
from src.webcam_pipeline import WebcamCamera

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Select camera
    if args.camera_type == "oak":
        cam = OakCamera(width=args.cam_width, height=args.cam_height, fps=args.fps)
    elif args.camera_type == "webcam":
        cam = WebcamCamera(src=0, width=args.cam_width, height=args.cam_height, fps=args.fps)
    else:
        raise ValueError("Unsupported camera type. Use 'oak' or 'webcam'.")

    print("[INFO] Press 'c' to capture | Press 'q' to quit")

    img_index = 0
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        if args.flip:
            frame = cv2.flip(frame, -1)

        cv2.imshow("Capture Preview", cv2.resize(frame, (960, 540)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save image
            filename = f"capture_{img_index}.png"
            filepath = os.path.join(args.output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"[INFO] Image saved: {filepath}")
            img_index += 1
        elif key == ord('q'):
            break
        else:
            cam.cam_params_control(key)  # only active for OakCamera

    cam.stop_cam()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images using OAK-1 or webcam.")
    parser.add_argument("--camera_type", type=str, choices=["oak", "webcam"], required=True, help="Select camera: 'oak' or 'webcam'")
    parser.add_argument("--output_dir", type=str, default="example_images", help="Directory to save captured images")
    parser.add_argument("--cam_width", type=int, default=3840, help="Camera width (default: 3840)")
    parser.add_argument("--cam_height", type=int, default=2160, help="Camera height (default: 2160)")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--flip", action="store_true", help="Flip the camera vertically if it's mounted upside-down")
    args = parser.parse_args()
    main(args)
