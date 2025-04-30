# src/webcam_pipeline.py

import time
import cv2
import numpy as np
from threading import Thread


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


class WebcamCamera:
    def __init__(self, src=0, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.src = src
        self.frame = None
        self.isNewFrame = False
        self.thread_active = True

        self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.thread_active:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    self.frame = frame
                    self.isNewFrame = True
            time.sleep(1 / self.fps)

    def get_frame(self):
        if self.isNewFrame:
            self.isNewFrame = False
            return self.frame.copy()
        return None

    def stop(self):
        self.thread_active = False
        self.thread.join()
        self.capture.release()
        cv2.destroyAllWindows()

    def cam_params_control(self, key):
        pass  # OpenCV VideoCapture does not easily allow manual ISO/exposure control from keyboard
