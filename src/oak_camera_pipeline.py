# src/oak_camera_pipeline.py

import time
import cv2
import depthai as dai
import numpy as np
from threading import Thread


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


class OakCamera:
    def __init__(self, width=3840, height=2160, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.lens_pos = 135
        self.exp_time = 20000
        self.sens_iso = 800
        self.awb_lock = False
        self.sharpness = 0
        self.frame = None
        self.isNewFrame = False
        self.thread_active = True

        self.device = dai.Device()
        self.pipeline = dai.Pipeline()

        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setFps(fps)
        cam_rgb.initialControl.setManualFocus(self.lens_pos)
        cam_rgb.initialControl.setManualExposure(self.exp_time, self.sens_iso)
        cam_rgb.initialControl.setSharpness(self.sharpness)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("isp")
        cam_rgb.isp.link(xout.input)

        ctrl_in = self.pipeline.create(dai.node.XLinkIn)
        ctrl_in.setStreamName("control")
        ctrl_in.out.link(cam_rgb.inputControl)

        self.device.startPipeline(self.pipeline)
        self.output_queue = self.device.getOutputQueue(name="isp", maxSize=4, blocking=False)
        self.control_queue = self.device.getInputQueue("control")

        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.thread_active:
            frame_data = self.output_queue.tryGet()
            if frame_data:
                self.frame = frame_data.getCvFrame()
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
        self.device.close()
        cv2.destroyAllWindows()

    def cam_params_control(self, key):
        ctrl = dai.CameraControl()
        if key == ord(','):
            self.lens_pos = clamp(self.lens_pos - 3, 0, 255)
            ctrl.setManualFocus(self.lens_pos)
        elif key == ord('.'):
            self.lens_pos = clamp(self.lens_pos + 3, 0, 255)
            ctrl.setManualFocus(self.lens_pos)
        elif key == ord('i'):
            self.exp_time = clamp(self.exp_time - 500, 1, 33000)
            ctrl.setManualExposure(self.exp_time, self.sens_iso)
        elif key == ord('o'):
            self.exp_time = clamp(self.exp_time + 500, 1, 33000)
            ctrl.setManualExposure(self.exp_time, self.sens_iso)
        elif key == ord('h'):
            self.sens_iso = clamp(self.sens_iso - 50, 100, 1600)
            ctrl.setManualExposure(self.exp_time, self.sens_iso)
        elif key == ord('j'):
            self.sens_iso = clamp(self.sens_iso + 50, 100, 1600)
            ctrl.setManualExposure(self.exp_time, self.sens_iso)
        elif key == ord('9'):
            self.awb_lock = not self.awb_lock
            ctrl.setAutoWhiteBalanceLock(self.awb_lock)
        elif key == ord('0'):
            self.sharpness = clamp(self.sharpness + 1, 0, 4)
            ctrl.setSharpness(self.sharpness)

        self.control_queue.send(ctrl)
