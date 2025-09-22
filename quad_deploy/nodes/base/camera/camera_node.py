import time

import cv2
import numpy as np
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import CompressedImage


class CameraNode(BaseNode):
    def __init__(
        self,
        cam_type: str = "zed",
        scale: int = 4,
        freq_hz: int = 20,
        format: str = "jpeg",
        compression_rate: int = 80,
        initial_resolution: list = [1280, 720],
        *args,
        **kwargs,
    ):
        """Camera node to interface with different camera types and publish images.
        Args:
            cam_type (str): Type of camera to use ("zed" or "go2").
            scale (int): Scale factor for resizing the image.
                - scale = 1: original resolution (1280x720)
                - scale = 2: half resolution (640x360)
                - scale = 4: quarter resolution (320x180)
                - scale = 8: eighth resolution (160x90)
            freq_hz (int): Frequency to capture and publish images.
            format (str): Image format for publishing ("jpeg" or "png").
            compression_rate (int): Compression quality for the image (0-100).
        """
        super().__init__(*args, **kwargs)

        self.cam_type = cam_type
        self.initial_resolution = initial_resolution  # default resolution for both zed mini and go2 (w, h)
        self.compression_rate = compression_rate  # jpeg quality from 0 to 100
        self.img_pub = self.create_publisher(CompressedImage, "/geometry_msgs/image", 1)

        self.timer = self.create_timer(1 / freq_hz, self._timer_callback)
        self.img_msg = CompressedImage()
        self.img_msg.format = format

        if self.cam_type == "zed":
            self.logger.info("Using Zed Mini Camera.")
            from quad_deploy.nodes.base.camera.zed_mini.zed_camera import ZedCamera

            self.camera = ZedCamera(scale=scale, *args, **kwargs)
        elif self.cam_type == "go2":
            self.logger.info("Using Go2 Camera.")
            from quad_deploy.nodes.base.camera.go2_front.go2_camera import Go2Camera

            self.camera = Go2Camera(scale=scale, *args, **kwargs)
        else:
            self.logger.error(f"Unsupported camera type: {self.cam_type}. Supported types are 'zed' and 'go2'.")
            raise ValueError(f"Unsupported camera type: {self.cam_type}")

    def _timer_callback(self):
        start_capture_time = time.time()  # system time
        ret, img = self.camera.capture_image()  # img: numpy array, BGR format, has scaled to (1280//scale, 720//scale)
        if ret:
            # scale down the image
            self.img_msg.header.stamp.sec = int(start_capture_time)
            self.img_msg.header.stamp.nanosec = int((start_capture_time - int(start_capture_time)) * 1e9)
            _, jpeg_buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.compression_rate])
            self.img_msg.data = jpeg_buffer.tobytes()
            self.img_pub.publish(self.img_msg)  # publish bytes stream

        else:
            self.logger.warning("Failed to capture image from camera.")
