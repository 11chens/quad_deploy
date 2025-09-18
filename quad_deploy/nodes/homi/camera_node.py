import time

import cv2
import numpy as np
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import CompressedImage

from quad_deploy.nodes.camera.zed_mini.zed_camera import ZedCamera
from quad_deploy.nodes.camera.go2_front.go2_camera import Go2Camera

class CameraNode(BaseNode):
    def __init__(self, cam_type: str = "zed", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cam_type = cam_type

        self.img_pub = self.create_publisher(CompressedImage, "/geometry_msgs/image/compressed", 1)

        freq_hz = 20
        self.timer = self.create_timer(1 / freq_hz, self._timer_callback)
        self.img_msg = CompressedImage()
        self.img_msg.format = "jpeg"

        if self.cam_type == "zed":
            self.logger.info("Using Zed Mini Camera.")
            self.camera = ZedCamera(resolution_mode="VGA", depth_mode="NEURAL")
        elif self.cam_type == "go2":
            self.logger.info("Using Go2 Camera.")
            self.camera = Go2Camera()
        else:
            self.logger.error(f"Unsupported camera type: {self.cam_type}. Supported types are 'zed' and 'go2'.")
            raise ValueError(f"Unsupported camera type: {self.cam_type}")


    def _timer_callback(self):
        ret, img = self.camera.capture_image()
        if ret:
            # scale down the image
            img = cv2.resize(img, (1280 // 2, 720 // 2))
            self.img_msg.header.stamp = self.manager.get_clock().now().to_msg()
            self.img_msg.data = np.array(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]).tobytes()
            self.img_pub.publish(self.img_msg)
        else:
            self.logger.warning("Failed to capture image from camera.")
