import time

import cv2
import numpy as np
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import CompressedImage

from nodes.zed.zed_camera import ZedCamera


class CameraNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_pub = self.create_publisher(CompressedImage, "/geometry_msgs/image/compressed", 1)

        freq_hz = 20
        self.timer = self.create_timer(1 / freq_hz, self._timer_callback)
        self.img_msg = CompressedImage()
        self.img_msg.format = "jpeg"

        self.zed_camera = ZedCamera(resolution_mode="VGA", depth_mode="NEURAL")

    def _timer_callback(self):
        ret, img = self.zed_camera.capture_image()
        if ret:
            self.img_msg.header.stamp = self.manager.get_clock().now().to_msg()
            self.img_msg.data = np.array(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]).tobytes()
            self.img_pub.publish(self.img_msg)
        else:
            self.get_logger().warning("Failed to capture image from ZED camera.")


class ImageViewer(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/image/compressed", self.callback, 1
        )

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow("Camera Image", cv_image)
        cv2.waitKey(1)
