import os
import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager
from ros_base.node.base_node import BaseNode

from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent as HomiLocoAgent
from quad_deploy.utils.logger import CustomLogger
from quad_deploy.utils.parse_args import parse_arguments
from sensor_msgs.msg import CompressedImage
import cv2
from quad_deploy.utils.math_utils import CircularBuffer

class ImageViewer(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/image/compressed", self.image_callback, 1
        )

        self.subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/mask", self.mask_callback, 1
        )
        self.cv_image = None
        self.cv_image_hist = CircularBuffer(10)  # append image for buffer, 20 Hz
        self.green_image = None  # 5 Hz

    def mask_callback(self, msg):
        # msg.data is consised of 1 or 0
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # cv2.imshow("Mask Image", cv_image*255)
        # cv2.waitKey(1)

        # Draw mask
        if self.cv_image is None:
            return
        self.green_image = np.zeros_like(self.cv_image)
        mask_np_arr = np.frombuffer(msg.data, np.uint8)
        cv_mask = cv2.imdecode(mask_np_arr, cv2.IMREAD_COLOR)

        self.green_image[:, :, :] = (0, 185, 118)
        self.green_image[cv_mask == 0] = 0

        delay_cv_image = self.cv_image_hist.buffer[-4] if  self.cv_image_hist.buffer is not None else self.cv_image
        image = cv2.addWeighted(delay_cv_image, 0.4, self.green_image, 0.6, 0)
        cv2.imshow("Mixed Image", image)
        cv2.waitKey(1)

    def image_callback(self, msg):
        # msg.data is between 0 and 255
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cv_image_hist.append(self.cv_image)


class ViewerManager(BaseManager):
    def __init__(
        self,
        node_name="ViewerManager",
        *args,
        **kwargs,
    ):
        super().__init__(node_name=node_name, *args, **kwargs)


def main(args=None):
    nodes_dict = {
        "viewer": ImageViewer,
    }

    rclpy.init()

    homi_robot_node = ViewerManager(
        # ros_base args
        nodes_dict=nodes_dict,
        custom_logger=CustomLogger,
    )

    homi_robot_node.start_main_loop()


if __name__ == "__main__":
    args = parse_arguments()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 8888)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args=args)
