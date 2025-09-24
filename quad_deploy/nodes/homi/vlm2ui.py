import time

import cv2
import numpy as np
from ros_base.nodes.base_node import BaseNode
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from quad_deploy.utils.math_utils import CircularBuffer


class VLM2UIBridge(BaseNode):
    def __init__(self, show_raw_image=False, *args, **kwargs):
        """Node to bridge between VLM outputs and UI inputs/outputs.
        Args:
            show_raw_image (bool): Whether to display the raw camera image in a separate window.
        """
        super().__init__(*args, **kwargs)

        self.show_raw_image = show_raw_image
        self.image_subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/image", self.image_callback, 1
        )

        self.mask_subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/mask", self._mask_callback, 1
        )

        self.cv_image = None
        self.cv_image_hist = CircularBuffer(10)  # append image for buffer, 20 Hz, 50 ms
        self.image_timestamp_hist = CircularBuffer(10)  # append timestamp for buffer, 20 Hz, 50 ms
        self.green_image = None  # ~10 Hz, 110 ms

        self.turn_pub = self.create_publisher(String, "/control/turn", 1)
        self.ui_ready_pub = self.create_publisher(Bool, "/control/ui_ready", 1)
        self.inquiry_sub = self.create_subscription(Bool, "/control/inquiry", self._inquiry_callback, 1)

        self.ui_ready_msg = Bool()
        self.ui_ready = False
        self.turn_msg = String()
        self.inquiry = False
        self.turn = ""

    def publish_ui_ready(self, ui_ready: bool):
        if ui_ready != self.ui_ready:
            self.ui_ready = ui_ready
            self.ui_ready_msg.data = self.ui_ready
            self.ui_ready_pub.publish(self.ui_ready_msg)
            self.logger.info(f"""[Pub] ui_ready: {self.ui_ready}.""")

    def publish_turn(self, turn: str):
        self.turn = turn
        self.turn_msg.data = self.turn
        self.turn_pub.publish(self.turn_msg)
        self.logger.info(f"""[Pub] turn: {self.turn}.""")

    def _inquiry_callback(self, msg: Bool):
        inquiry = msg.data
        self.inquiry = inquiry
        self.logger.info(f"""[Sub] inquiry: {inquiry}.""")

    def _mask_callback(self, msg):
        # Draw mask
        if self.cv_image is None:
            return
        self.green_image = np.zeros_like(self.cv_image)
        mask_np_arr = np.frombuffer(msg.data, np.uint8)
        cv_mask = cv2.imdecode(mask_np_arr, cv2.IMREAD_COLOR)

        self.green_image[:, :, :] = (0, 185, 118)
        self.green_image[cv_mask == 0] = 0

        delay_cv_image = self.cv_image_hist.buffer[-2] if self.cv_image_hist.buffer is not None else self.cv_image
        image = cv2.addWeighted(delay_cv_image, 0.5, self.green_image, 0.5, 0)

        # delay_timestamp = self.image_timestamp_hist.buffer[-4] if self.image_timestamp_hist.buffer is not None else 0
        # mask_timestamp = msg.header.stamp.nanosec
        # self.logger.info(f"mask_timestamp: {mask_timestamp*1e-6:.4f} ms")
        # self.logger.info(f"delay_timestamp4: {self.image_timestamp_hist.buffer[-4].item()*1e-6:.4f} ms")
        # self.logger.info(f"Loop delay: {loop_delay*1000:.1f} ms, Handle delay: {handle_delay*1000:.1f} ms")
        cv2.imshow("Mixed Image", image)
        cv2.waitKey(1)

    def image_callback(self, msg):
        # msg.data is between 0 and 255
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.image_timestamp = msg.header.stamp.nanosec
        self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.cv_image_hist.append(self.cv_image)
        self.image_timestamp_hist.append(self.image_timestamp)

        if self.show_raw_image:
            cv2.imshow("Raw Image", self.cv_image)
            key = cv2.waitKey(1)

            if key == ord("q"):
                self.logger.info("Quitting...")
                cv2.destroyWindow("Raw Image")
                self.logger.info("Destroyed Raw Image window.")
