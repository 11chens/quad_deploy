import time

import cv2
import numpy as np
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import CompressedImage

from quad_deploy.utils.math_utils import CircularBuffer


class ImageViewer(BaseNode):
    def __init__(self, show_raw_image=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_raw_image = show_raw_image
        self.image_subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/image", self.image_callback, 1
        )

        self.mask_subscription = self.create_subscription(CompressedImage, "/geometry_msgs/mask", self.mask_callback, 1)
        self.cv_image = None
        self.cv_image_hist = CircularBuffer(10)  # append image for buffer, 20 Hz, 50 ms
        self.image_timestamp_hist = CircularBuffer(10)  # append timestamp for buffer, 20 Hz, 50 ms
        self.green_image = None  # 8 Hz, 125 ms

    def mask_callback(self, msg):
        loop_delay = time.monotonic() - self.start_time if hasattr(self, "start_time") else 0
        self.start_time = time.monotonic()
        # Draw mask
        if self.cv_image is None:
            return
        self.green_image = np.zeros_like(self.cv_image)
        mask_np_arr = np.frombuffer(msg.data, np.uint8)
        cv_mask = cv2.imdecode(mask_np_arr, cv2.IMREAD_COLOR)

        self.green_image[:, :, :] = (0, 185, 118)
        self.green_image[cv_mask == 0] = 0

        delay_cv_image = self.cv_image_hist.buffer[-4] if self.cv_image_hist.buffer is not None else self.cv_image
        delay_timestamp = self.image_timestamp_hist.buffer[-4] if self.image_timestamp_hist.buffer is not None else 0
        mask_timestamp = msg.header.stamp.nanosec
        image = cv2.addWeighted(delay_cv_image, 0.4, self.green_image, 0.6, 0)
        self.end_time = time.monotonic()
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
