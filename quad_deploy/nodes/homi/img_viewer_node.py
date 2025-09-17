from sensor_msgs.msg import CompressedImage
from quad_deploy.utils.math_utils import CircularBuffer
from ros_base.node.base_node import BaseNode
import cv2
import numpy as np

class ImageViewer(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/image/compressed", self.image_callback, 1
        )

        self.mask_subscription = self.create_subscription(
            CompressedImage, "/geometry_msgs/mask", self.mask_callback, 1
        )
        self.cv_image = None
        self.cv_image_hist = CircularBuffer(10)  # append image for buffer, 20 Hz
        self.green_image = None  # 5 Hz

    def mask_callback(self, msg):
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
