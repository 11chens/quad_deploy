import time

import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class Go2Camera:
    def __init__(self, scale: int = 2, channel: bool = False, *args, **kwargs):
        """Initialize the Go2 camera interface.
        Args:
            scale (int): Scale factor for resizing the image. Default is 2.
            # default resolution is 1280*720
            # scale = 2: 640*360
            # scale = 4: 320*180
            channel (bool): Whether to initialize the network channel. Default is False.
        """
        if channel:
            ChannelFactoryInitialize(0, "eth0")
        self.scale = scale
        self.client = VideoClient()  # Create a video client
        self.client.SetTimeout(3.0)
        self.client.Init()

    def capture_image(self, *args, **kwargs):
        code, data = self.client.GetImageSample()
        if code != 0:
            print("Get image sample error. code:", code)
            return False, None
        # Convert to numpy image
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if self.scale != 1:
            image = cv2.resize(image, (image.shape[1] // self.scale, image.shape[0] // self.scale))
        return True, image


if __name__ == "__main__":
    cam = Go2Camera()
    NUM_ITR = 100
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    handle_start = time.monotonic()
    for i in range(NUM_ITR):
        _, image = cam.capture_image()
        cv2.imshow("image", image)
        cv2.waitKey(1)
    cv2.destroyWindow("image")

    handle_end = time.monotonic()
    handle_time = handle_end - handle_start
    handle_time /= NUM_ITR

    # print result
    print(f"handle query time: {handle_time*1e3:.3f} ms")
