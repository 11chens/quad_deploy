import time

import cv2
import numpy as np
from geometry_msgs.msg import Point, PointStamped
from ros_base.nodes.base_node import BaseNode
from ros_base.utils.math_utils import CircularBuffer
from ros_base.utils.realsense_config import RealsenseConfig
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String


class VLM2UIBridge(BaseNode):
    def __init__(self, show_raw_image=False, *args, **kwargs):
        """Node to bridge between VLM outputs and UI inputs/outputs.
        Args:
            show_raw_image (bool): Whether to display the raw camera image in a separate window.
        """
        super().__init__(*args, **kwargs)

        self.show_raw_image = show_raw_image

        if self.show_raw_image:
            img_topic = "/camera/color/image_raw"
            self.image_subscription = self.create_subscription(Image, img_topic, self._img_callback, 1)

        # self.mask_subscription = self.create_subscription(
        #     CompressedImage, "/geometry_msgs/mask", self._mask_callback, 1
        # )

        self.cv_image = None
        self.cv_image_hist = CircularBuffer(10)  # append image for buffer, 20 Hz, 50 ms
        self.image_timestamp_hist = CircularBuffer(10)  # append timestamp for buffer, 20 Hz, 50 ms
        self.green_image = None  # ~10 Hz, 110 ms

        self.P_img_sub = self.create_subscription(PointStamped, "/geometry_msgs/p_img", self._perception_callback, 1)
        self.p_img_filter_sub = self.create_subscription(
            PointStamped, "/geometry_msgs/p_img_filtered", self._p_img_filter_callback, 1
        )
        self.P_img = None  # (u, v, depth)
        self.P_img_filtered = None  # (u, v, depth)

        # Visualization state
        self.vis_width, self.vis_height = RealsenseConfig.img_width, RealsenseConfig.img_height
        self.vis_canvas = np.zeros((self.vis_height, self.vis_width, 3), dtype=np.uint8)
        if not self.show_raw_image:
            self.create_timer(0.05, self._show_p_img_window)  # 20 Hz visualization

    def _p_img_filter_callback(self, msg: PointStamped):
        self.P_img_filtered = [msg.point.x, msg.point.y, msg.point.z]  # (u, v, depth)

    def _perception_callback(self, msg: PointStamped):
        self.P_img = [msg.point.x, msg.point.y, msg.point.z]  # (u, v, depth)

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

        # delay_cv_image = self.cv_image_hist.buffer[-2] if self.cv_image_hist.buffer is not None else self.cv_image
        image = cv2.addWeighted(self.cv_image, 0.5, self.green_image, 0.5, 0)

        if self.P_img is not None:
            self.draw_info_on_img(
                image, f"P_img: ({self.P_img[0]:.2f}, {self.P_img[1]:.2f}, {self.P_img[2]:.2f} m)", position=(10, 20)
            )

        if self.P_img_filtered is not None:
            # draw P_img_filtered circle
            u = int(self.P_img_filtered[0] * self.cv_image.shape[1])
            v = int(self.P_img_filtered[1] * self.cv_image.shape[0])
            cv2.circle(image, (u, v), 5, (255, 0, 0), -1)  # blue circle

            self.draw_info_on_img(
                image,
                f"P_img_filt: ({self.P_img_filtered[0]:.2f}, {self.P_img_filtered[1]:.2f},"
                f" {self.P_img_filtered[2]:.2f} m)",
                position=(10, 40),
            )

        cv2.imshow("Mixed Image", image)
        cv2.waitKey(1)

    def draw_info_on_img(self, image, text, position=(10, 20)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = position
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 1

        cv2.putText(
            image,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        return image

    def _img_callback(self, msg):
        if self.show_raw_image:
            self.cv_image = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, -1))
            if self.P_img is not None:
                # draw P_img circle
                u = int(self.P_img[0] * self.cv_image.shape[1])
                v = int(self.P_img[1] * self.cv_image.shape[0])
                cv2.circle(self.cv_image, (u, v), 5, (0, 255, 0), -1)  # green circle
                self.draw_info_on_img(
                    self.cv_image,
                    f"P_img: ({self.P_img[0]:.2f}, {self.P_img[1]:.2f}, {self.P_img[2]:.2f} m)",
                    position=(10, 20),
                )

            if self.P_img_filtered is not None:
                # draw P_img_filtered circle
                u = int(self.P_img_filtered[0] * self.cv_image.shape[1])
                v = int(self.P_img_filtered[1] * self.cv_image.shape[0])
                cv2.circle(self.cv_image, (u, v), 5, (255, 0, 0), -1)  # blue circle

                self.draw_info_on_img(
                    self.cv_image,
                    f"P_img_filt: ({self.P_img_filtered[0]:.2f}, {self.P_img_filtered[1]:.2f},"
                    f" {self.P_img_filtered[2]:.2f} m)",
                    position=(10, 40),
                )

            cv2.imshow("Raw Image", self.cv_image)
            key = cv2.waitKey(1)

            if key == ord("q"):
                self.logger.info("Quitting...")
                cv2.destroyWindow("Raw Image")
                self.logger.info("Destroyed Raw Image window.")

    def _show_p_img_window(self):
        """Draw `P_img` and `P_img_filtered` onto a blank 640x360 canvas and show it.

        Assumes `P_img` and `P_img_filtered` are normalized in [0,1] for (u,v).
        """
        # Clear canvas
        self.vis_canvas.fill(0)

        width, height = self.vis_width, self.vis_height

        # Draw raw P_img (green)
        if self.P_img is not None:
            try:
                ux = int(np.clip(self.P_img[0], 0.0, 1.0) * (width - 1))
                vy = int(np.clip(self.P_img[1], 0.0, 1.0) * (height - 1))
                cv2.circle(self.vis_canvas, (ux, vy), 6, (0, 255, 0), -1)
                self.draw_info_on_img(
                    self.vis_canvas,
                    f"P_img: ({self.P_img[0]:.2f}, {self.P_img[1]:.2f}, {self.P_img[2]:.2f} m)",
                    position=(10, 20),
                )
            except Exception:
                pass

        # Draw filtered P_img (blue)
        if self.P_img_filtered is not None:
            try:
                uxf = int(np.clip(self.P_img_filtered[0], 0.0, 1.0) * (width - 1))
                vyf = int(np.clip(self.P_img_filtered[1], 0.0, 1.0) * (height - 1))
                cv2.circle(self.vis_canvas, (uxf, vyf), 6, (255, 0, 0), -1)
                self.draw_info_on_img(
                    self.vis_canvas,
                    f"P_img_filt: ({self.P_img[0]:.2f}, {self.P_img[1]:.2f}, {self.P_img[2]:.2f} m)",
                    position=(10, 40),
                )
            except Exception:
                pass

        # Show
        try:
            cv2.imshow("P Image Window", self.vis_canvas)
            cv2.waitKey(1)
        except Exception:
            # Some environments (headless) may fail to create windows; ignore silently
            pass


def main():
    import rclpy
    from ros_base.utils.args_debug import add_debug_mode

    parser = add_debug_mode(listen_port=8888)
    parser.add_argument("--raw", action="store_true", help="Show raw image window.")
    args = parser.parse_args()

    rclpy.init()
    vlm2ui_node = VLM2UIBridge(node_name="vlm2ui_bridge", show_raw_image=args.raw)
    try:
        vlm2ui_node.start_spin_standalone()
    except KeyboardInterrupt:
        pass
    # finally block removed to avoid double destruction as start_spin_standalone handles it


if __name__ == "__main__":
    main()
