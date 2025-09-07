import time

import numpy as np
from geometry_msgs.msg import Point, Pose
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import Image

from utils.camera_sensor import CameraSensor
from utils.math_utils import quat_rotate_inverse


class ZedNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img_pub = self.create_publisher(Image, "/geometry_msgs/Image", 1)
        self.P_img_pub = self.create_publisher(Point, "/geometry_msgs/p_img", 1)

        freq_hz = 10
        self.timer = self.create_timer(1 / freq_hz, self._sim_timer_callback)
        self.img_msg = Image()
        self.P_img_msg = Point()

        # TODO: camera config, setup Cutie

        self._sim_config()

    def _timer_callback(self):
        # TODO: get img from camera, Cutie process, and publish
        # TODO: get point from img, compute P_img(u,v,depth), and publish
        self.img_pub.publish(self.img_msg)
        self.P_img_pub.publish(self.P_img_msg)

    def _sim_config(self):
        self.goal_pub = self.create_publisher(Point, "/mujoco/goal", 1)
        self.pose_sub = self.create_subscription(Pose, "/mujoco/pose", self._pose_callback, 1)

        self.camera_sim = CameraSensor()
        self.P_world_msg = Point()
        self.P_world = [5.0, 2.0, 0.1]
        self.position = np.array([0.0, 0.0, 0.0])
        self.start_time = time.monotonic()
        self.depth = 0
        self.quat = np.quaternion(1.0, 0.0, 0.0, 0.0)

    def _sim_timer_callback(self):
        # self.logger.info(f"loop_time: {(time.monotonic() - self.start_time)*1e3} ms")
        # self.start_time = time.monotonic()

        pos_diff = self.P_world - self.position
        self.P_base = quat_rotate_inverse(self.quat, pos_diff)
        self.P_cam, P_img_xy = self.camera_sim.transform(self.P_base)
        self.out_of_view = (P_img_xy == -1).any(axis=-1)
        self.depth = np.where(
            self.out_of_view, np.ones_like(self.P_cam[-1]) * -1, self.P_cam[-1]  # set to -1 if out of view
        )

        self.P_img_msg.x = P_img_xy[0]
        self.P_img_msg.y = P_img_xy[1]
        self.P_img_msg.z = float(self.depth)

        self.img_pub.publish(self.img_msg)
        self.P_img_pub.publish(self.P_img_msg)

        self.P_world_msg.x, self.P_world_msg.y, self.P_world_msg.z = self.P_world[0], self.P_world[1], self.P_world[2]
        self.goal_pub.publish(self.P_world_msg)

        # self.camera_sim.visualize_img_coords(P_base=self.P_base, P_camera=self.P_cam, P_image=P_img_xy)

        # self.end_time = time.monotonic()
        # self.logger.info(f"handle_time: {(self.end_time - self.start_time)*1e3} ms")

        # self.logger.info(f"self_pos: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")
        # self.logger.info(f"P_base: ({self.P_base[0]:.2f}, {self.P_base[1]:.2f}, {self.P_base[2]:.2f})")
        # self.logger.info(f"P_cam: ({self.P_cam[0]:.2f}, {self.P_cam[1]:.2f}, {self.P_cam[2]:.2f})")
        # self.logger.info(f"P_img: ({self.P_img_msg.x:.2f}, {self.P_img_msg.y:.2f}, {self.P_img_msg.z:.2f})")

    def _pose_callback(self, msg: Pose):
        self.position = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        self.quat = np.quaternion(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
