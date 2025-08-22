import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool

from utils.math_utils import CircularBuffer


class HomiSubscriber:
    def __init__(self, ros_manager=None):
        self.P_img_hist_ = CircularBuffer(5)
        self.start_sub = ros_manager.create_subscription(Bool, "/control/start", self._start_control_callback, 10)
        self.P_img_sub = ros_manager.create_subscription(Point, "/geometry_msgs/p_img", self._perception_callback, 10)

    def _start_control_callback(self, msg: Bool):
        self.start_ = msg.data

    def _perception_callback(self, msg: Point):
        self.P_img_ = [msg.x, msg.y, msg.z]  # (u, v, depth)
        self.P_img_hist_.append(self.P_img_)
