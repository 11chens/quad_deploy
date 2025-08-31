import numpy as np
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from utils.math_utils import CircularBuffer


class LidarSubscriber:
    def __init__(self, ros_manager: Node = None):
        # subscriber
        self.ros_manager = ros_manager
        self.logger = self.ros_manager.logger

        self.rays_hist_ = CircularBuffer(5)
        self.ray_sub = ros_manager.create_subscription(LaserScan, "/rays", self._perception_callback, 10)
        self.pose_sub = ros_manager.create_subscription(Pose2D, "/pose", self._odom_callback, 10)

    def _perception_callback(self, msg: LaserScan):
        self.rays_ = np.asarray(msg.ranges, dtype=np.float32)
        self.rays_hist_.append(self.rays_)

    def _odom_callback(self, msg: Pose2D):
        self.pose_ = np.array([msg.x, msg.y, msg.theta], dtype=np.float32)
