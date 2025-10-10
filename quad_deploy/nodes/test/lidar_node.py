import numpy as np
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from ros_base.nodes.base_node import BaseNode
from ros_base.utils.math_utils import CircularBuffer
from sensor_msgs.msg import LaserScan


class LidarSubscriber(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rays_hist_ = CircularBuffer(5)
        self.ray_sub = self.create_subscription(LaserScan, "/rays", self._perception_callback, 1)
        self.pose_sub = self.create_subscription(Pose2D, "/pose", self._odom_callback, 1)

    def _perception_callback(self, msg: LaserScan):
        self.rays_ = np.asarray(msg.ranges, dtype=np.float32)
        self.rays_hist_.append(self.rays_)

    def _odom_callback(self, msg: Pose2D):
        self.pose_ = np.array([msg.x, msg.y, msg.theta], dtype=np.float32)
