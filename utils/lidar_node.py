import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from utils.math_utils import CircularBuffer


class LidarSubscriber:

    def __init__(self, rosnode=None):
        self.rays_hist_ = CircularBuffer(5)
        self.ray_sub = rosnode.create_subscription(LaserScan, '/rays', self._perception_callback, 10)
        self.pose_sub = rosnode.create_subscription(Pose2D, '/pose', self._odom_callback, 10)

    def _perception_callback(self, msg: LaserScan):
        try:
            self.rays_ = np.asarray(msg.ranges, dtype=np.float32)
            self.rays_ = np.log2(np.clip(self.rays_, 0.1, 5.0))
            self.rays_hist_.append(self.rays_)
        except Exception as e:
            self.get_logger().error(f"Perception data processing error: {str(e)}")

    def _odom_callback(self, msg: Pose2D):
        try:
            self.pose_ = np.array([msg.x, msg.y, msg.theta], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Pose data processing error: {str(e)}")
