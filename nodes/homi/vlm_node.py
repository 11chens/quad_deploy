import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String


class VLMSubscriber:
    def __init__(self, ros_manager=None):
        self.start_sub = ros_manager.create_subscription(Bool, "/control/start", self._start_control_callback, 10)
        self.grasp_sub = ros_manager.create_subscription(Bool, "/control/grasp", self._grasp_control_callback, 10)
        self.turn_sub = ros_manager.create_subscription(String, "/control/turn", self._turn_control_callback, 10)
        self.P_img_sub = ros_manager.create_subscription(Point, "/geometry_msgs/p_img", self._perception_callback, 10)

    def _start_control_callback(self, msg: Bool):
        self.start = msg.data

    def _grasp_control_callback(self, msg: Bool):
        self.grasp = msg.data

    def _turn_control_callback(self, msg: Bool):
        # -90 degree (-1)
        if msg.data == "turn right":
            self.target_yaw = -1.57
        # +90 degree (1)
        elif msg.data == "turn left":
            self.target_yaw = 1.57
        else:
            self.target_yaw = 0.0
        # TODO:
        # -180 degree (-2)

        # +180 degree (+2)

    def _perception_callback(self, msg: Point):
        self.P_img = [msg.x, msg.y, msg.z]  # (u, v, depth)
