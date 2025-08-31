from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, String


class VLMSubNode:
    def __init__(self, ros_manager: Node = None):
        # subscriber
        self.ros_manager = ros_manager
        self.logger = self.ros_manager.logger
        self.start_sub = ros_manager.create_subscription(Bool, "/control/start", self._start_control_callback, 10)
        self.grasp_sub = ros_manager.create_subscription(Bool, "/control/grasp", self._grasp_control_callback, 10)
        self.turn_sub = ros_manager.create_subscription(String, "/control/turn", self._turn_control_callback, 10)
        self.P_img_sub = ros_manager.create_subscription(Point, "/geometry_msgs/p_img", self._perception_callback, 10)

        self.start = False
        self.grasp = False
        self.turn = ""

    def _start_control_callback(self, msg: Bool):
        if msg.data:
            self.logger.info(f"""[Sub] Start: {msg.data}.""")
        self.start = msg.data

    def _grasp_control_callback(self, msg: Bool):
        if self.grasp != msg.data:
            self.logger.info(f"""[Sub] Grasp: {msg.data}.""")
        self.grasp = msg.data

    def _turn_control_callback(self, msg: Bool):
        if msg.data != "":
            self.logger.info(f"""[Sub] Turn: {msg.data}.""")
        self.turn = msg.data

        # -90 degree (-1)
        if self.turn == "turn right":
            self.target_yaw = -1.57
        # +90 degree (1)
        elif self.turn == "turn left":
            self.target_yaw = 1.57
        else:
            self.target_yaw = 0.0
        # TODO:
        # -180 degree (-2)

        # +180 degree (+2)

    def _perception_callback(self, msg: Point):
        self.P_img = [msg.x, msg.y, msg.z]  # (u, v, depth)
