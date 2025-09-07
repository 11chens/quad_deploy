from geometry_msgs.msg import Point
from rclpy.node import Node
from ros_base.node.base_node import BaseNode
from std_msgs.msg import Bool, String


class VLM2BobotBridge(BaseNode):
    """Subscribe to the messages from VLM and publish messages back"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # subscriber
        self.turn_sub = self.create_subscription(String, "/control/turn", self._turn_control_callback, 1)
        self.start_sub = self.create_subscription(Bool, "/control/start", self._start_control_callback, 1)
        self.grasp_sub = self.create_subscription(Bool, "/control/grasp", self._grasp_control_callback, 1)
        self.P_img_sub = self.create_subscription(Point, "/geometry_msgs/p_img", self._perception_callback, 1)
        self.turn = ""
        self.start = False
        self.grasp = False
        self.gripper_start = False

        # publisher
        self.ready_pub = self.create_publisher(Bool, "/control/ready", 1)
        self.ready_msg = Bool()
        self.ready = False
        self.turn_done_pub = self.create_publisher(Bool, "/control/turn_done", 1)
        self.turn_done_msg = Bool()
        self.turn_done = False
        self.grasp_done_pub = self.create_publisher(Bool, "/control/grasp_done", 1)
        self.grasp_done_msg = Bool()
        self.grasp_done = False

        self.target_yaw = 0.0
        self.initial_yaw = 0.0

    def _start_control_callback(self, msg: Bool):
        start = msg.data
        self.logger.info(f"""[Sub] start: {start}.""")
        self.start = start

    def _grasp_control_callback(self, msg: Bool):
        grasp = msg.data
        self.logger.info(f"""[Sub] grasp: {grasp}.""")
        self.grasp = grasp
        self.gripper_start = True

    def _turn_control_callback(self, msg: Bool):
        turn = msg.data
        self.logger.info(f"""[Sub] turn: {turn}.""")
        self.turn = turn

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
        self.initial_yaw = self.nodes["robot"].euler_rpy[2]

    def _perception_callback(self, msg: Point):
        self.P_img = [msg.x, msg.y, msg.z]  # (u, v, depth)

    def publish_ready(self, ready):
        if not self.ready and ready:  # False -> True
            self.logger.info(f"""[Pub] ready: {ready}.""")
            self.ready = ready
            self.ready_msg.data = ready
            self.ready_pub.publish(self.ready_msg)

    def publish_turn_done(self, turn_done):
        if not self.turn_done and turn_done:  # False -> True
            self.logger.info(f"""[Pub] turn_done: {turn_done}.""")
            self.turn_done = turn_done
            self.turn_done_msg.data = turn_done
            self.turn_done_pub.publish(self.turn_done_msg)

    def publish_grasp_done(self, grasp_done):
        if not self.grasp_done and grasp_done:  # False -> True
            self.logger.info(f"""[Pub] grasp_done: {grasp_done}.""")
            self.grasp_done = grasp_done
            self.grasp_done_msg.data = grasp_done
            self.grasp_done_pub.publish(self.grasp_done_msg)

    def reset(self):
        self.start = False
        self.grasp = False
        self.turn = ""
        self.ready = False
        self.turn_done = False
        self.grasp_done = False
        self.gripper_start = False
        self.target_yaw = 0.0
        self.initial_yaw = self.nodes["robot"].euler_rpy[2]
