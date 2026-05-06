from geometry_msgs.msg import Point, PointStamped, PolygonStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_base.nodes.base_node import BaseNode
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class VLM2BobotBridge(BaseNode):
    """Subscribe to the messages from VLM and publish messages back"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # subscriber
        self.turn_sub = self.create_subscription(Float32, "/control/turn", self._turn_control_callback, 1)
        self.grasp_sub = self.create_subscription(Bool, "/control/grasp", self._grasp_control_callback, 1)
        self.P_img_sub = self.create_subscription(
            PointStamped, "/geometry_msgs/p_img_filtered", self._p_img_callback, 1
        )
        self.vlm_done_sub = self.create_subscription(Bool, "/control/vlm_done", self._vlm_done_callback, 1)
        self.sigma_filter_sub = self.create_subscription(
            PolygonStamped, "/geometry_msgs/sigma_points_filtered", self._sigma_points_callback, 1
        )
        self.object_ready_sub = self.create_subscription(Bool, "/control/object_ready", self._object_ready_callback, 1)

        self.turn = None
        self.grasp = None
        self.P_img = None  # (u, v, depth)
        self.vlm_done = False
        self.sigma_3d_cam = None  # List of [x, y, z] in camera frame
        self.object_ready = False

        # publisher
        self.rl_ready_pub = self.create_publisher(Bool, "/control/rl_ready", 1)
        self.rl_ready_msg = Bool()
        self.rl_ready = False
        self.turn_done_pub = self.create_publisher(Bool, "/control/turn_done", 1)
        self.turn_done_msg = Bool()
        self.turn_done = False
        self.grasp_done_pub = self.create_publisher(Bool, "/control/grasp_done", 1)
        self.grasp_done_msg = Bool()
        self.grasp_done = False

        self.euler_pub = self.create_publisher(PointStamped, "/control/euler_rpy", 1)
        self.euler_msg = PointStamped()

    def _grasp_control_callback(self, msg: Bool):
        grasp = msg.data
        self.logger.info(f"""[Sub] grasp: {grasp}.""")
        self.grasp = grasp

    def _turn_control_callback(self, msg: Float32):
        turn = msg.data
        self.yaw_diff = turn
        self.logger.info(f"""[Sub] turn: {turn}.""")
        self.turn = turn  # Store the raw message value or flag

    def _sigma_points_callback(self, msg: PolygonStamped):
        points_cam = []
        for p in msg.polygon.points:
            points_cam.append([p.x, p.y, p.z])
        self.sigma_3d_cam = points_cam  # List of [x, y, z] in camera frame

    def _p_img_callback(self, msg: PointStamped):
        self.P_img = [msg.point.x, msg.point.y, msg.point.z]  # (u, v, depth)
        self.missing = self.P_img[0] == -1.0 and self.P_img[1] == -1.0 and self.P_img[2] == -1.0

    def _vlm_done_callback(self, msg: Bool):
        vlm_done = msg.data
        self.vlm_done = vlm_done
        self.logger.log_once(f"""[Sub] vlm_done: {vlm_done}.""")

    def _object_ready_callback(self, msg: Bool):
        object_ready = msg.data
        self.object_ready = object_ready
        self.logger.log_once(f"""[Sub] object_ready: {object_ready}.""")

    def publish_rl_ready(self, rl_ready: bool):
        if not self.rl_ready and rl_ready:  # False -> True
            self.rl_ready = rl_ready
            self.rl_ready_msg.data = rl_ready
            self.rl_ready_pub.publish(self.rl_ready_msg)
            self.logger.log_once(f"""[Pub] rl_ready: {rl_ready}.""")

    def publish_turn_done(self, turn_done: bool):
        if not self.turn_done and turn_done:  # False -> True
            self.turn_done = turn_done
            self.turn_done_msg.data = turn_done
            self.turn_done_pub.publish(self.turn_done_msg)
            self.logger.log_once(f"""[Pub] turn_done: {turn_done}.""")

    def publish_grasp_done(self, grasp_done: bool):
        if not self.grasp_done and grasp_done:  # False -> True
            self.grasp_done = grasp_done
            self.grasp_done_msg.data = grasp_done
            self.grasp_done_pub.publish(self.grasp_done_msg)
            self.logger.log_once(f"""[Pub] grasp_done: {grasp_done}.""")

    def publish_robot_euler_rpy(self, euler_rpy):
        self.euler_msg.header.stamp = self.get_clock().now().to_msg()
        self.euler_msg.point.x = float(euler_rpy[0])
        self.euler_msg.point.y = float(euler_rpy[1])
        self.euler_msg.point.z = float(euler_rpy[2])
        self.euler_pub.publish(self.euler_msg)

    def reset(self):
        self.start = False
        self.grasp = None
        self.turn = None  # Reset turn flag
        self.rl_ready = False
        self.turn_done = False
        self.grasp_done = False
        self.vlm_done = False
        self.object_ready = False
        self.P_img = None
        self.yaw_diff = None
        self.sigma_3d_cam = None
