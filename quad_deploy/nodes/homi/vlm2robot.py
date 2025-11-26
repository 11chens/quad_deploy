from geometry_msgs.msg import Point, PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_base.nodes.base_node import BaseNode
from std_msgs.msg import Bool, String


class VLM2BobotBridge(BaseNode):
    """Subscribe to the messages from VLM and publish messages back"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # subscriber
        self.turn_sub = self.create_subscription(String, "/control/turn", self._turn_control_callback, 1)
        self.grasp_sub = self.create_subscription(Bool, "/control/grasp", self._grasp_control_callback, 1)
        self.P_img_sub = self.create_subscription(PointStamped, "/geometry_msgs/p_img", self._perception_callback, 1)
        self.vlm_done_sub = self.create_subscription(Bool, "/control/vlm_done", self._vlm_done_callback, 1)

        self.turn = ""
        self.grasp = None
        self.P_img = None  # (u, v, depth)
        self.vlm_done = False

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

        self.twist_pub = self.create_publisher(Twist, "/control/twist", 1)
        self.twist_msg = Twist()

        self.odom_pub = self.create_publisher(Odometry, "/visual_slam/tracking/odometry", 1)
        self.odom_msg = Odometry()

        self.target_yaw = 0.0
        self.initial_yaw = 0.0

    def _grasp_control_callback(self, msg: Bool):
        grasp = msg.data
        self.logger.info(f"""[Sub] grasp: {grasp}.""")
        self.grasp = grasp

    def _turn_control_callback(self, msg: String):
        turn = msg.data
        self.logger.info(f"""[Sub] turn: {turn}.""")
        self.turn = turn

        # -90 degree (-1)
        if self.turn.lower() == "right":
            self.target_yaw = -1.57
        # +90 degree (1)
        elif self.turn.lower() == "left":
            self.target_yaw = 1.57
        else:
            self.target_yaw = 0.0
        # TODO:
        # -180 degree (-2)

        # +180 degree (+2)
        self.initial_yaw = self.nodes["robot"].euler_rpy[2]

    def _perception_callback(self, msg: PointStamped):
        self.P_img = [msg.point.x, msg.point.y, msg.point.z]  # (u, v, depth)

    def _vlm_done_callback(self, msg: Bool):
        vlm_done = msg.data
        self.vlm_done = vlm_done
        self.logger.info(f"""[Sub] vlm_done: {vlm_done}.""")

    def publish_rl_ready(self, rl_ready: bool):
        if not self.rl_ready and rl_ready:  # False -> True
            self.rl_ready = rl_ready
            self.rl_ready_msg.data = rl_ready
            self.rl_ready_pub.publish(self.rl_ready_msg)
            self.logger.info(f"""[Pub] rl_ready: {rl_ready}.""")

    def publish_turn_done(self, turn_done: bool):
        if not self.turn_done and turn_done:  # False -> True
            self.turn_done = turn_done
            self.turn_done_msg.data = turn_done
            self.turn_done_pub.publish(self.turn_done_msg)
            self.logger.info(f"""[Pub] turn_done: {turn_done}.""")

    def publish_grasp_done(self, grasp_done: bool):
        if not self.grasp_done and grasp_done:  # False -> True
            self.grasp_done = grasp_done
            self.grasp_done_msg.data = grasp_done
            self.grasp_done_pub.publish(self.grasp_done_msg)
            self.logger.info(f"""[Pub] grasp_done: {grasp_done}.""")

    def publish_robot_twist(self, lin_vel, ang_vel):
        self.twist_msg.linear.x = float(lin_vel[0])
        self.twist_msg.linear.y = float(lin_vel[1])
        self.twist_msg.linear.z = float(lin_vel[2])
        self.twist_msg.angular.x = float(ang_vel[0])
        self.twist_msg.angular.y = float(ang_vel[1])
        self.twist_msg.angular.z = float(ang_vel[2])
        self.twist_pub.publish(self.twist_msg)

    def publish_robot_odom(self, position=None, orientation_quat=None, lin_vel=None, ang_vel=None):
        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "robot_base"
        self.odom_msg.pose.pose.position.x = float(position[0]) if position is not None else 0.0
        self.odom_msg.pose.pose.position.y = float(position[1]) if position is not None else 0.0
        self.odom_msg.pose.pose.position.z = float(position[2]) if position is not None else 0.0
        self.odom_msg.pose.pose.orientation.w = float(orientation_quat[0]) if orientation_quat is not None else 1.0
        self.odom_msg.pose.pose.orientation.x = float(orientation_quat[1]) if orientation_quat is not None else 0.0
        self.odom_msg.pose.pose.orientation.y = float(orientation_quat[2]) if orientation_quat is not None else 0.0
        self.odom_msg.pose.pose.orientation.z = float(orientation_quat[3]) if orientation_quat is not None else 0.0
        self.odom_msg.twist.twist.linear.x = float(lin_vel[0]) if lin_vel is not None else 0.0
        self.odom_msg.twist.twist.linear.y = float(lin_vel[1]) if lin_vel is not None else 0.0
        self.odom_msg.twist.twist.linear.z = float(lin_vel[2]) if lin_vel is not None else 0.0
        self.odom_msg.twist.twist.angular.x = float(ang_vel[0]) if ang_vel is not None else 0.0
        self.odom_msg.twist.twist.angular.y = float(ang_vel[1]) if ang_vel is not None else 0.0
        self.odom_msg.twist.twist.angular.z = float(ang_vel[2]) if ang_vel is not None else 0.0
        self.odom_pub.publish(self.odom_msg)

    def reset(self):
        self.start = False
        self.grasp = None
        self.turn = ""
        self.rl_ready = False
        self.turn_done = False
        self.grasp_done = False
        self.vlm_done = False
        self.P_img = None
        self.target_yaw = 0.0
        self.initial_yaw = self.nodes["robot"].euler_rpy[2]
