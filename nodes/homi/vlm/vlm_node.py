import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class VLMNode:
    def __init__(self, ros_manager=None):
        # publisher
        self.start_pub = ros_manager.create_publisher(Bool, "/control/start", 10)
        self.grasp_pub = ros_manager.create_publisher(Bool, "/control/grasp", 10)
        self.turn_pub = ros_manager.create_publisher(String, "/control/turn", 10)
        # subscriber
        self.done_sub = ros_manager.create_subscription(Bool, "/control/done", self._done_control_callback, 10)
        self.img_sub = ros_manager.create_subscription(Image, "/geometry_msgs/Image", self._zed_img_callback, 10)

        freq_hz = 10
        self.timer = ros_manager.create_timer(1 / freq_hz, self._timer_callback)
        self.start_msg = Bool()
        self.start = self.start_msg.data
        self.grasp_msg = Bool()
        self.grasp = self.grasp_msg.data
        self.turn_msg = String()
        self.turn = self.turn_msg.data
        self.done = False

    def _timer_callback(self):
        self.start_msg.data = self.start
        self.grasp_msg.data = self.grasp
        self.turn_msg.data = self.turn

        self.start_pub.publish(self.start_msg)
        self.grasp_pub.publish(self.grasp_msg)
        self.turn_pub.publish(self.turn_msg)

    def _done_control_callback(self, msg: Bool):
        # get done signal from RL
        self.done = msg.data

    def _zed_img_callback(self, msg: Image):
        # get processed img from zed_pub_node
        self.img = msg.data
