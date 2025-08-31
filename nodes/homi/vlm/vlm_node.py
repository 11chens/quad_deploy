import time

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

        self.start_msg = Bool()
        self.start = self.start_msg.data
        self.grasp_msg = Bool()
        self.grasp = self.grasp_msg.data
        self.turn_msg = String()
        self.turn = self.turn_msg.data
        self.done = False
        self.ros_manager = ros_manager
        self.logger = self.ros_manager.logger

    def publish(self):
        self.start_msg.data = self.start
        self.grasp_msg.data = self.grasp
        self.turn_msg.data = self.turn

        self.start_pub.publish(self.start_msg)
        self.grasp_pub.publish(self.grasp_msg)
        self.turn_pub.publish(self.turn_msg)

    def _done_control_callback(self, msg: Bool):
        # get done signal from RL
        self.done = msg.data
        self.timestamp = time.perf_counter()

    def _zed_img_callback(self, msg: Image):
        # get processed img from zed_pub_node
        self.img = msg.data
