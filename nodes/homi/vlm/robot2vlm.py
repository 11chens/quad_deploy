import time

import numpy as np
from ros_base.node.base_node import BaseNode
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class Robot2VLMBridge(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # publisher
        self.start_pub = self.create_publisher(Bool, "/control/start", 1)
        self.turn_pub = self.create_publisher(String, "/control/turn", 1)
        self.grasp_pub = self.create_publisher(Bool, "/control/grasp", 1)
        # subscriber
        self.ready_sub = self.create_subscription(Bool, "/control/ready", self._ready_control_callback, 1)
        self.turn_done_sub = self.create_subscription(Bool, "/control/turn_done", self._turn_done_control_callback, 1)
        self.grasp_done_sub = self.create_subscription(
            Bool, "/control/grasp_done", self._grasp_done_control_callback, 1
        )
        self.img_sub = self.create_subscription(Image, "/geometry_msgs/Image", self._zed_img_callback, 1)
        self.ready = False
        self.turn_done = False
        self.grasp_done = False

        self.start_msg = Bool()
        self.start = self.start_msg.data
        self.grasp_msg = Bool()
        self.grasp = self.grasp_msg.data
        self.turn_msg = String()
        self.turn = self.turn_msg.data

    def publish_turn(self, turn):
        self.logger.info(f"""[Pub] turn: {turn}.""")
        self.turn = turn
        self.turn_msg.data = turn
        self.turn_pub.publish(self.turn_msg)

    def publish_start(self, start):
        self.logger.info(f"""[Pub] start: {start}.""")
        self.start = start
        self.start_msg.data = start
        self.start_pub.publish(self.start_msg)

    def publish_grasp(self, grasp):
        self.logger.info(f"""[Pub] grasp: {grasp}.""")
        self.grasp = grasp
        self.grasp_msg.data = grasp
        self.grasp_pub.publish(self.grasp_msg)

    def _ready_control_callback(self, msg: Bool):
        ready = msg.data
        if self.ready != ready:  # new message
            self.logger.info(f"""[Sub] ready: {ready}.""")
            self.ready = ready

    def _turn_done_control_callback(self, msg: Bool):
        turn_done = msg.data
        if self.turn_done != turn_done:  # new message
            self.logger.info(f"""[Sub] turn_done: {turn_done}.""")
            self.turn_done = turn_done

    def _grasp_done_control_callback(self, msg: Bool):
        grasp_done = msg.data
        if self.grasp_done != grasp_done:  # new message
            self.logger.info(f"""[Sub] grasp_done: {grasp_done}.""")
            self.grasp_done = grasp_done

    def _zed_img_callback(self, msg: Image):
        # get processed img from zed_pub_node
        self.img = msg.data
