import os
import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from utils.logger import CustomLogger


class Gripper:
    """Gripper control."""

    def __init__(
        self,
    ):
        level = "DEBUG"
        self.logger = CustomLogger(level=level)
        self.last_grasp = False
        self.timestamp = 0
        self.exe_start = False
        self.dt = 0.02  # in seconds
        self.duration = 2  # in seconds, simulate the gripper execution

    def grasp_handle(self, grasp):
        # callback at 50 Hz (0.02s)
        if not self.last_grasp and grasp:  # False -> True
            self.exe_start = True
            # TODO: call gripper function to grasp
            self.logger.info("Start grasping.")

        if self.last_grasp and not grasp:  # True -> False
            self.exe_start = True
            # TODO: call gripper function to release
            self.logger.info("Start releasing.")

        if self.timestamp == self.duration / self.dt:
            self.timestamp = 0
            self.done = True
            self.exe_start = False
        else:
            # Important: reset done to avoid publishing self.nodes["robot_pub"].done = True
            self.done = False

        if self.exe_start:
            self.timestamp += 1
        self.last_grasp = grasp

    def start_handlers(self):
        pass
