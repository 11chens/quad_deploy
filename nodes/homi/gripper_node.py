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
        self.logger = CustomLogger()

    def grasp_handle(self, grasp):
        if grasp:  # last grasp -> curr grasp
            # TODO: call gripper function to grasp
            # TODO: wait until grasping is done
            self.done = True
            self.logger.info("Start grasping.")
        else:
            # Important: reset done to avoid publishing self.nodes["robot_pub"].done = True
            self.done = False

    def start_handlers(self):
        pass
