import os
import time

import numpy as np


class HomiTurnAgent:
    def __init__(
        self,
        logdir=None,
        robot_node=None,
    ):
        self.robot_node = robot_node

    def infer(self):
        # TODO: design action, according to the robot state
        # (vx, vy, vyaw, pitch), e.g. (0.0, 0.0, 1.0, 0.0)
        action = None
        return action

    def step(self):
        # Importrant: action is executed at 50 Hz in main_loop
        action = self.infer()
        return action, None, None, self.done

    def reset(self):
        self.wireless = False

    @property
    def done(self):
        # TODO: done = (self.target_yaw - self.yaw) < threshold
        # Important: yaw state is based on the robots's initial pose, but not current pose.
        return False

    @property
    def target_yaw(self):
        self.robot_node.nodes["vlm"].target_yaw
