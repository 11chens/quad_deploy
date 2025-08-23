import os
import time

import numpy as np


class HomiNavAgent:
    def __init__(
        self,
        logdir=None,
        robot_node=None,
    ):
        self.robot_node = robot_node
        self.vlm = self.robot_node.nodes["vlm"]

    def infer(self):
        # TODO: design action, according to the P_img and robot state
        # (vx, vy, vyaw, pitch), e.g. (0.5, 0.0, 0.0, 0.25)
        action = None
        # stop moving, but keep the pose
        # action[:3] = 0.0 if self.grasp else action[:3]
        return action

    def step(self):
        # Importrant: action is executed at 50 Hz in main_loop
        action = self.infer()
        return action, None, None, self.done

    def reset(self):
        self.wireless = False

    @property
    def P_img(self):
        return self.vlm.P_img

    @property
    def grasp(self):
        return self.vlm.grasp

    @property
    def done(self):
        return False
