import os
import time

import numpy as np

from agents.base_agent import BaseAgent
from nodes.robot_node import UnitreeGo2


class HomiRLTurnAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
    ):
        super().__init__(logdir, robot_node)

        self.vlm = self.robot_node.nodes["vlm"]
        self.loco_agent = self.robot_node.agents["loco"]

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
        self.loco_agent.wireless = False

    @property
    def done(self):
        # TODO: done = (self.target_yaw - self.yaw) < threshold
        # Important: yaw state is based on the robots's initial pose, but not current pose.
        return False

    @property
    def target_yaw(self):
        self.robot_node.nodes["vlm"].target_yaw
