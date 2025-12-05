import os
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.math_utils import warp2pi

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.config.base_agent_cfg import BaseAgentCfg
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class HomiTurnAgent(BaseRLAgent):
    def __init__(self, cfg=BaseAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes["vlm"]
        self.loco_agent: HomiLocoAgent = self.agents["loco"]

        self.yaw_threshold = 0.05
        self.max_yaw_vel = 1.0
        self.min_yaw_vel = 0.5

        self.k_p = 0.5
        self.target_yaw = 0.0
        self.initial_yaw = 0.0

    def parse_config(self):
        super().parse_config()

    def infer(self):
        # Use absolute yaw control
        # self.target_yaw is the absolute target yaw in world frame
        # self.robot.euler_rpy[2] is the current absolute yaw
        if self.target_yaw is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        yaw_diff = warp2pi(self.target_yaw - self.robot.euler_rpy[2])

        if np.abs(yaw_diff) < self.yaw_threshold:
            desired_yaw_vel = 0.0
        else:
            desired_yaw_vel = (
                yaw_diff
                / np.abs(yaw_diff)
                * np.clip(np.abs(self.k_p * yaw_diff), a_min=self.min_yaw_vel, a_max=self.max_yaw_vel)
            )

        action = np.array([0.0, 0.0, desired_yaw_vel, 0.0], dtype=np.float32)

        return action

    def step(self):
        action = self.infer()
        self.loco_agent.pre_cmds = action
        action, _, _, _ = self.loco_agent.step()
        return action, None, None, self.done

    def handle(self):
        self.target_yaw = self.vlm.target_yaw
        super().handle()
        self.vlm.publish_turn_done(self.done)

    def reset(self):
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto

    @property
    def done(self):
        if self.vlm.turn is not None:
            yaw_diff = warp2pi(self.target_yaw - self.robot.euler_rpy[2])
            return bool(abs(yaw_diff) < self.yaw_threshold)
        else:
            return False

    @property
    def curr_yaw(self):
        return self.robot.euler_rpy[2]
