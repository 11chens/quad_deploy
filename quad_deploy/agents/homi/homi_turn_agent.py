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
        self.yaw_diff = 0.0
        self.start_turn_time = None
        self.duration = 3.0

    def parse_config(self):
        super().parse_config()

    def infer(self):
        if self.yaw_diff is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        desired_yaw_vel = (
            self.yaw_diff
            / np.abs(self.yaw_diff)
            * np.clip(np.abs(self.k_p * self.yaw_diff), a_min=self.min_yaw_vel, a_max=self.max_yaw_vel)
        )
        action = np.array([0.0, 0.0, desired_yaw_vel, 0.0], dtype=np.float32)

        return action

    def step(self):
        action = self.infer()
        self.loco_agent.pre_cmds = action
        action, _, _, _ = self.loco_agent.step()
        return action, None, None, self.done

    def handle(self):
        self.yaw_diff = self.vlm.yaw_diff
        super().handle()
        self.vlm.publish_turn_done(self.done)

    def reset(self):
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.start_turn_time = time.time()

    @property
    def done(self):
        if self.vlm.turn is not None:
            return time.time() - self.start_turn_time > self.duration
        else:
            return False
