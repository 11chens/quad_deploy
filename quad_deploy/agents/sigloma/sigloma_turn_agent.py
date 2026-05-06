import os
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.math_utils import warp2pi

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.sigloma.sigloma_loco_agent import SigLoMaLocoAgent
from quad_deploy.config.base_agent_cfg import BaseAgentCfg
from quad_deploy.nodes.sigloma.vlm2robot import VLM2BobotBridge


class SigLoMaTurnAgent(BaseRLAgent):
    def __init__(self, cfg=BaseAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes.get("vlm")
        self.loco_agent: SigLoMaLocoAgent = self.agents.get("loco")

        self.yaw_threshold = 0.05
        self.max_yaw_vel = 1.0
        self.min_yaw_vel = 0.5

        self.k_p = 0.75
        self.target_yaw_diff = 0.0
        self.start_turn_time = None
        self.duration = 3.0

    def parse_config(self):
        super().parse_config()

    def infer(self):
        if self.done or self.yaw_diff is None:
            # Return zero action when done or yaw_diff is None
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        desired_yaw_vel = (
            self.yaw_diff
            / np.abs(self.yaw_diff)
            * np.clip(np.abs(self.k_p * self.yaw_diff), a_min=self.min_yaw_vel, a_max=self.max_yaw_vel)
        )
        action = np.array([0.0, 0.0, desired_yaw_vel, 0.0], dtype=np.float32)

        return action

    def step(self):
        action_ = self.infer()
        self.loco_agent.pre_cmds = action_
        action, _, _, _ = self.loco_agent.step()

        if self.done:
            # Ensure zero action is sent when done
            if self.timestamp % 100 == 0:
                self.logger.info(f"Turn action done, yaw_diff: {self.yaw_diff:.3f}, yaw_cmd: {action_[2]}.")
                self.loco_agent.pre_cmds = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return action, None, None, self.done

    def handle(self):
        super().handle()
        self.vlm.publish_turn_done(self.done)

    def reset(self):
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.start_turn_time = time.time()
        self.robot_yaw_start = self.robot.euler_rpy[2]
        self.target_yaw_diff = self.vlm.yaw_diff

    @property
    def yaw_diff(self):
        if self.vlm.yaw_diff is not None:
            current_yaw = self.robot.euler_rpy[2]
            desired_yaw = self.robot_yaw_start + self.vlm.yaw_diff
            yaw_diff = warp2pi(desired_yaw - current_yaw)
            return yaw_diff
        else:
            return None

    @property
    def done(self):
        if hasattr(self, "vlm") and self.vlm is not None and self.vlm.turn is not None:
            # Combine time-based and yaw_diff-based conditions
            yaw_aligned = abs(self.yaw_diff) < self.yaw_threshold if self.yaw_diff is not None else False
            return bool(yaw_aligned)  # Ensure the return value is always a boolean
        else:
            return False
