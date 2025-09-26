import os
import time

import numpy as np
import onnxruntime as ort

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.config.homi.homi_nav_agent_cfg import HomiNavAgentCfg
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class HomiNavAgent(BaseRLAgent):
    def __init__(self, cfg=HomiNavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes["vlm"]
        self.loco_agent: HomiLocoAgent = self.agents["loco"]
        self.cfg: HomiNavAgentCfg

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dim: 17:
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 3
            (self.timer, 1.0),  # dim 1
            (self.last_action, 1.0),  # dim 4
        ]

    def parse_config(self):
        super().parse_config()
        self.num_actions = self.cfg.num_actions
        self.pixel_gain = self.cfg.pixel_gain
        self.cx_norm = self.cfg.cx_norm
        self.cy_norm = self.cfg.cy_norm

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "nav_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.input_name = self.policy.get_inputs()[0].name
        self.output_name = self.policy.get_outputs()[0].name

    def infer(self):
        _actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        action = self.policy.run([self.output_name], {self.input_name: _actor_input})[0]
        action = action[0]
        return action

    def debug_info(self):
        self.logger.info(f"[Nav] P_img: ({self.commands[0]:.5f}, {self.commands[1]:.5f}, {self.commands[2]:.5f})")
        self.logger.info(
            f"[Nav] Base lin vel: ({self.loco_agent.base_lin_vel_pred[0]:.5f},"
            f" {self.loco_agent.base_lin_vel_pred[1]:.5f}, {self.loco_agent.base_lin_vel_pred[2]:.5f})"
        )
        self.logger.info(
            f"[Nav] Base ang vel: ({self.robot.base_ang_vel[0]:.5f}, {self.robot.base_ang_vel[1]:.5f},"
            f" {self.robot.base_ang_vel[2]:.5f})"
        )
        self.logger.info(
            f"[Nav] Proj gravity: ({self.robot.projected_gravity[0]:.5f}, {self.robot.projected_gravity[1]:.5f},"
            f" {self.robot.projected_gravity[2]:.5f})"
        )
        self.logger.info(f"[Nav] Timer: {self.timer[0]:.5f}")
        self.logger.info(
            f"[Nav] Last action: ({self.last_action[0]:.5f}, {self.last_action[1]:.5f}, {self.last_action[2]:.5f},"
            f" {self.last_action[3]:.5f})"
        )

    def step(self):
        self.get_observation()
        action = self.infer()
        if self.state == "gripper_start":
            action[:3] = 0.0  # stop moving when gripper is workimin_actionng, only keep the pitch command
        self.loco_agent.pre_cmds = np.clip(action, self.cfg.min_action, self.cfg.max_action)
        action, _, _, _ = self.loco_agent.step()
        self.nav_timestamp += 1  # each step is 0.02s

        # if self.timestamp % 100 == 0:
        #     self.debug_info()

        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.nav_timestamp = 0

    @property
    def done(self):
        return False

    @property
    def commands(self):
        _u = np.tanh(self.pixel_gain * (self.vlm.P_img[0] - self.cx_norm))
        _v = np.tanh(self.pixel_gain * (self.vlm.P_img[1] - self.cy_norm))
        _depth = self.vlm.P_img[2]
        return np.array(
            [_u, _v, _depth],
            dtype=np.float32,
        )

    @property
    def last_action(self):
        return self.loco_agent.pre_cmds

    @property
    def timer(self):
        return np.array([min(self.nav_timestamp / self.max_episode_length, 1.0)], dtype=np.float32)
        # return np.array([0.5], dtype=np.float32)
