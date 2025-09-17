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
        # total dim: 16
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 3
            (self.last_action, 1.0),  # dim 4
        ]

    def parse_obs_config(self, cfg):
        super().parse_obs_config(cfg)
        self.num_actions = self.cfg.num_actions
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

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

    def step(self):
        self.get_observation()
        action = self.infer()
        self.loco_agent.pre_cmds = np.clip(action, self.cfg.min_action, self.cfg.max_action)
        action, _, _, _ = self.loco_agent.step()

        # if self.timestamp % 10 == 0:
        #     self.logger.info(f"[Nav] P_img: ({self.commands[0]:.5f}, {self.commands[1]:.5f}, {self.commands[2]:.5f})")

        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto

    @property
    def done(self):
        return False

    @property
    def commands(self):
        return np.array(
            self.vlm.P_img,
            dtype=np.float32,
        )

    @property
    def last_action(self):
        return self.loco_agent.pre_cmds
