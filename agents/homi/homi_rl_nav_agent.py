import os
import time

import numpy as np
import onnxruntime as ort

from agents.base_agent import BaseAgent
from config.homi.homi_nav_agent_cfg import HomiNavAgentCfg
from nodes.robot_node import UnitreeGo2


class HomiRLNavAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
    ):
        super().__init__(logdir, robot_node)

        self.vlm = self.robot_node.nodes["vlm"]
        self.loco_agent = self.robot_node.agents["loco"]

        self.parse_obs_config(HomiNavAgentCfg)
        self.load_model()

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dim: 16
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot_node.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot_node.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 4
            (self.last_action, 1.0),  # dim 3
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
        self.loco_agent.pre_cmds = action
        action, _, _, _ = self.loco_agent.step()
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot_node.auto

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
    def last_actiom(self):
        return self.loco_agent.pre_cmds
