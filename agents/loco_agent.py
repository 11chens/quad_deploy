import os

import numpy as np
import onnxruntime as ort

from agents.base_agent import BaseAgent
from config.loco_agent_cfg import LocoAgentCfg
from nodes.robot_node import UnitreeGo2


class LocoAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
    ):
        super().__init__(logdir, robot_node)

        self.parse_obs_config(LocoAgentCfg())
        self.load_model()

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "loco_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.output_names = [output.name for output in self.policy.get_outputs()]
        self.input_name = self.policy.get_inputs()[0].name
        self.base_lin_vel_pred = np.zeros(3, dtype=np.float32)

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dimension: 45
        self.observation_components = [
            (self.robot_node.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot_node.projected_gravity, 1.0),  # dim 3
            (self.commands, self.commands_scale),  # dim 3
            (self.robot_node.dof_pos_rel, self.obs_scale.dof_pos),  # dim 12
            (self.robot_node.dof_vel, self.obs_scale.dof_vel),  # dim 12
            (self.robot_node.last_action, 1.0),  # dim 12
        ]

    def parse_obs_config(self, cfg):
        super().parse_obs_config(cfg)

    def infer(self):
        _actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        actions, vel_pred = self.policy.run(self.output_names, {self.input_name: _actor_input})
        actions = actions[0]
        self.base_lin_vel_pred[:2] = vel_pred[0]
        return actions

    def step(self):
        self.get_observation()
        action = self.infer()
        if (self.robot_node.timestamp) % 200 == 0:
            self.robot_node.logger.debug(
                f"Cx: {self.commands[0]:.2f}, Cy: {self.commands[1]:.2f}, Cyaw: {self.commands[2]:.2f} "
            )
            self.robot_node.logger.debug(
                f"Vx: {self.base_lin_vel_pred[0].item():.2f}, Vy: {self.base_lin_vel_pred[1].item():.2f}, Vyaw:"
                f" {self.robot_node.base_ang_vel[2:].item():.2f} "
            )
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        self.wireless = True

    @property
    def done(self):
        return False

    @property
    def commands(self):
        return self.update_comannds()
