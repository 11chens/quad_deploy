import os

import numpy as np
import onnxruntime as ort

from agents.base_agent import BaseAgent
from config.loco_pitch_agent_cfg import LocoPitchAgentCfg
from nodes.robot_node import UnitreeGo2
from utils.math_utils import CircularBuffer


class LocoPitchAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
    ):
        super().__init__(logdir, robot_node)
        self.cfg = LocoPitchAgentCfg()

        self.parse_obs_config()
        self.prepare_obs_terms()

        self.load_model()

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "loco_pitch_model", "model.onnx")
        self.policy_loco = ort.InferenceSession(onnx_path)
        self.output_names = [output.name for output in self.policy_loco.get_outputs()]
        self.input_name = self.policy_loco.get_inputs()[0].name

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dimension: 47
        self.observation_components = [
            (self.robot_node.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot_node.projected_gravity, 1.0),  # dim 3
            (self.commands, self.commands_scale),  # dim 4
            (self.robot_node.base_euler[1:2], self.obs_scale.pitch),  # dim 1
            (self.robot_node.dof_pos_rel, self.obs_scale.dof_pos),  # dim 12
            (self.robot_node.dof_vel, self.obs_scale.dof_vel),  # dim 12
            (self.robot_node.last_action, 1.0),  # dim 12
        ]

    def parse_obs_config(self):
        super().parse_obs_config()
        self.obs_scale = self.cfg.obs_scale
        self.num_commands = self.cfg.num_commands
        self.num_props = self.cfg.num_props
        self.len_history = self.cfg.len_history
        self.commands_scale = np.array(self.cfg.obs_scale.commands_scale, dtype=np.float32)
        self.pre_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.commands = np.zeros(self.num_commands, dtype=np.float32)
        self.obs_buf = np.zeros(self.num_props, dtype=np.float32)
        self.obs_hist = CircularBuffer(self.len_history)

        self.min_cmds = np.array(self.cfg.min_cmds, dtype=np.float32)
        self.max_cmds = np.array(self.cfg.max_cmds, dtype=np.float32)

    def infer_loco(self):
        _actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        actions, vel_pred = self.policy_loco.run(self.output_names, {self.input_name: _actor_input})
        actions = actions[0]
        self.base_lin_vel[:2] = vel_pred[0] * 0.5  # scale
        return actions

    def step(self):
        self.update_commands()
        self.get_observation()
        action = self.infer_loco()
        if (self.robot_node.timestamp) % 200 == 0:
            self.robot_node.logger.debug(
                f"Cx: {self.commands[0]:.2f}, Cy: {self.commands[1]:.2f}, Cyaw: {self.commands[2]:.2f} "
            )
            self.robot_node.logger.debug(
                f"Vx: {self.base_lin_vel[0].item():.2f}, Vy: {self.base_lin_vel[1].item():.2f}, Vyaw:"
                f" {self.robot_node.base_ang_vel[2:].item():.2f} "
            )
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()

    @property
    def done(self):
        return False
