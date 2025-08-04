import os
import time

import numpy as np
import onnxruntime as ort

from agents.base import BaseAgent
from robot_real import UnitreeGo2
from utils.math_utils import CircularBuffer


class LocomotionAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
        smooth_factor_loco=[0.1, 0.1, 0.1],
        smooth_factor_stop=[0.3, 0.3, 0.3],
        min_cmds=[-0.5, -0.8, -1.0],
        max_cmds=[1.5, 0.8, 1.0],
        dead_zone=0.2,
    ):
        super().__init__(logdir, robot_node)
        self.obs_buf = np.zeros(45, dtype=np.float32)
        self.obs_hist = CircularBuffer(10)
        self._actor_input = np.zeros(450, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)

        self.smooth_factor_loco = np.array(smooth_factor_loco, dtype=np.float32)
        self.smooth_factor_stop = np.array(smooth_factor_stop, dtype=np.float32)
        self.min_cmds = np.array(min_cmds, dtype=np.float32)
        self.max_cmds = np.array(max_cmds, dtype=np.float32)
        self.dead_zone = dead_zone
        self.load_model()

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "locomotion_model", "model.onnx")
        self.policy_loco = ort.InferenceSession(onnx_path)
        self.output_names = [output.name for output in self.policy_loco.get_outputs()]
        self.input_name = self.policy_loco.get_inputs()[0].name

    def joystick_to_commands(self):
        self.pre_commands[0] = self.robot_node.joystick.cmd_vx * self.max_cmds[0]
        self.pre_commands[1] = self.robot_node.joystick.cmd_vy * self.max_cmds[1]
        self.pre_commands[2] = self.robot_node.joystick.cmd_vyaw * self.max_cmds[2]
        self.post_commands()

    def post_commands(self):
        smooth_factors = np.where(
            np.linalg.norm(self.pre_commands) >= self.dead_zone, self.smooth_factor_loco, self.smooth_factor_stop
        )
        self.commands = self.commands * (1 - smooth_factors) + self.pre_commands * smooth_factors
        self.commands = np.clip(self.commands, self.min_cmds, self.max_cmds)

    def get_observation(self):
        """Extract from the buffers and build the 1d observation tensor
        Each get ... obs function does not do the obs_scale multiplication.
        """

        # self.obs_buf[:3] = self.robot_node.base_ang_vel_filter * self.obs_scale.ang_vel
        self.obs_buf[:3] = self.robot_node.base_ang_vel * self.obs_scale.ang_vel
        self.obs_buf[3:6] = self.robot_node.projected_gravity
        self.obs_buf[6:9] = self.commands * self.commands_scale
        self.obs_buf[9:21] = self.robot_node.dof_pos_rel * self.obs_scale.dof_pos
        self.obs_buf[21:33] = self.robot_node.dof_vel * self.obs_scale.dof_vel
        self.obs_buf[33:45] = self.robot_node.last_action
        if np.isnan(self.obs_buf).any():
            self.robot_node.logger.error("obs has nan")
        self.obs_hist.append(self.obs_buf)

    def infer_loco(self):
        self._actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        actions, vel_pred = self.policy_loco.run(self.output_names, {self.input_name: self._actor_input})
        actions = actions[0]
        self.base_lin_vel[:2] = vel_pred[0] * 0.5  # scale
        return actions

    def step(self):
        self.joystick_to_commands()
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
