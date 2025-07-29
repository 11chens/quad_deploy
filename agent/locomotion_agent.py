import numpy as np
import os
import onnxruntime as ort
import time
from utils.math_utils import CircularBuffer
from agent.base import BaseAgent
from robot_real import UnitreeGo2


class LocomotionAgent(BaseAgent):

    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
        smooth_factor_loco=[0.1, 0.1, 0.1],
        smooth_factor_stop=[0.3, 0.3, 0.3],
        min_cmds=[-0.5, -0.8, -1.25],
        max_cmds=[1.5, 0.8, 1.25],
        dead_zone=0.1,
    ):
        super().__init__(logdir, robot_node)
        self.obs_buf = np.zeros(45, dtype=np.float32)
        self.obs_hist = CircularBuffer(10)
        self._actor_input = np.zeros(48, dtype=np.float32)
        self._base_lin_vel = np.zeros(3, dtype=np.float32)
        self._vxy_pred = np.zeros(2, dtype=np.float32)

        self.smooth_factor_loco = np.array(smooth_factor_loco, dtype=np.float32)
        self.smooth_factor_stop = np.array(smooth_factor_stop, dtype=np.float32)
        self.min_cmds = np.array(min_cmds, dtype=np.float32)
        self.max_cmds = np.array(max_cmds, dtype=np.float32)
        self.dead_zone = dead_zone
        self.load_model()

    def load_model(self):
        models = ['body_latest', 'encoder_vel']
        self.ort_sessions = {}
        for name in models:
            onnx_path = os.path.join(self.logdir, "locomotion_model", f"{name}.onnx")
            ort_session = ort.InferenceSession(onnx_path)
            ort_session._model_name = name
            self.ort_sessions[name] = ort_session
        self.encoder_vel = self.ort_sessions['encoder_vel']
        self.policy_loco = self.ort_sessions['body_latest']

    def post_commands(self):
        smooth_factors = np.where(
            np.linalg.norm(self.pre_commands) >= self.dead_zone, self.smooth_factor_loco, self.smooth_factor_stop)
        self.commands = self.commands * (1 - smooth_factors) + self.pre_commands * smooth_factors
        self.commands = np.clip(self.commands, self.min_cmds, self.max_cmds)

    def get_observation(self):
        """ Extract from the buffers and build the 1d observation tensor
        Each get ... obs function does not do the obs_scale multiplication.
        """
        self.obs_buf[:3] = self.robot_node.base_ang_vel * self.obs_scale.ang_vel
        self.obs_buf[3:6] = self.robot_node.projected_gravity
        self.obs_buf[6:9] = self.commands * self.commands_scale
        self.obs_buf[9:21] = self.robot_node.dof_pos_rel * self.obs_scale.dof_pos
        self.obs_buf[21:33] = self.robot_node.dof_vel * self.obs_scale.dof_vel
        self.obs_buf[33:45] = self.robot_node.last_action
        self.obs_hist.append(self.obs_buf)

    def infer_loco(self):
        self._vxy_pred = self.encoder_vel.run(None,\
             {self.encoder_vel.get_inputs()[0].name: self.obs_hist.buffer.reshape(-1)})[0]
        self._actor_input[:2] = self._vxy_pred
        self._actor_input[2:47] = self.obs_buf
        self._actor_input[47:] = self.robot_node.base_ang_vel[2:] * self.obs_scale.ang_vel
        actions = self.policy_loco.run(None,\
             {self.policy_loco.get_inputs()[0].name: self._actor_input})[0]
        return actions

    def step(self):
        self.pre_commands[0] = self.robot_node.joystick.cmd_vx * self.max_cmds[0]
        self.pre_commands[1] = self.robot_node.joystick.cmd_vy * self.max_cmds[1]
        self.pre_commands[2] = self.robot_node.joystick.cmd_vyaw * self.max_cmds[2]
        self.post_commands()
        self.get_observation()
        action = self.infer_loco()

        return action, None, None, False

    def reset(self):
        self.obs_hist.reset()

    @property
    def base_lin_vel(self):
        self._base_lin_vel[:2] = 0.5 * self._vxy_pred
        return self._base_lin_vel
