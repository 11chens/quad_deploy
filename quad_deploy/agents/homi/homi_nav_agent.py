import os
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.math_utils import CircularBuffer

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
            (self.robot.euler_rpy[1:2], 1.0),  # dim 1
            # (self.commands, 1.0),  # dim 2/3
            (self.last_action, 1.0),  # dim 4
        ]

    def parse_config(self):
        super().parse_config()
        self.len_history = self.cfg.len_history
        self.nav_length_history = self.cfg.nav_length_history
        self.num_commands = self.cfg.num_commands
        self.num_actions = self.cfg.num_actions
        self.pixel_gain = self.cfg.pixel_gain
        self.cx_norm = self.cfg.cx_norm
        self.cy_norm = self.cfg.cy_norm
        self.cmds_hist = CircularBuffer(self.nav_length_history)

    def get_observation(self):
        """Build the 1D observation array by concatenating scaled components.

        Returns:
            np.ndarray: The observation buffer with scaled values.
        """
        self.prepare_obs_terms()
        start = 0
        for component, scale in self.observation_components:
            end = start + component.shape[0]
            self.obs_buf[start:end] = component * scale
            start = end
        self.obs_hist.append(self.obs_buf)
        self.cmds_hist.append(self.commands)
        return self.obs_buf

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "nav_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.input_name = self.policy.get_inputs()[0].name
        self.output_name = self.policy.get_outputs()[0].name

    def infer(self):
        _obs_hist_flat = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        _cmds_hist_flat = np.expand_dims(self.cmds_hist.buffer.reshape(-1), axis=0)
        _actor_input = np.concatenate([_obs_hist_flat, _cmds_hist_flat], axis=1).astype(np.float32)
        action = self.policy.run([self.output_name], {self.input_name: _actor_input})[0]
        action = action[0]
        return action

    def debug_info(self, infos=None):
        for info in infos:
            if info == "P_img":
                P_img = self.vlm.P_img
                self.logger.info(f"[Nav] P_img: u={P_img[0]:.3f}, v={P_img[1]:.3f}, depth={P_img[2]:.3f}")
            elif info == "base_lin_vel":
                base_lin_vel = self.loco_agent.base_lin_vel_pred
                self.logger.info(
                    f"[Nav] base_lin_vel: x={base_lin_vel[0]:.3f}, y={base_lin_vel[1]:.3f}, z={base_lin_vel[2]:.3f}"
                )
            elif info == "base_ang_vel":
                base_ang_vel = self.robot.base_ang_vel
                # self.logger.info(f"[Nav] base_ang_vel: x={base_ang_vel[0]:.3f}, y={base_ang_vel[1]:.3f}, z={base_ang_vel[2]:.3f}")
                self.logger.info(f"[Nav] V_yaw: {base_ang_vel[2]:.3f}")
            elif info == "euler_rpy":
                euler_rpy = self.robot.euler_rpy
                self.logger.info(
                    f"[Nav] euler_rpy: roll={euler_rpy[0]:.3f}, pitch={euler_rpy[1]:.3f}, yaw={euler_rpy[2]:.3f}"
                )
            elif info == "last_action":
                last_action = self.last_action
                # self.logger.info(f"[Nav] last_action: vx={last_action[0]:.3f}, vy={last_action[1]:.3f}, vyaw={last_action[2]:.3f}, pitch={last_action[3]:.3f}")
                self.logger.info(f"[Nav] C_yaw: {last_action[2]:.3f}")

    def step(self):
        self.get_observation()
        action = self.infer()
        if self.state == "gripper_start":
            action[:3] = 0.0  # stop moving when gripper is working, only keep the pitch command
        self.loco_agent.pre_cmds = np.clip(action, self.cfg.min_action, self.cfg.max_action)
        action, _, _, _ = self.loco_agent.step()
        self.nav_timestamp += 1  # each step is 0.02s

        # if self.timestamp % 10 == 0:
        #     self.debug_info(infos=["base_ang_vel", "last_action"])

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
        _u = np.tanh(self.pixel_gain * (self.vlm.P_img[0] - self.cx_norm)) * 0.5 + 0.5
        _v = np.tanh(self.pixel_gain * (self.vlm.P_img[1] - self.cy_norm)) * 0.5 + 0.5

        _depth = self.vlm.P_img[2]
        commands_ = np.array(
            [_u, _v],
            dtype=np.float32,
        )
        if self.num_commands == 3:
            commands_ = np.append(commands_, _depth)
        return commands_

    @property
    def last_action(self):
        return self.loco_agent.pre_cmds

    @property
    def timer(self):
        return np.array([min(self.nav_timestamp / self.max_episode_length, 1.0)], dtype=np.float32)
        # return np.array([0.5], dtype=np.float32)
