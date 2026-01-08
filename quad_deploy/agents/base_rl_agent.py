from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from ros_base.agents.base_agent import BaseAgent
from ros_base.nodes.wireless.wireless_sdk import JoystickSDKNode
from ros_base.utils.math_utils import CircularBuffer

from quad_deploy.config.base_agent_cfg import BaseAgentCfg
from quad_deploy.nodes.sdk.robot_go2_sdk import UnitreeGo2SDKNode


class BaseRLAgent(BaseAgent):
    """
    Base class for agents in the quad deployment system.
    This class defines the interface that all agents must implement.
    """

    def __init__(self, logdir: str = None, cfg: BaseAgentCfg = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.robot: UnitreeGo2SDKNode = self.nodes["robot"]
        self.joystick: JoystickSDKNode = self.nodes["joystick"]

        self.logdir = logdir
        self.cfg = cfg
        self.parse_config()
        self.load_model()

    def parse_config(self):
        if self.cfg is None:
            self.decimation = 4
            self.dt = self.decimation / self.node_freq_hz  # infer timer (0.02s, 50Hz)
            return

        self.obs_scale = self.cfg.obs_scale
        self.smooth_factor = self.cfg.smooth_factor
        self.post_clip = self.cfg.post_clip
        self.dead_zone = self.cfg.dead_zone
        self.min_cmds = np.array(self.cfg.min_cmds, dtype=np.float32)
        self.max_cmds = np.array(self.cfg.max_cmds, dtype=np.float32)

        self.obs_scale = self.cfg.obs_scale
        self.num_commands = self.cfg.num_commands
        self.num_props = self.cfg.num_props
        self.len_history = self.cfg.len_history
        self.commands_scale = np.array(self.cfg.obs_scale.commands_scale, dtype=np.float32)

        self.joy_cmds = np.zeros(self.num_commands, dtype=np.float32)
        self.pre_cmds = np.zeros(self.num_commands, dtype=np.float32)
        self.post_cmds = np.zeros(self.num_commands, dtype=np.float32)
        self.obs_buf = np.zeros(self.num_props, dtype=np.float32)
        self.obs_hist = CircularBuffer(self.len_history)
        self.step_dt = (
            1 / self.node_freq_hz
        )  # step time (0.005s, 200Hz) or (0.02s, 50Hz), different implementation may have different step_dt
        # Important: differ from simulation (fixed 4), here decimation is according to dt and step_dt, so that node_freq_hz can be changed freely
        self.dt = 0.02  # infer time (0.02s, 50Hz), usually fixed
        self.decimation = int(self.dt / self.step_dt)
        self.max_episode_length_s = self.cfg.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.wireless = True
        self.observation_components = []

    def load_model(self):
        pass

    def joystick_to_commands(self):
        # TODO: put it to joystick node
        self.joy_cmds[0] = self.joystick.cmd_vx * self.max_cmds[0]
        self.joy_cmds[1] = self.joystick.cmd_vy * self.max_cmds[1]
        self.joy_cmds[2] = self.joystick.cmd_vyaw * self.max_cmds[2]
        if hasattr(self.obs_scale, "pitch"):
            self.joy_cmds[3] = self.joystick.cmd_pitch * self.max_cmds[3]
        self.pre_cmds = self.joy_cmds

    def post_commands(self):
        # self.pre_cmds *= np.linalg.norm(self.pre_cmds) >= self.dead_zone
        self.post_cmds = self.post_cmds * (1 - self.smooth_factor) + self.pre_cmds * self.smooth_factor
        if self.post_clip:
            self.post_cmds = np.clip(self.post_cmds, self.min_cmds, self.max_cmds)
        return self.post_cmds

    def update_commands(self):
        # wireless is False: get commands from high level output
        # wireless is True: get commands from joystick
        if self.wireless:
            self.joystick_to_commands()
        return self.post_commands()

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
        return self.obs_buf

    def prepare_obs_terms(self):
        """Prepare the observation terms for the agent."""
        pass

    def handle(self):
        if self.timestamp % self.decimation == 0:
            action, p_gains, d_gains, done = self.step()
        else:
            action, p_gains, d_gains, done = None, None, None, None
        self.robot.send_action(action, p_gains, d_gains)

    @abstractmethod
    def step(self):
        """Run the agent's main loop."""
        return None, None, None, self.done

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        pass

    @property
    def done(self):
        return False
