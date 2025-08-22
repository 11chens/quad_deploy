from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from config.base_agent_cfg import BaseAgentCfg
from utils.math_utils import CircularBuffer


class BaseAgent(ABC):
    """
    Base class for agents in the quad deployment system.
    This class defines the interface that all agents must implement.
    """

    def __init__(self, logdir=None, robot_node=None):
        self.logdir = logdir
        self.robot_node = robot_node

    # def prepare_obs_terms(self):
    #     """ Define observation components and their corresponding scale factors."""
    #     self.observation_components = []

    def parse_obs_config(self, cfg):
        self.cfg = cfg
        self.obs_scale = self.cfg.obs_scale
        self.smooth_factor = self.cfg.smooth_factor
        self.dead_zone = self.cfg.dead_zone
        self.min_cmds = np.array(self.cfg.min_cmds, dtype=np.float32)
        self.max_cmds = np.array(self.cfg.max_cmds, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.observation_components = None

        self.obs_scale = self.cfg.obs_scale
        self.num_commands = self.cfg.num_commands
        self.num_props = self.cfg.num_props
        self.len_history = self.cfg.len_history
        self.commands_scale = np.array(self.cfg.obs_scale.commands_scale, dtype=np.float32)

        self.commands = np.zeros(self.num_commands, dtype=np.float32)
        self.pre_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.obs_buf = np.zeros(self.num_props, dtype=np.float32)
        self.obs_hist = CircularBuffer(self.len_history)

    def update_commands(self):
        if not hasattr(self, "wireless"):
            self.wireless = True
        if self.wireless:
            self.pre_commands[0] = self.robot_node.joystick.cmd_vx * self.max_cmds[0]
            self.pre_commands[1] = self.robot_node.joystick.cmd_vy * self.max_cmds[1]
            self.pre_commands[2] = self.robot_node.joystick.cmd_vyaw * self.max_cmds[2]
            if hasattr(self.obs_scale, "pitch"):
                self.pre_commands[3] = self.robot_node.joystick.cmd_pitch * self.max_cmds[3]
        self.post_commands()

    def post_commands(self):
        self.pre_commands *= np.linalg.norm(self.pre_commands) >= self.dead_zone
        self.commands = self.commands * (1 - self.smooth_factor) + self.pre_commands * self.smooth_factor
        self.commands = np.clip(self.commands, self.min_cmds, self.max_cmds)

    def get_observation(self) -> np.ndarray:
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

    @abstractmethod
    def step(self):
        """Run the agent's main loop."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        pass

    @property
    def done(self):
        return False
