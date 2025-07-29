from abc import ABC, abstractmethod
import numpy as np
from robot_cfgs import RobotCfgs


class BaseAgent(ABC):
    """
    Base class for agents in the quad deployment system.
    This class defines the interface that all agents must implement.
    """

    def __init__(self, logdir=None, robot_node=None):
        self.logdir = logdir
        self.robot_node = robot_node
        self.parse_obs_config()

    def parse_obs_config(self):
        self.commands_scale = []
        self.obs_scale = RobotCfgs.AgentCfg.obs_scale
        self.commands_scale = np.array([self.obs_scale.lin_vel, self.obs_scale.lin_vel, self.obs_scale.ang_vel],
                                       dtype=np.float32)
        self.num_commands = RobotCfgs.AgentCfg.num_commands
        self.commands = np.zeros(self.num_commands, dtype=np.float32)
        self.pre_commands = np.zeros(self.num_commands, dtype=np.float32)

    @abstractmethod
    def step(self):
        """Run the agent's main loop."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent. This is a placeholder for any reset logic if needed."""
        pass
