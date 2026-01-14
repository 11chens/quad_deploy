import os

import numpy as np
import onnxruntime as ort

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.config.homi.homi_loco_agent_cfg import HomiLocoAgentCfg


class HomiLocoAgent(BaseRLAgent):
    def __init__(self, cfg=HomiLocoAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "loco_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.output_names = [output.name for output in self.policy.get_outputs()]
        self.input_name = self.policy.get_inputs()[0].name
        self.base_lin_vel_pred = np.zeros(3, dtype=np.float32)

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dimension: 47
        # Use tuple instead of list to avoid allocation overhead
        self.observation_components = (
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, self.commands_scale),  # dim 4
            (self.robot.euler_rpy[1:2], 1.0),  # dim 1
            (self.robot.dof_pos_rel, self.obs_scale.dof_pos),  # dim 12
            (self.robot.dof_vel, self.obs_scale.dof_vel),  # dim 12
            (self.robot.last_action, 1.0),  # dim 12
        )

    def parse_config(self):
        super().parse_config()

    def debug_info(self, infos=None):
        for info in infos:
            if info == "base_lin_vel":
                base_lin_vel = self.base_lin_vel_pred
                self.logger.info(
                    f"[Loco] base_lin_vel: x={base_lin_vel[0]:.3f}, y={base_lin_vel[1]:.3f}, z={base_lin_vel[2]:.3f}"
                )
            elif info == "commands":
                commands = self.commands
                self.logger.info(
                    f"[Loco] commands: x={commands[0]:.3f}, y={commands[1]:.3f}, z={commands[2]:.3f},"
                    f" pitch={commands[3]:.3f}"
                )
            elif info == "base_ang_vel":
                base_ang_vel = self.robot.base_ang_vel
                self.logger.info(
                    f"[Loco] base_ang_vel: x={base_ang_vel[0]:.3f}, y={base_ang_vel[1]:.3f}, z={base_ang_vel[2]:.3f}"
                )
            elif info == "pitch":
                pitch = self.robot.euler_rpy[1]
                self.logger.info(f"[Loco] pitch: {pitch:.3f}")

    def infer(self):
        _actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        actions, vel_pred = self.policy.run(self.output_names, {self.input_name: _actor_input})
        actions = actions[0]
        self.base_lin_vel_pred = vel_pred[0]
        return actions

    def step(self):
        self.get_observation()
        action = self.infer()
        # if (self.timestamp) % 1 == 0:
        #     self.debug_info(infos=["commands", "base_lin_vel", "base_ang_vel", "pitch"])
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        self.wireless = True

    @property
    def done(self):
        return False

    @property
    def commands(self):
        # wireless is False: get commands from high level output
        # wireless is True: get commands from joystick
        return self.update_commands()
