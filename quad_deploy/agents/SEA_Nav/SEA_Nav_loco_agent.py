import os

import numpy as np
import onnxruntime as ort

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.config.SEA_Nav.SEA_Nav_loco_agent_cfg import SEA_Nav_LocoAgentCfg


class SEA_Nav_LocoAgent(BaseRLAgent):
    """Low-level locomotion (velocity-tracking) agent.

    Consumes the high-level command from the Nav agent (or the joystick in
    human-teleop mode) and outputs 12 joint targets to the robot.

    The Nav agent emits a 3-channel ``(vx, vy, vyaw)`` command; this agent
    pads it to the 4-channel ``(vx, vy, vyaw, pitch)`` command its policy was
    trained with, holding the pitch channel at ``cfg.command_pitch``.

    The ONNX policy outputs both ``action[12]`` (joint targets) and
    ``vel_pred[3]`` (estimated base linear velocity, with ``z`` always zero).
    ``vel_pred`` is exposed as :attr:`base_lin_vel_pred` so the Nav agent can
    use it as the ``base_lin_vel`` slot of its own observation.

    The post-processing pipeline (clip + EMA + clip) is owned by the Nav
    agent; the Loco-side ``post_commands`` is configured as an identity
    pass-through (``smooth_factor=1.0``, ``post_clip=False``).
    """

    def __init__(self, cfg=SEA_Nav_LocoAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)
        self.command_pitch = float(getattr(cfg, "command_pitch", 0.0))

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "loco_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.output_names = [output.name for output in self.policy.get_outputs()]
        self.input_name = self.policy.get_inputs()[0].name
        # Public attribute consumed by the Nav agent's observation buffer;
        # initialized to zeros so the first Nav tick (before any Loco
        # inference has happened) reads a valid 3-vector.
        self.base_lin_vel_pred = np.zeros(3, dtype=np.float32)

    def prepare_obs_terms(self):
        """Build the 47-dim single-step observation::

        base_ang_vel(3, *0.25) + projected_gravity(3, *1.0)
        + commands(4, *[2, 2, 0.25, 1]) + pitch(1, *1.0)
        + dof_pos_rel(12, *1.0) + dof_vel(12, *0.05) + last_action(12, *1.0)
        """
        self.observation_components = (
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # 3
            (self.robot.projected_gravity, 1.0),  # 3
            (self.commands, self.commands_scale),  # 4 - nav (or joystick) command
            (self.robot.euler_rpy[1:2], 1.0),  # 1 - pitch
            (self.robot.dof_pos_rel, self.obs_scale.dof_pos),  # 12
            (self.robot.dof_vel, self.obs_scale.dof_vel),  # 12
            (self.robot.last_action, 1.0),  # 12
        )

    def parse_config(self):
        super().parse_config()

    def set_nav_command(self, command):
        """Adapt the Nav agent's 3-D ``(vx, vy, vyaw)`` command to this
        policy's 4-D ``(vx, vy, vyaw, pitch)`` command interface.

        The pitch channel is filled with ``command_pitch`` and the full
        command is clipped to the hardware limits before being handed to the
        observation buffer.
        """
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        if command.shape[0] != 3:
            raise ValueError(f"SEA-Nav command must be 3-D, got shape {command.shape}")

        adapted = np.zeros(self.num_commands, dtype=np.float32)
        adapted[:3] = command
        adapted[3] = self.command_pitch
        np.clip(adapted, self.min_cmds, self.max_cmds, out=adapted)
        self.pre_cmds = adapted

    def infer(self):
        _actor_input = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0)
        actions, vel_pred = self.policy.run(self.output_names, {self.input_name: _actor_input})
        actions = actions[0]
        # vel_pred is the estimated (vx, vy) with the z component zero-padded
        # inside the export wrapper. The 3-D shape lets the Nav agent drop it
        # directly into its base_lin_vel observation slot.
        self.base_lin_vel_pred = vel_pred[0]
        return actions

    def step(self):
        self.get_observation()
        action = self.infer()
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        self.wireless = True
        self.base_lin_vel_pred[:] = 0.0

    @property
    def done(self):
        return False

    @property
    def commands(self):
        # wireless is False: get commands from high-level Nav output
        # wireless is True : get commands from joystick (sim/teleop)
        return self.update_commands()
