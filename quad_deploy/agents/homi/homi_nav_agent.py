import os
import time

import numpy as np
import onnxruntime as ort

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.config.homi.homi_nav_agent_cfg import HomiNavAgentCfg
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class HomiNavAgent(BaseRLAgent):
    def __init__(self, cfg=HomiNavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes.get("vlm")
        self.loco_agent: HomiLocoAgent = self.agents.get("loco")
        self.cfg: HomiNavAgentCfg
        self.loco_agent.post_clip = self.cfg.post_clip
        self.orig_actions = np.zeros(self.cfg.num_actions, dtype=np.float32)

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dim: 3 + 3 + 3 + 21 + 1 + 4 = 35:
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 3 * 7
            (self.task_flag, 1.0),  # dim 1
            (self.last_action, 1.0),  # dim 4
        ]

    def parse_config(self):
        super().parse_config()
        self.num_commands = self.cfg.num_commands
        self.num_actions = self.cfg.num_actions

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "nav_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)

        # Get input/output names
        self.input_names = [inp.name for inp in self.policy.get_inputs()]
        self.output_names = [out.name for out in self.policy.get_outputs()]

        # Expected inputs: ['obs']
        # Expected outputs: ['action']

    def infer(self):
        _actor_input = self.obs_hist.buffer.reshape(1, -1)

        # Run inference
        # Inputs: {input_name: obs}
        inputs = {
            self.input_names[0]: _actor_input,
        }

        outputs = self.policy.run(self.output_names, inputs)

        # Outputs: [action]
        action = outputs[0][0]

        return action

    def debug_info(self, infos=None):
        for info in infos:
            if info == "base_lin_vel":
                base_lin_vel = self.loco_agent.base_lin_vel_pred
                self.logger.info(
                    f"[Nav] base_lin_vel: x={base_lin_vel[0]:.3f}, y={base_lin_vel[1]:.3f}, z={base_lin_vel[2]:.3f}"
                )
            elif info == "commands":
                commands = self.commands
                self.logger.info(f"[Nav] sigma points: x={commands[0]:.3f}, y={commands[1]:.3f}, z={commands[2]:.3f}")
            elif info == "base_ang_vel":
                base_ang_vel = self.robot.base_ang_vel
                self.logger.info(
                    f"[Nav] base_ang_vel: x={base_ang_vel[0]:.3f}, y={base_ang_vel[1]:.3f}, z={base_ang_vel[2]:.3f}"
                )
            elif info == "last_action":
                last_action = self.last_action
                self.logger.info(
                    f"[Nav] last_action: vx={last_action[0]:.3f}, vy={last_action[1]:.3f}, vyaw={last_action[2]:.3f},"
                    f" pitch={last_action[3]:.3f}"
                )

    def step(self):
        self.get_observation()
        action = self.infer()
        if self.state == "gripper_start":
            action[:3] = 0.0  # stop moving when gripper is working, only keep the pitch command

        self.orig_actions = np.clip(
            action, [-3.0] * self.num_actions, [3.0] * self.num_actions
        )  # in case the model outputs large values
        self.loco_agent.pre_cmds = np.clip(self.orig_actions, self.cfg.min_action, self.cfg.max_action)

        action, _, _, _ = self.loco_agent.step()
        self.nav_timestamp += 1  # each step is 0.02s

        if self.nav_timestamp % 10 == 0:
            self.debug_info(infos=["commands", "last_action", "base_lin_vel", "base_ang_vel"])

        return action, None, None, self.done

    def reset(self):
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.nav_timestamp = 0

    @property
    def done(self):
        return False

    @property
    def commands(self):
        # reshape: from self.vlm.sigma_3d_cam: :List of [x, y, z] to np.array of shape (L, 3) to (3L,)
        return np.array(
            self.vlm.sigma_3d_cam,
            dtype=np.float32,
        ).reshape(-1)

    @property
    def task_flag(self):
        return np.array(
            [not self.vlm.grasp],  # Pick: 0, Place: 1
            dtype=np.float32,
        )

    @property
    def last_action(self):
        return self.orig_actions
