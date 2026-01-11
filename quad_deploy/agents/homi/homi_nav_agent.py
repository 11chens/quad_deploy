import os
import sys
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
        self.loco_agent.smooth_factor = self.cfg.smooth_factor
        self.orig_actions = np.zeros(self.cfg.num_actions, dtype=np.float32)

        self.log_data = []
        self.log_path = os.path.join(os.path.expanduser("~"), "homi_nav_log.npz")

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dimension:  3 + 3 + 3 + 9 + 1 + 4 = 23
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 9
            (self.task_flag, 1.0),  # dim 1
            (self.last_action, 1.0),  # dim 4
        ]

        # obs_now = torch.cat([
        #     self.base_lin_vel_pred * self.obs_scales.lin_vel, # 3
        #     self.base_ang_vel * self.obs_scales.ang_vel, # 3
        #     self.projected_gravity, # 3
        #     self.nav_commands, # 15/21/9 (Ground Truth Vision)
        #     self.task_flags, # 1
        #     self.orig_nav_actions, # 4
        #     ], dim=-1)

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
            elif info == "orig_actions":
                orig_actions = self.orig_actions
                self.logger.info(
                    f"[Nav] orig_actions: vx={orig_actions[0]:.3f}, vy={orig_actions[1]:.3f},"
                    f" vyaw={orig_actions[2]:.3f}, pitch={orig_actions[3]:.3f}"
                )
            elif info == "pitch":
                pitch = self.robot.euler_rpy[1]
                self.logger.info(f"[Nav] pitch: {pitch:.3f}")

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

        # Logging
        if self.robot.sim_run:
            if self.nav_timestamp % 5 == 0:
                # Use dictionary for structured logging
                log_step = {
                    "base_lin_vel": np.array(self.loco_agent.base_lin_vel_pred).flatten(),  # 3
                    "base_ang_vel": np.array(self.robot.base_ang_vel).flatten(),  # 3
                    "euler_rpy": np.array(self.robot.euler_rpy).flatten(),  # 3
                    "projected_gravity": np.array(self.robot.projected_gravity).flatten(),  # 3
                    "nav_commands": self.commands.flatten(),  # 3 * 3
                    "task_flag": self.task_flag.flatten(),  # 1
                    "actions": self.orig_actions.flatten(),  # 4
                }
                self.log_data.append(log_step)

            if self.nav_timestamp == 300:
                # Convert list of dicts to dict of arrays for np.savez
                save_dict = {k: np.array([step[k] for step in self.log_data]) for k in self.log_data[0].keys()}
                np.savez(self.log_path, **save_dict)
                self.logger.important(f"[Nav] Saved navigation log to {self.log_path}")

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
            self.vlm.sigma_3d_cam[0 : self.num_commands // 3],
            dtype=np.float32,
        ).reshape(-1)

    @property
    def task_flag(self):
        if self.vlm.grasp is None:
            grasp_flag = 0.0
        else:
            grasp_flag = float(not self.vlm.grasp)
        return np.array(
            [grasp_flag],  # Pick: 0, Place: 1
            dtype=np.float32,
        )

    @property
    def last_action(self):
        return self.loco_agent.post_cmds
