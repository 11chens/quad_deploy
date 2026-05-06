import os
import sys
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.decorators import profile_latency
from ros_base.utils.math_utils import CircularBuffer

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.sigloma.sigloma_loco_agent import SigLoMaLocoAgent
from quad_deploy.config.sigloma.sigloma_nav_agent_cfg import SigLoMaNavAgentCfg
from quad_deploy.nodes.sigloma.gripper_node import GripperNode
from quad_deploy.nodes.sigloma.vlm2robot import VLM2BobotBridge


class SigLoMaNavAgent(BaseRLAgent):
    def __init__(self, cfg=SigLoMaNavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes.get("vlm")
        self.gripper: GripperNode = self.nodes.get("gripper")
        self.loco_agent: SigLoMaLocoAgent = self.agents.get("loco")
        self.cfg: SigLoMaNavAgentCfg
        self.loco_agent.post_clip = self.cfg.post_clip
        self.loco_agent.smooth_factor = self.cfg.smooth_factor
        self.orig_actions = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self._cached_commands = None

        self.log_data = []
        self.log_path = os.path.join(os.path.expanduser("~"), "sigloma_nav_log.npz")

        # Pre-allocate clip bounds to avoid creating lists every step
        self._clip_lower = np.array([-3.0] * self.cfg.num_actions, dtype=np.float32)
        self._clip_upper = np.array([3.0] * self.cfg.num_actions, dtype=np.float32)

        # EMA Filter State
        self.filtered_commands = np.zeros(self.cfg.num_commands, dtype=np.float32)

        # TCN Settings
        self.tcn_buffer = CircularBuffer(self.cfg.nav_len_history)

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dimension:  3 + 3 + 3 + 9 + 1 + 4 = 23
        # Use tuple instead of list to avoid allocation overhead
        self.observation_components = (
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 9
            (self.task_flag, 1.0),  # dim 1
            (self.last_action, 1.0),  # dim 4
        )

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
        mlp_input = self.obs_hist.buffer.reshape(1, -1)
        tcn_input = self.tcn_buffer.buffer.reshape(1, -1)

        # Combined Input: [TCN_Flattened, MLP_Flattened]
        # Match ActorCriticTCN: extract_obs expects TCN part first, MLP part last
        _actor_input = np.concatenate([tcn_input, mlp_input], axis=1)

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
            elif info == "task_flag":
                task_flag = self.task_flag
                self.logger.info(f"[Nav] task_flag (grasp_state): {task_flag[0]:.1f}")

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

        # Update TCN Buffer (every 10 steps / 200ms)
        if self.nav_timestamp % self.cfg.nav_update_interval == 0:
            self.tcn_buffer.append(self.commands)

    @profile_latency(
        cycle_threshold_ms=100.0,
        process_threshold_ms=50.0,
        log_interval_s=3.0,
        debug=True,
        check_capture_latency=False,
    )
    def step(self):
        self._cached_commands = None
        self.get_observation()
        action = self.infer()
        if self.state == "gripper_start":
            action[:3] = 0.0  # stop moving when gripper is working, only keep the pitch command

        # Use pre-allocated bounds instead of creating lists every time
        np.clip(action, self._clip_lower, self._clip_upper, out=self.orig_actions)
        self.loco_agent.pre_cmds = np.clip(self.orig_actions, self.cfg.min_action, self.cfg.max_action)

        action, _, _, _ = self.loco_agent.step()
        self.nav_timestamp += 1  # each step is 0.02s

        return action, None, None, self.done

    def reset(self):
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.nav_timestamp = 0

        # Initialize filter with current raw observation to avoid transient
        raw_cmds = np.array(
            self.vlm.sigma_3d_cam[0 : self.num_commands // 3],
            dtype=np.float32,
        ).reshape(-1)
        self.filtered_commands = raw_cmds

        # Reset Signal Processing and Buffers
        self.tcn_buffer.reset()
        self.obs_hist.reset()
        self.task_flag = np.array([self.gripper.grasp_state], dtype=np.float32)

    @property
    def done(self):
        return False

    @property
    def commands(self):
        if self._cached_commands is None:
            raw_cmds = np.array(
                self.vlm.sigma_3d_cam[0 : self.num_commands // 3],
                dtype=np.float32,
            ).reshape(-1)

            # Apply Optional EMA Filter
            if self.cfg.enable_ema_filter:
                is_place = self.task_flag[0] > 0.5
                alpha = self.cfg.ema_alpha if is_place else 1.0

                self.filtered_commands = (1.0 - alpha) * self.filtered_commands + alpha * raw_cmds
                self._cached_commands = self.filtered_commands.copy()
            else:
                self._cached_commands = raw_cmds

        return self._cached_commands

    @property
    def last_action(self):
        return self.loco_agent.post_cmds
