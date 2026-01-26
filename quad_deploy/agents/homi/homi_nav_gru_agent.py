import os
import sys
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.decorators import profile_latency

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.config.homi.homi_nav_agent_cfg import HomiNavAgentCfg
from quad_deploy.nodes.homi.gripper_node import GripperNode
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class HomiNavGruAgent(BaseRLAgent):
    def __init__(self, cfg=HomiNavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes.get("vlm")
        self.gripper: GripperNode = self.nodes.get("gripper")
        self.loco_agent: HomiLocoAgent = self.agents.get("loco")
        self.cfg: HomiNavAgentCfg
        self.loco_agent.post_clip = self.cfg.post_clip
        self.loco_agent.smooth_factor = self.cfg.smooth_factor
        self.orig_actions = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self._cached_commands = None

        self.log_data = []
        self.log_path = os.path.join(os.path.expanduser("~"), "homi_nav_gru_log.npz")

        # Pre-allocate clip bounds to avoid creating lists every step
        self._clip_lower = np.array([-3.0] * self.cfg.num_actions, dtype=np.float32)
        self._clip_upper = np.array([3.0] * self.cfg.num_actions, dtype=np.float32)

        # RNN Hidden State
        self.hidden_state = None

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

        # Initialize hidden state
        self.reset_hidden_state()
        self.logger.important(f"[NavGru] Loaded GRU model from {onnx_path}")

    def reset_hidden_state(self):
        # Find h_in shape from onnx metadata
        # Expected input names: ['obs', 'h_in']
        for inp in self.policy.get_inputs():
            if inp.name == "h_in":
                # shape is usually [num_layers, batch_size, hidden_size]
                dims = inp.shape
                # Replace dynamic dimensions (strings) with 1
                dims = [d if isinstance(d, int) else 1 for d in dims]
                self.hidden_state = np.zeros(dims, dtype=np.float32)
                return

        # Fallback if h_in not found (should not happen for GRU agent)
        self.logger.warning("[NavGru] 'h_in' not found in model inputs! Is this a GRU model?")
        self.hidden_state = None

    def infer(self):
        # Inputs: {obs: [1, flattened_len], h_in: [layers, 1, hidden]}
        _actor_input = self.obs_hist.buffer.reshape(1, -1).astype(np.float32)

        inputs = {
            self.input_names[0]: _actor_input,
        }

        if self.hidden_state is not None:
            inputs["h_in"] = self.hidden_state

        outputs = self.policy.run(self.output_names, inputs)

        # Outputs: [action, h_out]
        action = outputs[0][0]

        if len(outputs) > 1:
            self.hidden_state = outputs[1]

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
        if hasattr(self, "policy"):
            self.reset_hidden_state()

    @property
    def done(self):
        return False

    @property
    def commands(self):
        if self._cached_commands is None:
            # reshape: from self.vlm.sigma_3d_cam: :List of [x, y, z] to np.array of shape (L, 3) to (3L,)
            self._cached_commands = np.array(
                self.vlm.sigma_3d_cam[0 : self.num_commands // 3],
                dtype=np.float32,
            ).reshape(-1)
        return self._cached_commands

    @property
    def task_flag(self):
        # 1.0: place (closed), 0.0: pick (open)
        grasp_flag = float(self.gripper.grasp_state)

        # Avoid creating new numpy array every time
        if not hasattr(self, "_task_flag_buf"):
            self._task_flag_buf = np.zeros(1, dtype=np.float32)

        self._task_flag_buf[0] = grasp_flag
        return self._task_flag_buf

    @property
    def last_action(self):
        return self.loco_agent.post_cmds
