import os
import time

import numpy as np
import onnxruntime as ort
from ros_base.utils.math_utils import CircularBuffer

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.config.homi.homi_nav_agent_cfg import HomiNavAgentCfg
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class HomiNavRnnAgent(BaseRLAgent):
    def __init__(self, cfg=HomiNavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.vlm: VLM2BobotBridge = self.nodes["vlm"]
        self.loco_agent: HomiLocoAgent = self.agents["loco"]
        self.cfg: HomiNavAgentCfg
        
        # RNN Hidden States
        self.rnn_hidden_size = 256 # Default, should match training
        self.rnn_num_layers = 1
        self.num_latent_int = 16
        self.num_latent_ext = 16
        
        # Initialize hidden states (h_in, c_in for LSTM)
        # Shape depends on ONNX export. Usually (num_layers, batch=1, hidden_size)
        self.h_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)
        self.c_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        # total dim: 19:
        self.observation_components = [
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # dim 3
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # dim 3
            (self.robot.projected_gravity, 1.0),  # dim 3
            (self.robot.euler_rpy, 1.0),  # dim 3
            (self.commands, 1.0),  # dim 3
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
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.policy.get_inputs()]
        self.output_names = [out.name for out in self.policy.get_outputs()]
        
        # Expected inputs: ['obs', 'h_in', 'c_in'] (order may vary)
        # Expected outputs: ['action', 'h_out', 'c_out']

    def infer(self):
        # Prepare flattened history buffers
        _obs_hist_flat = np.expand_dims(self.obs_hist.buffer.reshape(-1), axis=0) # (1, num_props * history_len)
        _cmds_hist_flat = np.expand_dims(self.cmds_hist.buffer.reshape(-1), axis=0) # (1, num_commands * nav_history_len)
        
        # Concatenate to form the full observation input
        _actor_input = np.concatenate([_obs_hist_flat, _cmds_hist_flat], axis=1).astype(np.float32)
        
        # Run inference
        # Inputs: {input_name: obs, h_in_name: h, c_in_name: c}
        inputs = {
            self.input_names[0]: _actor_input,
            self.input_names[1]: self.h_in,
            self.input_names[2]: self.c_in
        }
        
        outputs = self.policy.run(self.output_names, inputs)
        
        # Outputs: [action, h_out, c_out]
        action = outputs[0][0]
        self.h_in = outputs[1] # Update hidden state
        self.c_in = outputs[2] # Update cell state
        
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
            elif info == "commands":
                commands = self.commands
                if self.num_commands == 3:
                    self.logger.info(f"[Nav] commands: u={commands[0]:.3f}, v={commands[1]:.3f}, depth={commands[2]:.3f}")
                else:
                    self.logger.info(f"[Nav] commands: u={commands[0]:.3f}, v={commands[1]:.3f}")
            elif info == "base_ang_vel":
                base_ang_vel = self.robot.base_ang_vel
                self.logger.info(f"[Nav] base_ang_vel: x={base_ang_vel[0]:.3f}, y={base_ang_vel[1]:.3f}, z={base_ang_vel[2]:.3f}")
            elif info == "euler_rpy":
                euler_rpy = self.robot.euler_rpy
                self.logger.info(
                    f"[Nav] euler_rpy: roll={euler_rpy[0]:.3f}, pitch={euler_rpy[1]:.3f}, yaw={euler_rpy[2]:.3f}"
                )
            elif info == "last_action":
                last_action = self.last_action
                self.logger.info(f"[Nav] last_action: vx={last_action[0]:.3f}, vy={last_action[1]:.3f}, vyaw={last_action[2]:.3f}, pitch={last_action[3]:.3f}")

    def step(self):
        self.get_observation()
        action = self.infer()
        if self.state == "gripper_start":
            action[:3] = 0.0  # stop moving when gripper is working, only keep the pitch command
        self.loco_agent.pre_cmds = np.clip(action, self.cfg.min_action, self.cfg.max_action)
        action, _, _, _ = self.loco_agent.step()
        self.nav_timestamp += 1  # each step is 0.02s

        if self.timestamp % 10 == 0:
            self.debug_info(infos=["commands", "P_img", "last_action", "base_lin_vel", "base_ang_vel", "euler_rpy"])

        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        # wireless = False: override the joystick commands
        self.loco_agent.wireless = not self.robot.auto
        self.nav_timestamp = 0
        
        # Reset RNN states
        self.h_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)
        self.c_in = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_size), dtype=np.float32)

    @property
    def done(self):
        return False

    @property
    def commands(self):
        if self.vlm.missing:
            return np.array([-1.0, -1.0] + ([-1.0] if self.num_commands == 3 else []), dtype=np.float32)
        
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
