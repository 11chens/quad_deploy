import os
import time

import numpy as np
import onnxruntime as ort

from agents.base_agent import BaseAgent
from config.nav_agent_cfg import NavAgentCfg
from nodes.robot_node import UnitreeGo2
from utils.math_utils import transform_global_xy_to_robot_xy


class NavAgent(BaseAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
    ):
        super().__init__(logdir, robot_node)

        self.lidar = self.robot_node.nodes["lidar"]
        self.loco_agent = self.robot_node.agents["loco"]
        self.parse_obs_config(NavAgentCfg)

        self._actor_input = np.zeros(self.cfg.num_actor_obs, dtype=np.float32)

        self.robot_node.logger.info("Waiting for high state message")
        while not (hasattr(self.lidar, "pose_") and hasattr(self.lidar, "rays_")):
            time.sleep(0.1)
        self.robot_node.logger.info("High state message received, the navigation agent is ready!")

        self.load_model()

    def prepare_obs_terms(self):
        """Define observation components and their corresponding scale factors."""
        self.observation_components = [
            (self.robot_node.projected_gravity, 1.0),
            (self.loco_agent.pre_commands, self.loco_agent.commands_scale),
            (self.loco_agent.base_lin_vel, self.obs_scale.lin_vel),
            (self.robot_node.base_ang_vel, self.obs_scale.ang_vel),
        ]

    def parse_obs_config(self, cfg):
        super().parse_obs_config(cfg)

        self.goal_world = np.array(self.cfg.goal_world, dtype=np.float32)
        self.sigma = self.cfg.sigma

    def load_model(self):
        models = ["policy", "encoder_prop", "encoder_rays"]
        self.ort_sessions = {}
        for name in models:
            onnx_path = os.path.join(self.logdir, "nav_model", f"{name}.onnx")
            ort_session = ort.InferenceSession(onnx_path)
            ort_session._model_name = name
            self.ort_sessions[name] = ort_session
        self.policy = self.ort_sessions["policy"]
        self.encoder_prop = self.ort_sessions["encoder_prop"]
        self.encoder_rays = self.ort_sessions["encoder_rays"]

    def infer(self):
        latent_prop = self.encoder_prop.run(
            None, {self.encoder_prop.get_inputs()[0].name: self.obs_hist.buffer.reshape(-1)}
        )[0]
        latent_rays = self.encoder_rays.run(None, {self.encoder_rays.get_inputs()[0].name: self.rays_hist.reshape(-1)})[
            0
        ]

        self._actor_input[:12] = self.obs_buf
        self._actor_input[12:43] = self.rays
        self._actor_input[43:59] = latent_prop
        self._actor_input[59:75] = latent_rays
        self._actor_input[75:77] = self.goal_base

        actions = self.policy.run(None, {self.policy.get_inputs()[0].name: self._actor_input})[0]
        return actions

    def step(self):
        self.goal_base = transform_global_xy_to_robot_xy(self.goal_world, self.robot_pos, self.robot_yaw)
        self.get_observation()
        actions = self.infer()
        self.loco_agent.pre_commands = actions
        action, _, _, _ = self.loco_agent.step()
        if (self.robot_node.timestamp) % 200 == 0:
            self.robot_node.logger.debug(
                f"Goal in Base: ({self.goal_base[0].item():.2f}, {self.goal_base[1].item():.2f})"
            )
            self.robot_node.logger.debug(
                f"Base Pose: ({self.robot_pos[0].item():.2f}, {self.robot_pos[1].item():.2f},"
                f" {self.robot_yaw.item():.2f})"
            )
        return action, None, None, self.done

    def reset(self):
        self.obs_hist.reset()
        self.lidar.rays_hist_.reset()
        self.loco_agent.wireless = False

    @property
    def robot_pos(self):
        return self.lidar.pose_[:2]

    @property
    def robot_yaw(self):
        return self.lidar.pose_[2]

    @property
    def rays_hist(self):
        return np.log2(np.clip(self.lidar.rays_hist_.buffer, 0.1, 5.0))

    @property
    def rays(self):
        return np.log2(np.clip(self.lidar.rays_, 0.1, 5.0))

    @property
    def done(self):
        return np.linalg.norm(self.goal_base) < self.sigma
