import os
import time

import numpy as np
import onnxruntime as ort

from agents.locomotion_agent import LocomotionAgent
from robot_real import UnitreeGo2
from utils.math_utils import CircularBuffer, transform_global_xy_to_robot_xy


class NavigationAgent(LocomotionAgent):
    def __init__(
        self,
        logdir: str,
        robot_node: UnitreeGo2,
        goal_world=[5.0, 0.0],
        sigma=0.1,
    ):
        super().__init__(logdir, robot_node)

        self.obs_buf_nav = np.zeros(12, dtype=np.float32)
        self.obs_hist_nav = CircularBuffer(10)
        self.lidar = self.robot_node.ros_manager.lidar_node

        self.sigma = sigma
        self._actor_input_nav = np.zeros(77, dtype=np.float32)
        self.goal_world = np.array(goal_world, dtype=np.float32)

        self.robot_node.logger.info("Waiting for high state message")
        while not (hasattr(self.lidar, "pose_") and hasattr(self.lidar, "rays_")):
            time.sleep(0.1)
        self.robot_node.logger.info("High state message received, the navigation agent is ready!")

        self.load_model_nav()

    def load_model_nav(self):
        models = ["policy", "encoder_prop", "encoder_rays"]
        self.ort_sessions = {}
        for name in models:
            onnx_path = os.path.join(self.logdir, "navigation_model", f"{name}.onnx")
            ort_session = ort.InferenceSession(onnx_path)
            ort_session._model_name = name
            self.ort_sessions[name] = ort_session
        self.policy_nav = self.ort_sessions["policy"]
        self.encoder_prop_nav = self.ort_sessions["encoder_prop"]
        self.encoder_rays_nav = self.ort_sessions["encoder_rays"]

    def get_observation_nav(self):
        self.goal_base = transform_global_xy_to_robot_xy(self.goal_world, self.robot_pos, self.robot_yaw)
        self.obs_buf_nav[:3] = self.robot_node.projected_gravity
        self.obs_buf_nav[3:6] = self.pre_commands * self.commands_scale
        self.obs_buf_nav[6:9] = self.base_lin_vel * self.obs_scale.lin_vel
        self.obs_buf_nav[9:12] = self.robot_node.base_ang_vel * self.obs_scale.ang_vel
        self.obs_hist_nav.append(self.obs_buf_nav)

    def infer_nav(self):
        latent_prop = self.encoder_prop_nav.run(
            None, {self.encoder_prop_nav.get_inputs()[0].name: self.obs_hist_nav.buffer.reshape(-1)}
        )[0]
        latent_rays = self.encoder_rays_nav.run(
            None, {self.encoder_rays_nav.get_inputs()[0].name: self.rays_hist.reshape(-1)}
        )[0]

        self._actor_input_nav[:12] = self.obs_buf_nav
        self._actor_input_nav[12:43] = self.rays
        self._actor_input_nav[43:59] = latent_prop
        self._actor_input_nav[59:75] = latent_rays
        self._actor_input_nav[75:77] = self.goal_base

        actions_nav = self.policy_nav.run(None, {self.policy_nav.get_inputs()[0].name: self._actor_input_nav})[0]
        return actions_nav

    def step(self):
        self.get_observation_nav()
        actions_nav = self.infer_nav()
        self.pre_commands = actions_nav
        self.post_commands()
        self.get_observation()
        action = self.infer_loco()
        if (self.robot_node.timestamp) % 100 == 0:
            self.robot_node.logger.debug(
                f"Goal in Base: ({self.goal_base[0].item():.2f}, {self.goal_base[1].item():.2f})"
            )
            self.robot_node.logger.debug(
                f"Base Pose: ({self.robot_pos[0].item():.2f}, {self.robot_pos[1].item():.2f},"
                f" {self.robot_yaw.item():.2f})"
            )
        return action, None, None, self.done

    def reset(self):
        self.obs_hist_nav.reset()
        self.lidar.rays_hist_.reset()

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
