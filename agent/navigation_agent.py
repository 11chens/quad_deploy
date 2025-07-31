import os

import numpy as np
import onnxruntime as ort

from agent.locomotion_agent import LocomotionAgent
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

        self.sigma = sigma
        self._actor_input_nav = np.zeros(77, dtype=np.float32)
        self.goal_world = np.array(goal_world, dtype=np.float32)
        self.ros_node = robot_node.ros_node
        self.timestamp = 0

        self.load_model()

    def load_model(self):
        super().load_model()
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
        self.goal_base = transform_global_xy_to_robot_xy(
            self.goal_world, self.robot_node.ros_node.lidar_node.pose_[:2], self.robot_node.ros_node.lidar_node.pose_[2]
        )
        self.obs_buf_nav[:3] = self.robot_node.projected_gravity
        self.obs_buf_nav[3:6] = self.commands * self.commands_scale
        self.obs_buf_nav[6:9] = self.base_lin_vel * self.obs_scale.lin_vel
        self.obs_buf_nav[9:12] = self.robot_node.base_ang_vel * self.obs_scale.ang_vel
        self.obs_hist_nav.append(self.obs_buf_nav)

    def infer_nav(self):
        latent_prop = self.encoder_prop_nav.run(
            None, {self.encoder_prop_nav.get_inputs()[0].name: self.obs_hist_nav.buffer.reshape(-1)}
        )[0]
        latent_rays = self.encoder_rays_nav.run(
            None, {self.encoder_rays_nav.get_inputs()[0].name: self.rays_hist.buffer.reshape(-1)}
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
        done = np.linalg.norm(self.goal_base) < self.sigma
        self.timestamp += 1
        if self.timestamp % 100 == 0:
            self.robot_node.logger.debug(
                f"Goal in Base: ({self.goal_base[0].item():.2f}, {self.goal_base[1].item():.2f})"
            )
            self.robot_node.logger.debug(
                f"Base Pose: ({self.robot_pos[0].item():.2f}, {self.robot_pos[1].item():.2f},"
                f" {self.robot_yaw.item():.2f})"
            )

        return action, None, None, done

    def reset(self):
        self.rays_hist.reset()
        self.obs_hist_nav.reset()

    @property
    def robot_pos(self):
        return self.robot_node.ros_node.lidar_node.pose_[:2]

    @property
    def robot_yaw(self):
        return self.robot_node.ros_node.lidar_node.pose_[2:]

    @property
    def rays_hist(self):
        return self.robot_node.ros_node.lidar_node.rays_hist_

    @property
    def rays(self):
        return self.robot_node.ros_node.lidar_node.rays_
