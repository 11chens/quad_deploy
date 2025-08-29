import os
import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from utils.logger import CustomLogger


class UnitreeGo2MPC:
    """Go2 robot sport client control."""

    def __init__(self):
        self.logger = CustomLogger()
        self.sport_client = SportClient()
        self.sport_client.Init()
        self.gripper = gripper()
        sp_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sp_sub.Init(self._sport_state_callback, 10)
        time.sleep(1.0)

    def send_action(self, action, p_gains, d_gains):
        # TODO: call sport_client to execute (vx, vy, vyaw, pitch) on real robot
        self.sport_client.Move(action[0], action[1], action[2])
        self.sport_client.Euler(action[3])
        self.logger.log_throttle(f"action: {action}", seconds=3)

    def publish_done(self, agent_done):
        if "vlm" in self.nodes:
            self.nodes["vlm"].done = self.grasp_done if self.grasp_done else agent_done

    def grasp_handle(self):
        if "vlm" in self.nodes:
            if self.nodes["vlm"].grasp:  # last grasp -> curr grasp
                # TODO: call gripper function to grasp
                self.gripper.close()
                # TODO: wait until grasping is done
                self.logger.info("Start grasping.")
                time.sleep(2.0)  # wait for grasping
                self.grasp_done = True
                self.logger.info("Grasping done.")
            else:
                # Important: reset done to avoid publishing self.nodes["vlm"].done = True
                self.grasp_done = False

    def _sport_state_callback(self, msg: SportModeState_):
        self.high_state = msg
        self.imu_state = msg.imu_state
        self.position = msg.position
        self.body_height = msg.body_height
        self.velocity = msg.velocity
        self.yaw_speed = msg.yaw_speed

    @property
    def base_ang_vel(self):
        return np.array(
            self.high_state.imu_state.gyroscope,
            dtype=np.float32,
        )

    @property
    def base_euler(self):
        return np.array(
            self.high_state.imu_state.rpy,
            dtype=np.float32,
        )


class gripper:
    def __init__(self):
        pass

    def open(self):
        pass

    def close(self):
        pass
