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

    def __init__(
        self,
        dry_run=True,
        sim_run=True,
        base_height=0.30,
        sport_state_topic="rt/sportmodestate",
    ):
        self.sim_run = sim_run
        self.sport_state_topic = sport_state_topic
        self.base_height = base_height
        level = "DEBUG" if (dry_run or sim_run) else "INFO"
        self.logger = CustomLogger(level=level)
        self.sport_client = SportClient()
        self.sport_client.Init()

        self.action = None
        self.p_gains = None
        self.d_gains = None

    def send_action(self, action, p_gains, d_gains):
        """Call sport_client to execute (vx, vy, vyaw, pitch) on real robot."""
        self.sport_client.Move(action[0], action[1], action[2])
        self.sport_client.Euler(pitch=action[3])
        self.sport_client.BodyHeight(height=self.base_height)
        self.logger.log_throttle(f"action: {action}", seconds=3)

    def _sport_state_callback(self, msg: SportModeState_):
        self.high_state = msg
        self.imu_state = msg.imu_state
        self.position = msg.position
        self.body_height = msg.body_height
        self.velocity = msg.velocity
        self.yaw_speed = msg.yaw_speed

    def start_handlers(self):
        sp_sub = ChannelSubscriber(self.sport_state_topic, SportModeState_)
        sp_sub.Init(self._sport_state_callback, 10)
        time.sleep(1.0)

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
