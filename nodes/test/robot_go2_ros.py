import os
import sys
import time

import numpy as np
from ros_base.node.base_node import BaseNode
from unitree_go.msg import LowCmd, LowState
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

from config.robot_cfgs import RobotCfg
from utils.get_crc import CRC
from utils.math_utils import VectorLPFilter, quat_rotate_inverse


class UnitreeGo2ROS(BaseNode):
    """A proxy implementation of the real Go2 robot."""

    def __init__(
        self,
        robot_class_name="Go2",
        dry_run=True,
        sim_run=True,
        auto=False,
        safe_check=False,
        dof_pos_protect_ratio=1.0,
        low_state_topic="/lowstate",
        low_cmd_topic="/lowcmd",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.robot_class_name = robot_class_name
        self.dry_run = dry_run
        self.sim_run = sim_run
        self.auto = auto
        self.safe_check = safe_check
        self.dof_pos__protect_ratio = dof_pos_protect_ratio

        self.low_state_topic = low_state_topic
        self.low_cmd_topic = low_cmd_topic
        self.crc = CRC()
        self.parse_config()
        self.init_buffers()
        self.start_handlers()

    def parse_config(self):
        """parse, set attributes from config dict, initialize buffers to speed up the computation"""

        self.NUM_DOF = getattr(RobotCfg, self.robot_class_name).NUM_DOF
        self.NUM_ACTIONS = getattr(RobotCfg, self.robot_class_name).NUM_ACTIONS
        self.dof_names = getattr(RobotCfg, self.robot_class_name).dof_names
        self.dof_map = getattr(RobotCfg, self.robot_class_name).dof_map
        self.default_joint_angles = getattr(RobotCfg, self.robot_class_name).default_joint_angles
        self.stiffness = getattr(RobotCfg, self.robot_class_name).stiffness
        self.damping = getattr(RobotCfg, self.robot_class_name).damping
        self.action_scale = getattr(RobotCfg, self.robot_class_name).action_scale
        self.computer_clip_torque = getattr(RobotCfg, self.robot_class_name).computer_clip_torque

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = np.zeros(3)
        self.gravity_vec[self.up_axis_idx] = -1
        self.torque_limits = getattr(RobotCfg, self.robot_class_name).torque_limits

        self.p_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]  # set p_gains in simulation order
            for k, v in self.stiffness.items():
                if k in name:
                    self.p_gains.append(v)
                    break

        self.d_gains = []
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]  # set d_gains in simulation order
            for k, v in self.damping.items():
                if k in name:
                    self.d_gains.append(v)
                    break

        self.default_dof_pos = np.zeros(self.NUM_DOF, dtype=np.float32)
        for i in range(self.NUM_DOF):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.default_joint_angles[name]

        self.p_gains = np.array(self.p_gains, dtype=np.float32)
        self.d_gains = np.array(self.d_gains, dtype=np.float32)

        self.joint_limits_high = getattr(RobotCfg, self.robot_class_name).joint_limits_high
        self.joint_limits_low = getattr(RobotCfg, self.robot_class_name).joint_limits_low
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.dof_pos__protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.dof_pos__protect_ratio
        self.action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.ang_vel_filter_ = VectorLPFilter(0.02, cutoff_freq=3.0, num_channels=3)

        self.reindex(self.torque_limits)
        self.reindex(self.default_dof_pos)
        self.reindex(self.joint_pos_protect_high)
        self.reindex(self.joint_pos_protect_low)

    def init_buffers(self):
        """Initialize buffers to speed up the computation"""
        self.dof_pos_ = np.zeros(self.NUM_DOF, dtype=np.float32)
        self.dof_vel_ = np.zeros(self.NUM_DOF, dtype=np.float32)
        self.base_ang_vel_ = np.zeros(3, dtype=np.float32)
        self.euler_rpy_ = np.zeros(3, dtype=np.float32)
        self.quat_wxyz_ = np.quaternion(1, 0, 0, 0)
        self.action = np.zeros(self.NUM_DOF, dtype=np.float32)

    def start_handlers(self):
        """Start the handlers for the unitree robot."""
        self.low_state_sub = self.create_subscription(LowState, self.low_state_topic, self._low_state_callback, 10)
        self.low_cmd = LowCmd()
        self.low_cmd_pub = self.create_publisher(LowCmd, self.low_cmd_topic, 10)
        self.init_motors()
        self.init_client()

    def reindex(self, sim_data):
        temp_sim_data = sim_data.copy()
        for sim_idx in range(self.NUM_DOF):
            real_idx = self.dof_map[sim_idx]
            sim_data[real_idx] = temp_sim_data[sim_idx].item()

    def clip_by_torque_limit(self, actions_scaled):
        """Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        p_limits_low = (-self.torque_limits) + self.d_gains * self.dof_vel_
        p_limits_high = (self.torque_limits) + self.d_gains * self.dof_vel_
        actions_low = (p_limits_low / self.p_gains) - self.default_dof_pos + self.dof_pos_
        actions_high = (p_limits_high / self.p_gains) - self.default_dof_pos + self.dof_pos_

        return np.clip(actions_scaled, actions_low, actions_high)

    def send_action(self, action=None, p_gains=None, d_gains=None):
        """Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        Thus, the actions has the batch dimension, whose size is 1.
        """
        self.action = action if action is not None else self.action
        self.p_gains = p_gains if p_gains is not None else self.p_gains
        self.d_gains = d_gains if d_gains is not None else self.d_gains
        if self.computer_clip_torque:
            clipped_scaled_action = self.action * self.action_scale
            clipped_scaled_action = self.clip_by_torque_limit(self.action * self.action_scale)
        else:
            self.logger.warning("Computer Clip Torque is False, the robot may be damaged.")
            clipped_scaled_action = self.action * self.action_scale
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos
        self._publish_motors_cmd(robot_coordinates_action, p_gains, d_gains)

    @property
    def base_ang_vel(self):
        self.base_ang_vel_[0] = self.low_state.imu_state.gyroscope[0]
        self.base_ang_vel_[1] = self.low_state.imu_state.gyroscope[1]
        self.base_ang_vel_[2] = self.low_state.imu_state.gyroscope[2]
        return self.base_ang_vel_

    @property
    def base_ang_vel_filter(self):
        self.ang_vel_filter_.update(self.base_ang_vel)
        return self.ang_vel_filter_.get_values()

    @property
    def euler_rpy(self):
        self.euler_rpy_[0] = self.low_state.imu_state.rpy[0]
        self.euler_rpy_[1] = self.low_state.imu_state.rpy[0]
        self.euler_rpy_[2] = self.low_state.imu_state.rpy[2]
        return self.euler_rpy_

    @property
    def projected_gravity(self):
        self.quat_wxyz_.w = self.low_state.imu_state.quaternion[0]
        self.quat_wxyz_.x = self.low_state.imu_state.quaternion[1]
        self.quat_wxyz_.y = self.low_state.imu_state.quaternion[2]
        self.quat_wxyz_.z = self.low_state.imu_state.quaternion[3]

        return quat_rotate_inverse(
            self.quat_wxyz_,
            self.gravity_vec,
        ).astype(
            np.float32
        )  # shape (3,)

    @property
    def last_action(self):
        return self.action  # shape (NUM_ACTIONS,)

    @property
    def dof_pos_rel(self):
        """Get the joint position relative to the default joint position"""
        return self.dof_pos_ - self.default_dof_pos

    @property
    def dof_pos(self):
        """Get the joint position"""
        return self.dof_pos_

    @property
    def dof_vel(self):
        return self.dof_vel_

    def _low_state_callback(self, msg: LowState):
        """store and handle proprioception data"""
        self.low_state = msg  # keep the latest low state
        # refresh dof_pos and dof_vel
        for i in range(self.NUM_DOF):
            self.dof_pos_[i] = self.low_state.motor_state[i].q
        for i in range(self.NUM_DOF):
            self.dof_vel_[i] = self.low_state.motor_state[i].dq
        # automatic safety check
        if self.safe_check:
            for i in range(self.NUM_DOF):
                if (
                    self.dof_pos_[i] > self.joint_pos_protect_high[i]
                    or self.dof_pos_[i] < self.joint_pos_protect_low[i]
                ):
                    self.logger.warning(
                        f"Joint {i}, position out of range at {self.low_state.motor_state[i].q}", once=True
                    )
                    self.logger.warning("The motors and this process shuts down.", once=True)
                    self.turn_off_motors()
                    # raise SystemExit()

    def _publish_motors_cmd(self, robot_coordinates_action, p_gains, d_gains):
        """Publish the joint commands to the robot legs in robot coordinates system.
        action: shape (NUM_DOF,), in simulation order.
        """
        if p_gains is None:
            p_gains = self.p_gains
        if d_gains is None:
            d_gains = self.d_gains

        for i in range(self.NUM_DOF):
            if self.dry_run:
                self.low_cmd.motor_cmd[i].mode = 0x00
            self.low_cmd.motor_cmd[i].q = float(robot_coordinates_action[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = float(p_gains[i])
            self.low_cmd.motor_cmd[i].kd = float(d_gains[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # self.low_cmd.crc = get_crc(self.low_cmd)
        self.low_cmd_pub.publish(self.low_cmd)

    def init_motors(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = getattr(RobotCfg, self.robot_class_name).PosStopF
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].dq = getattr(RobotCfg, self.robot_class_name).VelStopF
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # self.low_cmd.crc = get_crc(self.low_cmd)
        self.low_cmd_pub.publish(self.low_cmd)

    def turn_off_motors(self):
        """Turn off the motors"""
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].mode = 0x00
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # self.low_cmd.crc = get_crc(self.low_cmd)

        self.low_cmd_pub.publish(self.low_cmd)

    def init_client(self):
        """Close Unitree sport client, prepare for RL control"""
        if self.sim_run:
            return
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
