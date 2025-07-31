import numpy as np


class RobotCfgs:
    class Go2:
        HIGHLEVEL = 0xEE
        LOWLEVEL = 0xFF
        TRIGERLEVEL = 0xF0
        PosStopF = 2.146e9
        VelStopF = 16000.0

        NUM_DOF = 12
        NUM_ACTIONS = 12
        dof_map = [  # from isaacgym simulation joint order to real robot joint order
            3,
            4,
            5,
            0,
            1,
            2,
            9,
            10,
            11,
            6,
            7,
            8,
        ]
        dof_names = [  # NOTE: order matters. This list is the order in simulation.
            "FL_hip_joint",  # 0， 0.1
            "FL_thigh_joint",  # 1， 0.8
            "FL_calf_joint",  # 2， -1.5
            "FR_hip_joint",  # 3， -0.1
            "FR_thigh_joint",  # 4， 0.8
            "FR_calf_joint",  # 5，-1.5
            "RL_hip_joint",  # 6， 0.1
            "RL_thigh_joint",  # 7， 1.0
            "RL_calf_joint",  # 8， -1.5
            "RR_hip_joint",  # 9， -0.1
            "RR_thigh_joint",  # 10， 1.0
            "RR_calf_joint",  # 11， -1.5
        ]
        joint_limits_high = np.array(
            [
                1.0472,
                3.4907,
                -0.83776,
                1.0472,
                3.4907,
                -0.83776,
                1.0472,
                4.5379,
                -0.83776,
                1.0472,
                4.5379,
                -0.83776,
            ],
            dtype=np.float32,
        )
        joint_limits_low = np.array(
            [
                -1.0472,
                -1.5708,
                -2.7227,
                -1.0472,
                -1.5708,
                -2.7227,
                -1.0472,
                -0.5236,
                -2.7227,
                -1.0472,
                -0.5236,
                -2.7227,
            ],
            dtype=np.float32,
        )
        torque_limits = np.array(
            [
                25,
                40,
                40,
                25,
                40,
                40,
                25,
                40,
                40,
                25,
                40,
                40,
            ],
            dtype=np.float32,
        )
        turn_on_motor_mode = [0x01] * 12

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        stiffness = {"joint": 30.0}  # [N*m/rad]
        damping = {"joint": 0.75}  # [N*m*s/rad]
        action_scale = 0.25
        computer_clip_torque = True

    class AgentCfg:
        num_commands = 3

        class obs_scale:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
