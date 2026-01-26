import os
import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager, register_multiprocess_nodes
from ros_base.nodes.camera.camera_node import CameraNode
from ros_base.nodes.wireless.wireless_sdk import JoystickSDKNode as JoystickNode
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.logger import CustomLogger
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent as HomiLocoAgent
from quad_deploy.agents.homi.homi_nav_agent import HomiNavAgent
from quad_deploy.agents.homi.homi_nav_gru_agent import HomiNavGruAgent as HomiNavAgent
from quad_deploy.agents.homi.homi_turn_agent import HomiTurnAgent
from quad_deploy.agents.stand_agent import StandAgent
from quad_deploy.nodes.homi.gripper_node import GripperNode
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge
from quad_deploy.nodes.sdk.robot_go2_sdk import UnitreeGo2SDKNode as UnitreeGo2Node
from quad_deploy.utils.parse_args import parse_arguments


class HomiRunSDK(BaseManager):
    def __init__(
        self,
        wait_robot=True,
        wait_vlm=True,
        node_name="HomiRunSDK",
        *args,
        **kwargs,
    ):
        """Main class to run the Homi robot for manipulation tasks using VLM and joystick.
        Args:
            wait_robot (bool): Whether to wait for the robot to return lowstate before starting.
            wait_vlm (bool): Whether to wait for VLM to return highstate before starting navigation.
            node_name (str): Name of the ROS2 node.
        """
        super().__init__(node_name=node_name, *args, **kwargs)

        self.wait_robot = wait_robot
        self.wait_vlm = wait_vlm

        self.curr_agent_r: StandAgent = self.agents["stand"]

        self.robot: UnitreeGo2Node = self.nodes["robot"]
        self.gripper: GripperNode = self.nodes["gripper"]

        self.vlm: VLM2BobotBridge = self.nodes["vlm"]
        self.joystick: JoystickNode = self.nodes["joystick"]

    def get_state_switch(self):
        """Determine if we need to switch to a different agent based on the done flag, joystick or VLM outputs.
        Return None for not switching, or the name of the agent to switch to.
        """
        if self.joystick.L2:
            self.logger.warning("Robot will shut down motors.")
            return "emergency"

        if self.joystick.L1 and self.state == "emergency":
            self.logger.info("Robot will recovery.")
            return "recovery"

        if self.joystick.R2:
            self.logger.info("The autonomous control is [OFF]. Please control the robot using joystck.")
            return "human_teleop"

        if self.joystick.B:
            grasp = not self.gripper.grasp_state  # Toggle grasp state
            self.gripper.handle(grasp=grasp)
            return None

        # ================ Switch RL agent ================ #
        if (self.state == "cold_start" or self.state == "recovery") and self.agents["stand"].done:
            self.logger.log_once("[stand] agent returns done, waiting for press [X] to switch.")
            if self.joystick.X:
                return "human_teleop"
            return None

        if self.state == "human_teleop" and self.joystick.R1:
            self.logger.important(
                "[loco] The autonomous control is [ON]. Please pay attention to the safety of the robot."
            )
            return "turn"

        if self.state == "turn" and self.wait_vlm:
            if not self.robot.sim_run:
                self.logger.log_once("Waiting for VLM message: <sigma_3d_cam> and <object_ready>.")
                if self.vlm.sigma_3d_cam is not None and self.vlm.object_ready:
                    self.logger.info("VLM message <sigma_3d_cam> and <object_ready> received, starting navigation!")
                    return "navigation"
            else:
                self.logger.log_once("Waiting for VLM message: <sigma_3d_cam>.")
                if self.vlm.sigma_3d_cam is not None:
                    self.logger.info("VLM message <sigma_3d_cam> received, starting navigation!")
                    return "navigation"

        if self.state == "navigation" and self.joystick.A:
            return "gripper_start"

        if self.state == "gripper_start" and self.gripper.done:
            self.vlm.publish_grasp_done(grasp_done=True)
            return "turn"  # after grasp done, turn to box

        if self.state == "gripper_done" and self.vlm.vlm_done:
            return "turn"

        return None

    def state_handle(self, switch_to_state):
        if switch_to_state == "emergency":
            self.robot.turn_off_motors()

        elif switch_to_state == "recovery":
            self.curr_agent_r = self.agents["stand"]
            self.curr_agent_r.reset()
            self.logger.reset()
            self.timestamp = 0
            self.robot.init_motors()

        elif switch_to_state == "cold_start":
            self.curr_agent_r = self.agents["stand"]
            self.curr_agent_r.reset()

        elif switch_to_state == "human_teleop":
            self.curr_agent_r = self.agents["loco"]
            self.curr_agent_r.reset()

        elif switch_to_state == "turn":
            self.logger.reset()
            self.vlm.reset()
            self.curr_agent_r = self.agents["turn"]
            self.curr_agent_r.reset()
            self.vlm.publish_rl_ready(rl_ready=True)

        elif switch_to_state == "navigation":
            self.curr_agent_r = self.agents["nav"]
            self.curr_agent_r.reset()

        elif switch_to_state == "gripper_start":
            self.gripper.start_time = self.timestamp
            grasp = not self.gripper.grasp_state  # Toggle grasp state
            self.gripper.handle(grasp=grasp)

        # elif switch_to_state == "gripper_done":
        #     self.vlm.publish_grasp_done(grasp_done=True)

        if not self.state == "emergency":
            self.curr_agent_r.handle()
            self.vlm.publish_robot_euler_rpy(euler_rpy=self.robot.euler_rpy)

    def handshake(self):
        if self.wait_robot:
            self.logger.log_once("Waiting for robot low state message")
            if hasattr(self.robot, "low_state"):
                self.logger.info("Low state message received, the robot is ready to go")
                return True
            return False


def update_dict(
    args=None, nodes_dict: dict = {}, agents_dict: dict = {}, mp_nodes_dict: dict = {}, cmds_dict: dict = {}
):
    """Update the dicts for debugging in a simulated environment."""
    if not args.nosimrun:
        args.cam_type = "none"
        args.gripper = "none"
        cmds_dict["keyboard"] = (
            # "bash -c 'LD_LIBRARY_PATH=$HOME/miniforge3/envs/humble/lib:$LD_LIBRARY_PATH; source ~/ros2_ws/install/setup.bash; ros2 run keyboard keyboard' &"
            "bash -c 'source ~/ros2_ws/install/setup.bash; ros2 run keyboard keyboard' &"
        )

        from quad_deploy.nodes.sdk.keyboard_sdk import KeyboardSDKNode as KeyboardNode

        nodes_dict.update({"keyboard": KeyboardNode})

    if args.cam_type.lower() == "none":
        mp_nodes_dict.pop("camera")

    if args.gripper.lower() == "none":
        cmds_dict["sim_port"] = "socat -d -d pty,raw,echo=0,link=/tmp/pty10 pty,raw,echo=0,link=/tmp/pty11 &"

    return nodes_dict, agents_dict, mp_nodes_dict, cmds_dict


def main(args=None):
    nodes_dict = {
        "robot": UnitreeGo2Node,
        "vlm": VLM2BobotBridge,
        "gripper": GripperNode,
        "joystick": JoystickNode,
    }
    agents_dict = {
        "stand": StandAgent,
        "loco": HomiLocoAgent,
        "nav": HomiNavAgent,
        "turn": HomiTurnAgent,
    }

    mp_nodes_dict = {"camera": CameraNode}

    logdir = "~/Data/onboard_data/onnx_models/homi"

    nodes_dict, agents_dict, mp_nodes_dict, cmds_dict = update_dict(
        args=args, nodes_dict=nodes_dict, agents_dict=agents_dict, mp_nodes_dict=mp_nodes_dict
    )

    # Importantly, sub processes call rclpy.init() first, then the main process can call rclpy.init(), because rclpy can only be initialized once.
    processes = register_multiprocess_nodes(mp_nodes_dict, cmds_dict)

    rclpy.init()

    homi_robot_node = HomiRunSDK(
        # ros_base args
        nodes_dict=nodes_dict,
        agents_dict=agents_dict,
        node_freq_hz=50 if args.nosimrun else 200,
        start_state="cold_start",
        logdir=os.path.expanduser(logdir),
        custom_logger=CustomLogger,
        # log_freq=True,
        # custom args
        auto=args.auto,
        dry_run=not args.nodryrun,
        sim_run=not args.nosimrun,
        wait_robot=args.wait_robot,
        wait_vlm=args.wait_vlm,
        gripper_type=args.gripper,
        cam_type=args.cam_type,
    )

    homi_robot_node.start_main_loop_timer(processes)


if __name__ == "__main__":
    custom_parameters = [
        {"name": "--wait_robot", "type": bool, "default": True, "help": "Waiting for robot return lowstate."},
        {"name": "--wait_vlm", "type": bool, "default": True, "help": "Waiting for VLM return highstate."},
        {
            "name": "--gripper",
            "type": str,
            "default": "None",
            "help": "Deciding what type of gripper to use (two_fingers, three_fingers, None).",
        },
        {"name": "--cam_type", "type": str, "default": "None", "help": "Camera type to use (zed, go2, None)."},
    ]

    args = parse_arguments(custom_parameters)

    if not args.nosimrun:
        ChannelFactoryInitialize(1, "lo")
        add_debug_mode(args=args, listen_port=7777)  # local attach
    else:
        ChannelFactoryInitialize(0, "eth0")
        add_debug_mode(args=args, listen_port=9999)  # unitree_wireless
        # add_debug_mode(args=args, listen_port=7777)  # unitree_wire

    main(args=args)
