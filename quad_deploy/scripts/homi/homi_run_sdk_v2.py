import os

import rclpy
from ros_base.manager.base_manager import BaseManager, register_multiprocess_nodes
from ros_base.nodes.camera.camera_node import CameraNode
from ros_base.nodes.wireless.wireless_sdk import JoystickSDKNode as JoystickNode
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.logger import CustomLogger
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent
from quad_deploy.agents.homi.homi_nav_agent import HomiNavAgent
from quad_deploy.agents.homi.homi_turn_agent import HomiTurnAgent
from quad_deploy.agents.stand_agent import StandAgent

# The Logic Processor (FSM)
from quad_deploy.handlers.homi_handler import HomiHandler
from quad_deploy.nodes.homi.gripper_node import GripperNode
from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge

# Nodes & Agents
from quad_deploy.nodes.sdk.robot_go2_sdk import UnitreeGo2SDKNode as UnitreeGo2Node
from quad_deploy.utils.parse_args import parse_arguments


class HomiRunSDKV2(BaseManager):
    """
    Refactored Runner for Homi.
    Logic and state transitions are now in HomiHandler.
    This class focus purely on high-level orchestration and handshaking.
    """

    def __init__(self, *args, **kwargs):
        # 1. Inject the handler class before init
        kwargs["handlers_class"] = HomiHandler
        super().__init__(*args, **kwargs)

        # 2. Setup Handshake Rules (Co-design: using the new rules API)
        wait_robot = kwargs.get("wait_robot", True)
        wait_vlm = kwargs.get("wait_vlm", True)

        if wait_robot:
            self.add_handshake_rule("Robot Connection", lambda: hasattr(self.nodes.get("robot"), "low_state"))

        if wait_vlm:
            self.add_handshake_rule(
                "VLM Bridge",
                lambda: (self.nodes.get("vlm") and self.nodes.get("vlm").sigma_3d_cam is not None)
                or (self.nodes.get("robot") and self.nodes.get("robot").sim_run),
            )


def setup_environment(args):
    """Encapsulated environment setup."""
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
    cmds_dict = {}

    # Handle Simulation specifics
    if not args.nosimrun:
        from quad_deploy.nodes.sdk.keyboard_sdk import KeyboardSDKNode

        nodes_dict["keyboard"] = KeyboardSDKNode
        # Ensure `source ~/ros2_ws/install/setup.bash` first
        cmds_dict["keyboard"] = "bash -c 'ros2 run keyboard keyboard' "
        print(
            "Note: Please ensure ROS2 environment is sourced for keyboard node:\n`source ~/ros2_ws/install/setup.bash`"
        )

    if args.cam_type.lower() == "none":
        mp_nodes_dict.pop("camera", None)

    if args.gripper.lower() == "none":
        cmds_dict["sim_port"] = "socat -d -d pty,raw,echo=0,link=/tmp/pty10 pty,raw,echo=0,link=/tmp/pty11 &"

    return nodes_dict, agents_dict, mp_nodes_dict, cmds_dict


def main(args):
    nodes_dict, agents_dict, mp_nodes_dict, cmds_dict = setup_environment(args)

    # 1. Spawn sub-processes first (ROS requirement for rclpy.init order)
    processes = register_multiprocess_nodes(mp_nodes_dict, cmds_dict)

    # 2. Main Process initialization
    rclpy.init()

    # Model path setup
    logdir = os.path.expanduser("~/Data/onboard_data/onnx_models/homi")

    # 3. Start Orchestrator
    manager = HomiRunSDKV2(
        node_name="HomiOrchestrator",
        nodes_dict=nodes_dict,
        agents_dict=agents_dict,
        node_freq_hz=50 if args.nosimrun else 200,
        start_state="cold_start",
        logdir=logdir,  # Fixed: Path for ONNX models
        custom_logger=CustomLogger,
        main_loop_timer=True,
        # Custom parameters passed to Handler/Nodes
        wait_robot=args.wait_robot,
        wait_vlm=args.wait_vlm,
        sim_run=not args.nosimrun,
        auto=args.auto,
        gripper_type=args.gripper,
        cam_type=args.cam_type,
    )

    manager.start_main_loop_timer(processes)


if __name__ == "__main__":
    custom_params = [
        {"name": "--wait_robot", "type": bool, "default": True, "help": "Wait for robot hardware."},
        {"name": "--wait_vlm", "type": bool, "default": True, "help": "Wait for VLM software."},
        {"name": "--gripper", "type": str, "default": "None", "help": "Gripper type."},
        {"name": "--cam_type", "type": str, "default": "None", "help": "Camera type."},
    ]
    args = parse_arguments(custom_params)
    (
        # Unitree specific init
        ChannelFactoryInitialize(1, "lo")
        if not args.nosimrun
        else ChannelFactoryInitialize(0, "eth0")
    )
    add_debug_mode(args=args, listen_port=9999 if args.nosimrun else 7777)

    main(args)
