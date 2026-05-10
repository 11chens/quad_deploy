import os

import rclpy
from ros_base.manager.base_manager import BaseManager, register_multiprocess_nodes
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.logger import CustomLogger
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from quad_deploy.agents.sigloma.auto_trigger_agent import AutoTriggerAgent
from quad_deploy.agents.sigloma.sigloma_loco_agent import SigLoMaLocoAgent
from quad_deploy.agents.sigloma.sigloma_nav_agent import SigLoMaNavAgent
from quad_deploy.agents.sigloma.sigloma_turn_agent import SigLoMaTurnAgent
from quad_deploy.agents.stand_agent import StandAgent

# The Logic Processor (FSM)
from quad_deploy.handlers.sigloma_handler import SigLoMaHandler

# Nodes & Agents
from quad_deploy.nodes.sdk.robot_go2_sdk import UnitreeGo2SDKNode as UnitreeGo2Node
from quad_deploy.nodes.sdk.wireless_sdk import JoystickSDKNode as JoystickNode
from quad_deploy.nodes.sigloma.gripper_node import GripperNode
from quad_deploy.nodes.sigloma.vlm2robot import VLM2BobotBridge
from quad_deploy.utils.parse_args import get_base_parser


class SigLoMaRunSDK(BaseManager):
    """
    Refactored Runner for SigLoMa.
    Logic and state transitions are now in SigLoMaHandler.
    This class focus purely on high-level orchestration and handshaking.
    """

    def __init__(self, *args, **kwargs):
        # 1. Inject the handler class before init
        kwargs["handlers_class"] = SigLoMaHandler
        super().__init__(*args, **kwargs)

        # 2. Setup Handshake Rules (Co-design: using the new rules API)
        wait_robot = kwargs.get("wait_robot", True)
        wait_vlm = kwargs.get("wait_vlm", False)

        if wait_robot:
            self.add_handshake_rule("Robot Connection", lambda: hasattr(self.nodes.get("robot"), "low_state"))

    def init_custom_variables(self):
        # Override to add anything else
        pass


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
        "loco": SigLoMaLocoAgent,
        "nav": SigLoMaNavAgent,
        "turn": SigLoMaTurnAgent,
        "auto_trigger": AutoTriggerAgent,
    }
    mp_nodes_dict = {}
    cmds_dict = {}

    # Handle Simulation specifics
    if args.sim_run:
        args.has_grasp_servo = False
        args.has_rotation_servo = False
        from quad_deploy.nodes.sdk.keyboard_sdk import KeyboardSDKNode

        nodes_dict["keyboard"] = KeyboardSDKNode
        # Ensure `source ~/ros2_ws/install/setup.bash` first
        cmds_dict["keyboard"] = "bash -c 'source ~/ros2_ws/install/setup.bash && ros2 run keyboard keyboard' &"
        print(
            "Note: Please ensure ROS2 environment is sourced for keyboard node:\n`source ~/ros2_ws/install/setup.bash`"
        )

    if args.sim_gripper or args.sim_run:
        # Ensure socat is installed and available: `sudo apt install socat`
        cmds_dict["sim_port"] = "socat -d -d pty,raw,echo=0,link=/tmp/pty10 pty,raw,echo=0,link=/tmp/pty11 &"

    return nodes_dict, agents_dict, mp_nodes_dict, cmds_dict


def main(args):
    nodes_dict, agents_dict, mp_nodes_dict, cmds_dict = setup_environment(args)

    # 1. Spawn sub-processes first (ROS requirement for rclpy.init order)
    processes = register_multiprocess_nodes(mp_nodes_dict, cmds_dict)

    # 2. Main Process initialization
    rclpy.init()

    # Model path setup
    logdir = os.path.expanduser(args.data)

    # 3. Start Orchestrator
    manager = SigLoMaRunSDK(
        node_name="SigLoMaOrchestrator",
        nodes_dict=nodes_dict,
        agents_dict=agents_dict,
        node_freq_hz=200 if args.sim_run else 50,
        start_state="cold_start",
        logdir=logdir,
        custom_logger=CustomLogger,
        # Custom parameters passed to Handler/Nodes
        wait_robot=args.wait_robot,
        wait_vlm=args.wait_vlm,
        sim_run=args.sim_run,
        dry_run=args.dry_run,
        auto=args.auto,
        has_rotation=args.has_rotation_servo,
        has_grasp=args.has_grasp_servo,
        use_sim_gripper=True if (args.sim_gripper or args.sim_run) else False,
        gripper_port=args.port,
        auto_trigger_verbose=args.auto_trigger_verbose,
    )

    manager.start_main_loop_timer(processes)


if __name__ == "__main__":
    parser = get_base_parser(description="SigLoMa SDK ")
    parser.add_argument("--wait_robot", action="store_true", default=True, help="Wait for robot hardware.")
    parser.add_argument("--wait_vlm", action="store_true", default=True, help="Wait for VLM software.")
    parser.add_argument(
        "--has_rotation_servo", action="store_true", default=False, help="Enable rotation servo capability."
    )
    parser.add_argument(
        "--has_grasp_servo", action="store_true", default=True, help="Enable real grasp servo capability."
    )
    parser.add_argument("--sim_gripper", action="store_true", default=False, help="Use simulated gripper port.")
    parser.add_argument(
        "--port", type=str, default="/dev/ttyUSB0", help="Gripper serial port, choose /dev/ttyUSB0 or /dev/ttyUSB1."
    )
    parser.add_argument(
        "--data", type=str, default="~/Data/onboard_data/onnx_models/sigloma", help="Directory of model."
    )
    parser.add_argument(
        "--auto_trigger_verbose",
        action="store_true",
        default=False,
        help="Print periodic auto-trigger condition details.",
    )
    args = parser.parse_args()
    (
        # Unitree specific init
        ChannelFactoryInitialize(1, "lo")
        if args.sim_run
        else ChannelFactoryInitialize(0, "eth0")
    )
    add_debug_mode(args=args, listen_port=7777 if args.sim_run else 9999)

    main(args)
