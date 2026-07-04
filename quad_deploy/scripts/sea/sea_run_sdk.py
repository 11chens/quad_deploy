import os

import rclpy
from ros_base.manager.base_manager import BaseManager, register_multiprocess_nodes
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.logger import CustomLogger
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from quad_deploy.agents.sea.sea_loco_agent import SEALocoAgent
from quad_deploy.agents.sea.sea_nav_agent import SEANavAgent
from quad_deploy.agents.stand_agent import StandAgent
from quad_deploy.config.sea.sea_nav_agent_cfg import SEANavAgentCfg

# The Logic Processor (FSM)
from quad_deploy.handlers.sea_handler import SEAHandler

# Nodes
from quad_deploy.nodes.sdk.robot_go2_sdk import UnitreeGo2SDKNode as UnitreeGo2Node
from quad_deploy.nodes.sdk.wireless_sdk import JoystickSDKNode as JoystickNode
from quad_deploy.nodes.sea.pose_sub_node import PoseSubNode
from quad_deploy.nodes.sea.rays_sub_node import RaysSubNode
from quad_deploy.utils.parse_args import get_base_parser


class SEARunSDK(BaseManager):
    """Top-level orchestrator for SEA-Nav deployment.

    Logic and state transitions live in :class:`SEAHandler`; this class
    only registers nodes / agents and runs the handshake.
    """

    def __init__(self, *args, **kwargs):
        kwargs["handlers_class"] = SEAHandler
        super().__init__(*args, **kwargs)

        wait_robot = kwargs.get("wait_robot", True)
        if wait_robot:
            self.add_handshake_rule("Robot Connection", lambda: hasattr(self.nodes.get("robot"), "low_state"))

    def init_custom_variables(self):
        pass


def setup_environment(args):
    """Build the node, agent and multiprocess dictionaries the manager
    expects, plus any auxiliary external commands (keyboard listener etc.)
    needed for the current run mode.
    """
    nodes_dict = {
        "robot": UnitreeGo2Node,
        "joystick": JoystickNode,
        "rays_sub": RaysSubNode,
        "pose_sub": PoseSubNode,
    }
    agents_dict = {
        "stand": StandAgent,
        "loco": SEALocoAgent,
        "nav": SEANavAgent,
    }
    mp_nodes_dict = {}
    cmds_dict = {}

    if args.sim_run:
        from quad_deploy.nodes.sdk.keyboard_sdk import KeyboardSDKNode

        nodes_dict["keyboard"] = KeyboardSDKNode
        # Ensure `source ~/ros2_ws/install/setup.bash` first
        cmds_dict["keyboard"] = "bash -c 'source ~/ros2_ws/install/setup.bash && ros2 run keyboard keyboard' &"
        print(
            "Note: Please ensure ROS2 environment is sourced for keyboard node:\n`source ~/ros2_ws/install/setup.bash`"
        )

    return nodes_dict, agents_dict, mp_nodes_dict, cmds_dict


def maybe_override_goal_world(args):
    """Override ``SEANavAgentCfg.goal_world`` from cmdline arguments
    before the Nav agent is instantiated.

    The Nav agent reads and caches ``cfg.goal_world`` in ``__init__``, so the
    override has to happen on the cfg class itself prior to manager
    construction.
    """
    if args.goal_x is None and args.goal_y is None:
        return
    gx = args.goal_x if args.goal_x is not None else SEANavAgentCfg.goal_world[0]
    gy = args.goal_y if args.goal_y is not None else SEANavAgentCfg.goal_world[1]
    SEANavAgentCfg.goal_world = [float(gx), float(gy)]


def main(args):
    nodes_dict, agents_dict, mp_nodes_dict, cmds_dict = setup_environment(args)
    maybe_override_goal_world(args)

    # Loud banner so the operator can quickly tell whether motor commands will
    # actually drive the robot or are masked out by dry_run=True.
    motor_note = "MOTOR ENABLED" if not args.dry_run else "MOTOR DISABLED (dry_run, motor mode=0x00)"
    print(
        "\n========== SEA-Nav RunSDK ==========\n"
        f"  sim_run = {args.sim_run}   dry_run = {args.dry_run}   auto = {args.auto}\n"
        f"  data    = {args.data}\n"
        f"  goal    = ({args.goal_x}, {args.goal_y})  (None means cfg default)\n"
        f"  safe_stop_timeout_s={args.safe_stop_timeout_s}, safe_stop_recover_s={args.safe_stop_recover_s}\n"
        f"  --> {motor_note}\n"
        "====================================\n"
    )

    processes = register_multiprocess_nodes(mp_nodes_dict, cmds_dict)

    rclpy.init()

    logdir = os.path.expanduser(args.data)

    # The rays / pose subscriber kwargs live on the Nav cfg so that the Nav
    # agent and the two subscriber nodes share a single source of truth. The
    # manager passes every kwarg below to each registered node; nodes that
    # don't recognise a kwarg ignore it.
    nav_cfg = SEANavAgentCfg

    manager = SEARunSDK(
        node_name="SEANavOrchestrator",
        nodes_dict=nodes_dict,
        agents_dict=agents_dict,
        node_freq_hz=200 if args.sim_run else 50,
        start_state="cold_start",
        logdir=logdir,
        custom_logger=CustomLogger,
        # ---- Common runtime flags ----
        wait_robot=args.wait_robot,
        sim_run=args.sim_run,
        dry_run=args.dry_run,
        auto=args.auto,
        # ---- Rays subscriber kwargs (forwarded by manager to RaysSubNode) ----
        rays_topic=nav_cfg.rays_topic,
        rays_msg_type=nav_cfg.rays_msg_type,
        rays_num_rays=nav_cfg.rays_num_rays,
        rays_range_min=nav_cfg.rays_range_min,
        rays_range_max=nav_cfg.rays_range_max,
        rays_frame_id=nav_cfg.rays_frame_id,
        rays_timeout_s=nav_cfg.rays_timeout_s,
        rays_expected_freq_hz=nav_cfg.rays_expected_freq_hz,
        # ---- Pose subscriber kwargs (forwarded by manager to PoseSubNode) ----
        pose_topic=nav_cfg.pose_topic,
        pose_frame_id=nav_cfg.pose_frame_id,
        pose_timeout_s=nav_cfg.pose_timeout_s,
        pose_expected_freq_hz=nav_cfg.pose_expected_freq_hz,
        # ---- Safe-stop FSM thresholds (forwarded to SEAHandler) ----
        safe_stop_timeout_s=args.safe_stop_timeout_s,
        safe_stop_recover_s=args.safe_stop_recover_s,
    )

    manager.start_main_loop_timer(processes)


if __name__ == "__main__":
    parser = get_base_parser(description="SEA-Nav SDK")
    parser.add_argument("--wait_robot", action="store_true", default=True, help="Wait for robot hardware.")
    parser.add_argument(
        "--data",
        type=str,
        default="~/Data/onboard_data/onnx_models/sea_nav",
        help="Directory of ONNX models (must contain nav_model/model.onnx and loco_model/model.onnx).",
    )
    parser.add_argument(
        "--goal_x",
        type=float,
        default=None,
        help="Override SEANavAgentCfg.goal_world[0] (world frame x in meters).",
    )
    parser.add_argument(
        "--goal_y",
        type=float,
        default=None,
        help="Override SEANavAgentCfg.goal_world[1] (world frame y in meters).",
    )
    parser.add_argument(
        "--safe_stop_timeout_s",
        type=float,
        default=1.0,
        help="navigation -> safe_stop after this many seconds of stale perception.",
    )
    parser.add_argument(
        "--safe_stop_recover_s",
        type=float,
        default=0.5,
        help="safe_stop -> navigation requires fresh perception for this long + an R1 press.",
    )
    args = parser.parse_args()

    (ChannelFactoryInitialize(1, "lo") if args.sim_run else ChannelFactoryInitialize(0, "eth0"))
    add_debug_mode(args=args, listen_port=7778 if args.sim_run else 9998)

    main(args)
