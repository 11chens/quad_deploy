import os
import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager

from agents.homi.homi_loco_agent import HomiLocoAgent as HomiLocoAgent
from agents.homi.homi_nav_agent import HomiNavAgent
from agents.homi.homi_turn_agent import HomiTurnAgent
from agents.stand_agent import StandAgent
from nodes.homi.camera_node import CameraNode, ImageViewer
from nodes.homi.gripper_node import GripperNode
from nodes.homi.vlm2robot import VLM2BobotBridge
from nodes.ros.robot_go2_ros import UnitreeGo2ROS
from nodes.ros.wireless_ros import JoystickRosNode
from utils.logger import CustomLogger
from utils.parse_args import parse_arguments


class ViewerManager(BaseManager):
    def __init__(
        self,
        node_name="ViewerManager",
        *args,
        **kwargs,
    ):
        super().__init__(node_name=node_name, *args, **kwargs)


def main(args=None):
    nodes_dict = {
        "camera": CameraNode,
        "viewer": ImageViewer,
    }

    if not args.nosimrun:
        from nodes.ros.keyboard_ros import KeyboardRos

        nodes_dict.update({"keyboard": KeyboardRos})

    rclpy.init()

    homi_robot_node = ViewerManager(
        # ros_base args
        nodes_dict=nodes_dict,
        custom_logger=CustomLogger,
    )

    homi_robot_node.start_main_loop()


if __name__ == "__main__":
    args = parse_arguments()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 7890)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args=args)
