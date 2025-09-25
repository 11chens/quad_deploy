import os
import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager
from ros_base.utils.logger import CustomLogger

from quad_deploy.agents.homi.homi_loco_agent import HomiLocoAgent as HomiLocoAgent
from quad_deploy.nodes.homi.vlm2ui import VLM2UIBridge
from quad_deploy.utils.parse_args import parse_arguments


class UIManager(BaseManager):
    def __init__(
        self,
        node_name="UIManager",
        *args,
        **kwargs,
    ):
        """Main class to manage the UI interactions for the Homi robot using VLM. This manager is executed on the PC generally.
        Args:
            node_name (str): Name of the ROS2 node.
        """
        super().__init__(node_name=node_name, *args, **kwargs)

        self.vlm: VLM2UIBridge = self.nodes["vlm"]
        self.vlm.publish_ui_ready(True)

    def main_loop(self):
        if self.vlm.inquiry:
            ret = input("Which direction to turn? (left/right): ").strip().upper()
            self.vlm.publish_turn(ret)
            self.vlm.inquiry = False


def main(args=None):
    nodes_dict = {
        "vlm": VLM2UIBridge,
    }

    rclpy.init()

    homi_robot_node = UIManager(
        # ros_base args
        nodes_dict=nodes_dict,
        custom_logger=CustomLogger,
        node_freq_hz=10,
        # ImageViewer args
        show_raw_image=args.show_raw,
    )

    homi_robot_node.start_main_loop_timer()


if __name__ == "__main__":
    custom_parameters = [
        {"name": "--show_raw", "action": "store_true", "default": False, "help": "Show raw image"},
    ]
    args = parse_arguments(custom_parameters)

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 8888)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args=args)
