import sys
import time

import numpy as np
from base_run import BaseRun
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from agents.loco_agent import LocoAgent
from agents.nav_agent import NavAgent
from agents.stand_agent import StandAgent
from nodes.lidar_node import LidarSubscriber


class Go2NavRun(BaseRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_agent_switch(self, done: bool) -> str | None:
        """Determine if we need to switch to a different agent based on the done flag and Joystick.
        Return None for not switching, or the name of the agent to switch to.
        """
        if self.curr_agent is self.agents["stand"] and done:
            self.logger.log_throttle("Current stand agent returns done, waiting for press [X] to switch", 5)
            if self.joystick.X:
                return "loco"
            return None

        if self.curr_agent is self.agents["loco"]:
            if self.joystick.R1:
                return "nav"
            return None

        if self.curr_agent is self.agents["nav"]:
            if self.joystick.R2 or done:
                return "loco"
            return None

        return None


def main(args=None):
    agents_dict = {
        "stand": StandAgent,
        "loco": LocoAgent,
        "nav": NavAgent,
    }
    nodes_dict = {
        "lidar": LidarSubscriber,
    }

    go2_nav_node = Go2NavRun(
        log_dir=args.logdir,
        sim_run=not args.nosimrun,
        startup_ros=True,
        agents_dict=agents_dict,
        nodes_dict=nodes_dict,
        dry_run=not args.nodryrun,
    )
    global_start_time = time.perf_counter()

    while True:
        go2_nav_node.main_loop()
        if go2_nav_node.timestamp % 1000 == 0:
            frequency = go2_nav_node.timestamp / (time.perf_counter() - global_start_time)
            go2_nav_node.logger.info(f"frequency: {frequency:.2f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Go2 robot.")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--nodryrun", action="store_true", default=False, help="Disable dry run mode."
    )  # default: False, --nodryrun:True
    parser.add_argument(
        "--logdir",
        type=str,
        default="models/onnx_models",
        help="Common directory for user's data (absolute path).",
    )
    parser.add_argument(
        "--nosimrun", action="store_true", default=False, help="Enable simulation."
    )  # default: False, --nosimrun:True
    parser.add_argument(
        "--navrun", action="store_true", help="Enable navigation agent."
    )  # default: False, --navrun:True
    args = parser.parse_args()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 7890)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    if not args.nosimrun:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, "eth0")

    main(args=args)
