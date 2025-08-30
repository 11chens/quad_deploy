import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from agents.homi.homi_loco_agent import HomiLocoAgent
from agents.homi.homi_rl_nav_agent import HomiRLNavAgent
from agents.homi.homi_rl_turn_agent import HomiRLTurnAgent
from agents.stand_agent import StandAgent
from nodes.homi.gripper_node import Gripper
from nodes.homi.robot_pub_node import RobotPubNode
from nodes.homi.vlm_sub_node import VLMSubNode
from scripts.base_run import BaseRun


class HomiRLRun(BaseRun):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gripper = Gripper()

        self.logger.info("Waiting for VLM message")
        while (
            not (hasattr(self.nodes["vlm"], "P_img"))
            or not (hasattr(self.nodes["vlm"], "start"))
            or not (hasattr(self.nodes["vlm"], "grasp"))
            or not (hasattr(self.nodes["vlm"], "turn"))
        ):
            time.sleep(0.1)
        self.logger.info("VLM message received, the robot is ready!")

        for agent_name, agent_class in self.agents_dict.items():
            self.agent_warm_up(agent_name)

    # ---- user's custom function --- #

    def grasp_handle(self):
        self.gripper.grasp_handle(self.nodes["vlm"].grasp)

    def publish_infos(self):
        # self.joystick.R1: autonomous control start
        self.nodes["robot_pub"].done = self.joystick.R1 or self.agents["homi_turn"].done or self.gripper.done

    def get_agent_switch(self, done: bool) -> str | None:
        """Determine if we need to switch to a different agent based on the done flag, joystick or VLM outputs.
        Return None for not switching, or the name of the agent to switch to.
        """
        if self.joystick.R2:
            self.logger.info("The autonomous control is [OFF]. Please remotely control the robot.")
            return "loco"

        if self.curr_agent is self.agents["stand"] and done:
            self.logger.log_throttle("Current stand agent returns done, waiting for press [X] to switch.", 5)
            if self.joystick.X:
                return "loco"
            return None

        if self.curr_agent is self.agents["loco"]:
            if self.joystick.R1:
                self.logger.info("The autonomous control is [ON]. Please pay attention to the safety of the robot.")
                return "homi_turn"
            return None

        if self.curr_agent is self.agents["homi_turn"] and self.nodes["robot_pub"].done:  # turn done
            self.logger.log_throttle("Turn done, waiting for VLM to publish start.", 3)
            if self.nodes["vlm"].start:
                return "homi_nav"
            return None

        if self.curr_agent is self.agents["homi_nav"] and self.nodes["robot_pub"].done:  # grasp done
            self.logger.log_throttle("Task completed", 3)
            return "homi_turn"

        return None

    def main_loop(self) -> None:
        """Main loop that runs the state machine to control the robot."""
        loop_start_time = time.perf_counter()
        self.emergency_handle()
        self.grasp_handle()
        if not self.EMERGENCY:  # 200 Hz
            if self.timestamp % 4 == 0:  # 50 Hz
                action, p_gains, d_gains, done = self.curr_agent.step()
            else:
                action, p_gains, d_gains, done = self.action, self.p_gains, self.d_gains, self.curr_agent.done

            switch_to_agent = self.get_agent_switch(done)
            if switch_to_agent is not None:
                self.logger.info(f"Switching to agent: {switch_to_agent}")
                self.curr_agent = self.agents[switch_to_agent]
                self.curr_agent.reset()

            self.send_action(action=action, p_gains=p_gains, d_gains=d_gains)
            self.publish_infos()

        loop_delay = time.perf_counter() - loop_start_time
        time.sleep(max(self.dt - loop_delay, 0))
        self.timestamp += 1


def main(args=None):
    agents_dict = {
        "stand": StandAgent,
        "loco": HomiLocoAgent,
        "homi_nav": HomiRLNavAgent,
        "homi_turn": HomiRLTurnAgent,
    }
    nodes_dict = {
        "vlm": VLMSubNode,
        "robot_pub": RobotPubNode,
    }

    homi_rl_node = HomiRLRun(
        log_dir=args.logdir,
        startup_ros=True,
        start_agent="stand",
        agents_dict=agents_dict,
        nodes_dict=nodes_dict,
        dry_run=not args.nodryrun,
        sim_run=not args.nosimrun,
    )
    global_start_time = time.perf_counter()

    while True:
        homi_rl_node.main_loop()
        if homi_rl_node.timestamp % 1000 == 0:
            frequency = homi_rl_node.timestamp / (time.perf_counter() - global_start_time)
            homi_rl_node.logger.info(f"frequency: {frequency:.2f} Hz")


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
        default="models/onnx_models/homi",
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
