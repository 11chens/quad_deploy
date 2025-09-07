import os
import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from agents.homi.homi_loco_agent import HomiLocoAgent as HomiLocoAgent
from agents.homi.homi_nav_agent import HomiNavAgent
from agents.homi.homi_turn_agent import HomiTurnAgent
from agents.stand_agent import StandAgent
from nodes.homi.gripper_node import GripperNode
from nodes.homi.vlm2robot import VLM2BobotBridge
from nodes.robot_go2 import UnitreeGo2Node
from nodes.wireless_node import Go2JoystickSubscriber
from utils.logger import CustomLogger
from utils.parse_args import parse_arguments


class HomiRun(BaseManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.curr_agent_r: StandAgent = self.agents["stand"]

        self.robot: UnitreeGo2Node = self.nodes["robot"]
        self.gripper: GripperNode = self.nodes["gripper"]

        self.vlm: VLM2BobotBridge = self.nodes["vlm"]
        self.joystick: Go2JoystickSubscriber = self.nodes["joystick"]

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
            self.curr_agent_r = self.agents["turn"]
            self.curr_agent_r.reset()
            self.vlm.reset()
            self.vlm.publish_ready(ready=True)

        elif switch_to_state == "navigation":
            self.curr_agent_r = self.agents["nav"]
            self.curr_agent_r.reset()

        elif switch_to_state == "gripper_start":
            self.gripper.start_time = self.timestamp
            self.gripper.handle(grasp=self.vlm.grasp)

        if not self.state == "emergency":
            self.curr_agent_r.handle()

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

        # ================ Switch RL agent ================ #
        if self.state == "cold_start" or self.state == "recovery" and self.agents["stand"].done:
            self.logger.log_throttle("[stand] agent returns done, waiting for press [X] to switch.", 5)
            if self.joystick.X:
                return "human_teleop"
            return None

        if self.state == "human_teleop" and self.joystick.R1:
            self.logger.important(
                "[loco] The autonomous control is [ON]. Please pay attention to the safety of the robot."
            )
            return "turn"

        if self.state == "turn" and self.vlm.start:
            return "navigation"

        if self.state == "navigation" and self.vlm.gripper_start:
            return "gripper_start"

        if self.state == "gripper_start" and self.gripper.done:
            self.logger.info("Gripper done, task completed")
            return "turn"

        return None

    def handshake(self):
        self.logger.info("Waiting for VLM message")
        while not hasattr(self.vlm, "P_img"):
            time.sleep(0.01)
        self.logger.info("VLM message received, the VLM is ready!")


def main(args=None):
    nodes_dict = {
        "robot": UnitreeGo2Node,
        "vlm": VLM2BobotBridge,
        "gripper": GripperNode,
        "joystick": Go2JoystickSubscriber,
    }
    agents_dict = {
        "stand": StandAgent,
        "loco": HomiLocoAgent,
        "nav": HomiNavAgent,
        "turn": HomiTurnAgent,
    }

    logdir = "~/Data/onboard_data/onnx_models/homi"

    if not args.nosimrun:
        from nodes.keyboard_node import KeyboardNode

        nodes_dict.update({"keyboard": KeyboardNode})

    rclpy.init()

    homi_robot_node = HomiRun(
        # ros_base args
        nodes_dict=nodes_dict,
        agents_dict=agents_dict,
        node_freq_hz=200,
        start_state="cold_start",
        logdir=os.path.expanduser(logdir),
        custom_logger=CustomLogger,
        # custom args
        auto=args.auto,
        dry_run=not args.nodryrun,
        sim_run=not args.nosimrun,
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

    if not args.nosimrun:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, "eth0")

    main(args=args)
