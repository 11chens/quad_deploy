import sys
import threading
import time

import numpy as np
import rclpy
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

from agents.base import BaseAgent
from agents.locomotion_agent import LocomotionAgent
from agents.navigation_agent import NavigationAgent
from agents.stand_agent import StandAgent
from nodes.wireless_node import Go2JoystickSubscriber
from robot_real import UnitreeGo2


class Go2NavRun(UnitreeGo2):
    def __init__(self, simrun, navrun, agents_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simrun = simrun
        self.navrun = navrun
        self.agents_dict = agents_dict
        self.agents = {}
        self.timestamp = 0
        self.curr_agent = None
        self.EMERGENCY = False

        if self.simrun or self.navrun:
            from nodes.ros_manager import RosManager

            self.ros_manager = RosManager(
                lidar=self.navrun,
                key=self.simrun,
            )
            ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_manager,), daemon=True)
            ros_thread.start()

        if self.navrun:
            self.agents_dict["nav"] = NavigationAgent

    def register_agent(self, name: str, agent: BaseAgent):
        """Register the agent to the robot node.
        Args:
            name (str): The name of the agent.
            agent (BaseAgent): The agent to register.
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Expected BaseAgent, got {type(agent)}")
        self.agents[name] = agent
        self.logger.info(f"Registered Agent: {name}")

    def create_and_register_agent(self, logdir: str, agent_name: str, agent_class: BaseAgent):
        try:
            agent = agent_class(logdir=logdir, robot_node=self)
            self.register_agent(agent_name, agent)
            self.logger.info(f"Successfully registered {agent_name} agent")
        except Exception as e:
            self.logger.error(f"Failed to create/register agent for {agent_name}: {str(e)}")

    def init_client(self):
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def start_handlers(self, start_agent: str = "stand"):
        super().start_handlers()
        if self.simrun:
            self.logger.debug(f"Start up keyboard node")
            self.joystick = self.ros_manager.key_node
        else:
            self.logger.debug(f"Start up joystick node")
            self.joystick = Go2JoystickSubscriber()

        self.curr_agent = self.agents[start_agent]
        self.curr_agent.reset()

    def agent_warm_up(self, agent_name):
        infer_start_time = time.perf_counter()
        num_warm_iter = 50
        for i in range(num_warm_iter):
            _, _, _, _ = self.agents[agent_name].step()
        delay = (time.perf_counter() - infer_start_time) / num_warm_iter
        self.logger.debug(f"[{agent_name}] Infer delay: {delay*1e3:.3f} ms")

    def get_agent_switch(self, done: bool) -> str | None:
        """Determine if we need to switch to a different agent based on the done flag and Joystick.
        Return None for not switching, or the name of the agent to switch to.
        """
        if self.curr_agent is self.agents["stand"] and done:
            self.logger.log_throttle("Current stand agent returns done, waiting for press [X] to switch", 5)
            if self.joystick.X:
                return "loco"
            return None

        if "nav" in self.agents_dict:
            if self.curr_agent is self.agents["loco"]:
                if self.joystick.R1:
                    return "nav"
                return None

            if self.curr_agent is self.agents["nav"]:
                if self.joystick.R2 or done:
                    return "loco"
                return None

        return None

    def emergency_handle(self):
        if self.joystick.L2:
            self.EMERGENCY = True
            self.turn_off_motors()
            self.logger.warning("L2 is pressed, The motors shuts down.")
            self.joystick.reset()

        if self.joystick.L1 and self.EMERGENCY:
            self.EMERGENCY = False
            self.logger.info("L1 is pressed, robot will recovery.")
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()

    def main_loop(self) -> None:
        """Main loop that runs the state machine to control the robot."""
        self.emergency_handle()
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


def main(args=None):
    agents_dict = {
        "stand": StandAgent,
        "loco": LocomotionAgent,
    }
    go2_nav_node = Go2NavRun(
        simrun=not args.nosimrun,
        navrun=args.navrun,
        agents_dict=agents_dict,
        dry_run=not args.nodryrun,
    )

    if args.nosimrun:
        go2_nav_node.init_client()
        go2_nav_node.init_motors()

    for agent_name, agent_class in go2_nav_node.agents_dict.items():
        go2_nav_node.create_and_register_agent(logdir=args.logdir, agent_name=agent_name, agent_class=agent_class)

    go2_nav_node.start_handlers()
    go2_nav_node.agent_warm_up("loco")
    go2_nav_node.agent_warm_up("nav")

    # input("Press enter to start")

    dt = 0.005
    global_start_time = time.perf_counter()

    while True:
        loop_start_time = time.perf_counter()
        go2_nav_node.main_loop()
        loop_delay = time.perf_counter() - loop_start_time
        time.sleep(max(dt - loop_delay, 0))
        go2_nav_node.timestamp += 1
        if go2_nav_node.timestamp % 100 == 0:
            frequency = go2_nav_node.timestamp / (time.perf_counter() - global_start_time)
            # go2_nav_node.logger.debug(f"frequency: {frequency:.2f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Go2 robot.")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--nodryrun", action="store_true", help="Disable dry run mode."
    )  # default: False, --nodryrun:True
    parser.add_argument(
        "--logdir",
        type=str,
        default="example/quad_deploy/models/onnx_models",
        help="Common directory for user's data (absolute path).",
    )
    parser.add_argument("--nosimrun", action="store_true", help="Enable simulation.")  # default: False, --nosimrun:True
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
        rclpy.init()
    else:
        ChannelFactoryInitialize(0, "eth0")

    main(args=args)
