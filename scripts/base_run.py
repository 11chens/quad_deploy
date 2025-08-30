import sys
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from agents.loco_agent import LocoAgent
from agents.stand_agent import StandAgent
from nodes.robot_node import UnitreeGo2
from nodes.wireless_node import Go2JoystickSubscriber


class BaseRun(UnitreeGo2):
    def __init__(
        self,
        log_dir=None,
        startup_ros=True,
        start_agent="stand",
        agents_dict={},
        nodes_dict={},
        num_warm_iter=50,
        dt=0.005,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.log_dir = log_dir
        self.startup_ros = startup_ros
        self.start_agent = start_agent
        self.agents_dict = agents_dict
        self.nodes_dict = nodes_dict
        self.num_warm_iter = num_warm_iter
        self.dt = dt
        self.agents = {}
        self.nodes = {}
        self.timestamp = 0
        self.EMERGENCY = False

        self.register_node()
        self.register_agent()

        self.start_handlers()

        # for agent_name, agent_class in self.agents_dict.items():
        #     self.agent_warm_up(agent_name)

    def register_node(self):
        if not self.startup_ros:
            return
        else:
            import rclpy

            rclpy.init()
            from nodes.ros_manager import RosManager

            self.ros_manager = RosManager()
            for node_name, node_class in self.nodes_dict.items():
                self.nodes[node_name] = node_class(ros_manager=self.ros_manager)
                self.logger.info(f"Successfully registered {node_name} node")

            ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_manager,), daemon=True)
            ros_thread.start()

    def register_agent(self):
        for agent_name, agent_class in self.agents_dict.items():
            self.agents[agent_name] = agent_class(logdir=self.log_dir, robot_node=self)
            self.logger.info(f"Successfully registered {agent_name} agent")

    def start_handlers(self):
        super().start_handlers()
        if not self.sim_run:
            self.joystick = Go2JoystickSubscriber()
        else:
            from nodes.keyboard_node import KeyboardSubscriber

            self.joystick = KeyboardSubscriber(ros_manager=self.ros_manager)

        self.curr_agent = self.agents[self.start_agent]
        self.curr_agent.reset()

    def agent_warm_up(self, agent_name):
        infer_start_time = time.perf_counter()
        for i in range(self.num_warm_iter):
            _, _, _, _ = self.agents[agent_name].step()
        delay = (time.perf_counter() - infer_start_time) / self.num_warm_iter
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

        return None

    def emergency_handle(self):
        if self.joystick.L2 and not self.EMERGENCY:
            self.EMERGENCY = True
            self.turn_off_motors()
            self.logger.warning("L2 is pressed, The motors shuts down.")

        if self.joystick.L1 and self.EMERGENCY:
            self.EMERGENCY = False
            self.logger.info("L1 is pressed, robot will recovery.")
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()
            self.timestamp = 0
            self.init_motors()

    def main_loop(self) -> None:
        """Main loop that runs the state machine to control the robot."""
        loop_start_time = time.perf_counter()
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

        loop_delay = time.perf_counter() - loop_start_time
        time.sleep(max(self.dt - loop_delay, 0))
        self.timestamp += 1


def main(args=None):
    agents_dict = {
        "stand": StandAgent,
        "loco": LocoAgent,
    }
    nodes_dict = {}

    go2_base_node = BaseRun(
        log_dir=args.logdir,
        startup_ros=not args.nosimrun,
        start_agent="stand",
        agents_dict=agents_dict,
        nodes_dict=nodes_dict,
        dry_run=not args.nodryrun,
        sim_run=not args.nosimrun,
    )
    global_start_time = time.perf_counter()

    while True:
        go2_base_node.main_loop()
        if go2_base_node.timestamp % 1000 == 0:
            frequency = go2_base_node.timestamp / (time.perf_counter() - global_start_time)
            go2_base_node.logger.info(f"frequency: {frequency:.2f} Hz")


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
