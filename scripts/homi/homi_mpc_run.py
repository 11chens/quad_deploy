import sys
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from agents.homi.homi_nav_agent import HomiNavAgent
from agents.homi.homi_turn_agent import HomiTurnAgent
from nodes.homi.mpc_node import UnitreeGo2MPC
from nodes.homi.robot_pub_node import RobotPubNode
from nodes.homi.vlm_sub_node import VLMSubNode
from nodes.wireless_node import Go2JoystickSubscriber


class HomiMPCRun(UnitreeGo2MPC):
    def __init__(
        self, log_dir=None, agents_dict={}, nodes_dict={}, startup_ros=True, num_warm_iter=50, dt=0.005, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.agents_dict = agents_dict
        self.nodes_dict = nodes_dict
        self.startup_ros = startup_ros
        self.log_dir = log_dir
        self.num_warm_iter = num_warm_iter
        self.dt = dt
        self.agents = {}
        self.nodes = {}
        self.timestamp = 0
        self.curr_agent = None
        self.EMERGENCY = False

        self.register_node()
        self.register_agent()

        self.start_handlers(start_agent="homi_turn")

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

    def start_handlers(self, start_agent: str = "stand"):
        super().start_handlers()
        if not self.sim_run:
            self.joystick = Go2JoystickSubscriber()
        else:
            from nodes.keyboard_node import KeyboardSubscriber

            self.joystick = KeyboardSubscriber(ros_manager=self.ros_manager)
        self.curr_agent = self.agents[start_agent]
        self.curr_agent.reset()

    def agent_warm_up(self, agent_name):
        infer_start_time = time.perf_counter()
        for i in range(self.num_warm_iter):
            _, _, _, _ = self.agents[agent_name].step()
        delay = (time.perf_counter() - infer_start_time) / self.num_warm_iter
        self.logger.debug(f"[{agent_name}] Infer delay: {delay*1e3:.3f} ms")

    # ---- user's custom function --- #

    def get_agent_switch(self, done: bool) -> str | None:
        """Determine if we need to switch to a different agent based on the done flag, joystick or VLM outputs.
        Return None for not switching, or the name of the agent to switch to.
        """
        if self.curr_agent is self.agents["homi_turn"] and self.nodes["robot_pub"].done:  # turn done
            self.logger.log_throttle("homi_turn agent returns done, waiting for VLM to publish start", 1)
            if self.nodes["vlm"].start:
                return "homi_nav"
            return None

        if self.curr_agent is self.agents["homi_nav"] and self.nodes["robot_pub"].done:  # grasp done
            return "homi_turn"
        return None

    def emergency_handle(self):
        if self.joystick.L2 and not self.EMERGENCY:
            self.EMERGENCY = True
            self.logger.warning("L2 is pressed, The motors shuts down.")

        if self.joystick.L1 and self.EMERGENCY:
            self.EMERGENCY = False
            self.logger.info("L1 is pressed, robot will recovery.")

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
            self.publish_done(done)

        loop_delay = time.perf_counter() - loop_start_time
        time.sleep(max(self.dt - loop_delay, 0))
        self.timestamp += 1


def main(args=None):
    agents_dict = {
        "homi_nav": HomiNavAgent,
        "homi_turn": HomiTurnAgent,
    }
    nodes_dict = {
        "vlm": VLMSubNode,
        "robot_pub": RobotPubNode,
    }

    go2_base_node = HomiMPCRun(
        agents_dict=agents_dict,
        nodes_dict=nodes_dict,
    )
    global_start_time = time.perf_counter()

    while True:
        go2_base_node.main_loop()
        if go2_base_node.timestamp % 1000 == 0:
            frequency = go2_base_node.timestamp / (time.perf_counter() - global_start_time)
            go2_base_node.logger.debug(f"frequency: {frequency:.2f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Go2 robot.")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 7890)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    # ChannelFactoryInitialize(0, "eth0")
    ChannelFactoryInitialize(1, "lo")

    main(args=args)
