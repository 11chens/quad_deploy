import sys
import time

import numpy as np
import rclpy
from ros_base.manager.base_manager import BaseManager

from nodes.homi.vlm.robot2vlm import Robot2VLMBridge
from nodes.homi.vlm.zed_node import ZedNode


class MockVLMRun(BaseManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.zed: ZedNode = self.nodes["zed"]
        self.vlm_node: Robot2VLMBridge = self.nodes["vlm_node"]

        self.turn_list = ["", "turn left", "turn right"]
        self.gripper_list = [True, False]

    def handshake(self):
        self.logger.info("Waiting for RL controller start up")
        while not (self.vlm_node.ready):
            time.sleep(0.01)
        self.logger.info("RL message received, the VLM is ready!")

    def main_loop(self):
        if self.timestamp == 2 * self.node_freq_hz:
            turn = self.turn_list[0]
            self.vlm_node.publish_turn(turn)
            self.logger.info(f"""--------------------------""")
            self.logger.info(f"""{self.timestamp/self.node_freq_hz}s | turn: {self.vlm_node.turn}""")

        elif self.timestamp == 6 * self.node_freq_hz:
            start = True
            self.vlm_node.publish_start(start)
            self.logger.info(f"""{self.timestamp/self.node_freq_hz}s | start: {self.vlm_node.start}""")

        elif self.timestamp == 16 * self.node_freq_hz:
            grasp = self.gripper_list[0]
            # self.vlm_node.publish_grasp(grasp)
            self.logger.info(f"""{self.timestamp/self.node_freq_hz}s | grasp: {self.vlm_node.grasp}""")

        elif self.timestamp == 20 * self.node_freq_hz:
            turn_1 = self.turn_list[1]
            turn_2 = self.turn_list[2]
            self.turn_list[2] = turn_1
            self.turn_list[1] = turn_2
            grasp_0 = self.gripper_list[0]
            grasp_1 = self.gripper_list[1]
            self.gripper_list[1] = grasp_0
            self.gripper_list[0] = grasp_1
            self.timestamp = 0
            P_world_x = self.zed.P_world[0]
            P_world_y = self.zed.P_world[1]
            P_world_z = self.zed.P_world[2]
            self.zed.P_world[0] = -P_world_x
            self.zed.P_world[1] = -P_world_y
            self.zed.P_world[2] = P_world_z
            self.logger.info(f"""Task Done !!! """)

        self.update_timestamp()


def main(args=None):
    node_name = "MockVLMRun"
    nodes_dict = {
        "vlm_node": Robot2VLMBridge,
        "zed": ZedNode,
    }
    node_freq_hz = 10

    rclpy.init()
    mock_vlm_runner = MockVLMRun(
        node_name=node_name,
        nodes_dict=nodes_dict,
        node_freq_hz=node_freq_hz,
    )
    mock_vlm_runner.start_main_loop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock VLM running.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 8901)
        print(f"Process: {sys.argv[:]}")
        print(f"Is waiting for attach at {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args=args)
