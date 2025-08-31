import sys
import threading
import time

import numpy as np
import rclpy

from nodes.homi.vlm.vlm_node import VLMNode
from nodes.homi.vlm.zed_pub_node import ZedPublisher
from nodes.ros_manager import RosManager


def reset_msg(vlm_node):
    vlm_node.turn = ""
    vlm_node.start = False


def VLM_MPC_main():
    nodes = {}
    ros_manager = RosManager()

    nodes["vlm_node"] = VLMNode(ros_manager=ros_manager)
    nodes["zed_pub"] = ZedPublisher(ros_manager=ros_manager)

    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_manager,), daemon=True)
    ros_thread.start()
    global_timestamp = 0
    turn_list = ["", "turn left", "turn right"]

    nodes["zed_pub"].P_img_msg.x = 0.2
    nodes["zed_pub"].P_img_msg.y = 0.4
    nodes["zed_pub"].P_img_msg.z = 3.5

    start = True
    grasp = False
    freq_hz = 10

    print("Waiting for RL controller start up")
    while not nodes["vlm_node"].done:
        nodes["vlm_node"].publish()
    print("RL message received, the VLM is ready!")

    while True:
        if global_timestamp == 2 * freq_hz:
            nodes["vlm_node"].turn = turn_list[1]
            print(f"""--------------------------""")
            print(f"""{global_timestamp/freq_hz}s | turn: {nodes["vlm_node"].turn}""")
        elif global_timestamp == 6 * freq_hz:
            nodes["vlm_node"].start = start
            print(f"""{global_timestamp/freq_hz}s | start: {nodes["vlm_node"].start}""")
        elif global_timestamp == 12 * freq_hz:
            nodes["vlm_node"].grasp = not grasp
            print(f"""{global_timestamp/freq_hz}s | grasp: {nodes["vlm_node"].grasp}""")
        elif global_timestamp == 16 * freq_hz:
            turn_1 = turn_list[1]
            turn_2 = turn_list[2]
            turn_list[2] = turn_1
            turn_list[1] = turn_2
            global_timestamp = 0
            grasp = not grasp
        else:
            reset_msg(nodes["vlm_node"])

        nodes["vlm_node"].publish()

        time.sleep(1 / freq_hz)
        global_timestamp += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Go2 robot.")
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

    rclpy.init()
    VLM_MPC_main()
