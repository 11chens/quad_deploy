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
    vlm_node.grasp = False


def VLM_MPC_main():
    nodes = {}
    ros_manager = RosManager()

    nodes["vlm_node"] = VLMNode(ros_manager=ros_manager)
    nodes["zed_pub"] = ZedPublisher(ros_manager=ros_manager)

    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_manager,), daemon=True)
    ros_thread.start()
    global_timestamp = 0
    turn_string = "turn left"

    nodes["zed_pub"].P_img_msg.x = 0.2
    nodes["zed_pub"].P_img_msg.y = 0.4
    nodes["zed_pub"].P_img_msg.z = 3.5

    freq_hz = 10

    print("Waiting for RL controller start up")
    while not nodes["vlm_node"].done:
        time.sleep(0.1)
    print("RL message received, the VLM is ready!")

    while True:
        if global_timestamp == 3 * freq_hz:
            nodes["vlm_node"].turn = turn_string
            print(f"""turn: {nodes["vlm_node"].turn}""")
        elif global_timestamp == 10 * freq_hz:
            nodes["vlm_node"].start = True
            print(f"""start: {nodes["vlm_node"].start}""")
        elif global_timestamp == 15 * freq_hz:
            nodes["vlm_node"].grasp = True
            print(f"""grasp: {nodes["vlm_node"].grasp}""")
        elif global_timestamp == 20 * freq_hz:
            turn_string = "turn right"
            global_timestamp = 0
        else:
            reset_msg(nodes["vlm_node"])

        time.sleep(1 / freq_hz)
        global_timestamp += 1


if __name__ == "__main__":
    rclpy.init()
    VLM_MPC_main()
