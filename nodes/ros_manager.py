from rclpy.node import Node

from nodes.keyboard_node import KeyboardSubscriber
from nodes.lidar_node import LidarSubscriber


class RosManager(Node):
    def __init__(
        self,
        lidar=False,
        key=False,
    ):
        super().__init__("Sensors")

        self.lidar_node = LidarSubscriber(ros_mangager=self) if lidar else None
        self.key_node = KeyboardSubscriber(ros_mangager=self) if key else None
