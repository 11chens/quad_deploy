from rclpy.node import Node

from utils.keyboard_node import KeyboardSubscriber
from utils.lidar_node import LidarSubscriber


class NodeHandle(Node):
    def __init__(
        self,
        lidar=False,
        key=False,
    ):
        super().__init__("Sensors")

        self.lidar_node = LidarSubscriber(rosnode=self) if lidar else Node
        self.key_node = KeyboardSubscriber(rosnode=self) if key else Node
