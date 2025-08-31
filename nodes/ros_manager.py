from rclpy.node import Node


class RosManager(Node):
    def __init__(
        self,
    ):
        super().__init__("RosManager")

        self.logger = self.get_logger()
