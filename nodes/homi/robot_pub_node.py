from rclpy.node import Node
from std_msgs.msg import Bool


class RobotPubNode:
    def __init__(self, ros_manager: Node = None):
        # subscriber
        self.ros_manager = ros_manager
        self.logger = self.ros_manager.logger
        # publisher
        self.done_pub = ros_manager.create_publisher(Bool, "/control/done", 10)
        self.done_msg = Bool()
        self.done = self.done_msg.data

    def publish(self):
        if self.done:
            self.logger.info(f"""[Pub] Done: {self.done}.""")
        self.done_msg.data = self.done
        self.done_pub.publish(self.done_msg)
