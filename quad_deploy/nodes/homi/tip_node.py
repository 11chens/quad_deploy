from ros_base.node.base_node import BaseNode
from std_msgs.msg import Bool, String


class TipNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_pub = self.create_publisher(String, "/control/turn", 1)
        self.ui_ready_pub = self.create_publisher(Bool, "/control/ui_ready", 1)
        self.inquiry_sub = self.create_subscription(Bool, "/control/inquiry", self.inquiry_callback, 1)

        self.ui_ready_msg = Bool()
        self.ui_ready = False
        self.turn_msg = String()
        self.inquiry = False
        self.turn = ""

    def publish_ui_ready(self, ui_ready: bool):
        if ui_ready != self.ui_ready:
            self.ui_ready = ui_ready
            self.ui_ready_msg.data = self.ui_ready
            self.ui_ready_pub.publish(self.ui_ready_msg)
            self.logger.info(f"""[Pub] ui_ready: {self.ui_ready}.""")

    def publish_turn(self, turn: str):
        self.turn = turn
        self.turn_msg.data = self.turn
        self.turn_pub.publish(self.turn_msg)
        self.logger.info(f"""[Pub] turn: {self.turn}.""")

    def inquiry_callback(self, msg: Bool):
        inquiry = msg.data
        self.logger.info(f"""[Sub] inquiry: {inquiry}.""")
        self.inquiry = inquiry
