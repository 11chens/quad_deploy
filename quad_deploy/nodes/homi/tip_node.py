from std_msgs.msg import String, Bool
from ros_base.node.base_node import BaseNode

class TipNode(BaseNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tip_pub = self.create_publisher(String, "/control/tip", 1)
        self.inquiry_sub = self.create_subscription(Bool, "/control/inquiry", self.inquiry_callback, 1)

        self.tip_msg = String()
        self.inquiry = False

    def publish_tip(self, tip_str: str):
        self.tip_msg.data = tip_str
        self.tip_pub.publish(self.tip_msg)

    def inquiry_callback(self, inquiry: Bool):
        self.inquiry = inquiry.data