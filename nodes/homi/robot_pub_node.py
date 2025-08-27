from std_msgs.msg import Bool


class RobotPubNode:
    def __init__(self, ros_manager=None):
        # publisher
        self.done_pub = ros_manager.create_publisher(Bool, "/control/done", 10)

        freq_hz = 10
        self.timer = ros_manager.create_timer(1 / freq_hz, self._timer_callback)
        self.done_msg = Bool()
        self.done = self.done_msg.data

    def _timer_callback(self):
        self.done_msg.data = self.done
        self.done_pub.publish(self.done_msg)
