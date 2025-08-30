from keyboard_msgs.msg import Key
from rclpy.node import Node


class KeyboardSubscriber:
    def __init__(self, ros_manager: Node = None):
        # subscriber
        self.ros_manager = ros_manager
        self.logger = self.ros_manager.get_logger()
        self._init_keys()
        self.keydown_sub = ros_manager.create_subscription(Key, "/keydown", self._keydown_callback, 1)

    def _init_keys(self):
        self.S = False
        self.X = False
        self.R1 = False
        self.R2 = False
        self.L1 = False
        self.L2 = False

        self.up = False
        self.down = False
        self.right = False
        self.left = False

        self.cmd_vx = 0
        self.cmd_vy = 0
        self.cmd_vyaw = 0
        self.cmd_pitch = 0

    def _keydown_callback(self, msg: Key):
        self.S = msg.code == 115  # S
        self.X = msg.code == 120  # X

        self.L1 = msg.code == 113  # Q
        self.R1 = msg.code == 101  # E
        self.L2 = msg.code == 97  # A
        self.R2 = msg.code == 100  # D

        self.up = msg.code == 273
        self.down = msg.code == 274
        self.right = msg.code == 275
        self.left = msg.code == 276

        self.pitch_up = msg.code == 53 or msg.code == 261  # 5
        self.pitch_down = msg.code == 50 or msg.code == 258  # 2

        self.cmd_vx += float(self.up)
        self.cmd_vx -= float(self.down)

        self.cmd_vyaw += float(self.left)
        self.cmd_vyaw -= float(self.right)

        self.cmd_pitch += float(self.pitch_down)
        self.cmd_pitch -= float(self.pitch_up)

        self.cmd_vx = max(min(self.cmd_vx, 1.0), -1)
        self.cmd_vyaw = max(min(self.cmd_vyaw, 1.0), -1)
        self.cmd_pitch = max(min(self.cmd_pitch, 1.0), -1)

    def reset(self):
        self._init_keys()
