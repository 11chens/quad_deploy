from ros_base.node.base_node import BaseNode
from unitree_go.msg import WirelessController

from utils.button_code import WirelessButtons


class JoystickRosNode(BaseNode):
    """Class to handle Unitree go2 joystick inputs for controlling the robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._init_keys()
        self.WirelessButtons = WirelessButtons
        self.joy_stick_topic = "rt/wirelesscontroller"
        self.joy_stick_sub = self.create_subscription(
            WirelessController, self.joy_stick_topic, self._joy_stick_callback, 1
        )

    def _joy_stick_callback(self, msg: WirelessController):
        """Update joystick state based on the received message."""
        self.cmd_vx = msg.ly
        self.cmd_vy = -msg.lx
        self.cmd_vyaw = -msg.rx
        self.cmd_pitch = -msg.ry
        self.L2 = bool(msg.keys & self.WirelessButtons.L2)
        self.L1 = bool(msg.keys & self.WirelessButtons.L1)
        self.R2 = bool(msg.keys & self.WirelessButtons.R2)
        self.R1 = bool(msg.keys & self.WirelessButtons.R1)
        self.A = bool(msg.keys & self.WirelessButtons.A)
        self.X = bool(msg.keys & self.WirelessButtons.X)

    def _init_keys(self):
        self.R1 = False
        self.L1 = False
        self.start = False
        self.select = False
        self.R2 = False
        self.L2 = False
        self.F1 = False
        self.F2 = False
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.up = False
        self.right = False
        self.down = False
        self.left = False
        self.cmd_vx = 0
        self.cmd_vy = 0
        self.cmd_vyaw = 0
        self.cmd_pitch = 0

    def reset(self):
        self._init_keys()
