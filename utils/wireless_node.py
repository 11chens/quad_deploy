from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.core.channel import ChannelSubscriber


class WirelessButtons:
    R1 = 0b00000001  # 1
    L1 = 0b00000010  # 2
    start = 0b00000100  # 4
    select = 0b00001000  # 8
    R2 = 0b00010000  # 16
    L2 = 0b00100000  # 32
    F1 = 0b01000000  # 64
    F2 = 0b10000000  # 128
    A = 0b100000000  # 256
    B = 0b1000000000  # 512
    X = 0b10000000000  # 1024
    Y = 0b100000000000  # 2048
    up = 0b1000000000000  # 4096
    right = 0b10000000000000  # 8192
    down = 0b100000000000000  # 16384
    left = 0b1000000000000000  # 32768


class Go2JoystickSubscriber:
    """Class to handle Unitree go2 joystick inputs for controlling the robot."""

    def __init__(self):
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
        self.WirelessButtons = WirelessButtons
        self.joy_stick_topic = "rt/wirelesscontroller"
        self.joy_stick_sub = ChannelSubscriber(self.joy_stick_topic, WirelessController_)
        self.joy_stick_sub.Init(self._joy_stick_callback, 1)

    def _joy_stick_callback(self, msg: WirelessController_):
        """Update joystick state based on the received message."""
        self.cmd_vx = msg.ly
        self.cmd_vy = -msg.lx
        self.cmd_vyaw = -msg.rx
        self.L2 = msg.keys & self.WirelessButtons.L2
        self.L1 = msg.keys & self.WirelessButtons.L1
        self.R2 = msg.keys & self.WirelessButtons.R2
        self.R1 = msg.keys & self.WirelessButtons.R1
        self.A = msg.keys & self.WirelessButtons.A
        self.X = msg.keys & self.WirelessButtons.X

    def reset(self):
        pass
