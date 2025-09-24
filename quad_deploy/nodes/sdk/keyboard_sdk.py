from keyboard_msgs.msg import Key
from ros_base.nodes.base_node import BaseNode
from ros_base.utils.button_code import WirelessButtons
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


class KeyboardSDKNode(BaseNode):
    def __init__(
        self, keyboard_topic: str = "/keydown", joy_stick_topic: str = "rt/wirelesscontroller", *args, **kwargs
    ):
        """Class to handle keyboard inputs and publish them as joystick commands (used for simulation in Mujoco).
        Args:
            keyboard_topic (str): ROS2 topic to subscribe for keyboard inputs (get from ros2 keyboard package).
            joy_stick_topic (str): Topic to publish joystick commands (get from unitree sdk).
        """
        super().__init__(*args, **kwargs)

        self.WirelessButtons = WirelessButtons()
        self.keydown_sub = self.create_subscription(Key, keyboard_topic, self._keydown_callback, 1)
        self.joy_stick_topic = joy_stick_topic
        self.joy_stick_pub = ChannelPublisher(self.joy_stick_topic, WirelessController_)
        self.joy_stick_pub.Init()
        self.joy_stick_msg = unitree_go_msg_dds__WirelessController_()
        self.create_timer(0.005, self.publish)
        self._init_keys()

        self.cmd_vx = 0
        self.cmd_vy = 0
        self.cmd_vyaw = 0
        self.cmd_pitch = 0

    def _init_keys(self):
        self.S = False
        self.X = False
        self.R1 = False
        self.R2 = False
        self.L1 = False
        self.L2 = False

        self.up = False
        self.down = False
        self.vyaw_right = False
        self.vyaw_left = False

    def _keydown_callback(self, msg: Key):
        self.X = msg.code == 120  # X
        self.X *= self.WirelessButtons.X

        self.L1 = msg.code == 113  # Q
        self.L1 *= self.WirelessButtons.L1
        self.R1 = msg.code == 101  # E
        self.R1 *= self.WirelessButtons.R1

        self.L2 = msg.code == 97  # A
        self.L2 *= self.WirelessButtons.L2
        self.R2 = msg.code == 100  # D
        self.R2 *= self.WirelessButtons.R2

        self.up = msg.code == 273
        self.down = msg.code == 274
        self.vyaw_right = msg.code == 275
        self.vyaw_left = msg.code == 276

        self.vy_left = msg.code == 257 or msg.code == 49  # 1
        self.vy_right = msg.code == 259 or msg.code == 51  # 3

        self.pitch_up = msg.code == 53 or msg.code == 261  # 5
        self.pitch_down = msg.code == 50 or msg.code == 258  # 2

        self.cmd_vx += float(self.up)
        self.cmd_vx -= float(self.down)

        self.cmd_vyaw += float(self.vyaw_left)
        self.cmd_vyaw -= float(self.vyaw_right)

        self.cmd_vy += float(self.vy_left)
        self.cmd_vy -= float(self.vy_right)

        self.cmd_pitch += float(self.pitch_down)
        self.cmd_pitch -= float(self.pitch_up)

        self.cmd_vx = max(min(self.cmd_vx, 1.0), -1)
        self.cmd_vy = max(min(self.cmd_vy, 1.0), -1)
        self.cmd_vyaw = max(min(self.cmd_vyaw, 1.0), -1)
        self.cmd_pitch = max(min(self.cmd_pitch, 1.0), -1)

    def scan_keys(self):
        if self.L1:
            return self.L1
        if self.L2:
            return self.L2
        if self.R1:
            return self.R1
        if self.R2:
            return self.R2
        if self.X:
            return self.X

        return 0

    def publish(self):
        self.joy_stick_msg.ly = self.cmd_vx
        self.joy_stick_msg.lx = -self.cmd_vy
        self.joy_stick_msg.rx = -self.cmd_vyaw
        self.joy_stick_msg.ry = -self.cmd_pitch

        self.joy_stick_msg.keys = self.scan_keys()

        self.joy_stick_pub.Write(self.joy_stick_msg)
        self.reset()

    def reset(self):
        self._init_keys()
