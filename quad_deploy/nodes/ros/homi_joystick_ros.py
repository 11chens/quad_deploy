from ros_base.node.base_node import BaseNode
from unitree_go.msg import WirelessController

from quad_deploy.utils.button_code import WirelessButtons
from quad_deploy.nodes.ros.wireless_ros import JoystickRosNode as BaseJoystickRosNode

class HomiJoystickRosNode(BaseJoystickRosNode):
    """Class to handle Unitree go2 joystick inputs for controlling the robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm = self.nodes["vlm"]

    def _joy_stick_callback(self, msg: WirelessController):
        """Update joystick state based on the received message."""
        super()._joy_stick_callback(msg)
        if self.X:
            self.loco_ready = True
        if self.R1:
            self.vlm.publish_ready(ready=True)