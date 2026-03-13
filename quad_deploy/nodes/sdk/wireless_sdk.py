import time

from ros_base.nodes.base_node import BaseNode
from ros_base.utils.args_debug import add_debug_mode
from ros_base.utils.button_code import WirelessButtons
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


class JoystickSDKNode(BaseNode):
    """
    Class to handle Unitree GO2 joystick inputs for controlling the robot.

    This class provides both current button states and edge detection for button presses.
    Button names can be accessed directly as properties (e.g., joystick_node.A returns True
    if the A button was just pressed).

    Supported buttons: R1, L1, start, select, R2, L2, F1, F2, A, B, X, Y, up, right, down, left
    """

    def __init__(self, joy_stick_topic: str = "rt/wirelesscontroller", channel: bool = False, *args, **kwargs):
        if channel:
            ChannelFactoryInitialize()  # SDK init
        else:
            super().__init__(*args, **kwargs)  # ROS2 Node init, CANNOT init SDK with ROS2 Node (standalone mode)

        # Get all button names from WirelessButtons class
        self.button_names = [
            attr
            for attr in dir(WirelessButtons)
            if not attr.startswith("__") and isinstance(getattr(WirelessButtons, attr), int)
        ]

        self._init_keys()
        self._init_previous_keys()  # Initialize previous button states
        self.WirelessButtons = WirelessButtons
        self.joy_stick_topic = joy_stick_topic
        self.joy_stick_sub = ChannelSubscriber(self.joy_stick_topic, WirelessController_)
        self.joy_stick_sub.Init(self._joy_stick_callback, 1)

    def _joy_stick_callback(self, msg: WirelessController_):
        """
        Update joystick state based on the received message.

        Args:
            msg (WirelessController_): The wireless controller message containing button states
        """
        # Update previous button states
        self._update_previous_keys()

        # Update current joystick analog values
        self.cmd_vx = msg.ly
        self.cmd_vy = -msg.lx
        self.cmd_vyaw = -msg.rx
        self.cmd_pitch = msg.ry

        # Auto-bind all buttons from WirelessButtons class
        for button_name in self.button_names:
            button_code = getattr(self.WirelessButtons, button_name)
            setattr(self, f"raw_{button_name}", bool(msg.keys & button_code))

    def _init_keys(self):
        """
        Initialize all button states to False and joystick analog values to 0.
        """
        # Initialize all buttons to False
        for button_name in self.button_names:
            setattr(self, f"raw_{button_name}", False)

        # Initialize joystick analog data
        self.cmd_vx = 0
        self.cmd_vy = 0
        self.cmd_vyaw = 0
        self.cmd_pitch = 0

    def _init_previous_keys(self):
        """
        Initialize previous button states and edge detection flags for edge detection.
        """
        for name in self.button_names:
            setattr(self, f"prev_{name}", False)
            setattr(self, f"edge_triggered_press_{name}", False)  # Press edge trigger flag
            setattr(self, f"edge_triggered_release_{name}", False)  # Release edge trigger flag

    def _update_previous_keys(self):
        """
        Save the previous state before updating current button states.
        """
        for name in self.button_names:
            setattr(self, f"prev_{name}", getattr(self, f"raw_{name}", False))

    def _is_pressed(self, button_name: str) -> bool:
        """
        Detect if a button was just pressed (transition from not pressed to pressed).
        Uses state locking mechanism to prevent repeated triggers.

        Args:
            button_name (str): Button name, e.g., 'A', 'X', 'R1', etc.

        Returns:
            bool: True if the button was just pressed, False otherwise
        """
        current = getattr(self, f"raw_{button_name}", False)
        previous = getattr(self, f"prev_{button_name}", False)
        press_flag = f"edge_triggered_press_{button_name}"
        release_flag = f"edge_triggered_release_{button_name}"

        # Detect rising edge (False to True transition)
        if current and not previous and not getattr(self, press_flag, False):
            setattr(self, press_flag, True)  # Set press edge trigger flag
            setattr(self, release_flag, False)  # Reset release edge flag
            return True

        # Reset press edge trigger flag when button is released
        if not current:
            setattr(self, press_flag, False)

        return False

    def _is_released(self, button_name: str) -> bool:
        """
        Detect if a button was just released (transition from pressed to not pressed).
        Uses state locking mechanism to prevent repeated triggers.

        Args:
            button_name (str): Button name, e.g., 'A', 'X', 'R1', etc.

        Returns:
            bool: True if the button was just released, False otherwise
        """
        current = getattr(self, f"raw_{button_name}", False)
        previous = getattr(self, f"prev_{button_name}", False)
        release_flag = f"edge_triggered_release_{button_name}"
        press_flag = f"edge_triggered_press_{button_name}"

        # Detect falling edge (True to False transition)
        if not current and previous and not getattr(self, release_flag, False):
            setattr(self, release_flag, True)  # Set release edge trigger flag
            setattr(self, press_flag, False)  # Reset press edge flag
            return True

        # Reset release edge trigger flag when button is pressed again
        if current:
            setattr(self, release_flag, False)

        return False

    def __getattr__(self, name):
        """
        Dynamic property access for button states.
        Accessing joystick_node.A returns True if A button was just pressed.

        Args:
            name (str): Button name to check

        Returns:
            bool: True if the button was just pressed, False otherwise

        Raises:
            AttributeError: If the attribute name is not a valid button name
        """
        if name in self.button_names:
            if name.startswith("raw_"):
                return getattr(self, name, False)
            return self._is_pressed(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def is_released(self, button_name: str) -> bool:
        """
        Public method to check if a button was just released.

        Args:
            button_name (str): Button name to check

        Returns:
            bool: True if the button was just released, False otherwise
        """
        return self._is_released(button_name)

    def reset(self):
        """
        Reset all button states and edge detection flags to their initial values.
        """
        self._init_keys()
        self._init_previous_keys()


def main():
    """
    Test button edge detection and debounce functionality.
    """
    print("=== Button Debounce Test Program ===")
    print("This program checks button states at 100Hz frequency")
    print("If debounce works correctly, each button press will only print once")
    print("Please press any buttons on the controller to test")
    print("Press Ctrl+C to exit")
    print("-" * 40)

    try:
        joystick_node = JoystickSDKNode(channel=True)

        # Get all button names
        test_buttons = joystick_node.button_names
        print(f"Detected buttons: {', '.join(test_buttons)}")
        print("-" * 40)

        loop_count = 0
        while True:
            loop_count += 1

            # Display status every 1000 loops (10 seconds)
            if loop_count % 1000 == 0:
                print(f"Program running normally... (loop count: {loop_count})")

            # Test button press detection using direct property access
            for button in test_buttons:
                if getattr(joystick_node, button):  # This calls __getattr__ -> _is_pressed
                    print(f"[{loop_count:06d}] ✓ {button} button pressed!")

                if joystick_node.is_released(button):
                    print(f"[{loop_count:06d}] ✗ {button} button released!")

            # 100Hz loop frequency
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nProgram exited")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    add_debug_mode(listen_port=9999)  # unitree_wireless
    main()
