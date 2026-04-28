import os
import sys
import time

import numpy as np
import serial
from ros_base.nodes.base_node import BaseNode

from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class GripperNode(BaseNode):
    def __init__(
        self, has_rotation=True, has_grasp=True, use_sim_gripper=False, gripper_port="/dev/ttyUSB0", *args, **kwargs
    ):
        """Node to control the gripper via serial communication.
        Args:
            has_rotation: bool, whether there is a rotation servo.
            has_grasp: bool, whether there is a grasp servo.
            use_sim_gripper: bool, whether to use simulated serial port.
        """
        super().__init__(*args, **kwargs)

        self.has_rotation = has_rotation
        self.has_grasp = has_grasp
        self.use_sim_gripper = use_sim_gripper

        # Servo Configuration
        self.GRASP_SERVO_CHANNEL = 1
        self.ROTATION_SERVO_CHANNEL = 4

        # State definitions
        self.GRASP_ANGLE_CLOSE = 180
        self.GRASP_ANGLE_OPEN = 50

        self.ROTATION_ANGLE_HORIZONTAL = 90
        self.ROTATION_ANGLE_VERTICAL = 0

        self.gripper_port = gripper_port
        self.parse_config()
        self.duration = 3.0  # duration to finish grasp or release action, in seconds
        self.start_time = None
        self.grasp_state = None
        self.rotation_state = "horizontal"

        self.serial_port = serial.Serial(
            port=self.gripper_port,
            baudrate=115200,  # Baud rate, can be modified as needed
            timeout=1,
        )

        self.handle(grasp=False)  # initialize to released state
        if self.has_rotation:
            self.rotate_gripper(state="horizontal")  # initialize to horizontal state

    def _generate_servo_cmd(self, channel, angle):
        """Generate hex command bytes for physical servo control (protocol: $A<Angle>#)."""
        servo_id = chr(ord("A") + channel - 1)
        angle_str = f"{angle:03d}"
        return bytes([0x24, ord(servo_id), ord(angle_str[0]), ord(angle_str[1]), ord(angle_str[2]), 0x23])

    def servo_rotate(self, channel, angle):
        if not (1 <= channel <= 24):
            self.logger.error("Servo channel must be between 1 and 24.")
            return
        if not (0 <= angle <= 180):
            self.logger.error("Servo angle must be between 0 and 180.")
            return

        datasend = self._generate_servo_cmd(channel, angle)
        try:
            if self.serial_port.is_open:
                self.serial_port.write(datasend)
                self.logger.info(f"Servo {channel} rotated to {angle} degrees.")
        except Exception as e:
            self.logger.error(f"Failed to rotate servo {channel}: {e}")

    def rotate_gripper(self, state: str):
        """Rotate gripper to 'horizontal' or 'vertical' state."""
        if not self.has_rotation:
            self.logger.warning("Rotation servo is disabled. Ignoring rotate command.")
            return

        if state not in ["horizontal", "vertical"]:
            self.logger.error(f"Invalid rotation state: {state}")
            return

        target_angle = self.ROTATION_ANGLE_HORIZONTAL if state == "horizontal" else self.ROTATION_ANGLE_VERTICAL
        self.servo_rotate(channel=self.ROTATION_SERVO_CHANNEL, angle=target_angle)
        self.rotation_state = state
        self.logger.info(f"Gripper rotated to {state} state (angle {target_angle}).")

    def toggle_rotation(self):
        """Toggle gripper rotation between horizontal and vertical."""
        if not self.has_rotation:
            self.logger.warning("Rotation servo is disabled. Ignoring toggle command.")
            return
        target_state = "vertical" if self.rotation_state == "horizontal" else "horizontal"
        self.rotate_gripper(target_state)

    def parse_config(self):
        """Parse configuration for different gripper types."""
        if self.use_sim_gripper:
            # sim port: socat -d -d pty,raw,echo=0,link=/tmp/pty10 pty,raw,echo=0,link=/tmp/pty11
            self.grasp_data = bytes([0x7B, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF8, 0x7D])  # grasp command
            self.release_data = bytes(
                [0x7B, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF9, 0x7D]
            )  # release command
            self.gripper_port = "/tmp/pty10"
        elif self.has_grasp:
            self.grasp_data = self._generate_servo_cmd(self.GRASP_SERVO_CHANNEL, self.GRASP_ANGLE_CLOSE)
            self.release_data = self._generate_servo_cmd(self.GRASP_SERVO_CHANNEL, self.GRASP_ANGLE_OPEN)
        else:
            self.grasp_data = b""
            self.release_data = b""

    def send_hex_to_serial_port(self, hex_data):
        """Send hex data to serial port for gripper control."""
        try:
            # Ensure serial port is open
            if self.serial_port.is_open:
                # self.logger.info(f"Connected to {self.serial_port.name}")

                # Send hex data
                self.serial_port.write(hex_data)
                # self.logger.info(f"Sent hex data: {hex_data.hex(' ')}")

        except serial.SerialException as e:
            self.logger.error(f"Serial port error: {e}")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")

    def handle(self, grasp):
        """Handle the gripper action based on the grasp command."""
        if not self.has_grasp and not self.use_sim_gripper:
            # self.logger.warning("Grasp servo is disabled. Ignoring grasp command.")
            self.start_time = self.timestamp
            return

        if self.start_time is None:
            self.start_time = self.timestamp

        self.grasp_state = grasp

        target_data = self.grasp_data if grasp else self.release_data
        action_name = "grasping" if grasp else "releasing"

        try:
            if target_data:
                self.send_hex_to_serial_port(target_data)
                self.logger.info(f"Start {action_name} - sent command to gripper.")
        except Exception as e:
            self.logger.error(f"Failed to execute {action_name} command: {e}")

    @property
    def done(self):
        return (self.timestamp - self.start_time) > (self.duration * self.node_freq_hz)
