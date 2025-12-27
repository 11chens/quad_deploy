import os
import sys
import time

import numpy as np
import serial
from ros_base.nodes.base_node import BaseNode

from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class GripperNode(BaseNode):
    def __init__(self, gripper_type="two_fingers", *args, **kwargs):
        """Node to control the gripper via serial communication.
        Supported gripper types: "two_fingers", "three_fingers", or "None" (sim serial port).
        """
        super().__init__(*args, **kwargs)

        # Configure serial port
        self._support_gripper_types = ["two_fingers", "three_fingers"]
        self.gripper_type = gripper_type
        self.parse_config()
        self.duration = 2  # duration to finish grasp or release action, in seconds
        self.start_time = None

    def parse_config(self):
        """Parse configuration for different gripper types."""
        if self.gripper_type == "two_fingers":
            self.grasp_data = bytes([0x24, 0x41, 0x30, 0x39, 0x38, 0x23])  # grasp command
            self.release_data = bytes([0x24, 0x41, 0x30, 0x35, 0x30, 0x23])  # release command
            self.port = "/dev/ttyUSB0"

        elif self.gripper_type == "three_fingers":
            self.grasp_data = bytes([0x7B, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF8, 0x7D])  # grasp command
            self.release_data = bytes(
                [0x7B, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF9, 0x7D]
            )  # release command
            self.port = "/dev/ttyACM0"

        # sim port: socat -d -d pty,raw,echo=0,link=/tmp/pty10 pty,raw,echo=0,link=/tmp/pty11
        else:
            self.grasp_data = bytes([0x7B, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF8, 0x7D])  # grasp command
            self.release_data = bytes(
                [0x7B, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF9, 0x7D]
            )  # release command
            self.port = "/tmp/pty10"

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
        if self.gripper_type not in self._support_gripper_types:
            self.logger.warning(f"Gripper type '{self.gripper_type}' not supported. No action taken.")
            self.start_time = self.timestamp
            return

        if not hasattr(self, "serial_port"):
            # First in, initialize serial port and start time
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=115200,  # Baud rate, can be modified as needed
                timeout=1,
            )
            self.start_time = self.timestamp

        if grasp:  # True: pick, False: place
            try:
                self.send_hex_to_serial_port(self.grasp_data)
                self.logger.info("Start grasping - sent grasp command to gripper.")
            except Exception as e:
                self.logger.error(f"Failed to execute grasp command: {e}")
        else:
            try:
                self.send_hex_to_serial_port(self.release_data)
                self.logger.info("Start releasing - sent release command to gripper.")
            except Exception as e:
                self.logger.error(f"Failed to execute release command: {e}")

    @property
    def done(self):
        return (self.timestamp - self.start_time) > (self.duration * self.node_freq_hz)
