import os
import sys
import time

import numpy as np
import serial
from ros_base.nodes.base_node import BaseNode

from quad_deploy.nodes.homi.vlm2robot import VLM2BobotBridge


class GripperNode(BaseNode):
    def __init__(self, gripper_type="two_fingers", port="/dev/ttyUSB0", *args, **kwargs):
        """Node to control the gripper via serial communication.
        Supported gripper types: "two_fingers", or "None" (sim serial port).
        """
        super().__init__(*args, **kwargs)

        # Configure serial port
        self._support_gripper_types = ["two_fingers"]
        self.gripper_type = gripper_type
        self.parse_config()
        self.duration = 3.0  # duration to finish grasp or release action, in seconds
        self.start_time = None
        self.grasp_state = None
        self.port = port

        self.serial_port = serial.Serial(
            port=self.port,
            baudrate=115200,  # Baud rate, can be modified as needed
            timeout=1,
        )

        self.handle(grasp=False)  # initialize to released state

        # Initialize servo 4 to 170 degrees
        self.servo4_state = 170
        try:
            self.servo_rotate(channel=4, angle=170)
            self.logger.info("Servo channel 4 initialized to 170 degrees.")
        except Exception as e:
            self.logger.error(f"Failed to initialize servo 4: {e}")

    def servo_rotate(self, channel, angle):
        if not (1 <= channel <= 24):
            self.logger.error("Servo channel must be between 1 and 24.")
            return
        if not (0 <= angle <= 180):
            self.logger.error("Servo angle must be between 0 and 180.")
            return
        servo_id = chr(ord("A") + channel - 1)
        angle_str = f"{angle:03d}"
        datasend = [0x24, ord(servo_id), ord(angle_str[0]), ord(angle_str[1]), ord(angle_str[2]), 0x23]
        try:
            if self.serial_port.is_open:
                self.serial_port.write(bytes(datasend))
                self.logger.info(f"Servo {channel} rotated to {angle} degrees.")
        except Exception as e:
            self.logger.error(f"Failed to rotate servo {channel}: {e}")

    def toggle_servo4(self):
        """Toggle servo 4 between 80 and 170 degrees."""
        target_angle = 80 if self.servo4_state == 170 else 170
        self.servo_rotate(channel=4, angle=target_angle)
        self.logger.info(f"Servo 4 toggled to {target_angle} degrees.")
        self.servo4_state = target_angle

    def parse_config(self):
        """Parse configuration for different gripper types."""
        if self.gripper_type == "two_fingers":
            # Gripper Control Table (Two Fingers type)
            # Protocol: $A<Angle># where Angle is 3 digits (000-108)
            # Physical Limits: 000 (Max Open) to 108 (Max Close)
            self.grasp_data = bytes([0x24, 0x41, 0x31, 0x30, 0x38, 0x23])  # grasp command
            self.release_data = bytes([0x24, 0x41, 0x30, 0x35, 0x30, 0x23])  # release command

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
        # if self.gripper_type not in self._support_gripper_types:
        #     self.logger.warning(f"Gripper type '{self.gripper_type}' not supported. No action taken.")
        #     self.start_time = self.timestamp
        #     return

        if self.start_time is None:
            self.start_time = self.timestamp

        self.grasp_state = grasp

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
