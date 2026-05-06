import os

import numpy as np
import onnxruntime as ort
from ros_base.agents.base_agent import BaseAgent

from quad_deploy.nodes.sigloma.gripper_node import GripperNode
from quad_deploy.nodes.sigloma.vlm2robot import VLM2BobotBridge


class ArmAgent(BaseAgent):
    def __init__(self, cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gripper: GripperNode = self.nodes["gripper"]
        self.vlm: VLM2BobotBridge = self.nodes["vlm"]

    def step(self):
        pass

    def reset(self):
        pass

    def handle(self):
        action, done = self.step()
        self.publish_to_gripper(action)
        self.publish_to_vlm(done)

    def publish_to_gripper(self, action):
        self.gripper.publish(action)

    def publish_to_vlm(self, done):
        self.vlm.publish_grasp_done(done)

    @property
    def done(self):
        return False
