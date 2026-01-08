import time

from ros_base.handlers.base_handlers import BaseHandlers


class HomiHandler(BaseHandlers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Shortcuts for Homi components
        self.robot = self.nodes.get("robot")
        self.gripper = self.nodes.get("gripper")
        self.vlm = self.nodes.get("vlm")
        self.joystick = self.nodes.get("joystick")

        # State Control
        self.curr_agent = self.agents.get("stand")
        self.wait_vlm = kwargs.get("wait_vlm", True)

    def handle(self):
        """Main state machine logic moved from Manager."""
        new_state = self.get_state_transition()

        if new_state:
            self.state = new_state  # This uses the setter in BaseHandlers to update Manager.state
            self.on_state_enter(new_state)

        # Periodic execution
        if self.state != "emergency":
            if self.curr_agent:
                self.curr_agent.handle()

            if self.robot:
                self.vlm.publish_robot_euler_rpy(euler_rpy=self.robot.euler_rpy)

    def get_state_transition(self):
        """Logic to decide state switches."""
        if not self.joystick:
            return None

        if self.joystick.L2:
            self.logger.warning("Robot emergency shutdown requested.")
            return "emergency"

        if self.joystick.L1 and self.state == "emergency":
            self.logger.info("Robot recovery requested.")
            return "recovery"

        if self.joystick.R2:
            self.logger.info("Human teleop mode activated.")
            return "human_teleop"

        # RL Switch Logic
        current_state = self.state

        if (current_state in ["cold_start", "recovery"]) and self.agents["stand"].done:
            self.logger.log_once("[stand] agent finished. Press [X] to move to teleop.")
            if self.joystick.X:
                return "human_teleop"

        if current_state == "human_teleop" and self.joystick.R1:
            self.logger.important("Autonomous control ON. Starting [turn] agent.")
            return "turn"

        if current_state == "turn" and self.wait_vlm:
            if self.vlm.sigma_3d_cam is not None and (self.vlm.object_ready or self.robot.sim_run):
                self.logger.info("VLM targets received, starting navigation!")
                return "navigation"

        if current_state == "navigation" and self.joystick.A:
            return "gripper_start"

        if current_state == "gripper_start" and self.gripper.done:
            self.vlm.publish_grasp_done(grasp_done=True)
            return "turn"

        if current_state == "gripper_done" and self.vlm.vlm_done:
            return "turn"

        return None

    def on_state_enter(self, state):
        """Initialization logic for each state."""
        self.logger.important(f"Entering State: {state}")

        if state == "emergency":
            self.robot.turn_off_motors()

        elif state == "recovery":
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()
            self.logger.reset()
            self.manager.timestamp = 0  # Accessing manager for specific housekeeping
            self.robot.init_motors()

        elif state == "cold_start":
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()

        elif state == "human_teleop":
            self.curr_agent = self.agents["loco"]
            self.curr_agent.reset()

        elif state == "turn":
            self.logger.reset()
            self.vlm.reset()
            self.curr_agent = self.agents["turn"]
            self.curr_agent.reset()
            self.vlm.publish_rl_ready(rl_ready=True)

        elif state == "navigation":
            self.curr_agent = self.agents["nav"]
            self.curr_agent.reset()

        elif state == "gripper_start":
            self.gripper.start_time = self.manager.timestamp
            grasp = not self.gripper.grasp_state
            self.gripper.handle(grasp=grasp)
