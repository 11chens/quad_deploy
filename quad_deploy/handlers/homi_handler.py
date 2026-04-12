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
            self.logger.info("Robot will recovery.")
            return "recovery"

        if self.joystick.R2:
            self.logger.info("The autonomous control is [OFF]. Please control the robot using joystck.")
            return "human_teleop"

        # Toggle gripper rotation on B key
        if self.joystick.B and getattr(self.gripper, "has_rotation", False):
            if not hasattr(self, "_last_b_pressed"):
                self._last_b_pressed = False
            if not self._last_b_pressed:
                try:
                    self.gripper.toggle_rotation()
                    self.logger.info("Joystick B pressed: Gripper rotation toggled.")
                except Exception as e:
                    self.logger.error(f"Failed to toggle gripper rotation on B press: {e}")
                self._last_b_pressed = True
            return None
        else:
            if hasattr(self, "_last_b_pressed"):
                self._last_b_pressed = False

        if self.joystick.start and getattr(self.gripper, "has_grasp", False):
            grasp = not self.gripper.grasp_state  # Toggle grasp state
            self.gripper.handle(grasp=grasp)
        #     return None

        # RL Switch Logic
        current_state = self.state

        if (current_state in ["cold_start", "recovery"]) and self.agents["stand"].done:
            self.logger.log_once("[stand] agent returns done, waiting for press [X] to switch.")
            if self.joystick.X:
                return "human_teleop"
            return None

        if current_state == "human_teleop" and self.joystick.R1:
            self.logger.important(
                "[loco] The autonomous control is [ON]. Please pay attention to the safety of the robot."
            )
            return "turn"

        if current_state == "turn" and self.wait_vlm:
            if not getattr(self.robot, "sim_run", False):
                has_sigma = self.vlm.sigma_3d_cam is not None
                has_object = self.vlm.object_ready

                if has_sigma and has_object:
                    self.logger.info("VLM messages <sigma_3d_cam> and <object_ready> received, starting navigation!")
                    return "navigation"
                else:
                    wait_msgs = []
                    if not has_sigma:
                        wait_msgs.append("<sigma_3d_cam>")
                    if not has_object:
                        wait_msgs.append("<object_ready>")
                    ready_msgs = []
                    if has_sigma:
                        ready_msgs.append("<sigma_3d_cam>")
                    if has_object:
                        ready_msgs.append("<object_ready>")

                    log_msg = f"Waiting for VLM: {', '.join(wait_msgs)}."
                    if ready_msgs:
                        log_msg += f" (Received: {', '.join(ready_msgs)})"

                    self.logger.log_once(log_msg)
                    return None
            else:
                self.logger.log_once("Waiting for VLM message: <sigma_3d_cam>.")
                if self.vlm.sigma_3d_cam is not None:
                    self.logger.info("VLM message <sigma_3d_cam> received, starting navigation!")
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
