import time

import numpy as np
from ros_base.handlers.base_handlers import BaseHandlers


class SEAHandler(BaseHandlers):
    """Finite-state machine that orchestrates the SEA-Nav deployment.

    State transitions::

        cold_start    --(stand done + X)--> human_teleop
        human_teleop  -- R1 + perception fresh --> navigation
        navigation    -- perception stale > safe_stop_timeout_s --> safe_stop
        navigation    -- R2 --> human_teleop
        safe_stop     -- perception fresh + R1 --> navigation
        safe_stop     -- R2 --> human_teleop
        any state     -- L2 --> emergency
        emergency     -- L1 --> recovery
        recovery      --(stand done + X)--> human_teleop

    In ``safe_stop`` the locomotion command is zeroed every tick and the
    joystick is disabled, so the robot keeps standing regardless of operator
    input. The Nav agent itself also short-circuits to a zero command on the
    inner control loop when perception is stale; ``safe_stop`` is the
    state-level escalation when the outage persists.

    Hysteresis thresholds (defaults chosen to be well above the per-subscriber
    timeout so transient drops don't bounce the FSM):
      - ``safe_stop_timeout_s``: stale perception duration that triggers
        navigation -> safe_stop. Default 1.0 s.
      - ``safe_stop_recover_s``: fresh perception duration required before
        safe_stop -> navigation will accept an R1 press. Default 0.5 s.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Shortcuts to the nodes / agents this handler talks to.
        self.robot = self.nodes.get("robot")
        self.joystick = self.nodes.get("joystick")
        self.rays_sub = self.nodes.get("rays_sub")
        self.pose_sub = self.nodes.get("pose_sub")

        self.curr_agent = self.agents.get("stand")

        # Safe-stop hysteresis timers (in `time.monotonic()` seconds).
        self.safe_stop_timeout_s = float(kwargs.get("safe_stop_timeout_s", 1.0))
        self.safe_stop_recover_s = float(kwargs.get("safe_stop_recover_s", 0.5))
        self._unfresh_since = None  # when perception first went stale (navigation state)
        self._fresh_since = None  # when perception first recovered (safe_stop state)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def handle(self):
        new_state = self.get_state_transition()

        if new_state:
            self.state = new_state  # propagates to manager.state via setter
            self.on_state_enter(new_state)

        # safe_stop guarantee: force loco.pre_cmds=0 every tick so even a
        # nasty joystick input cannot drive the robot.
        if self.state == "safe_stop" and self.curr_agent is not None:
            if hasattr(self.curr_agent, "pre_cmds") and self.curr_agent.pre_cmds is not None:
                self.curr_agent.pre_cmds[:] = 0.0

        if self.state != "emergency" and self.curr_agent is not None:
            self.curr_agent.handle()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def get_state_transition(self):
        if self.joystick is None:
            return None

        # Global overrides
        if self.joystick.L2:
            self.logger.warning("Robot emergency shutdown requested.")
            return "emergency"

        if self.joystick.L1 and self.state == "emergency":
            self.logger.info("Robot will recover.")
            return "recovery"

        # R2 = manual takeover (from navigation or safe_stop)
        if self.joystick.R2 and self.state in ("navigation", "safe_stop"):
            self.logger.info("Manual takeover. Switching to human_teleop.")
            return "human_teleop"

        current_state = self.state

        if current_state in ("cold_start", "recovery") and self.agents["stand"].done:
            self.logger.log_once("[stand] agent returns done, press [X] to switch to human_teleop.")
            if self.joystick.X:
                return "human_teleop"
            return None

        if current_state == "human_teleop" and self.joystick.R1:
            if self._perception_fresh():
                self.logger.important("[nav] Entering navigation state.")
                return "navigation"
            self.logger.log_once(
                "[nav] Cannot start navigation: rays / pose subscription not fresh "
                "(check unitree_mujoco_ros publishers)."
            )
            return None

        if current_state == "navigation":
            return self._maybe_enter_safe_stop()

        if current_state == "safe_stop":
            return self._maybe_recover_from_safe_stop()

        return None

    def _maybe_enter_safe_stop(self):
        """In navigation: track how long perception has been stale; escalate to
        safe_stop after ``safe_stop_timeout_s`` of continuous staleness.
        """
        nav = self.agents.get("nav")
        # Prefer the Nav agent's own per-step flag; fall back to direct sub check
        # if Nav has not run yet.
        if nav is not None and hasattr(nav, "perception_fresh"):
            fresh = bool(nav.perception_fresh)
        else:
            fresh = self._perception_fresh()

        now = time.monotonic()
        if fresh:
            self._unfresh_since = None
            return None

        if self._unfresh_since is None:
            self._unfresh_since = now
            return None

        if now - self._unfresh_since >= self.safe_stop_timeout_s:
            self.logger.warning(
                f"[safe_stop] perception stale for {now - self._unfresh_since:.2f}s "
                f"(>= {self.safe_stop_timeout_s}s); entering safe_stop."
            )
            return "safe_stop"
        return None

    def _maybe_recover_from_safe_stop(self):
        """In safe_stop: require both fresh perception for ``safe_stop_recover_s``
        AND an explicit R1 press from the operator before returning to
        navigation. This double check is intentional - we never auto-resume.
        """
        if not self._perception_fresh():
            self._fresh_since = None
            return None

        now = time.monotonic()
        if self._fresh_since is None:
            self._fresh_since = now
            self.logger.log_once("[safe_stop] perception recovered; press [R1] to resume navigation.")
            return None

        if now - self._fresh_since >= self.safe_stop_recover_s and self.joystick.R1:
            self.logger.important("[nav] Resuming navigation from safe_stop.")
            return "navigation"
        return None

    def _perception_fresh(self):
        rays_fresh = self.rays_sub is not None and self.rays_sub.is_fresh()
        pose_fresh = self.pose_sub is not None and self.pose_sub.is_fresh()
        return rays_fresh and pose_fresh

    # ------------------------------------------------------------------
    # State entry hooks
    # ------------------------------------------------------------------
    def on_state_enter(self, state):
        self.logger.important(f"Entering State: {state}")

        if state == "emergency":
            self.robot.turn_off_motors()

        elif state == "recovery":
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()
            self.logger.reset()
            self.manager.timestamp = 0
            self.robot.init_motors()

        elif state == "cold_start":
            self.curr_agent = self.agents["stand"]
            self.curr_agent.reset()

        elif state == "human_teleop":
            self.curr_agent = self.agents["loco"]
            self.curr_agent.reset()
            # Reset safe-stop timers when leaving navigation / safe_stop.
            self._unfresh_since = None
            self._fresh_since = None

        elif state == "navigation":
            self.curr_agent = self.agents["nav"]
            self.curr_agent.reset()
            self._unfresh_since = None
            self._fresh_since = None

        elif state == "safe_stop":
            # Switch to Loco agent so we are not running Nav inference while
            # perception is unusable. Zero pre_cmds is enforced each tick by
            # handle() (see top of handle()).
            self.curr_agent = self.agents["loco"]
            self.curr_agent.reset()
            self.curr_agent.wireless = False  # ignore joystick during safe_stop
            self.curr_agent.pre_cmds = np.zeros(self.curr_agent.num_commands, dtype=np.float32)
            self._fresh_since = None
