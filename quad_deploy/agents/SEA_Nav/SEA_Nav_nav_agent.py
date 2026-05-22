import os

import numpy as np
import onnxruntime as ort
from ros_base.utils.decorators import profile_latency

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.agents.SEA_Nav.SEA_Nav_loco_agent import SEA_Nav_LocoAgent
from quad_deploy.config.SEA_Nav.SEA_Nav_nav_agent_cfg import SEA_Nav_NavAgentCfg
from quad_deploy.nodes.SEA_Nav.pose_sub_node import PoseSubNode
from quad_deploy.nodes.SEA_Nav.rays_sub_node import RaysSubNode


class SEA_Nav_NavAgent(BaseRLAgent):
    """High-level CBF-shielded navigation agent.

    Each control tick:
      1. Read ``self.loco_agent.base_lin_vel_pred`` (the locomotion encoder's
         linear-velocity estimate from the previous tick) as the agent's own
         base linear velocity.
      2. Read the robot's world pose from :class:`PoseSubNode` and rotate
         ``cfg.goal_world`` into the robot base frame.
      3. Assemble a 55-dim single-step observation
         ``prop(12) + log2(clip(rays, 0.1, 3.0))(41) + goal_local_xy(2)`` and
         push it into the 10-frame history buffer.
      4. Run the Nav ONNX policy (CBF-shielded velocity command, output dim 3).
      5. Post-process: ``clip(action, -3, 3)`` -> EMA(``smooth_factor``) ->
         ``clip(limit_vx, limit_vy, limit_vyaw)``.
      6. Push the filtered command to :attr:`loco_agent.pre_cmds` and step
         locomotion.

    Safety: if either ``rays_sub`` or ``pose_sub`` is stale, ``pre_cmds`` is
    forced to zero so the robot keeps standing. The handler additionally
    escalates to ``safe_stop`` when the outage persists.
    """

    def __init__(self, cfg=SEA_Nav_NavAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.loco_agent: SEA_Nav_LocoAgent = self.agents.get("loco")
        self.rays_sub: RaysSubNode = self.nodes.get("rays_sub")
        self.pose_sub: PoseSubNode = self.nodes.get("pose_sub")
        self.cfg: SEA_Nav_NavAgentCfg

        # Pre-allocated clip bounds to avoid per-step list allocations.
        self._pre_clip_lo = np.full(self.cfg.num_actions, self.cfg.pre_clip_lo, dtype=np.float32)
        self._pre_clip_hi = np.full(self.cfg.num_actions, self.cfg.pre_clip_hi, dtype=np.float32)
        self._limit_lo = np.array(self.cfg.min_action, dtype=np.float32)
        self._limit_hi = np.array(self.cfg.max_action, dtype=np.float32)

        # EMA buffer for the nav action. Initialized to zeros, reset on reset().
        self.filtered_action = np.zeros(self.cfg.num_actions, dtype=np.float32)

        # Per-step scratch buffers populated in prepare_obs_terms.
        self._rays_log2_buf = np.zeros(self.cfg.num_rays, dtype=np.float32)
        self._goal_local_buf = np.zeros(self.cfg.num_goal_obs, dtype=np.float32)
        self._goal_world = np.asarray(self.cfg.goal_world, dtype=np.float32)
        assert self._goal_world.shape == (2,), f"cfg.goal_world must be a 2-vector, got shape {self._goal_world.shape}"

        # Cached safety flag updated each step (for handler / debug_info).
        self.perception_fresh = False

        # Periodic debug print throttle (one line per ``log_every`` ticks).
        # Set ``cfg.log_every`` to 0 to silence.
        self._log_tick = 0
        self._log_every = int(getattr(self.cfg, "log_every", 50))  # default 50 ticks (= 1s at 50Hz)

    def parse_config(self):
        super().parse_config()
        self.num_actions = self.cfg.num_actions

    def load_model(self):
        onnx_path = os.path.join(self.logdir, "nav_model", "model.onnx")
        self.policy = ort.InferenceSession(onnx_path)
        self.input_name = self.policy.get_inputs()[0].name
        self.output_names = [out.name for out in self.policy.get_outputs()]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def prepare_obs_terms(self):
        """Build the 55-dim single-step observation::

            prop(12)        = projected_gravity(3)
                            + last_nav_command(3)
                            + base_lin_vel(3)         # = loco.base_lin_vel_pred
                            + base_ang_vel(3)
            rays(41)        = log2(clip(rays, 0.1, 3.0))
            goal_local(2)   = world->base(cfg.goal_world)

        The slot ordering is fixed: it must match what the Nav ONNX expects.
        """
        self._update_rays_log2()
        self._update_goal_local()

        self.observation_components = (
            (self.robot.projected_gravity, 1.0),  # 0:3   gravity
            (self.filtered_action, self.commands_scale),  # 3:6   last nav command
            (self.loco_agent.base_lin_vel_pred, self.obs_scale.lin_vel),  # 6:9   base_lin_vel
            (self.robot.base_ang_vel, self.obs_scale.ang_vel),  # 9:12  base_ang_vel
            (self._rays_log2_buf, self.obs_scale.rays_log2),  # 12:53 rays (log2)
            (self._goal_local_buf, self.obs_scale.goal_local),  # 53:55 goal_local_xy
        )

    def _update_rays_log2(self):
        """In-place: ``self._rays_log2_buf := log2(clip(rays_sub.rays, 0.1, 3.0))``.

        Matches the log2 + clip the policy was trained with. The lidar's
        physical range is capped at 3 m on the publisher side, so 3.0 is also
        the effective upper bound here.
        """
        np.clip(
            self.rays_sub.rays,
            self.cfg.rays_clip_min,
            self.cfg.rays_clip_max,
            out=self._rays_log2_buf,
        )
        np.log2(self._rays_log2_buf, out=self._rays_log2_buf)

    def _update_goal_local(self):
        """In-place: rotate ``goal_world - robot_world_xy`` into the robot
        base frame using ``-robot_world_yaw``.

        ``pose_sub`` provides the robot's pose in the world frame (from the
        simulator or external SLAM). The Nav agent owns the goal in the world
        frame and converts it to the base frame here so the policy sees the
        same representation as during training.
        """
        delta = self._goal_world - self.pose_sub.robot_world_xy  # (2,)
        yaw = float(self.pose_sub.robot_world_yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        # R(-yaw) @ delta = [[ c,  s], [-s, c]] @ delta
        self._goal_local_buf[0] = c * delta[0] + s * delta[1]
        self._goal_local_buf[1] = -s * delta[0] + c * delta[1]

    # ------------------------------------------------------------------
    # Inference & step
    # ------------------------------------------------------------------
    def infer(self):
        actor_input = self.obs_hist.buffer.reshape(1, -1)
        action = self.policy.run(self.output_names, {self.input_name: actor_input})[0]
        return action[0]  # (num_actions,)

    @profile_latency(
        cycle_threshold_ms=100.0,
        process_threshold_ms=50.0,
        log_interval_s=3.0,
        debug=True,
        check_capture_latency=False,
    )
    def step(self):
        """Run one control tick: build observation -> Nav inference -> filter
        -> push command to Loco -> Loco inference -> send action to robot.
        """
        self.get_observation()
        action = self.infer().astype(np.float32, copy=True)

        # Always run the EMA + clip pipeline so the filter state stays
        # consistent; perception-stale handling below decides whether the
        # filtered command is forwarded or zeroed out.
        # Use np.clip(...) returning a new array (instead of np.clip(out=...))
        # to stay portable across numpy versions.
        action_clipped = np.clip(action, self.cfg.pre_clip_lo, self.cfg.pre_clip_hi).astype(np.float32)
        ema = (1.0 - self.cfg.smooth_factor) * self.filtered_action + self.cfg.smooth_factor * action_clipped
        self.filtered_action = np.clip(ema, self._limit_lo, self._limit_hi).astype(np.float32)

        # Safe-stop short-circuit: if either perception stream is stale, force
        # zero commands. The handler can additionally escalate to a ``safe_stop``
        # state on prolonged outage based on this ``perception_fresh`` flag.
        self.perception_fresh = self.rays_sub.is_fresh() and self.pose_sub.is_fresh()
        if self.perception_fresh:
            self.loco_agent.pre_cmds = self.filtered_action.copy()
        else:
            self.loco_agent.pre_cmds = np.zeros(self.cfg.num_commands, dtype=np.float32)
            if hasattr(self.logger, "log_throttle"):
                self.logger.log_throttle(
                    "[SEA_Nav_NavAgent] perception not fresh "
                    f"(rays_dt={self.rays_sub.time_since_last_msg():.3f}s, "
                    f"pose_dt={self.pose_sub.time_since_last_msg():.3f}s); pre_cmds=0.",
                    seconds=1.0,
                    level="WARNING",
                )
            else:
                self.logger.warning(
                    "[SEA_Nav_NavAgent] perception not fresh; pre_cmds=0.",
                    once=True,
                )

        loco_action, _, _, _ = self.loco_agent.step()

        # Periodic [Nav] state line (1 Hz by default) so the operator can see
        # what the policy is commanding without sprinkling print() through the
        # control loop. Set cfg.log_every = 0 to silence.
        if self._log_every > 0:
            self._log_tick += 1
            if self._log_tick % self._log_every == 0:
                fa = self.filtered_action
                v = self.loco_agent.base_lin_vel_pred
                gl = self._goal_local_buf
                xy = self.pose_sub.robot_world_xy if self.pose_sub is not None else (0.0, 0.0)
                yaw = float(self.pose_sub.robot_world_yaw) if self.pose_sub is not None else 0.0
                self.logger.info(
                    f"[Nav] cmd=({fa[0]:+.3f}, {fa[1]:+.3f}, {fa[2]:+.3f}) "
                    f"vel_pred=({v[0]:+.3f}, {v[1]:+.3f}) "
                    f"pose=({xy[0]:+.3f}, {xy[1]:+.3f}, yaw={yaw:+.3f}) "
                    f"goal_local=({gl[0]:+.3f}, {gl[1]:+.3f}) "
                    f"fresh={self.perception_fresh}"
                )

        return loco_action, None, None, self.done

    def reset(self):
        # Navigation state must always consume high-level nav commands. Human
        # teleop still restores joystick control through SEA_Nav_LocoAgent.reset().
        self.loco_agent.wireless = False
        self.filtered_action[:] = 0.0
        self.obs_hist.reset()
        self._rays_log2_buf[:] = 0.0
        self._goal_local_buf[:] = 0.0
        self.perception_fresh = False

    def debug_info(self, infos=None):
        for info in infos or ():
            if info == "base_lin_vel":
                v = self.loco_agent.base_lin_vel_pred
                self.logger.info(f"[Nav] base_lin_vel_pred: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
            elif info == "base_ang_vel":
                w = self.robot.base_ang_vel
                self.logger.info(f"[Nav] base_ang_vel: ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")
            elif info == "commands":
                c = self.filtered_action
                self.logger.info(f"[Nav] filtered_action (commands): ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
            elif info == "goal_local":
                g = self._goal_local_buf
                self.logger.info(
                    f"[Nav] goal_local_xy: ({g[0]:.3f}, {g[1]:.3f}); goal_world={self._goal_world.tolist()}"
                )
            elif info == "rays":
                r = self.rays_sub.rays
                self.logger.info(
                    f"[Nav] rays: min={float(r.min()):.3f} m, max={float(r.max()):.3f} m, "
                    f"fresh={self.rays_sub.is_fresh()}"
                )
            elif info == "pose":
                xy = self.pose_sub.robot_world_xy
                yaw = self.pose_sub.robot_world_yaw
                self.logger.info(
                    f"[Nav] robot_world: xy=({xy[0]:.3f}, {xy[1]:.3f}), yaw={yaw:.3f}, fresh={self.pose_sub.is_fresh()}"
                )
            elif info == "perception_fresh":
                self.logger.info(f"[Nav] perception_fresh: {self.perception_fresh}")

    @property
    def done(self):
        return False
