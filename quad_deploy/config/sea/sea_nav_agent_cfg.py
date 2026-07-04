from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class SEANavAgentCfg(BaseAgentCfg):
    """Configuration for the SEA-Nav high-level (CBF-shielded) policy.

    The agent consumes a 55-dim single-step observation::

        prop(12) + log2(clip(rays, 0.1, 3.0))(41) + goal_local_xy(2)

    where ``prop`` = projected_gravity(3) + last nav command(3)
                  + base_lin_vel(3) + base_ang_vel(3),
    and the 10-frame history flattens to a 550-dim ONNX input.

    The ONNX outputs a 3-D velocity command ``(vx, vy, vyaw)`` (CBF inline);
    the agent then applies ``clip(-3, 3)`` -> EMA(``smooth_factor``) ->
    ``clip(limit_v*)`` before forwarding to the Loco agent.

    The goal is configured in the *world* frame (``goal_world``); the Nav
    agent rotates it into the robot base frame on each tick using the pose
    received on ``/pose``.
    """

    # ----- action / commands -----
    num_actions = 3  # nav outputs (vx, vy, vyaw)
    num_commands = 3  # same 3 channels feed into loco
    post_clip = False  # nav owns its own clip pipeline inside step()

    # ----- single-step observation decomposition -----
    num_base_props = 12  # gravity(3) + last_command(3) + base_lin_vel(3) + base_ang_vel(3)
    num_rays = 41
    num_goal_obs = 2
    num_props = num_base_props + num_rays + num_goal_obs  # 55
    len_history = 10

    # ----- nav action post-processing -----
    smooth_factor = 0.5  # EMA alpha
    pre_clip_lo = -3.0  # raw nav action is clipped to this range before the EMA
    pre_clip_hi = 3.0

    # Hardware velocity limits (m/s, rad/s). Applied after the EMA.
    limit_vx = [-0.5, 2.0]
    limit_vy = [-1.0, 1.0]
    limit_vyaw = [-1.0, 1.0]

    min_action = [limit_vx[0], limit_vy[0], limit_vyaw[0]]
    max_action = [limit_vx[1], limit_vy[1], limit_vyaw[1]]

    # Same bounds are used by the base joystick path.
    min_cmds = min_action
    max_cmds = max_action

    # ----- ray encoding -----
    # log2 + clip applied inside the Nav agent before the observation is built;
    # 3.0 m matches the lidar's physical range on both the simulator and the
    # real-robot driver.
    rays_clip_min = 0.1
    rays_clip_max = 3.0

    # ----- goal in the world frame -----
    # Matches the simulator's goal for ``--seed 42`` (any difficulty): the
    # spawn / goal are sampled before the obstacles, so the goal is
    # deterministic for a fixed seed. Override per trial via cmdline
    # (--goal_x / --goal_y) or by editing this field.
    goal_world = [1.94, 3.34]

    # ----- RaysSubNode kwargs (forwarded to the node by the manager) -----
    rays_topic = "/rays"
    rays_msg_type = "LaserScan"
    rays_num_rays = num_rays
    rays_range_min = rays_clip_min
    rays_range_max = rays_clip_max
    rays_frame_id = "base_link"
    rays_timeout_s = 0.5  # >= 5 x publisher period at 10 Hz
    rays_expected_freq_hz = 10.0

    # ----- PoseSubNode kwargs (forwarded to the node by the manager) -----
    pose_topic = "/pose"
    pose_frame_id = "world"
    pose_timeout_s = 0.5
    pose_expected_freq_hz = 10.0

    class obs_scale(BaseAgentCfg.obs_scale):
        # Scales applied when assembling the Nav observation. Note these are
        # different from the Loco agent's scales (which use lin_vel=2.0,
        # ang_vel=0.25) - do not unify them.
        lin_vel = 1.0
        ang_vel = 1.0
        dof_pos = 1.0  # unused (kept for BaseAgentCfg compatibility)
        dof_vel = 0.05  # unused (kept for BaseAgentCfg compatibility)
        rays_log2 = 1.0  # log2 already absorbs the scaling
        goal_local = 1.0
        commands_scale = [1.0, 1.0, 1.0]
