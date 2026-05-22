from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class SEA_Nav_LocoAgentCfg(BaseAgentCfg):
    """Configuration for the SEA-Nav low-level (velocity-tracking) policy.

    The agent consumes a 45-dim single-step observation::

        base_ang_vel(3) + projected_gravity(3) + commands(3)
        + dof_pos_rel(12) + dof_vel(12) + last_action(12)

    and the 10-frame history flattens to a 450-dim ONNX input. The ONNX
    outputs both the 12-D joint-target action and a 3-D ``vel_pred`` (the
    estimated base linear velocity); ``vel_pred`` is exposed on the agent as
    ``base_lin_vel_pred`` for the Nav agent to consume.
    """

    num_commands = 3  # (vx, vy, vyaw)
    num_props = 45
    len_history = 10

    # Hardware velocity limits (m/s, rad/s) - same as Nav action limits.
    min_cmds = [-0.5, -1.0, -1.0]
    max_cmds = [2.0, 1.0, 1.0]

    # The Nav agent owns the EMA + clip pipeline; this Loco-side
    # ``post_commands`` should be a pass-through:
    #   - smooth_factor = 1.0 disables the EMA  (post_cmds := pre_cmds)
    #   - post_clip = False disables the second clip
    smooth_factor = 1.0
    post_clip = False

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0  # unused by Loco obs (kept for BaseAgentCfg compatibility)
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [lin_vel, lin_vel, ang_vel]  # [2.0, 2.0, 0.25]
