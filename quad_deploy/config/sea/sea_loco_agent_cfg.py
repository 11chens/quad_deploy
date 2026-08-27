from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class SEALocoAgentCfg(BaseAgentCfg):
    """Configuration for the SEA-Nav low-level (velocity-tracking) policy.

    The agent consumes a 47-dim single-step observation::

        base_ang_vel(3) + projected_gravity(3) + commands(4) + pitch(1)
        + dof_pos_rel(12) + dof_vel(12) + last_action(12)

    and the 5-frame history flattens to a 235-dim ONNX input. The ONNX
    outputs both the 12-D joint-target action and a 3-D ``vel_pred`` (the
    estimated base linear velocity); ``vel_pred`` is exposed on the agent as
    ``base_lin_vel_pred`` for the Nav agent to consume.

    The Nav agent emits a 3-D ``(vx, vy, vyaw)`` command; the Loco agent pads
    it to the 4-D ``(vx, vy, vyaw, pitch)`` command the policy was trained
    with, filling the pitch channel with ``command_pitch``.
    """

    num_commands = 4  # (vx, vy, vyaw, pitch)
    num_props = 47
    len_history = 5

    # [vx, vy, vyaw, pitch] hardware command limits (m/s, m/s, rad/s, rad).
    min_cmds = [-0.5, -0.5, -1.0, -3.14 / 6 - 0.05]
    max_cmds = [1.0, 0.5, 1.0, 3.14 / 6 + 0.05]

    # Pitch (rad) padded onto the Nav agent's 3-D command. The Nav policy only
    # drives (vx, vy, vyaw), so the pitch channel is held at this constant.
    command_pitch = 0.0

    # The Nav agent owns the EMA + clip pipeline; this Loco-side
    # ``post_commands`` should be a pass-through:
    #   - smooth_factor = 1.0 disables the EMA  (post_cmds := pre_cmds)
    #   - post_clip = False disables the second clip
    smooth_factor = 1.0
    post_clip = False

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0  # no direct obs term; reused to scale the vx/vy command channels in commands_scale below
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        pitch = 1.0
        commands_scale = [lin_vel, lin_vel, ang_vel, pitch]  # [2.0, 2.0, 0.25, 1.0]
