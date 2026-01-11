from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class HomiNavAgentCfg(BaseAgentCfg):
    # (u, v, depth)
    # num_commands = 3 * 7  # 7 points
    num_commands = 3 * 3  # 3 points
    # num_commands = 3 # 1 point
    # (vx, vy, vyaw, pitch)
    num_actions = 4
    post_clip = False
    num_props = num_actions + num_commands + 1 + 9  # lin_vel(3), ang_vel(3), gravity(3)
    num_nav_commands = num_commands
    len_history = 5
    pixel_gain = 10
    cx_norm = 0.5
    cy_norm = 0.5
    max_episode_length_s = 9  # max episode length in seconds

    smooth_factor = 0.2

    limit_vx = [-0.5, 0.5]  # [m/s]
    limit_vy = [-0.5, 0.5]  # [m/s]
    limit_vyaw = [-0.5, 0.5]  # [rad/s]
    limit_pitch = [-3.14 / 6, 3.14 / 6]  # [rad]

    min_action = [limit_vx[0], limit_vy[0], limit_vyaw[0], limit_pitch[0]]
    max_action = [limit_vx[1], limit_vy[1], limit_vyaw[1], limit_pitch[1]]

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [1.0, 1.0, 1.0]  # (u, v, depth)
