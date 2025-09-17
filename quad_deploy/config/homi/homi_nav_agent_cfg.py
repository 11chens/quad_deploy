from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class HomiNavAgentCfg(BaseAgentCfg):
    # (u, v, depth)
    num_commands = 3
    # (vx, vy, vyaw, pitch)
    num_actions = 4
    num_props = 16
    len_history = 10
    dead_zone = 0.2

    limit_vx = [-0.0, 1.0]  # [m/s]
    limit_vy = [-0.05, 0.05]  # [m/s]
    limit_vyaw = [-1.0, 1.0]  # [rad/s]
    limit_pitch = [-0.5, 0.5]  # [rad]

    min_action = [limit_vx[0], limit_vy[0], limit_vyaw[0], limit_pitch[0]]
    max_action = [limit_vx[1], limit_vy[1], limit_vyaw[1], limit_pitch[1]]

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 2.0
        commands_scale = [1.0, 1.0, 1.0]  # (u, v, depth)
