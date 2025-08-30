from config.base_agent_cfg import BaseAgentCfg


class HomiNavAgentCfg(BaseAgentCfg):
    # (u, v, depth)
    num_commands = 3
    # (vx, vy, vyaw, pitch)
    num_actions = 4
    num_props = 16
    len_history = 10
    dead_zone = 0.2

    min_action = [-0.5, -0.1, -1.0, -0.5]
    max_action = [1.0, 0.1, 1.0, 0.5]

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 2.0
        commands_scale = [1.0, 1.0, 1.0]  # (u, v, depth)
