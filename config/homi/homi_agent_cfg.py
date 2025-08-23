from config.base_agent_cfg import BaseAgentCfg


class HomiAgentCfg(BaseAgentCfg):
    # [px, py]
    num_commands = 2
    num_props = 12
    num_actor_obs = 77
    len_history = 10
    smooth_factor = 0.1
    goal_world = [5.0, 0.0]
    min_cmds = [-0.5, -0.8, -1.0]
    max_cmds = [1.5, 0.8, 1.0]
    dead_zone = 0.2
    sigma = 0.1

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [1.0, 1.0]
