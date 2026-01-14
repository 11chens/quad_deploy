from quad_deploy.config.base_agent_cfg import BaseAgentCfg


class HomiLocoAgentCfg(BaseAgentCfg):
    num_commands = 4
    num_props = 47
    len_history = 5
    # [lin_vel, lin_vel, ang_vel, pitch]
    min_cmds = [-0.5, -0.5, -1.0, -3.14 / 6 - 0.05]
    max_cmds = [1.0, 0.5, 1.0, 3.14 / 6 + 0.05]

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        pitch = 1.0
        commands_scale = [lin_vel, lin_vel, ang_vel, pitch]
