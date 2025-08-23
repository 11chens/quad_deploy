from config.base_agent_cfg import BaseAgentCfg


class LocoPitchAgentCfg(BaseAgentCfg):
    num_commands = 4
    num_props = 47
    len_history = 10
    # [lin_vel, lin_vel, ang_vel, pitch]
    min_cmds = [-0.5, -0.8, -1.0, -0.5]
    max_cmds = [1.5, 0.8, 1.0, 0.5]

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        pitch = 1.0
        commands_scale = [lin_vel, lin_vel, ang_vel, pitch]
