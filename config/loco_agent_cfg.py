from config.base_agent_cfg import BaseAgentCfg


class LocoAgentCfg(BaseAgentCfg):
    num_commands = 3
    num_props = 45
    len_history = 10

    class obs_scale(BaseAgentCfg.obs_scale):
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [lin_vel, lin_vel, ang_vel]
