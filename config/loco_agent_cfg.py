class LocoAgentCfg:
    num_commands = 3
    num_props = 45
    len_history = 10
    smooth_factor = 0.1
    min_cmds = [-0.5, -0.8, -1.0]
    max_cmds = [1.5, 0.8, 1.0]
    dead_zone = 0.2

    class obs_scale:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [lin_vel, lin_vel, ang_vel]
