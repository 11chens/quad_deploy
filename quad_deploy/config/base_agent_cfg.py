class BaseAgentCfg:
    decimation = 4  # infer every 4 steps
    smooth_factor = 0.1
    dead_zone = 0.0
    min_cmds = [-0.5, -0.5, -1.0]
    max_cmds = [1.0, 0.5, 1.0]

    num_commands = 3
    num_props = 47
    len_history = 0
    max_episode_length_s = 20  # max episode length in seconds

    class obs_scale:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        commands_scale = [lin_vel, lin_vel, ang_vel]
