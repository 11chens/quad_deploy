class StandAgentCfg:
    startPos = [0.0] * 12
    # Target positions for standing up
    targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
    targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3]

    stand_up_joint_pos = [
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
        0.00571868,
        0.608813,
        -1.21763,
        -0.00571868,
        0.608813,
        -1.21763,
    ]

    stand_down_joint_pos = [
        0.0473455,
        1.22187,
        -2.44375,
        -0.0473455,
        1.22187,
        -2.44375,
        0.0473455,
        1.22187,
        -2.44375,
        -0.0473455,
        1.22187,
        -2.44375,
    ]
    # Duration for each phase of standing up
    duration_1 = 50  # 500 125
    duration_2 = 50  # 500 125
    duration_3 = 100  # 200
    duration_4 = 100  # 200

    # Standing parameters
    stand_kp = 60.0
    stand_kd = 5.0
