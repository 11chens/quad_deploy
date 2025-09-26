import numpy as np

from quad_deploy.agents.base_rl_agent import BaseRLAgent
from quad_deploy.config.stand_agent_cfg import StandAgentCfg


class StandAgent(BaseRLAgent):
    def __init__(self, cfg=StandAgentCfg, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)
        self.cfg: StandAgentCfg

    def parse_config(self):
        super().parse_config()
        self.startPos = self.cfg.startPos

        # Target positions for standing up
        self._targetPos_1 = self.cfg.targetPos_1
        self._targetPos_2 = self.cfg.targetPos_2

        self.stand_up_joint_pos = np.array(
            self.cfg.stand_up_joint_pos,
            dtype=np.float32,
        )
        self.stand_down_joint_pos = np.array(
            self.cfg.stand_down_joint_pos,
            dtype=np.float32,
        )

        # Duration for each phase of standing up
        self.duration_1 = self.cfg.duration_1
        self.duration_2 = self.cfg.duration_2
        self.duration_3 = self.cfg.duration_3
        self.duration_4 = self.cfg.duration_4

        # Standing parameters
        self.stand_kp = self.cfg.stand_kp
        self.stand_kd = self.cfg.stand_kd

        # Percentages for each phase
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0
        self.running_time = 0.0
        self.firstRun = True

        # NOTE: Dont assign values directly to avoid modifying the ground truth of robot
        self.loco_kp = self.robot.stiffness["joint"]
        self.loco_kd = self.robot.damping["joint"]
        self.p_gains_ = self.robot.p_gains.copy()
        self.d_gains_ = self.robot.d_gains.copy()
        self.final_dof_pos = self.robot.default_dof_pos.copy()
        self.robot_coordinates_action = np.zeros_like(self.final_dof_pos)

    def step(self):
        return self.real_stand_up()

    def real_stand_up(self):
        if self.firstRun:
            for i in range(self.robot.NUM_DOF):
                self.startPos[i] = self.robot.low_state.motor_state[i].q
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            self.logger.info("step into phase 1: move to targetPos1", once=True)
            for i in range(self.robot.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_1) * self.startPos[
                    i
                ] + self.percent_1 * self._targetPos_1[i]
                self.p_gains_[i] = self.stand_kp
                self.d_gains_[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 < 1):
            self.logger.info("step into phase 2: move to targetPos2", once=True)
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(self.robot.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_2) * self._targetPos_1[
                    i
                ] + self.percent_2 * self._targetPos_2[i]
                self.p_gains_[i] = self.stand_kp
                self.d_gains_[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.logger.info("step into phase 3: keep targetPos2", once=True)
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(self.robot.NUM_DOF):
                self.robot_coordinates_action[i] = self._targetPos_2[i]
                self.p_gains_[i] = self.stand_kp
                self.d_gains_[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 < 1):
            self.logger.info("step into phase 4: move to defaultPos", once=True)
            self.percent_4 += 1 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(self.robot.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_4) * self._targetPos_2[
                    i
                ] + self.percent_4 * self.final_dof_pos[i]
                self.p_gains_[i] = (1 - self.percent_4) * self.stand_kp + self.percent_4 * self.loco_kp
                self.d_gains_[i] = (1 - self.percent_4) * self.stand_kd + self.percent_4 * self.loco_kd

        else:
            self.logger.info("step into phase 5: keep defaultPos", once=True)
            for i in range(self.robot.NUM_DOF):
                self.robot_coordinates_action[i] = self.final_dof_pos[i]
                self.p_gains_[i] = self.loco_kp
                self.d_gains_[i] = self.loco_kd

        action = (self.robot_coordinates_action - self.final_dof_pos) / self.robot.action_scale
        return action, self.p_gains_, self.d_gains_, self.done

    def reset(self):
        self.firstRun = True
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0

    @property
    def done(self):
        return self.percent_4 == 1
