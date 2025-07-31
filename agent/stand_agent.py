import numpy as np

from agent.base import BaseAgent
from robot_real import UnitreeGo2


class StandAgent(BaseAgent):
    def __init__(self, logdir: None, robot_node: UnitreeGo2):
        super().__init__(logdir, robot_node)

        self.startPos = [0.0] * self.robot_node.NUM_DOF

        # Target positions for standing up
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3]

        self.stand_up_joint_pos = np.array(
            [
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
            ],
            dtype=np.float32,
        )

        self.stand_down_joint_pos = np.array(
            [
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
            ],
            dtype=np.float32,
        )

        # Duration for each phase of standing up
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 200
        self.duration_4 = 200

        self.firstRun = True
        # Percentages for each phase
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0

        # Standing parameters
        self.stand_kp = 60.0
        self.stand_kd = 5.0
        self.running_time = 0.0
        # NOTE: Dont assign values directly to avoid modifying the ground truth of robot
        self.loco_kp = self.robot_node.stiffness["joint"]
        self.loco_kd = self.robot_node.damping["joint"]
        self.p_gains = self.robot_node.p_gains.copy()
        self.d_gains = self.robot_node.d_gains.copy()
        self.final_dof_pos = self.robot_node.default_dof_pos.copy()
        self.robot_coordinates_action = np.zeros_like(self.final_dof_pos)

    def step(self):
        # if self.robot_node.simrun is True:
        #     return self.sim_stand_up()
        # else:
        #     return self.real_stand_up()
        return self.real_stand_up()

    def real_stand_up(self):
        if self.firstRun:
            for i in range(self.robot_node.NUM_DOF):
                self.startPos[i] = self.robot_node.low_state.motor_state[i].q
            self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            self.robot_node.logger.info("step into phase 1: move to targetPos1", once=True)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_1) * self.startPos[
                    i
                ] + self.percent_1 * self._targetPos_1[i]
                self.p_gains[i] = self.stand_kp
                self.d_gains[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 < 1):
            self.robot_node.logger.info("step into phase 2: move to targetPos2", once=True)
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_2) * self._targetPos_1[
                    i
                ] + self.percent_2 * self._targetPos_2[i]
                self.p_gains[i] = self.stand_kp
                self.d_gains[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.robot_node.logger.info("step into phase 3: keep targetPos2", once=True)
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = self._targetPos_2[i]
                self.p_gains[i] = self.stand_kp
                self.d_gains[i] = self.stand_kd

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 < 1):
            self.robot_node.logger.info("step into phase 4: move to defaultPos", once=True)
            self.percent_4 += 1 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_4) * self._targetPos_2[
                    i
                ] + self.percent_4 * self.final_dof_pos[i]
                self.p_gains[i] = (1 - self.percent_4) * self.stand_kp + self.percent_4 * self.loco_kp
                self.d_gains[i] = (1 - self.percent_4) * self.stand_kd + self.percent_4 * self.loco_kd

        else:
            self.robot_node.logger.info("step into phase 5: keep defaultPos", once=True)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = self.final_dof_pos[i]
                self.p_gains[i] = self.loco_kp
                self.d_gains[i] = self.loco_kd

        action = (self.robot_coordinates_action - self.final_dof_pos) / self.robot_node.action_scale
        done = self.percent_4 == 1
        return action, self.p_gains, self.d_gains, done

    def sim_stand_up(self):
        self.stand_kp = 50.0
        self.stand_kd = 3.5
        self.running_time += 0.005
        if self.running_time < 3.0:
            self.robot_node.logger.info("step into phase 1", once=True)
            phase = np.tanh(self.running_time / 1.2)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = (
                    phase * self.stand_up_joint_pos[i] + (1 - phase) * self.stand_down_joint_pos[i]
                )
                self.p_gains[i] = phase * 50.0 + (1 - phase) * 20.0
                self.d_gains[i] = 3.5
        elif self.percent_4 < 1:
            self.robot_node.logger.info("step into phase 2", once=True)
            self.percent_4 += 1 / 400
            self.percent_4 = min(self.percent_4, 1)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = (1 - self.percent_4) * self.stand_up_joint_pos[
                    i
                ] + self.percent_4 * self.final_dof_pos[i]
                self.p_gains[i] = (1 - self.percent_4) * self.stand_kp + self.percent_4 * self.loco_kp
                self.d_gains[i] = (1 - self.percent_4) * self.stand_kd + self.percent_4 * self.loco_kd
        else:
            self.robot_node.logger.info("step into phase 3", once=True)
            for i in range(self.robot_node.NUM_DOF):
                self.robot_coordinates_action[i] = self.final_dof_pos[i]
                self.p_gains[i] = self.loco_kp
                self.d_gains[i] = self.loco_kd
        action = (self.robot_coordinates_action - self.final_dof_pos) / self.robot_node.action_scale
        done = self.percent_4 == 1.0
        return action, self.p_gains, self.d_gains, done

    def reset(self):
        self.robot_node.logger.reset()
        self.firstRun = True
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0
