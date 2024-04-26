from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH
from spatialmath import SE3
from numpy import pi
from surgical_robotics_challenge.kinematics.psmIK import compute_FK
import numpy as np
import time
from accel_challenge.challenge1.tool import RPY2T, T_2_SE3


PI_2 = pi / 2

class PSM_KIN:
    def __init__(self):
        self.num_links = 7
        self.L_rcc = 4.389  # From dVRK documentation x 10
        self.L_tool = 4.16   # From dVRK documentation x 10
        self.L_pitch2yaw = 0.09  # Fixed length from the palm joint to the pinch joint
        self.L_yaw2ctrlpnt = 0.106  # Fixed length from the pinch joint to the pinch tip
        self.L_tool2rcm_offset = 0.229  # Delta between tool tip and the Remote Center of Motion

        # Joint limits
        self.qmin = np.deg2rad([-91.96, -60, 0.0, -175, -90, -85])
        self.qmax = np.deg2rad([91.96, 60, 0.0, 175, 90, 85])
        self.qmin[2] = 0
        self.qmax[2] = 2.4

        self.tool_T = np.array([
            [0, -1,  0, 0],
            [0,  0,  1, self.L_yaw2ctrlpnt],
            [-1, 0,  0, 0],
            [0,  0,  0, 1]
        ])

        self.build_kin()

    def build_kin(self):
        self.robot = DHRobot([
            RevoluteMDH(alpha=PI_2, a=0, d=0, offset=PI_2, qlim=np.array([self.qmin[0], self.qmax[0]])),
            RevoluteMDH(alpha=-PI_2, a=0, d=0, offset=-PI_2, qlim=np.array([self.qmin[1], self.qmax[1]])),
            PrismaticMDH(alpha=PI_2, a=0, theta=0, offset=-self.L_rcc, qlim=np.array([self.qmin[2], self.qmax[2]])),
            RevoluteMDH(alpha=0, a=0, d=self.L_tool, offset=0, qlim=np.array([self.qmin[3], self.qmax[3]])),
            RevoluteMDH(alpha=-PI_2, a=0, d=0, offset=-PI_2, qlim=np.array([self.qmin[4], self.qmax[4]])),
            RevoluteMDH(alpha=-PI_2, a=self.L_pitch2yaw, d=0, offset=-PI_2, qlim=np.array([self.qmin[5], self.qmax[5]]))
        ], name="PSM")

        self.robot.tool = SE3(self.tool_T)

    def fk(self, qs):
        assert len(qs) == 6
        return self.robot.fkine(qs)

    def ik(self, T_dsr, q0, method="LM"):
        assert len(q0) == 6
        if method == "LM":
            result = self.robot.ikine_LM(T=T_dsr, q0=q0)
            return result.q.tolist(), result.success
        elif method == "JNT_LMIT":
            result = self.robot.ikine_min(T=T_dsr, q0=q0, qlim=True)
            return result.q.tolist(), result.success
        else:
            raise NotImplementedError

    def jacob(self, qs):
        assert len(qs) == 6
        _qs = qs + [0]
        return self.robot.jacob0(_qs)

    def sample_q(self):
        """ Sample joint position within joint limits """
        return np.random.uniform(low=self.qlim[0], high=self.qlim[1], size=(6,)).tolist()

    @property
    def qlim(self):
        return self.robot.qlim[0], self.robot.qlim[1]

    def is_out_qlim(self, q, q_min_margin=None, q_max_margin=None, margin_ratio=None):
        _q = np.array(q)
        if margin_ratio is not None:
            _q_min_margin = _q_max_margin = (self.qlim[1] - self.qlim[0]) * margin_ratio
        else:
            _q_min_margin = np.array(q_min_margin) if q_min_margin is not None else np.zeros(len(q))
            _q_max_margin = np.array(q_max_margin) if q_max_margin is not None else np.zeros(len(q))

        result = np.logical_or(_q < self.qlim[0] + _q_min_margin, _q > self.qlim[1] - _q_max_margin)
        return np.sum(result) != 0, result


if __name__ == '__main__':
    kin = PSM_KIN()
    IS_FK_TEST = False
    IS_IK_TEST = True
    TEST_JNT_LIM = False
    
    if IS_FK_TEST:
        test_num = 10
        err = []
        for i in range(test_num):
            qs = np.random.uniform(low=-pi, high=pi, size=(6,)).tolist()
            T1 = SE3(compute_FK(qs, 7))
            T2 = kin.fk(qs)
            err.append(np.linalg.norm(T2 - T1))
        print(err)
        print("Error mean:", np.mean(np.array(err)))
        
    if IS_IK_TEST:
        test_num = 10
        err = []
        is_success = []
        ts = []
        error_bias = np.deg2rad(40)
        method = "JNT_LMIT"

        for i in range(test_num):
            q_dsr = kin.sample_q()
            q0 = (np.array(q_dsr) + error_bias * np.random.uniform(low=-1, high=1, size=(6,))).tolist()
            T_dsr = kin.fk(q_dsr)
            start = time.time()
            q, success = kin.ik(T_dsr=T_dsr, q0=q0, method=method)
            ts.append(time.time() - start)
            T = kin.fk(q)
            e = np.linalg.norm(T.t - T_dsr.t)
            err.append(e)
            is_success.append(success)

        print("Norm error of T:", np.mean(np.array(err)))
        print("Success norm error of T:", np.mean(np.array(err)[is_success]))
        print("Success rate:", sum(is_success), "/", test_num)
        print("Time elapsed for IK:", sum(ts) / test_num)

    if TEST_JNT_LIM:
        q = [10, 0, 0, 0, 0, 0]
        print("Is out of joint limits:", kin.is_out_qlim(q))
        T = kin.fk(q)
        print("Transformation matrix rotation part:", T.R)
        print("Rotation matrix for π/4 yaw:", RPY2T(0, 0, 0, 0, 0, pi/4).M)
        print("Conversion from transformation matrix to SE3:", T_2_SE3(RPY2T(0, 0, 0, 0, 0, pi/4)))