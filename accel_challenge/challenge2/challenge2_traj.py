from accel_challenge.challenge1.ros_client import ClientEngine
from PyKDL import Frame, Rotation, Vector
import PyKDL
from accel_challenge.challenge1.tool import RPY2T
from accel_challenge.challenge1.param import T_gt_n, T_hover_gt, NEEDLE_R, T_tip_n
from accel_challenge.challenge1.examples.calibrate import calibrate_joint_error
import time
import numpy as np
pi = np.pi

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool,Empty,Int32
import rospy
from pathlib import Path
from ambf_msgs.msg import RigidBodyState
from accel_challenge.challenge1.examples.calibrate import calibrate_joint_error, DLC_CONFIG_PATH_dict, ERROR_DATA_DIR


team_name = 'Amagi'
INTER_NUM = 50
INTER_NUM_SHORT = 250
engine = ClientEngine()

def pose_stamped_msg_to_frame(msg):
    return Frame(Rotation.Quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
                 Vector(msg.position.x, msg.position.y, msg.position.z))

def task_3_init_finish_cb(msg):
    global is_finish_init
    is_finish_init = True

def needle_pose_sub(msg):
    global T_nINw_reported, T_ecmINw
    T_nINe = pose_stamped_msg_to_frame(msg)
    T_nINw_reported = T_ecmINw * T_nINe
    T_nINw_reported.p[2] = 0.711629

def ecm_cb(msg):
    global T_ecmINw
    T_ecmINw = pose_stamped_msg_to_frame(msg.pose)

task_pub = rospy.Publisher(f'/surgical_robotics_challenge/completion_report/{team_name}/task3', Bool)
task3_init_pub = rospy.Publisher('/CRTK/scene/task_3_setup/init', Empty)
task3_init_finish_sub = rospy.Subscriber('/CRTK/scene/task_3_setup/ready', Empty, task_3_init_finish_cb, queue_size=1)
task3_estimation_num_pub = rospy.Publisher('/Amagi_msgs/estimation_num', Int32)
needle_pos_sub = rospy.Subscriber(f'/surgical_robotics_challenge/completion_report/{team_name}/task1/', PoseStamped, needle_pose_sub, queue_size=1)
ecm_sub = rospy.Subscriber('/ambf/env/CameraFrame/State', RigidBodyState, ecm_cb, queue_size=1)
T_ecmINw = Frame()
T_nINw_reported = Frame()

engine.add_clients(['psm1', 'psm2', 'ecm', 'scene'])
engine.start()
print("Reset pose...")

move_arm_list = ['psm2', 'psm1']
for move_arm in move_arm_list:
    joint_angle = 0.3
    engine.clients[move_arm].open_jaw(joint_angle)
    engine.clients[move_arm].joint_calibrate_offset = np.zeros(6)
    load_dict = {'dlc_config_path': DLC_CONFIG_PATH_dict[move_arm],
                 'keras_model_path': str(Path(ERROR_DATA_DIR) / move_arm / 'model.hdf5'),
                 'scalers_path': str(Path(ERROR_DATA_DIR) / move_arm / 'scalers.pkl')}
    error = calibrate_joint_error(_engine=engine, load_dict=load_dict, arm_name=move_arm)
    engine.clients[move_arm].joint_calibrate_offset = np.concatenate((error, np.zeros(3)))
    print(f"Predict joint offset: {engine.clients[move_arm].joint_calibrate_offset}")
    engine.clients[move_arm].reset_pose()

is_finish_init = False
time.sleep(0.3)
task3_init_pub.publish(Empty())

while not is_finish_init:
    pass

del task3_init_finish_sub

engine.clients['psm1'].grasp_point_offset = RPY2T(*[0, 0, -0.030, 0, 0, 0])
engine.clients['psm2'].grasp_point_offset = RPY2T(*[0, 0, -0.013, 0, 0, 0])
engine.clients['psm1'].reset_pose()
engine.clients['psm1'].open_jaw()
engine.clients['psm1'].wait()
engine.clients['ecm'].move_ecm_jp([0, 0, 0, 0])

T_n_w0 = engine.get_signal('scene', 'measured_needle_cp')
T_NEEDLE_GRASP_PSM2 = engine.clients['psm2'].T_g_w_msr.Inverse() * T_n_w0

engine.close_client('ecm')

x_origin, y_origin, z_origin = -0.211084, 0.560047 - 0.3, 0.706611 + 0.2
YAW = -0.8726640502948968
pose_origin_psm2 = RPY2T(0, 0.15, 0.1, 0, 0, 0) * RPY2T(0.2, 0, 0, 0, 0, 0) * \
                   RPY2T(x_origin, y_origin, z_origin, np.pi, -np.pi / 2, 0) * \
                   RPY2T(0, 0, 0, 0, 0, YAW) * RPY2T(0, 0, 0, -np.pi / 2, 0, 0)

needle_pos0 = [-0.15786, 0.0619, 0.7417]
needle_rpy0 = [0, 0, 0]
T_needle = RPY2T(*needle_pos0, *needle_rpy0)
T_g_w_init_dsr_PSM2 = T_needle * T_NEEDLE_GRASP_PSM2.Inverse()

needle_pos0 = [-0.21, -0.15, 0.7417]
needle_rpy0 = [0,0,0]
T_needle = RPY2T(*needle_pos0, *needle_rpy0)
T_gt_n_PSM1 = RPY2T(*[0.015,0.103,0, 0,0,0]) * RPY2T(*[0,0,0, -pi,0, np.deg2rad(10)])
T_g_w_init_dsr_PSM1 = T_needle * T_gt_n_PSM1

for i in range(4):
    T_ENTRY = engine.get_signal('scene', f"measured_entry{i+1}_cp")
    T_EXIT = engine.get_signal('scene', f"measured_exit{i+1}_cp")
    alpha = np.deg2rad(35)  # Tip visibility after penetrating exit hole
    beta = np.deg2rad(140)  # Angle to extract needle
    d = (T_ENTRY.p - T_EXIT.p).Norm()
    r = NEEDLE_R
    theta = np.arcsin(d / (2 * r))
    
    # Pivot frame calculation
    Rx_pivot_w = T_EXIT.p - T_ENTRY.p
    Rx_pivot_w = Rx_pivot_w / Rx_pivot_w.Norm()
    Rz_pivot_w = Vector(0, 0, 1)
    Ry_pivot_w = Rz_pivot_w * Rx_pivot_w
    p_pivot_w = (T_ENTRY.p + T_EXIT.p) / 2 + Vector(0, 0, r * np.cos(theta))
    T_pivot_w = Frame(Rotation(Rx_pivot_w, Ry_pivot_w, Rz_pivot_w), p_pivot_w)
    
    # Needle insertion interpolate trajectory frames
    INSERTION_ITPL_NUM = 400
    theta_list = np.linspace(theta, -theta - alpha, INSERTION_ITPL_NUM)
    T_tip_w_ITPL_lst = [T_pivot_w * RPY2T(0, 0, 0, 0, t, 0) * RPY2T(0, 0, -r, 0, 0, 0) * RPY2T(0, 0, 0, -np.pi / 2, 0, 0)
                        for t in theta_list]
    theta_extract_list = np.linspace(-theta - alpha, -theta - beta, INSERTION_ITPL_NUM)
    T_tip_w_ITPL_extract_lst = [T_pivot_w * RPY2T(0, 0, 0, 0, t, 0) * RPY2T(0, 0, -r, 0, 0, 0) * RPY2T(0, 0, 0, -np.pi / 2, 0, 0)
                                for t in theta_extract_list]
    T_NET_w = T_tip_w_ITPL_lst[0]  # Needle entry frame
    T_NEX_w = T_tip_w_ITPL_lst[-1]  # Needle exit frame
    print("entry dsr error 3:", T_NET_w.p - T_ENTRY.p)
    print("entry dsr error 3:", T_NEX_w.p - T_EXIT.p)
    
    # Movement and grasping logic for needles when i >= 1
    if i + 1 >= 2:
        time.sleep(15)
        start = time.time()
        print('T_nINw_reported: ', T_nINw_reported)
        print('T_nINw_real: ', engine.get_signal('scene', 'measured_needle_cp'))
        print("Hover to needle...")
        _, _, _Y = T_nINw_reported.M.GetRPY()
        _offset_theta = np.pi / 2
        if T_nINw_reported.M.UnitZ()[2] >= 0:
            print("Positive direction...")
        else:
            print("Negative direction...")
        _Y += _offset_theta
        grasp_R = Rotation.RPY(0, 0, _Y) * Rotation.RPY(np.pi, 0, 0)
        T_g_w_dsr = Frame(grasp_R, T_nINw_reported.p)

        direction = PyKDL.dot(grasp_R.UnitX(), T_nINw_reported.M.UnitY())
        print('Direction: ', direction)
        if direction > 0:
            print('Convert the grasp direction')
            _, _, cur_yaw = T_g_w_dsr.M.GetRPY()
            cur_yaw += -np.pi if cur_yaw > 0 else np.pi
            T_g_w_dsr = Frame(grasp_R, T_nINw_reported.p)

        print("Elapsed time: calculation and close redundant", time.time() - start)
        T_hover_gt_outside = RPY2T(-0.25, 0, 0.20, 0, 0, 0)
        T_g_w_dsr = T_hover_gt * T_g_w_dsr  # Desired tool pose
        T_HOVER_POSE_OUT = T_hover_gt_outside * T_g_w_dsr
        T_HOVER_POSE = T_g_w_dsr
        engine.clients['psm2'].servo_tool_cp(T_g_w_dsr, INTER_NUM * 12)
        engine.clients['psm2'].wait()
        print("Elapsed time: move to hover", time.time() - start)

        print("Approach needle...")
        T_target_w = T_nINw_reported * T_gt_n
        print("Grasp...")
        T_g_w_dsr = Frame(grasp_R, T_target_w.p)

        engine.clients['psm2'].servo_tool_cp(T_g_w_dsr, INTER_NUM * 6)
        engine.clients['psm2'].wait()
        time.sleep(2)
        T_NEEDLE_GRASP = T_g_w_dsr.Inverse() * T_nINw_reported  # Needle base pose w.r.t. gripper point
        print("Elapsed time: approaching", time.time() - start)

        engine.clients['psm2'].close_jaw()
        engine.clients['psm2'].wait()
        time.sleep(2.0)
        print("Elapsed time: grasp", time.time() - start)

        T_g_w_dsr = T_HOVER_POSE_OUT
        print("Hover...")
        engine.clients['psm2'].servo_tool_cp(T_g_w_dsr, INTER_NUM * 6)
        engine.clients['psm2'].wait()
        time.sleep(0.5)

        print("Elapsed time: lift object", time.time() - start)

    print("move to entry #{}..".format(i+1))

    T_tip_w_dsr = T_NET_w
    T_g_w_dsr_PSM2 = T_tip_w_dsr * T_tip_n.Inverse() * T_NEEDLE_GRASP_PSM2.Inverse()
    engine.clients['psm2'].servo_tool_cp(T_g_w_dsr_PSM2, INTER_NUM_SHORT * 2)
    engine.clients['psm2'].wait()
    time.sleep(0.5)

    print("insert..")
    T_tip_w_dsr_prv = T_tip_w_dsr
    for T_tip_w_dsr in T_tip_w_ITPL_lst:
        T_g_w_dsr_PSM2 = T_tip_w_dsr * T_tip_n.Inverse() * T_NEEDLE_GRASP_PSM2.Inverse()
        engine.clients['psm2'].servo_tool_cp(T_g_w_dsr_PSM2, interpolate_num=None, clear_queue=False)
    engine.clients['psm2'].wait()
    time.sleep(2)

    if i == 3:
        break

    # Left arm extract needle
    T_gt_n_PSM1 = RPY2T(0.015, 0.103, 0, 0, 0, 0) * RPY2T(0, 0, 0, -np.pi, 0, np.deg2rad(10))
    T_g_w_dsr_PSM1 = engine.clients['psm2'].T_g_w_msr * T_NEEDLE_GRASP_PSM2 * T_gt_n_PSM1
    engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1 * RPY2T(0, 0, -0.08, 0, 0, 0), INTER_NUM_SHORT)
    engine.clients['psm1'].open_jaw()
    engine.clients['psm1'].wait()
    time.sleep(2)
    engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1, INTER_NUM_SHORT)
    engine.clients['psm1'].wait()
    time.sleep(2)
    engine.clients['psm1'].close_jaw()
    engine.clients['psm1'].wait()
    time.sleep(3)
    T_NEEDLE_GRASP_PSM1 = engine.clients['psm1'].T_g_w_msr.Inverse() * engine.clients['psm2'].T_g_w_msr * T_NEEDLE_GRASP_PSM2
    engine.clients['psm2'].open_jaw()
    engine.clients['psm2'].wait()
    time.sleep(3)
    print("extract..")
    for T_tip_w_dsr in T_tip_w_ITPL_extract_lst:
        T_g_w_dsr_PSM1 = T_tip_w_dsr * T_tip_n.Inverse() * T_NEEDLE_GRASP_PSM1.Inverse()
        engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1, interpolate_num=None, clear_queue=False)

    engine.clients['psm1'].wait()
    time.sleep(1)

    # Avoid the suture
    engine.clients['psm2'].reset_pose()
    engine.clients['psm2'].wait()
    time.sleep(2.0)

    print("lift")
    T_g_w_dsr_PSM1 = RPY2T(0, 0, 0.1, 0, 0, 0) * T_g_w_dsr_PSM1
    engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1, INTER_NUM_SHORT)
    engine.clients['psm1'].wait()
    T_g_w_dsr_PSM1 = T_g_w_dsr_PSM1 * RPY2T(0, 0, 0, 0, 0, -np.pi / 4)
    engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1, INTER_NUM_SHORT)
    engine.clients['psm1'].wait()
    time.sleep(1)

    # Put the needle in ground and wait for estimation
    engine.clients['psm1'].servo_tool_cp(T_g_w_init_dsr_PSM1, INTER_NUM_SHORT)
    engine.clients['psm1'].wait()
    engine.clients['psm1'].open_jaw()
    engine.clients['psm1'].wait()
    time.sleep(1.0)
    engine.clients['psm1'].servo_tool_cp(T_g_w_dsr_PSM1, INTER_NUM_SHORT)  # back to previous position
    engine.clients['psm1'].wait()
    time.sleep(2.0)

    task3_estimation_num_pub.publish(Int32(i + 1))

time.sleep(2)
# Send finish signal
for _ in range(2):
    msg = Bool()
    msg.data = True
    task_pub.publish(msg)
    time.sleep(0.01)
time.sleep(0.5)

engine.clients['psm1'].open_jaw()
engine.clients['psm2'].open_jaw()
engine.clients['psm1'].wait()
engine.clients['psm2'].wait()
# Close engine
engine.close()