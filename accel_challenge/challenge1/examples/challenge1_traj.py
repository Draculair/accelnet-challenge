import time
import numpy as np
import rospy
import PyKDL
from pathlib import Path
from std_msgs.msg import Bool
from PyKDL import Frame, Rotation, Vector
from accel_challenge.challenge1.ros_client import ClientEngine
from accel_challenge.challenge1.tool import RPY2T
from accel_challenge.challenge1.param import T_gt_n, T_hover_gt, NEEDLE_R, T_tip_n
from accel_challenge.challenge1.examples.calibrate import calibrate_joint_error, DLC_CONFIG_PATH_dict, ERROR_DATA_DIR

start = time.time()

# Parameters and configurations
team_name = 'Amagi'
move_arm = 'psm2'
INTER_NUM = 100
INSERTION_ITPL_NUM = 150
thres_err = 0.1
pi = np.pi

# Initialize client engine
engine = ClientEngine()
engine.add_clients(['psm2', 'ecm', 'scene'])
engine.start()

print("Resetting pose...")
engine.clients[move_arm].reset_pose(walltime=None)

load_dict = {
    'dlc_config_path': DLC_CONFIG_PATH_dict[move_arm],
    'keras_model_path': str(Path(ERROR_DATA_DIR) / move_arm / 'model.hdf5'),
    'scalers_path': str(Path(ERROR_DATA_DIR) / move_arm / 'scalers.pkl')
}

engine.clients['ecm'].move_ecm_jp([0, 0, 0, 0])

task_pub = rospy.Publisher(f'/surgical_robotics_challenge/completion_report/{team_name}/task2', Bool, queue_size=1)
print("Elapsed time: init object", time.time() - start)

# Calibration process
engine.clients[move_arm].joint_calibrate_offset = np.zeros(6)
error = calibrate_joint_error(_engine=engine, load_dict=load_dict, arm_name=move_arm)
joint_calibrate_offset = np.concatenate((error, np.zeros(3)))
engine.clients[move_arm].joint_calibrate_offset = joint_calibrate_offset

print("Predicted joint offset:", joint_calibrate_offset)
print("Elapsed time: calibration", time.time() - start)

# Calculating grasp pose
T_n_w0 = engine.get_signal('scene', 'measured_needle_cp')
T_ENTRY = engine.get_signal('scene', 'measured_entry1_cp') 
T_EXIT = engine.get_signal('scene', 'measured_exit1_cp')
print("Hovering to needle...")
_, _, _Y = T_n_w0.M.GetRPY()
_offset_theta = pi / 2
_Y += _offset_theta
grasp_R = Rotation.RPY(0, 0, _Y) * Rotation.RPY(pi, 0, 0)
T_g_w_dsr = Frame(grasp_R, T_n_w0.p)

direction = PyKDL.dot(grasp_R.UnitX(), T_n_w0.M.UnitY())
print('Direction: ', direction)
if direction > 0:
    print('Converting the grasp direction')
    _, _, cur_yaw = T_g_w_dsr.M.GetRPY()
    angle_adjustment = -pi if cur_yaw > 0 else pi
    T_g_w_dsr = T_g_w_dsr * RPY2T(0, 0, 0, 0, 0, angle_adjustment)

engine.close_client('ecm')
engine.close_client('scene')
print("Elapsed time: calculation and close redundant", time.time() - start)



# Move to hover pose
T_g_w_dsr = T_hover_gt * T_g_w_dsr  # Desired tool pose
T_HOVER_POSE = T_g_w_dsr
engine.clients[move_arm].servo_tool_cp(T_g_w_dsr, INTER_NUM)
engine.clients[move_arm].wait()
T_g_w_dsr_prv = T_g_w_dsr
T_g_w_dsr = None
print("Elapsed time: move to hover", time.time() - start)

# Approach needle
print("Approach needle..")
T_target_w = T_n_w0 * T_gt_n
T_TARGET_POSE = T_target_w
print("Grasp..")
T_g_w_dsr = Frame(grasp_R, T_TARGET_POSE.p)
direction = PyKDL.dot(grasp_R.UnitX(), T_n_w0.M.UnitY())
print('Direction: ', direction)
if direction > 0:
    print('Convert the grasp direction')
    _, _, cur_yaw = T_g_w_dsr.M.GetRPY()
    angle_adjustment = -pi if cur_yaw > 0 else pi
    T_g_w_dsr = T_g_w_dsr * RPY2T(0, 0, 0, 0, 0, angle_adjustment)
engine.clients[move_arm].servo_tool_cp(T_g_w_dsr, INTER_NUM)
engine.clients[move_arm].wait()
T_g_w_dsr_prv = T_g_w_dsr
T_g_w_dsr = None
time.sleep(0.2)
_T_dsr = T_g_w_dsr_prv
_T_dsr2 = engine.clients['psm2'].T_g_w_dsr
T_NEEDLE_GRASP = T_g_w_dsr_prv.Inverse() * T_n_w0  # Needle base pose w.r.t. gripper point
print("Elapsed time: approaching", time.time() - start)

# Grasp
engine.clients[move_arm].close_jaw()
engine.clients[move_arm].wait()
time.sleep(0.2)
print("Elapsed time: grasp", time.time() - start)

# Lift needle
T_g_w_dsr = T_HOVER_POSE
print("Hover..")
engine.clients[move_arm].servo_tool_cp(T_g_w_dsr, INTER_NUM)
engine.clients[move_arm].wait()
T_g_w_dsr_prv = T_g_w_dsr
T_g_w_dsr = None
print("Elapsed time: lift object", time.time() - start)

# Some calculations for suture
alpha = np.deg2rad(35)  # 5mm tip must be seen after penetrating exit hole
d = (T_ENTRY.p - T_EXIT.p).Norm()
r = NEEDLE_R  # Needle radius
theta = np.arcsin(d / 2 / r)
# Pivot frame
Rx_pivot_w = T_EXIT.p - T_ENTRY.p
Rx_pivot_w /= Rx_pivot_w.Norm()
Rz_pivot_w = Vector(0, 0, 1)
Ry_pivot_w = Rz_pivot_w * Rx_pivot_w  # Y-axis = Z-axis cross product X-axis
p_pivot_w = (T_ENTRY.p + T_EXIT.p) / 2 + Vector(0, 0, r * np.cos(theta))
T_pivot_w = Frame(Rotation(Rx_pivot_w, Ry_pivot_w, Rz_pivot_w), p_pivot_w)
# Needle entry and exit frame
TR_n_pivot = RPY2T(0, 0, 0, -pi / 2, 0, 0)  # Rotation Frame from pivot to needle base
# Needle insertion interpolate trajectory frames
theta_list = np.linspace(theta, -theta - alpha, INSERTION_ITPL_NUM).tolist()
T_tip_w_ITPL_lst = [T_pivot_w * RPY2T(0, 0, 0, 0, theta, 0) * RPY2T(0, 0, -r, 0, 0, 0) * TR_n_pivot
                    for theta in theta_list]
T_NET_w = T_tip_w_ITPL_lst[0]  # Needle entry frame
T_NEX_w = T_tip_w_ITPL_lst[-1]  # Needle exit frame
print("Elapsed time: calculation", time.time() - start)

# Move needle to entry point #1
print("Move to entry #1..")
T_tip_w_dsr = T_NET_w
T_g_w_dsr = T_tip_w_dsr * T_tip_n.Inverse() * T_NEEDLE_GRASP.Inverse()
engine.clients[move_arm].servo_tool_cp(T_g_w_dsr, INTER_NUM)
engine.clients[move_arm].wait()
time.sleep(0.6)
print("Elapsed time: move to entry", time.time() - start)
time.sleep(0.4)

# Insert needle
print("Insert..")
T_tip_w_dsr_prv = T_tip_w_dsr
for T_tip_w_dsr in T_tip_w_ITPL_lst:
    T_g_w_dsr = T_tip_w_dsr * T_tip_n.Inverse() * T_NEEDLE_GRASP.Inverse()
    engine.clients[move_arm].servo_tool_cp(T_g_w_dsr, interpolate_num=None, clear_queue=False)
engine.clients[move_arm].wait()
time.sleep(0.1)
_T_dsr = T_g_w_dsr
_T_dsr2 = engine.clients['psm2'].T_g_w_dsr
_T_msr = engine.clients['psm2'].T_g_w_msr
print("Elapsed time: insert", time.time() - start)

# Send finish signal
for _ in range(2):
    msg = Bool()
    msg.data = True
    task_pub.publish(msg)
    time.sleep(0.1)
time.sleep(1)

# Self testing (comment out during evaluation)
engine.clients[move_arm].open_jaw()
engine.clients[move_arm].wait()
engine.clients[move_arm].reset_pose(walltime=None)
engine.clients[move_arm].wait()

# Close engine
engine.close()