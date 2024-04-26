## ros related
from sensor_msgs.msg import JointState, Image, PointCloud2
from geometry_msgs.msg import TransformStamped, PoseStamped
from rospy import Publisher, Subscriber, Rate, init_node, spin, get_published_topics, Rate, is_shutdown
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool

## common
from typing import List, Any
from time import sleep
from PyKDL import Frame, Rotation
from dataclasses import dataclass
import numpy as np
from time import sleep
from threading import Thread
import time
from queue import Queue

## ambf related
from ambf_msgs.msg import RigidBodyState
### gym_suture related
from accel_challenge.challenge1.tool import Quaternion2T, RPY2T, gen_interpolate_frames, SE3_2_T, T_2_SE3, PoseStamped2T
from accel_challenge.challenge1.param import grasp_point_offset
from accel_challenge.challenge1.kinematics import PSM_KIN
from surgical_robotics_challenge.kinematics.psmIK import compute_IK

from simple_pid import PID
import logging
logging.basicConfig(level=logging.INFO)

class ClientEngine:
    """
    An engine to manage clients that subscribe and publish CRTK related topics.
    """
    ROS_RATE = 100

    def __init__(self):
        self.clients = {}
        self.ros_node = init_node('ros_client_engine', anonymous=True)
        self.ros_rate = Rate(self.ROS_RATE)

    def add_clients(self, client_names: List[str]):
        for client_name in client_names:
            self._add_client(client_name)
        time.sleep(1)  # Wait for ROS subscribed topics to be ok

    def start(self):
        for client in self.clients.values():
            client.start()

    def close(self):
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                print(str(e))

    def close_client(self, name):
        self.clients[name].close()
        del self.clients[name]

    def get_signal(self, client_name, signal_name):
        self._is_has_client(client_name, raise_error=True)
        return self.clients[client_name].get_signal(signal_name)

    def _add_client(self, client_name):
        if client_name in ['psm1', 'psm2']:
            self.clients[client_name] = PSMClient(self.ros_node, client_name)
        elif client_name == 'ecm':
            self.clients[client_name] = ECMClient(self.ros_node)
        elif client_name == 'scene':
            self.clients[client_name] = SceneClient(self.ros_node)
        elif client_name == 'ambf':
            self.clients[client_name] = AmbfClient()
        else:
            raise NotImplementedError

    def _is_has_client(self, client_name, raise_error=False):
        result = client_name in self.clients
        if raise_error and not result:
            raise Exception(f"Client {client_name} has not been added")

    @property
    def client_names(self):
        return list(self.clients.keys())

@dataclass
class SignalData:
    """
    Data class for testing if signal is dead
    """
    data: Any
    counter: int


class BaseClient:
    """
    Define some common properties and methods for clients.
    """
    COUNTER_BUFFER = 500

    def __init__(self, ros_node, is_run_thread=False):
        self.ros_node = ros_node
        self.sub_signals = {}
        self.subs = {}
        self.pubs = {}
        self.is_alive = False
        self.is_run_thread = is_run_thread

        if self.is_run_thread:
            self.thread = Thread(target=self.run)
        else:
            self.thread = None

    def start(self):
        self.is_alive = True
        if self.is_run_thread:
            self.thread.start()

    def close(self):
        self.is_alive = False

    def _T2TransformStamped(self, T):
        """
        Convert Frame to ROS Message: TransformStamped.
        """
        rx, ry, rz, rw = T.M.GetQuaternion()
        msg = TransformStamped()
        msg.transform.rotation.x = rx
        msg.transform.rotation.y = ry
        msg.transform.rotation.z = rz
        msg.transform.rotation.w = rw
        msg.transform.translation.x = T.p.x()
        msg.transform.translation.y = T.p.y()
        msg.transform.translation.z = T.p.z()
        return msg

    def _TransformStamped2T(self, msg):
        """
        Convert ROS Message: TransformStamped to Frame.
        """
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        z = msg.transform.translation.z
        qx = msg.transform.rotation.x
        qy = msg.transform.rotation.y
        qz = msg.transform.rotation.z
        qw = msg.transform.rotation.w
        return Quaternion2T(x, y, z, qx, qy, qz, qw)

    def _RigidBodyState2T(self, msg):
        """
        Convert RigidBodyState message to Frame.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        return Quaternion2T(x, y, z, qx, qy, qz, qw)

    def topic_wrap(self, topic_name, raise_exception=True):
        """
        Wrapper for ROS topic name; check if the topic name exists and raise exception if it does not.
        """
        topics = [topic[0] for topic in get_published_topics()]
        if topic_name not in topics and raise_exception:
            raise Exception(f"Topic {topic_name} does not exist, please check if CRTK interface is running.")
        return topic_name

    def is_has_signal(self, signal_name: str):
        """
        Check if a signal name exists.
        """
        return signal_name in self.sub_signals

    def set_signal(self, signal_name: str, data):
        """
        Update signal data.
        """
        if not self.is_has_signal(signal_name):
            self.sub_signals[signal_name] = SignalData(data=data, counter=1)
        else:
            cnt = self.sub_signals[signal_name].counter
            self.sub_signals[signal_name].counter = cnt + 1 if cnt < self.COUNTER_BUFFER else self.COUNTER_BUFFER

    def get_signal(self, signal_name: str):
        """
        Get signal data.
        """
        if not self.is_has_signal(signal_name):
            print(f"Signal {signal_name} does not exist.")
            return None
        return self.sub_signals[signal_name].data

    def reset_signal_counter(self, signal_name: str):
        """
        Reset signal counter to 0.
        """
        if self.is_has_signal(signal_name):
            self.sub_signals[signal_name].counter = 0

    def run(self):
        raise NotImplementedError

    @property
    def sub_signals_names(self):
        return list(self.sub_signals.keys())

    @property
    def sub_topics_names(self):
        return list(self.subs.keys())

    @property
    def pub_topics_names(self):
        return list(self.pubs.keys())


class AmbfClient(BaseClient):
    def __init__(self, ros_node=None):
        super().__init__(ros_node, is_run_thread=False)
        from ambf_client import Client
        self.client = Client('gym_suture')
        self.client.connect()

    def close(self):
        self.client.clean_up()
        self.is_alive = False

    def reset_all(self):
        """Reset all, similar to ctrl+R."""
        w = self.client.get_world_handle()
        time.sleep(0.1)
        w.reset()  # Resets the whole world (Lights, Cams, Volumes, Rigid Bodies, Plugins etc)
        time.sleep(0.5)
        w.reset_bodies()  # Reset Static / Dynamic Rigid Bodies

    def reset_needle(self, repeat=3, accuracy=0.09):
        """Reset needle to starting position."""
        pos = [-0.20786757338201337, 0.5619611862776279, 0.7317253877244148]  # Hardcoded position
        rpy = [0.03031654271074325, 0.029994510295635185, -0.00018838556827461113]
        for i in range(repeat):
            self.reset_all() if i + 1 == repeat else self.set_needle_pose(pos=pos, rpy=rpy)
            _pos, _rpy = self.get_needle_pose()
            err = np.linalg.norm(np.array(_pos) - np.array(pos))
            if err < accuracy:  # Ensure needle stays within target
                break

    def set_needle_pose(self, pos: List[float] = None, rpy: List[float] = None):
        """Set needle position."""
        assert pos is not None or rpy is not None, 'Pos and rpy cannot be None at the same time'
        needle = self.client.get_obj_handle('Needle')
        if pos is not None:
            needle.set_pos(*pos)
        if rpy is not None:
            needle.set_rpy(*rpy)
        time.sleep(1)  # Wait for needle to move to position
        for _ in range(3):
            needle.set_force(0, 0, 0)
            needle.set_torque(0, 0, 0)
        time.sleep(0.3)

    def servo_cam_local(self, pos, quat, cam='cameraL'):
        """Servo camera w.r.t. ECM base frame."""
        assert cam == 'cameraL', 'Camera not implemented'
        cam = self.client.get_obj_handle('/ambf/env/cameras/cameraL')
        rpy = Rotation.Quaternion(*quat).GetRPY()
        cam.set_pos(*pos)
        cam.set_rpy(*rpy)

    def get_needle_pose(self):
        """Get the current position and orientation of the needle."""
        needle = self.client.get_obj_handle('Needle')
        pos = needle.get_pos()
        rpy = needle.get_rpy()
        return [pos.x, pos.y, pos.z], list(rpy)

 

class PSMClient(BaseClient):
    """
    PSM CRTK topics client for handling specific robotic arm functionalities.
    """
    arm_names = ['psm1', 'psm2']
    _q_dsr = None
    _jaw_dsr = None

    is_update_marker = False
    IK_MAX_NUM = 2

    reset_jnt_psm2 = [-0.5656515955924988, -0.15630173683166504, 1.3160043954849243, -2.2147457599639893, 0.8174221515655518, -1]
    reset_jnt_psm1 = [0.2574930501388846, -0.2637599054454396, 1.490778072887017, -2.3705447576048746, 0.3589815573742414, -1.0241148122485695]

    Kp_servo_jp_list = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    Kv_servo_jp_list = [0, 0, 0, 0, 0, 0]
    Ki_servo_jp_list = [10, 10, 10, 10, 10, 10]

    Mag_servo_jp = 2

    def __init__(self, ros_node, arm_name: str):
        super().__init__(ros_node, is_run_thread=True)
        if arm_name not in self.arm_names:
            raise Exception(f"Arm name should be in {self.arm_names}")

        self.joint_calibrate_offset = np.array([0,0,0, 0,0,0])
        self.arm_name = arm_name
        self.grasp_point_offset = grasp_point_offset
        self.pubs["servo_jp"] = Publisher(f'/CRTK/{arm_name}/servo_jp', JointState, queue_size=1)
        self.pubs["servo_jaw_jp"] = Publisher(f'/CRTK/{arm_name}/jaw/servo_jp', JointState, queue_size=1)
        self.subs["measured_base_cp"] = Subscriber(self.topic_wrap(f'/ambf/env/{arm_name}/baselink/State'), RigidBodyState, self._measured_base_cp_cb)
        self.subs["measured_js"] = Subscriber(self.topic_wrap(f'/CRTK/{arm_name}/measured_js'), JointState, self._measured_js_cb)
        self.ros_rate = Rate(120)  # 100hz

        self.kin = PSM_KIN()  # Kinematics model
        self._jaw_pub_queue = Queue()
        self._arm_pub_queue = Queue()

        self.pids = [PID(self.Kp_servo_jp_list[i], self.Ki_servo_jp_list[i], self.Kv_servo_jp_list[i], setpoint=0, output_limits=(-self.Mag_servo_jp, self.Mag_servo_jp), sample_time=0.01) for i in range(6)]

    def reset_pose(self, q_dsr=None, walltime=None):
        if self.arm_name == 'psm2':
            _q_dsr = q_dsr or self.reset_jnt_psm2
        elif self.arm_name == 'psm1':
            _q_dsr = q_dsr or self.reset_jnt_psm1
        self.servo_jp(_q_dsr, interpolate_num=100)
        self.wait(walltime=walltime)

    def wait(self, walltime=None, force_walltime=False):
        """ Wait until the queues for joint angles and jaw are empty or until the walltime is exceeded. """
        assert not (walltime is None and force_walltime), 'Walltime must be specified when force_walltime is True'
        start = time.time()
        while True:
            if walltime is not None and (time.time() - start) > walltime:
                self._arm_pub_queue.queue.clear()
                self._jaw_pub_queue.queue.clear()
                break
            if self._arm_pub_queue.empty() and self._jaw_pub_queue.empty() and not force_walltime:
                break
            self.ros_rate.sleep()

    def servo_tool_cp_local(self, T_g_b_dsr: Frame, interpolate_num=None, clear_queue=True):
        """ Move tool to desired pose, in Roll-Pitch-Yaw convention, w.r.t. base frame. """
        T0 = self.T_g_b_dsr
        if clear_queue:
            self._arm_pub_queue.queue.clear()
        if interpolate_num is not None:
            frames = gen_interpolate_frames(T0, T_g_b_dsr, interpolate_num)
            for frame in frames:
                _T_t_b_dsr = frame * self.grasp_point_offset.Inverse()
                self._arm_pub_queue.put(_T_t_b_dsr)
        else:
            _T_t_b_dsr = T_g_b_dsr * self.grasp_point_offset.Inverse()
            self._arm_pub_queue.put(_T_t_b_dsr)


    def servo_tool_cp(self, T_g_w_dsr: Frame, interpolate_num=None, clear_queue=True):
        """
        Move tool to desired pose, in Roll-Pitch-Yaw convention, w.r.t. world frame.
        """
        T_b_w = self.get_signal('measured_base_cp')
        T_g_b_dsr = T_b_w.Inverse() * T_g_w_dsr
        self.servo_tool_cp_local(T_g_b_dsr, interpolate_num, clear_queue)

    def servo_jaw_jp(self, jaw_jp_dsr: float, interpolate_num=None, clear_queue=True):
        """
        Move jaw joint position.
        """
        if clear_queue:
            self._jaw_pub_queue.queue.clear()
        if interpolate_num is None:
            self._jaw_pub_queue.put(jaw_jp_dsr)
        else:
            qs = np.linspace(self.jaw_dsr, jaw_jp_dsr, interpolate_num).tolist()
            for q in qs:
                self._jaw_pub_queue.put(q)

    def servo_jp(self, q_dsr, interpolate_num=None, clear_queue=True):
        """
        Perform joint positioning for the robotic arm.
        """
        q0 = self.q_dsr
        if clear_queue:
            self._arm_pub_queue.queue.clear()
        if interpolate_num is None:
            self._arm_pub_queue.put(q_dsr)
        else:
            qs = np.linspace(q0, q_dsr, interpolate_num).tolist()
            for q in qs:
                self._arm_pub_queue.put(q)

    def close_jaw(self):
        """
        Fully close the jaw of the robotic arm.
        """
        self.servo_jaw_jp(0, 200)

    def open_jaw(self, angle=None):
        """
        Open the jaw of the robotic arm either fully or to a specified angle.
        """
        self.servo_jaw_jp(angle if angle is not None else 0.4, 100)

    def get_T_g_w_from_js(self, qs):
        """
        Calculate global transform from joint states.
        """
        return self.get_signal('measured_base_cp') * SE3_2_T(self.kin.fk(qs)) * self.grasp_point_offset

    def ik_local(self, T_g_b, q0=None, ik_engine='surgical_challenge'):
        """
        Compute the inverse kinematics locally for the given transform and initial conditions.
        """
        if ik_engine == 'peter_corke':
            q0 = q0 or self.get_signal('measured_js')
            q_dsr, is_success = self.kin.ik(T_2_SE3(T_g_b), q0)
        elif ik_engine == 'surgical_challenge':
            q_dsr = compute_IK(T_g_b)
            is_success = True
        return q_dsr, is_success

    def ik(self, T_g_w, q0=None):
        """
        Compute the inverse kinematics for the given transform in the global frame.
        """
        T_b_w = self.get_signal('measured_base_cp')
        T_g_b = T_b_w.Inverse() * T_g_w
        q_dsr, is_success = self.ik_local(T_g_b, q0)
        return q_dsr, is_success

    def run(self):
        """
        Publish ROS topics at a fixed rate, like a controller.
        """
        while not is_shutdown() and self.is_alive:
            if not self._arm_pub_queue.empty():
                data = self._arm_pub_queue.get()
                if isinstance(data, list):
                    q_dsr = data
                else:
                    q0 = self.get_signal('measured_js')
                    q_dsr = None
                    for _ in range(self.IK_MAX_NUM):
                        q_dsr, is_success = self.ik_local(data, q0)
                        if is_success:
                            break

                if q_dsr is not None:
                    self._q_dsr = q_dsr
                else:
                    self._q_dsr = self.get_signal('measured_js')

                q_msr = self.get_signal('measured_js')
                e = np.array(self._q_dsr) - np.array(q_msr)
                q_delta = [-self.pids[i](e[i]) for i in range(6)]
                q_delta = np.array(q_delta)
                q_dsr_servo = self._q_dsr + q_delta - self.joint_calibrate_offset
                msg = JointState()
                msg.position = q_dsr_servo.tolist()
                self.pubs['servo_jp'].publish(msg)

            if not self._jaw_pub_queue.empty():
                data = self._jaw_pub_queue.get()
                self._jaw_dsr = data
                msg = JointState()
                msg.position = [data]
                self.pubs["servo_jaw_jp"].publish(msg)

            self.ros_rate.sleep()


    @property
    def jaw_dsr(self):
        """
        Desired position of the jaw, defaulting to 0.0 if not set.
        """
        return self._jaw_dsr or 0.0

    @property
    def q_dsr(self):
        """
        Desired joint states, defaulting to the measured joint states if not set.
        """
        return self._q_dsr or self.get_signal('measured_js')

    @property
    def T_g_b_dsr(self):
        """
        Compute the desired grasp point frame with respect to the base frame.
        """
        return SE3_2_T(self.kin.fk(self.q_dsr)) * self.grasp_point_offset

    @property
    def T_g_w_dsr(self):
        """
        Compute the desired grasp point frame with respect to the world frame.
        """
        return self.get_signal('measured_base_cp') * self.T_g_b_dsr

    @property
    def T_g_b_msr(self):
        """
        Measure the grasp point frame with respect to the base frame using the current joint states.
        """
        _q = self.get_signal('measured_js')
        return SE3_2_T(self.kin.fk(_q)) * self.grasp_point_offset

    @property
    def T_g_w_msr(self):
        """
        Measure the grasp point frame with respect to the world frame.
        """
        return self.get_signal('measured_base_cp') * self.T_g_b_msr

    def _measured_base_cp_cb(self, data):
        """
        Callback to handle updates of the base coordinate pose from RigidBodyState messages.
        """
        self.set_signal('measured_base_cp', self._RigidBodyState2T(data))

    def _measured_js_cb(self, data):
        """
        Callback to handle updates of joint states, adjusting for calibration offsets.
        """
        pos = np.array(data.position) + self.joint_calibrate_offset
        self.set_signal('measured_js', pos.tolist())

    def _is_grasp_cb(self, data):
        """
        Callback to update the grasp state based on Bool data.
        """
        self.set_signal('is_grasp', data.data)




class ECMClient(BaseClient):
    """
    ECM CRTK topics client for managing endoscopic camera manipulations.
    """
    def __init__(self, ros_node, is_left_cam=True, is_right_cam=False, 
                 is_left_point_cloud=False, is_right_point_cloud=False):
        super(ECMClient, self).__init__(ros_node)
        
        # ROS topics
        self.pubs['ecm_servo_jp'] = Publisher('/CRTK/ecm/servo_jp', JointState, queue_size=1)
        self.subs['camera_frame_state'] = Subscriber(self.topic_wrap('/CRTK/ecm/measured_cp'), PoseStamped, self._camera_frame_state_cb)

        if is_left_cam:
            self.subs['cameraL_image'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraL/ImageData'), numpy_msg(Image), self._cameraL_image_cb)
        if is_left_point_cloud:
            self.subs['cameraL_point_cloud'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraL/DepthData'), numpy_msg(PointCloud2), self._cameraL_image_depth_cb)
        if is_right_cam:
            self.subs['cameraR_image'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraR/ImageData'), numpy_msg(Image), self._cameraR_image_cb)
        if is_right_point_cloud:
            self.subs['cameraR_point_cloud'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraR/DepthData'), numpy_msg(PointCloud2), self._cameraR_image_depth_cb)
        
        self.ros_rate = Rate(5)

    def move_ecm_jp(self, pos: List[float], time_out=30, threshold=0.001, ratio=1, count_break=3):
        """
        Move camera joint position with blocking until motion completes or timeout occurs.
        """
        msg = JointState()
        msg.position = pos
        self.pubs["ecm_servo_jp"].publish(msg)

        start = time.time()
        T_prv = self.get_signal('camera_frame_state')
        cnt = 0

        while True:
            self.ros_rate.sleep()
            T = self.get_signal('camera_frame_state')
            if T is None or T_prv is None:
                continue

            deltaT = T_prv.Inverse() * T
            theta, _ = deltaT.M.GetRotAngle()
            dp = deltaT.p.Norm()

            if time.time() - start > time_out:
                break

            if dp + theta * ratio < threshold:
                cnt += 1
            else:
                cnt = 0

            if cnt >= count_break:
                break

            T_prv = T

    def _cameraL_image_cb(self, data):
        """
        Callback for left camera image data.
        """
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.set_signal('cameraL_image', img)

    def _cameraL_image_depth_cb(self, data):
        """
        Callback for left camera depth data.
        """
        self.set_signal('cameraL_point_cloud', data)

    def _cameraR_image_cb(self, data):
        """
        Callback for right camera image data.
        """
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.set_signal('cameraR_image', img)

    def _cameraR_image_depth_cb(self, data):
        """
        Callback for right camera depth data.
        """
        self.set_signal('cameraR_point_cloud', data)

    def _camera_frame_state_cb(self, data):
        """
        Callback for updating camera frame state based on PoseStamped data.
        """
        self.set_signal('camera_frame_state', PoseStamped2T(data))



class SceneClient(BaseClient):
    """
    Scene crtk topics client
    """
    def __init__(self, ros_node):
        super(SceneClient, self).__init__(ros_node)
        # ros topics
        self.subs['measured_needle_cp'] = Subscriber(self.topic_wrap('/CRTK/Needle/measured_cp'), PoseStamped, self._needle_cp_cb)
        self.subs['measured_entry1_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry1/measured_cp'), PoseStamped, self._measured_entry1_cp_cb)
        self.subs['measured_entry2_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry2/measured_cp'), PoseStamped, self._measured_entry2_cp_cb)
        self.subs['measured_entry3_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry3/measured_cp'), PoseStamped, self._measured_entry3_cp_cb)
        self.subs['measured_entry4_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry4/measured_cp'), PoseStamped, self._measured_entry4_cp_cb)

        self.subs['measured_exit1_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit1/measured_cp'), PoseStamped, self._measured_exit1_cp_cb)
        self.subs['measured_exit2_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit2/measured_cp'), PoseStamped, self._measured_exit2_cp_cb)
        self.subs['measured_exit3_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit3/measured_cp'), PoseStamped, self._measured_exit3_cp_cb)
        self.subs['measured_exit4_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit4/measured_cp'), PoseStamped, self._measured_exit4_cp_cb) 

    def _needle_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_needle_cp', PoseStamped2T(data))

    def _measured_entry1_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry1_cp', PoseStamped2T(data))

    def _measured_entry2_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry2_cp', PoseStamped2T(data))
        
    def _measured_entry3_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry3_cp', PoseStamped2T(data))

    def _measured_entry4_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry4_cp', PoseStamped2T(data))

    def _measured_exit1_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit1_cp', PoseStamped2T(data))

    def _measured_exit2_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit2_cp', PoseStamped2T(data))
        
    def _measured_exit3_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit3_cp', PoseStamped2T(data))

    def _measured_exit4_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit4_cp', PoseStamped2T(data))



if __name__ == "__main__":
    try:
        engine = ClientEngine()
        engine.add_clients(['psm2'])
        engine.start()

        # Initial joint positions for psm2
        q0 = [-0.5656515955924988, -0.15630173683166504, 1.3160043954849243, -2.2147457599639893, 0.8174221515655518, -1]
        sleep_time = 0.3

        # Operations on psm2
        client = engine.clients['psm2']
        client.servo_jp(q0, interpolate_num=100)
        client.close_jaw()
        client.sleep(sleep_time)

        for angles, action in [([0.2, 0, 0], 'open'), ([0, 0.2, 0], 'close'), ([0, 0, 0.1], 'open')]:
            T_g_w_msr = client.T_g_w_msr
            deltaT = RPY2T(*angles + [0, 0, 0])
            client.servo_tool_cp(deltaT * T_g_w_msr, interpolate_num=100)
            getattr(client, f'{action}_jaw')()
            client.sleep(sleep_time)

        logging.info("CTRL+C to stop.")
        spin()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        engine.close()