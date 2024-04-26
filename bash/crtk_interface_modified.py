from surgical_robotics_challenge.launch_crtk_interface import PSMCRTKWrapper, ECMCRTKWrapper, SceneCRTKWrapper, SceneManager, get_boolean_from_opt, Options
from ambf_client import Client
import time
import rospy
from surgical_robotics_challenge.psm_arm import PSM
from std_msgs.msg import Bool, Empty
from argparse import ArgumentParser
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rospy import Subscriber
from accel_challenge.challenge1.tool import PoseStamped2T, T2RPY

class PSMCRTKWrapperModified(PSMCRTKWrapper):
    def __init__(self, client, name, namespace, add_joint_errors=True):
        super().__init__(client, name, namespace)
        if not add_joint_errors:
            self.arm = PSM(client, name, add_joint_errors=False)
        self.measured_grasp_pub = rospy.Publisher(f'{namespace}/{name}/is_grasp', Bool, queue_size=1)

    def publish_gripper(self):
        msg = Bool(data=self.arm.grasped[0])
        self.measured_grasp_pub.publish(msg)

    def run(self):
        self.publish_js()
        self.publish_cs()
        self.publish_gripper()

class SceneCRTKWrapperModified(SceneCRTKWrapper):
    def __init__(self, client, namespace):
        super().__init__(client, namespace)
        self.sub_needle_cp = Subscriber('/CRTK/Needle/servo_cp', PoseStamped, self._sub_needle_cp_cb)
        self.sub_needle_zero_force = Subscriber('/CRTK/Needle/zero_force', Bool, self._sub_zero_force_cb)

    def _sub_needle_cp_cb(self, data):
        pos, rpy = T2RPY(PoseStamped2T(data))
        handle = self.scene.client.get_obj_handle('Needle')
        handle.set_pos(*pos)
        handle.set_rpy(*rpy)
        print("get msg needle cp")

    def _sub_zero_force_cb(self, data):
        if data.data:
            handle = self.scene.client.get_obj_handle('Needle')
            handle.set_force(0, 0, 0)
            handle.set_torque(0, 0, 0)
            print("get msg needle zero force")

class SceneManagerModified(SceneManager):
    def __init__(self, options):
        self.client = Client("ambf_surgical_sim_crtk_node")
        self.client.connect()
        time.sleep(0.2)
        self._components = []
        if options.run_psm_one:
            print("Launching CRTK-ROS Interface for PSM1 ")
            self.psm1 = PSMCRTKWrapperModified(self.client, 'psm1', options.namespace)
            self._components.append(self.psm1)
        if options.run_psm_two:
            print("Launching CRTK-ROS Interface for PSM2 ")
            self.psm2 = PSMCRTKWrapperModified(self.client, 'psm2', options.namespace)
            self._components.append(self.psm2)
        if options.run_psm_three:
            print("Launching CRTK-ROS Interface for PSM3 ")
            self.psm3 = PSMCRTKWrapperModified(self.client, 'psm3', options.namespace)
            self._components.append(self.psm3)
        if options.run_ecm:
            print("Launching CRTK-ROS Interface for ECM ")
            self.ecm = ECMCRTKWrapper(self.client, 'ecm', options.namespace)
            self._components.append(self.ecm)
        if options.run_scene:
            print("Launching CRTK-ROS Interface for Scene ")
            self.scene = SceneCRTKWrapperModified(self.client, options.namespace)
            self._components.append(self.scene)

        self._task_3_init_sub = rospy.Subscriber('/CRTK/scene/task_3_setup/init', Empty, self.task_3_setup_cb, queue_size=1)
        self._task_3_setup_ready_pub = rospy.Publisher('/CRTK/scene/task_3_setup/ready', Empty, queue_size=1)
        self._rate = rospy.Rate(options.rate)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--one', dest='run_psm_one', help='RUN PSM1', default=True)
    parser.add_argument('--two', dest='run_psm_two', help='RUN PSM2', default=True)
    parser.add_argument('--three', dest='run_psm_three', help='RUN PSM3', default=False)
    parser.add_argument('--ecm', dest='run_ecm', help='RUN ECM', default=True)
    parser.add_argument('--scene', dest='run_scene', help='RUN Scene', default=True)
    parser.add_argument('--ns', dest='namespace', help='Namespace', default='/CRTK')
    parser.add_argument('--rate', dest='rate', help='Rate of Publishing', default=120)

    parsed_args = parser.parse_args()
    options = Options()
    options.run_psm_one = get_boolean_from_opt(parsed_args.run_psm_one)
    options.run_psm_two = get_boolean_from_opt(parsed_args.run_psm_two)
    options.run_psm_three = get_boolean_from_opt(parsed_args.run_psm_three)
    options.run_ecm = get_boolean_from_opt(parsed_args.run_ecm)
    options.run_scene = get_boolean_from_opt(parsed_args.run_scene)
    options.namespace = parsed_args.namespace
    options.rate = parsed_args.rate

    sceneManager = SceneManagerModified(options)
    sceneManager.run()