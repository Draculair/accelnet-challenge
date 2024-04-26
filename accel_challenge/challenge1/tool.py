import cv2
import numpy as np
from PyKDL import Frame, Rotation, Vector
from geometry_msgs.msg import TransformStamped, PoseStamped
from spatialmath.pose3d import SE3
from pathlib import Path
from os.path import dirname, normpath
import ros_numpy
from typing import Tuple

def get_package_root_abs_path():
    return normpath(dirname(Path(__file__).parent.absolute()))

PACKAGE_ROOT_PATH = get_package_root_abs_path()
cam_width, cam_height = 1920, 1080

def resize_img(img, height):
    if isinstance(img, np.ndarray):
        method = cv2.INTER_AREA
        img = cv2.resize(img, dsize=(int(height / 0.5625), height), interpolation=method)
        return img
    else:
        raise NotImplementedError

def crop2square_img(img):
    size = img.shape
    margin = int(abs(size[0] - size[1]) / 2)
    if size[0] > size[1]:
        return img[margin:margin + size[1], :, :]
    else:
        return img[:, margin:margin + size[0], :]

def gen_interpolate_frames(T_origin, T_dest, num):
    T_delta = T_origin.Inverse() * T_dest
    angle, axis = T_delta.M.GetRotAngle()
    return [T_origin * Frame(Rotation.Rot(axis, angle * alpha), alpha * T_delta.p) for alpha in np.linspace(0, 1, num=num)]

def RPY2T(x, y, z, R, P, Y):
    return Frame(Rotation.RPY(R, P, Y), Vector(x, y, z))

def T2RPY(T: Frame) -> Tuple[np.array]:
    pos = [T.p.x(), T.p.y(), T.p.z()]
    rpy = list(T.M.GetRPY())
    return np.array(pos), np.array(rpy)

def Quaternion2T(x, y, z, rx, ry, rz, rw):
    return Frame(Rotation.Quaternion(rx, ry, rz, rw), Vector(x, y, z))

def PoseStamped2T(msg):
    x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    Qx, Qy, Qz, Qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
    return Quaternion2T(x, y, z, Qx, Qy, Qz, Qw)

def T2PoseStamped(T):
    msg = PoseStamped()
    rx, ry, rz, rw = T.M.GetQuaternion()
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = rx, ry, rz, rw
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = T.p.x(), T.p.y(), T.p.z()
    return msg

def TransformStamped2T(msg):
    x, y, z = msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z
    Qx, Qy, Qz, Qw = msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w
    return Quaternion2T(x, y, z, Qx, Qy, Qz, Qw)

def T2TransformStamped(T):
    msg = TransformStamped()
    rx, ry, rz, rw = T.M.GetQuaternion()
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = rx, ry, rz, rw
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = T.p.x(), T.p.y(), T.p.z()
    return msg

def SE3_2_T(SE3_obj):
    R = SE3_obj.R
    t = SE3_obj.t
    return Frame(Rotation(Vector(*R[:, 0]), Vector(*R[:, 1]), Vector(*R[:, 2])), Vector(*t))

def T_2_SE3(T):
    R = T.M
    Rx, Ry, Rz = R.UnitX(), R.UnitY(), R.UnitZ()
    t = T.p
    _T = np.array([[Rx[0], Ry[0], Rz[0], t[0]],
                   [Rx[1], Ry[1], Rz[1], t[1]],
                   [Rx[2], Ry[2], Rz[2], t[2]],
                   [0, 0, 0, 1]])
    return SE3(_T, check=False)

def T_2_arr(T):
    R = T.M
    Rx, Ry, Rz = R.UnitX(), R.UnitY(), R.UnitZ()
    t = T.p
    _T = np.array([[Rx[0], Ry[0], Rz[0], t[0]],
                   [Rx[1], Ry[1], Rz[1], t[1]],
                   [Rx[2], Ry[2], Rz[2], t[2]],
                   [0, 0, 0, 1]])
    return _T

def PointCloud2_2_xyzNimage(msg):
    _data = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    _data_rgb = ros_numpy.point_cloud2.split_rgb_field(_data)
    rgb = np.array([_data_rgb['r'], _data_rgb['g'], _data_rgb['b']]).T.reshape(cam_height, cam_width, 3)
    rgb = np.flip(rgb, axis=0)
    xyz = np.array([_data_rgb['x'], _data_rgb['y'], _data_rgb['z']]).T.reshape(cam_height, cam_width, 3)
    xyz = np.flip(xyz, axis=0)
    return (xyz, rgb)

def render_rgb(rgb_arr):
    print("Press 'q' to exit...")
    while True:
        cv2.imshow('Preview', rgb_arr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def render_rgb_xyz(rgb_arr, xyz_arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_arr.reshape(-1, 3))
    o3d.visualization.draw_geometries([pcd])
