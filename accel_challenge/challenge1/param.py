from PyKDL import Frame, Rotation, Vector
from accel_challenge.challenge1.tool import RPY2T
import numpy as np

# Grasping point offset w.r.t. tool frame
grasp_point_offset = Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, -0.008))

# Grasp target point offset w.r.t. needle frame
T_gt_n = Frame(Rotation.RPY(0, 0, 0), Vector(-0.1, 0.03, 0))
T_hover_gt = RPY2T(0, 0, 0.15, 0, 0, 0)

# Needle radius
NEEDLE_R = 0.103

# Tip frame w.r.t. needle base frame
T_tip_n = Frame(Rotation.RPY(0, 0, np.deg2rad(-30)), Vector(0.055, 0.083, 0))

# Camera parameters for tracking
cam_width, cam_height = 1920, 1080
fov_angle = np.deg2rad(1.2)
u0, v0 = cam_width / 2, cam_height / 2
kuv = cam_height / 2 / np.tan(fov_angle / 2)
f = 0.01
fuv = f * kuv
T_cam_project = np.array([
    [fuv, 0, u0, 0],
    [0, fuv, v0, 0],
    [0, 0, fuv, 0]
])