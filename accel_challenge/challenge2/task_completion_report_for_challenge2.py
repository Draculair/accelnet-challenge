from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32
import rospy
from surgical_robotics_challenge.camera import Camera
import tf_conversions.posemath as pm
from PyKDL import Frame, Rotation, Vector
from ambf_client import Client
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from argparse import ArgumentParser
import numpy as np
from skimage import morphology
import skimage
from detectron2.utils.logger import setup_logger
setup_logger()
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import time
import math
torch.backends.cudnn.benchmark = True

class TaskCompletionReport:
    def __init__(self, team_name):
        self._team_name = team_name
        try:
            rospy.init_node('challenge_report_node')
        except:
            pass
        prefix = '/surgical_robotics_challenge/completion_report/' + self._team_name
        self._task1_pub = rospy.Publisher(prefix + '/task1/', PoseStamped, queue_size=1)
        self._task2_pub = rospy.Publisher(prefix + '/task2/', Bool, queue_size=1)
        self._task3_pub = rospy.Publisher(prefix + '/task3/', Bool, queue_size=1)

    def task_1_report(self, pose):
        print(self._team_name, 'reporting task 1 complete with result:', pose)
        self._task1_pub.publish(pose)

    def task_2_report(self, complete):
        print(self._team_name, 'reporting task 2 complete with result:', complete)
        self._task2_pub.publish(complete)

    def task_3_report(self, complete):
        print(self._team_name, 'reporting task 3 complete with result:', complete)
        self._task3_pub.publish(complete)

class ImageLRSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.left_frame = None
        self.left_ts = None
        self.right_frame = None
        self.right_ts = None
        self.imgL_subs = rospy.Subscriber("/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback)
        self.imgR_subs = rospy.Subscriber("/ambf/env/cameras/cameraR/ImageData", Image, self.right_callback)
        rospy.sleep(1.5)  # Ensures subscribers and publishers are ready

    def left_callback(self, msg):
        try:
            self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def right_callback(self, msg):
        try:
            self.right_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

class RosCommunication:
    def __init__(self):
        self.estimation_num = 0
        self.rosCom_subs = rospy.Subscriber("/Amagi_msgs/estimation_num", Int32, self.estimation_numCallback)
        rospy.sleep(0.5)  # Ensures subscribers are ready

    def estimation_numCallback(self, msg):
        self.estimation_num = msg.data

def get_max_IoU(pred_bboxes, gt_bbox):
    """
    Return the maximum IoU score for a given ground truth bbox and predicted bboxes.
    
    :param pred_bboxes: Numpy array of predicted bounding box coordinates.
    :param gt_bbox: Ground truth bounding box coordinates as a tuple/list.
    :return: Tuple of all IoU scores, maximum IoU score, and index of maximum IoU score.
    """
    if pred_bboxes.shape[0] > 0:
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        inters = iw * ih
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    return overlaps, ovmax, jmax

def RT2Frame(R, t):
    """
    Convert rotation and translation matrices into a KDL Frame.
    
    :param R: Rotation matrix as a numpy array.
    :param t: Translation vector as a numpy array.
    :return: KDL Frame object.
    """
    frame = Frame()
    R_list = R.flatten()
    t_list = t.flatten()
    frame.M = Rotation(*R_list)
    frame.p = Vector(*t_list)
    return frame

def frame_to_pose_stamped_msg(frame):
    """
    Convert a KDL frame to a ROS PoseStamped message.
    
    :param frame: KDL Frame object.
    :return: PoseStamped message populated with position and orientation from the frame.
    """
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = frame.p[0]
    msg.pose.position.y = frame.p[1]
    msg.pose.position.z = frame.p[2]
    msg.pose.orientation.x = frame.M.GetQuaternion()[0]
    msg.pose.orientation.y = frame.M.GetQuaternion()[1]
    msg.pose.orientation.z = frame.M.GetQuaternion[2]
    msg.pose.orientation.w = frame.M.GetQuaternion[3]

    return msg

def get_cus_Cross(pred_bboxes, gt_bbox):
    """
    Calculate custom cross metric for predicted bboxes against a ground truth bbox.
    
    :param pred_bboxes: Numpy array of predicted bounding boxes.
    :param gt_bbox: Ground truth bounding box coordinates.
    :return: Array of custom cross metrics and indices where the metric exceeds a threshold.
    """
    ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
    iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
    ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
    iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    inters = iw * ih
    selfs = (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.)
    self_contain = inters / selfs
    threshold = 0.8
    indexs = np.argwhere(self_contain > threshold)

    return self_contain, indexs

def dist_betw_pts(refer_pts, pts):
    """
    Compute the sum of minimum distances between reference points and a set of points.
    
    :param refer_pts: Tensor of reference points.
    :param pts: Tensor of points to compare against the reference.
    :return: Tensor of sum errors for each set of points compared to the reference.
    """
    refer_ = refer_pts.unsqueeze(dim=2)
    real = pts.unsqueeze(dim=0).unsqueeze(dim=0)
    batch_error = real - refer_
    channel_dist = torch.norm(batch_error, dim=3)
    min_dist = torch.min(channel_dist, dim=1).values
    sum_error = torch.sum(min_dist, dim=1)
    return sum_error


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', action='store', dest='team_name', help='Team Name', default='Amagi')
    parser.add_argument('-e', action='store', dest='task_report', help='Task to evaluate (1,2, or 3)', default=1)
    parsed_args = parser.parse_args()

    print('Specified Arguments')
    print(parsed_args)

    task_report = TaskCompletionReport(parsed_args.team_name)
    imgLR_saver = ImageLRSaver()

    client = Client('surgical_robotics_task_report')
    client.connect()
    time.sleep(0.3)

    task_to_report = int(parsed_args.task_report)
    if task_to_report not in [1, 2, 3]:
        raise ValueError('ERROR! Acceptable task evaluation options (-e option) are 1, 2, or 3.')

    RosComm = RosCommunication()

    # Camera intrinsics calculation
    fvg = 1.2
    width, height = 1920, 1080
    f = height / (2 * np.tan(fvg / 2))

    intrinsic_params = np.zeros((3, 3))
    intrinsic_params[0, 0] = f
    intrinsic_params[1, 1] = f
    intrinsic_params[0, 2] = width / 2
    intrinsic_params[1, 2] = height / 2
    intrinsic_params[2, 2] = 1.0

    Kc = torch.tensor(intrinsic_params, dtype=torch.float32).cuda()

    print('Our method can achieve the summarization error under 1mm. \n'
          'If you cannot reproduce these results, try to use method 1, thux!')
    
    print('Acquire the stereo calibration on-the-fly ...')
    ambf_cam_l = Camera(client, "cameraL")
    ambf_cam_r = Camera(client, "cameraR")
    ambf_cam_frame = Camera(client, "CameraFrame")
    T_FL = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    T_FR = pm.toMatrix(ambf_cam_r.get_T_c_w())  # CamR to CamFrame
    
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    T_LR = np.linalg.inv(T_FL).dot(T_FR)
    T_r_l = np.dot(F.dot(T_LR), np.linalg.inv(F))
    T_l_f = F.dot(np.linalg.inv(T_FL))
    
    print('T_LR: \n', T_r_l)
    print('T_FL: \n', T_l_f)

    T_l_r = np.linalg.pinv(T_r_l)

    # Load models
    print('Loading models ...')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("needle_train",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0015
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Just one class for this example
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    print('---Model loaded!')

    # Get the stereo images
    pre_estimation_num = 0
    while not rospy.is_shutdown():
        # do the estimation job when receieve new signals
        cur_estimation_num = RosComm.estimation_num
        if cur_estimation_num > pre_estimation_num:
            imgL = imgLR_saver.left_frame
            imgR = imgLR_saver.right_frame

            predictionsL = predictor(imgL)
            predictionsR = predictor(imgR)

            # Process left image predictions
            maskL = predictionsL['instances'].pred_masks[0]
            keypointsL = predictionsL['instances'].pred_keypoints[0]
            scoresL = predictionsL['instances'].scores[0]
            [pts_yL, pts_xL] = torch.where(maskL)
            pixelsL = torch.hstack((pts_xL.reshape((-1, 1)), pts_yL.reshape((-1, 1))))

            # Process right image predictions
            keypointsR = predictionsR['instances'].pred_keypoints[0]
            scoresR = predictionsL['instances'].scores[0]

            # Enhanced needle mask extraction for left image
            condition1 = (imgL[:, :, 0] == imgL[:, :, 1])
            condition2 = (imgL[:, :, 1] == imgL[:, :, 2])
            condition3 = (imgL[:, :, 0] >= 10)
            needle_maskL = np.bitwise_and(condition1, condition2)
            maskL = np.bitwise_and(needle_maskL, condition3)

            # Enhanced needle mask extraction for right image
            needle_maskR = np.bitwise_and(imgR[:, :, 0] == imgR[:, :, 1], imgR[:, :, 1] == imgR[:, :, 2])
            maskR = np.bitwise_and(needle_maskR, imgR[:, :, 0] >= 10)

            # Analyze regions and handle predictions with masks
            label_imgL = skimage.measure.label(maskL, connectivity=2)
            propsL = skimage.measure.regionprops(label_imgL)
            label_imgR = skimage.measure.label(maskR, connectivity=2)
            propsR = skimage.measure.regionprops(label_imgR)

            gt_bboxR = predictionsR['instances'].pred_boxes[0].tensor.cpu().numpy().squeeze()
            pred_bboxesR = np.array([[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]] for prop in propsR])
            overlapsR, ovmaxR, jmaxR = get_max_IoU(pred_bboxesR, gt_bboxR)
            selflapsR, indsR = get_cus_Cross(pred_bboxesR, gt_bboxR)
            full_mask_idR = np.unique(np.concatenate([indsR, [jmaxR]]))[0]

            # Apply mask based on regions identified
            maskR = np.zeros_like(label_imgR)
            if full_mask_idR != 0:
                for i in range(full_mask_idR.shape[0]):
                    maskR = np.bitwise_or(maskR, label_imgR == (full_mask_idR[i] + 1))

            # Handle left image similarly as right
            gt_bboxL = predictionsL['instances'].pred_boxes[0].tensor.cpu().numpy().squeeze()
            pred_bboxesL = np.array([[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]] for prop in propsL])
            overlapsL, ovmaxL, jmaxL = get_max_IoU(pred_bboxesL, gt_bboxL)
            selflapsL, indsL = get_cus_Cross(pred_bboxesL, gt_bboxL)
            full_mask_idL = np.unique(np.concatenate([indsL, [jmaxL]]))[0]

            maskL = np.zeros_like(label_imgL)
            if full_mask_idL != 0:
                for i in range(full_mask_idL.shape[0]):
                    maskL = np.bitwise_or(maskL, label_imgL == (full_mask_idL[i] + 1))

            # Optional: Uniform sampling of pixels for performance management
            num_sample = 6000
            number_pixelsL, number_pixelsR = pixelsL.shape[0], pixelsR.shape[0]
            if number_pixelsL > num_sample:
                indicesL = np.random.choice(number_pixelsL, num_sample, replace=False)
                pixelsL = pixelsL[indicesL]
            if number_pixelsR > num_sample:
                indicesR = np.random.choice(number_pixelsR, num_sample, replace=False)
                pixelsR = pixelsR[indicesR]

            # Debugging prints to verify dimensions and sizes
            # print(f"pixelL/R size: {number_pixelsL}/{number_pixelsR}")

            # Additional constants or configurations for the model
            l_st2end = 0.182886  # Assuming some calibration or setup measure
            l_st2end_tensor = torch.tensor([l_st2end], device='cuda')

            parral_num_q1 = 4
            parral_num_q2 = 8
            parral_num = parral_num_q1 * parral_num_q2
            tensor_1_col = torch.ones((parral_num, 1), device='cuda')

            # Initializing joint angle ranges
            q1_ws = np.linspace(np.pi / 6, np.pi / 3 * 2, parral_num_q1)
            q2_ws = np.linspace(-np.pi, np.pi, parral_num_q2)
            q1_init, q2_init = np.meshgrid(q1_ws, q2_ws)
            q1_init = q1_init.reshape((-1,))
            q2_init = q2_init.reshape((-1,))

            q11 = torch.tensor(q1_init, dtype=torch.float32, device='cuda')
            q22 = torch.tensor(q2_init, dtype=torch.float32, device='cuda')

            # Determine initial positions based on keypoints
            if torch.max(torch.abs(keypointsL[0, :2] - keypointsL[1, :2])) > 10:
                ratio = 1.0
                s1_init, s2_init = keypointsL[0, :2], keypointsL[1, :2]
            elif torch.max(torch.abs(keypointsR[0, :2] - keypointsR[1, :2])) > 10:
                print('Using right frame keypoints instead...')
                ratio = 6.0
                s1_init, s2_init = keypointsR[0, :2], keypointsR[1, :2]
            else:
                print('Need to adjust the camera...')
                print('Calculating a coarse estimation for faster performance...')
                ratio = 2.0
                skeleton0 = morphology.skeletonize(maskL)
                [pty, ptx] = np.where(skeleton0)
                sk_pt1, sk_pt2 = torch.tensor([ptx[0], pty[0]], device='cuda'), torch.tensor([ptx[-1], pty[-1]], device='cuda')
                s1_init, s2_init = (sk_pt1, sk_pt2) if keypointsL[0, -1] > keypointsL[1, -1] else (sk_pt2, sk_pt1)
                print('Recalculated keypoints positions:', s1_init, s2_init)

            s1 = torch.ones((parral_num, 2), device='cuda') * s1_init
            s2 = torch.ones((parral_num, 2), device='cuda') * s2_init

            # Enabling gradients for optimization
            q11.requires_grad = True
            q22.requires_grad = True
            s1.requires_grad = True
            s2.requires_grad = True

            optimizer = torch.optim.Adam([
                {'params': q11, 'lr': 0.01},
                {'params': q22, 'lr': 0.01},
                {'params': s1, 'lr': 0.015 * ratio},
                {'params': s2, 'lr': 0.015 * ratio}
            ])

            # Reprojection geometry matrix setup
            R_l_r = torch.tensor(T_l_r[:3, :3], dtype=torch.float32, device='cuda')
            t_l_r = torch.tensor(T_l_r[:3, 3], dtype=torch.float32, device='cuda')
            M = torch.tensor([[1, 0, 0], [0, 1, 0], [-intrinsic_params[0, 2], -intrinsic_params[1, 2], intrinsic_params[0, 0]]], dtype=torch.float32, device='cuda')

            # Begin optimization process
            t_s = time.time()
            best_batch_error = torch.tensor([1e6], device='cuda')
            best_q1_hist = torch.tensor([math.pi], device='cuda')
            best_q2_hist = torch.tensor([2 * math.pi], device='cuda')
            best_cRn_hist = None
            best_ctn_hist = None
            tensor_1_col = torch.ones((parral_num, 1)).cuda()
            tensor_0 = torch.zeros(1).cuda()
            tensor_1 = torch.ones(1).cuda()
            a0 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(dim=0).cuda()
            a1 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).unsqueeze(dim=0).cuda()
            a2 = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32).unsqueeze(dim=0).cuda()

            ax = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.float32).unsqueeze(dim=0).cuda()
            ay = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(dim=0).cuda()
            az = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32).unsqueeze(dim=0).cuda()

            # Load needle model data
            Pt = np.loadtxt('../../model/needle_axis_keypoints.txt').T
            Pts = torch.tensor(Pt, dtype=torch.float32, device='cuda')
            print('Needle model loaded!')

            M = torch.tensor([[1, 0, 0], [0, 1, 0], [-intrinsic_params[0, 2], -intrinsic_params[1, 2], intrinsic_params[0, 0]]], dtype=torch.float32).cuda()

            for step in range(1000):  # 1500
                t_0 = time.time()
                s1_ = torch.cat((s1, tensor_1_col), dim=1)
                s2_ = torch.cat((s2, tensor_1_col), dim=1)
                c1 = torch.mm(s1_, M)  # Start point in camera coordinates
                c2 = torch.mm(s2_, M)  # End point in camera coordinates

                # Normalize vectors and calculate needed variables
                n = torch.cross(c2, c1) / torch.norm(torch.cross(c2, c1), dim=1, keepdim=True)
                alpha = torch.acos(torch.sum(c1 * c2, dim=1) / (torch.norm(c2, dim=1) * torch.norm(c1, dim=1)))
                h = l_st2end_tensor * torch.sin(q11)
                lambda2 = h / torch.sin(alpha)
                lambda1 = h / torch.tan(alpha) + torch.sign(q11 - math.pi / 2) * torch.sqrt(l_st2end_tensor**2 - h**2)

                m1 = lambda1.unsqueeze(dim=1) * c1 / torch.norm(c1, dim=1, keepdim=True)
                m2 = lambda2.unsqueeze(dim=1) * c2 / torch.norm(c2, dim=1, keepdim=True)

                x1e = (m2 - m1) / torch.norm(m2 - m1, dim=1, keepdim=True)
                y_temp = torch.cross(n, x1e, dim=1)
                y1e = y_temp / torch.norm(y_temp, dim=1, keepdim=True)

                # Compute rotation matrices
                Rcx = torch.bmm(x1e.unsqueeze(dim=1), ax)
                Rcy = torch.bmm(y1e.unsqueeze(dim=1), ay)
                Rcz = torch.bmm(n.unsqueeze(dim=1), az)
                Rc1 = Rcx + Rcy + Rcz

                # Rotation matrix for joint 2
                cos_q22 = torch.cos(q22).unsqueeze(dim=1).unsqueeze(dim=2)
                sin_q22 = torch.sin(q22).unsqueeze(dim=1).unsqueeze(dim=2)
                R12 = a0 + a1 * cos_q22 + a2 * sin_q22

                # Calculate final pose of the virtual needle w.r.t. the camera frame
                cRn = torch.matmul(Rc1, R12)
                ctn = 0.5 * (m1 + m2)

                # Projecting points onto the image plane
                temp = torch.matmul(cRn, Pts.unsqueeze(dim=0).expand(parral_num, -1, -1)) + ctn.unsqueeze(dim=2)
                cprr = temp / temp[:, 2, :].unsqueeze(dim=1)
                proj = torch.matmul(Kc, cprr[:, :3, :])  # x, y projections
                needle_proj_pts = torch.transpose(proj[:, :2, :], 1, 2)

                temp_R = torch.matmul(cRn, Pts.unsqueeze(dim=0).expand(parral_num, -1, -1)) + ctn.unsqueeze(dim=2)
                cprr_R = temp_R / temp_R[:, 2, :].unsqueeze(dim=1)
                proj_R = torch.matmul(Kc, cprr_R[:, :3, :])  # Right camera projection
                needle_proj_pts_R = torch.transpose(proj_R[:, :2, :], 1, 2)

                # Compute error and backpropagate
                batch_error = dist_betw_pts(needle_proj_pts, pixelsL) + dist_betw_pts(needle_proj_pts_R, pixelsR)
                batch_error = torch.where(torch.isnan(batch_error), torch.full_like(batch_error, 1e6), batch_error)
                loss = torch.sum(batch_error)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging and checking for improvements
                select = torch.argmin(batch_error)
                if batch_error[select] < best_batch_error:
                    best_batch_error = batch_error[select]
                    best_cRn_hist = cRn[select].detach()
                    best_ctn_hist = ctn[select].detach()

                if torch.isnan(q11[select]) or torch.isnan(q22[select]):
                    print('NaN detected, aborting optimization.')
                    break

                if step % 20 == 0:
                    print(f'Iteration {step}, min error: {batch_error[select].item():.2f}, q1: {q11[select].item() * 180 / math.pi:.2f}°, q2: {q22[select].item() * 180 / math.pi:.2f}°, selected index: {select.item()}')
                
                t_end = time.time()
                print(f'Optimization step took {t_end - t_0:.2f}s')


            t_e = time.time()
            print(f"Total computation time: {t_e - t_s:.2f}s.")
            print('*********************************************************************************************************************')
            print('*****************************************************Final Results***************************************************')
            
            # Select the best result based on minimum batch error
            select = torch.argmin(batch_error)
            min_error = batch_error[select].item()
            q1_deg = q11[select].item() * 180 / math.pi
            q2_deg = q22[select].item() * 180 / math.pi
            print(f'Minimum error: {min_error}. q1: {q1_deg:.2f}° / q2: {q2_deg:.2f}°')
            print(f's1 shape: {s1.shape}')
            print(f'Keypoint1: x: {s1[select, 0].item()} y: {s1[select, 1].item()} / Keypoint2: x: {s2[select, 0].item()} y: {s2[select, 1].item()}')
            print(f'Selected index: {select.item()}')
            print(f'Rotation Matrix cRn:\n{cRn[select, :, :]}')
            print(f'Translation Vector ctn: {ctn[select, :]}')
            print('*********************************************************************************************************************')
            print('*********************************************************************************************************************')

            # Clean up GPU memory
            torch.cuda.empty_cache()

            # Convert the results into frames and pose messages
            pose_frame = RT2Frame(cRn[select, :, :].detach().cpu().numpy(), ctn[select, :].detach().cpu().numpy())
            T_l_f_frame = RT2Frame(T_l_f[:3, :3], T_l_f[:3, 3])
            T_offset = np.array([[0.871982, 0.489403, -0.00136704, 0.00102883],
                                [-0.489405, 0.87198, -0.00184491, -0.04541],
                                [0.000289144, 0.00227791, 0.999932, 0.00043177],
                                [0, 0, 0, 1]])
            T_offset_frame = RT2Frame(T_offset[:3, :3], T_offset[:3, 3])
            
            # Perform transformations and send report
            final_pose = T_l_f_frame.Inverse() * pose_frame * T_offset_frame
            task_report.task_1_report(frame_to_pose_stamped_msg(final_pose))
            print('Final estimation pose:', final_pose)

            # Update the estimation number to manage loop or sequence flow
            pre_estimation_num = cur_estimation_num

            # Optional: Prepare projected and real points for further analysis or visualization
            proj_p2 = needle_proj_pts[select].detach().cpu().numpy()
            real_p = pixelsL.detach().cpu().numpy()
