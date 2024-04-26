import cv2
import os
from time import sleep
import numpy as np
from os import listdir
from os.path import join
from skimage import io
from skimage.util import img_as_ubyte
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.core import predict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class DLC_Predictor():
    def __init__(self, config_path, use_gpu=False):
        self.use_gpu = use_gpu
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if use_gpu:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        cfg = load_config(str(config_path))
        shuffle = 1
        trainingsetindex = 0
        trainFraction = cfg['TrainingFraction'][trainingsetindex]
        modelfolder = join(cfg["project_path"], str(auxiliaryfunctions.get_model_folder(trainFraction, shuffle, cfg)))
        path_test_config = join(modelfolder, 'test', 'pose_cfg.yaml')
        dlc_cfg = load_config(str(path_test_config))

        Snapshots = np.array([fn.split('.')[0] for fn in listdir(join(modelfolder, 'train')) if "index" in fn])
        snapshotindex = -1 if cfg['snapshotindex'] == 'all' else cfg['snapshotindex']
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        dlc_cfg['init_weights'] = join(modelfolder, 'train', Snapshots[snapshotindex])
        self.batch_size = cfg['batch_size']

        if use_gpu:
            self.sess, self.inputs, self.outputs = predict.setup_GPUpose_prediction(dlc_cfg)
            self.pose_tensor = predict.extract_GPUprediction(self.outputs, dlc_cfg)
        else:
            self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(dlc_cfg)
        self.dlc_cfg = dlc_cfg

    def predict(self, input_image, input_depth_xyz=None):
        if isinstance(input_image, np.ndarray):
            _image = np.expand_dims(input_image, axis=0)
        elif isinstance(input_image, str):
            image = io.imread(input_image, plugin='matplotlib')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = img_as_ubyte(image)
            _image = np.expand_dims(image, axis=0)
        else:
            raise NotImplementedError

        images = _image
        for _ in range(self.batch_size - 1):
            images = np.concatenate((images, _image), axis=0)
        if self.use_gpu:
            pose = self.sess.run(self.pose_tensor, feed_dict={self.inputs: images.astype(float)})
        else:
            pose = predict.getpose(np.squeeze(_image, axis=0), self.dlc_cfg, self.sess, self.inputs, self.outputs)
            pose[:, [0,1,2]] = pose[:, [1,0,2]]
        pose_list = [(pose[i][0], pose[i][1], pose[i][2]) for i in range(int(pose.shape[0]/self.batch_size))]

        if input_depth_xyz is None:
            return pose_list
        else:
            feature_xyz = [[input_depth_xyz[int(pose[i][0]), int(pose[i][1]), 0],
                            input_depth_xyz[int(pose[i][0]), int(pose[i][1]), 1],
                            input_depth_xyz[int(pose[i][0]), int(pose[i][1]), 2]] for i in range(len(pose_list))]
            return (np.array(pose_list), np.array(feature_xyz))
    
    def render(self, input_image_dir, annotes=None, circle_size=8):
        img = plt.imread(input_image_dir) if isinstance(input_image_dir, str) else input_image_dir
        fig, ax = plt.subplots(1)
        ax.grid(False)
        ax.set_aspect('equal')
        ax.imshow(img)
        if annotes is not None:
            for yy, xx, prob in annotes:
                circ = Circle((xx, yy), circle_size)
                ax.add_patch(circ)
        plt.show()


if __name__ == '__main__':
    from accel_challenge.challenge1.ros_client import ClientEngine
    from accel_challenge.challenge1.tool import PointCloud2_2_xyzNimage, render_rgb_xyz

    TEST_PREDICT = False
    TEST_PROJECTION = False
    TEST_DEPTH_IMAGE = False
    TEST_PREDICT_WITH_DEPTH = True

    if TEST_PREDICT:
        input_image = "/home/ben/code/robot/gym_suture/data/calibration/alpha-beta-0-0/{}.jpeg".format(8)
        pose_list = dlc_predict(input_image)
        print(pose_list)

    if TEST_PROJECTION:
        cam_model = CameraModel()
        print(cam_model.project_P_cam(1, 2, 3))

    if TEST_DEPTH_IMAGE:
        camera_engine = ClientEngine()
        camera_engine.add_clients(['ecm'])
        camera_engine.start()
        sleep(0.5)
        data = camera_engine.get_signal('ecm', 'cameraL_image_depth')
        import ros_numpy
        _data = ros_numpy.point_cloud2.pointcloud2_to_array(data)
        _data_rgb = ros_numpy.point_cloud2.split_rgb_field(_data)
        rgb = np.array([_data_rgb['r'], _data_rgb['g'], _data_rgb['b']]).T.reshape(1080, 1920, 3)
        rgb = np.flip(rgb, axis=0)
        while True:
            cv2.imshow('preview', rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        camera_engine.close()

    if TEST_PREDICT_WITH_DEPTH:
        camera_engine = ClientEngine()
        camera_engine.add_clients(['ecm'])
        camera_engine.start()
        sleep(0.5)
        data = camera_engine.get_signal('ecm', 'cameraL_image_depth')
        xyz, rgb = PointCloud2_2_xyzNimage(data)
        render_rgb_xyz(rgb, xyz)