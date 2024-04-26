import datetime
import io
import time
from time import sleep
from pathlib import Path

import numpy as np
import argparse
import cv2
from tqdm import tqdm
from rospy import Publisher, init_node, Rate, is_shutdown
from sklearn.preprocessing import StandardScaler 
from accel_challenge.challenge1.tool import RPY2T
from accel_challenge.challenge1.tracking import DLC_Predictor
import pickle
from numpy import pi
from sensor_msgs.msg import ChannelFloat32
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import device
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

ERROR_MAG_ARR = np.deg2rad([5, 5, 0.05, 0, 0, 0])
FPS = 10
TRAIN_TRAJ_SEED = 77
TEST_TRAJ_SEED = 66
ONLINE_TEST_TRAJ_SEED = 55
TRAJ_POINTS_NUMS = 600
NET_INTER_DIM_LIST = [400, 300, 200]
ERROR_DATA_DIR = '/home/draculair/accel-challenge/model/error_data'

DLC_CONFIG_PATH_PSM2 = "/home/draculair/accel-challenge/model/dlc/psm1/config.yaml"
DLC_CONFIG_PATH_PSM1 = "/home/draculair/accel-challenge/model/dlc/psm2/config.yaml"
DLC_CONFIG_PATH_dict = {'psm2': DLC_CONFIG_PATH_PSM2, 'psm1': DLC_CONFIG_PATH_PSM1}

x_origin, y_origin, z_origin = -0.211084, 0.260047, 0.906611
YAW = -0.8726640502948968
pose_origin_psm2 = RPY2T(0, 0.15, 0.1, 0, 0, 0) * RPY2T(0.2, 0, 0, 0, 0, 0) * RPY2T(x_origin, y_origin, z_origin, pi, -pi/2, 0) * RPY2T(0, 0, 0, 0, 0, YAW)
pose_origin_psm1 = RPY2T(0, 0.15, 0.1, 0, 0, 0) * RPY2T(0.2, 0, 0, 0, 0, 0) * RPY2T(x_origin, y_origin, z_origin, pi, pi/2, 0) * RPY2T(0, 0, 0, 0, 0, -YAW)
pose_origin_dict = {'psm2': pose_origin_psm2, 'psm1': pose_origin_psm1}

def cam_render_test(video_dir=None):
    if video_dir is not None:
        frame = engine.get_signal('ecm', 'cameraL_image')
        height, width, channel = frame.shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        file_name = 'calibrate_record' + timestamp + '.mp4'
        video = cv2.VideoWriter(str(video_dir / file_name), fourcc, float(FPS), (int(width), int(height)))

    while True:
        frame = engine.get_signal('ecm', 'cameraL_image')
        cv2.imshow('preview', frame)
        if video_dir is not None:
            video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(max(1 / FPS - 0.03, 0))

    cv2.destroyAllWindows()
    if video_dir is not None:
        video.release()

def set_error(arm_name, data):
    init_node('ros_client_engine', anonymous=True)
    error_pub = Publisher('/ambf/env/' + arm_name + '/errors_model/set_errors', ChannelFloat32, queue_size=1)
    sleep(0.5)
    msg = ChannelFloat32()
    msg.values = data
    error_pub.publish(msg)
    sleep(0.2)


def calibrate_joint_error(_engine, load_dict, arm_name='psm2'):
    models = {}
    _engine.clients[arm_name].servo_tool_cp(pose_origin_dict[arm_name], 100)
    start = time.time()
    with device('/cpu'):
        models['dlc_predictor'] = DLC_Predictor(load_dict['dlc_config_path'])
        print("DLC_Predictor initialize time:", time.time() - start)
        models['keras_model'] = load_model(load_dict['keras_model_path'])
        scalers = pickle.load(open(load_dict['scalers_path'], 'rb'))
    models['input_scaler'] = scalers['input_scaler']
    models['output_scaler'] = scalers['output_scaler']
    q = _engine.clients[arm_name].get_signal('measured_js')
    data = {}
    data['image'] = _engine.get_signal('ecm', 'cameraL_image')
    data['feature'] = models['dlc_predictor'].predict(data['image'])
    data['feature'] = np.array(data['feature'])[:, :2].reshape(1, -1)
    data['feature'] = models['input_scaler'].transform(data['feature'])
    data['err_pred'] = models['keras_model'].predict(data['feature'])
    data['err_pred'] = models['output_scaler'].inverse_transform(data['err_pred'])
    return data['err_pred'].reshape(-1)

def joint_error_test(seed, _engine, save_data_dir=None, load_dict=None, arm_name='psm2', is_predict=False):
    models = {}
    if load_dict is not None: 
        models['dlc_predictor'] = DLC_Predictor(load_dict['dlc_config_path'])
        models['keras_model'] = load_model(load_dict['keras_model_path'])
        scalers = pickle.load(open(load_dict['scalers_path'], 'rb'))
        models['input_scaler'] = scalers['input_scaler']
        models['output_scaler'] = scalers['output_scaler']
        print(models)

    _engine.clients[arm_name].reset_pose()
    _engine.clients[arm_name].wait()
    error_pub = Publisher('/ambf/env/' + arm_name + '/errors_model/set_errors', ChannelFloat32, queue_size=1)
    rate = Rate(100)
    sleep(0.5)  # wait a bit to initialize publisher
    def pub_error(data):
        msg = ChannelFloat32()
        msg.values = data
        for i in range(2):
            error_pub.publish(msg)
            rate.sleep()
    _engine.clients[arm_name].servo_tool_cp(pose_origin_dict[arm_name], 100)
    _engine.clients[arm_name].wait()

    rng_error = np.random.RandomState(seed)  # use local seed
    num = 0
    while not is_shutdown():
        num += 1
        if not is_predict:
            _error = rng_error.uniform(-ERROR_MAG_ARR, ERROR_MAG_ARR)
            q_msr = _engine.clients[arm_name].get_signal('measured_js')
            pub_error(_error)
            print(f'num: {num} error: {_error}, ctrl+c to stop')
            sleep(0.1)
        _engine.clients[arm_name].servo_tool_cp(pose_origin_dict[arm_name], 50)
        _engine.clients[arm_name].wait()
        sleep(3)

        if load_dict is not None:
            data = {}
            data['image'] = _engine.get_signal('ecm', 'cameraL_image')
            data['feature'] = models['dlc_predictor'].predict(data['image'])
            data['feature'] = np.array(data['feature'])[:, :2].reshape(1, -1)
            data['err_pred'] = models['keras_model'].predict(models['input_scaler'].transform(data['feature']))
            data['err_pred'] = models['output_scaler'].inverse_transform(data['err_pred'])
            print(data['err_pred'])
            print(_error[:3])

        if save_data_dir is not None:
            data_dict = {'error': _error, 'image': _engine.get_signal('ecm', 'cameraL_image')}
            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            filename = Path(save_data_dir) / f'error_{timestamp}.npz'
            with io.BytesIO() as f1:
                np.savez_compressed(f1, **data_dict)
                f1.seek(0)
                with open(filename, 'wb') as f2:
                    f2.write(f1.read())

def dlc_predict_test(config_path, test_image_dir):
    dlc_predictor = DLC_Predictor(config_path)
    annotes = dlc_predictor.predict(test_image_dir)
    print(annotes)
    dlc_predictor.render(test_image_dir, annotes=annotes)

def make_dataset(data_dir):
    directory = Path(data_dir).expanduser()
    data_buffer = []
    for filename in tqdm(reversed(sorted(directory.glob('**/*.npz')))):
        try:
            with filename.open('rb') as f:
                data = np.load(f, allow_pickle=True)
                data = {k: data[k] for k in data.keys()}
                data_buffer.append(data)
        except Exception as e:
            print(f'Could not load episode: {e}')
            continue

    xs = [np.array(d['feature'][:, :2]).reshape(-1) for d in data_buffer]
    ys = [np.array(d['error']).reshape(-1)[:3] for d in data_buffer]  # only the first 3 joints
    xs, ys = np.stack(xs, axis=0), np.stack(ys, axis=0)

    input_scaler, output_scaler = StandardScaler(), StandardScaler()
    input_scaler.fit(xs)
    output_scaler.fit(ys)

    xs_norm = input_scaler.transform(xs)
    ys_norm = output_scaler.transform(ys)
    return xs, ys, xs_norm, ys_norm, input_scaler, output_scaler

def make_features(load_dir, save_dir, dlc_config_path):
    load_dir = Path(load_dir).expanduser()
    dlc_predictor = DLC_Predictor(dlc_config_path)
    save_ft_names = ['error', 'feature']
    save_data_dir = Path(save_dir)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(reversed(sorted(load_dir.glob('**/*.npz')))):
        with filename.open('rb') as f:
            data = np.load(f, allow_pickle=True)
            data['feature'] = dlc_predictor.predict(data['image'])
            data_dict = {k: data[k] for k in save_ft_names}
            with io.BytesIO() as f1:
                np.savez_compressed(f1, **data_dict)
                f1.seek(0)
                save_file_dir = save_data_dir / (str(filename.stem) + '-ft.npz')
                with save_file_dir.open('wb') as f2:
                    f2.write(f1.read())

def train_mlp(train_data_dir, test_data_dir, save_model_dir):
    train_data = make_dataset(train_data_dir)
    test_data = make_dataset(test_data_dir)
    model = build_keras_model(input_dim=train_data[2].shape[1],
                              output_dim=train_data[3].shape[1], 
                              inter_dim_list=NET_INTER_DIM_LIST, 
                              lr=0.0001)

    callbacks = [
        ModelCheckpoint(filepath=str(Path(save_model_dir) / 'model.hdf5'), save_best_only=False),
        EarlyStopping(monitor='val_loss', patience=100),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    model.fit(train_data[2], train_data[3], validation_split=0.2, epochs=1000, verbose=2, callbacks=callbacks)
    test_pred = model.predict(test_data[2])
    test_pred_err = np.abs(test_data[5].inverse_transform(test_pred) - test_data[1])

    print(f"Test error mean: {np.mean(test_pred_err)}, std: {np.std(test_pred_err)}")
    print(f"Test error mean (Degrees): {np.rad2deg(np.mean(test_pred_err))}, std: {np.rad2deg(np.std(test_pred_err))}")
    print("Test data input shape:", test_data[0].shape)
    print("Test data input [1]:", test_data[0][0, :], test_data[2][0, :])
    print("Test data output [1]:", test_data[1][0, :], test_data[3][0, :])
    print("Test data predict output [1]:", test_data[5].inverse_transform(test_pred)[0, :], test_pred[0, :])

    pickle.dump({'input_scaler': train_data[4], 'output_scaler': train_data[5]}, open(Path(save_model_dir) / 'scalers.pkl', 'wb'))





def build_keras_model(input_dim, output_dim, inter_dim_list, lr, 
                        is_batchnormlize=False,
                        dropout_amount=None):
    model = Sequential()
    model.add(Dense(50, input_shape = (input_dim, ), kernel_initializer='he_normal'))
    if is_batchnormlize:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if not dropout_amount is None:
        model.add(Dropout(dropout_amount))

    for inter_dim in inter_dim_list:
        model.add(Dense(inter_dim, kernel_initializer='he_normal'))
        if is_batchnormlize:
            model.add(BatchNormalization())
        model.add(Activation('relu'))    
        if not dropout_amount is None:
            model.add(Dropout(dropout_amount))
    model.add(Dense(output_dim, kernel_initializer='he_normal'))
    print(model.summary())
    adam = optimizers.Adam(lr = lr)
    model.compile(optimizer = adam, loss =MeanSquaredError())
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, type=int, help='program type, 1 for recording, 2 for trajectory')
    parser.add_argument('--arm', required=True, type=str, help='Specify the robotic arm')
    args = parser.parse_args()
    arm_name = args.arm

    assert args.p in range(1, 12), "Program type out of expected range"
    is_no_engine = args.p in [7, 8, 9]

    engine = None
    if not is_no_engine:
        engine = ClientEngine()
        engine.add_clients(['ecm'])
        engine.start()

    operations = {
        1: lambda: cam_render_test(video_dir=Path(VIDEO_DIR) / arm_name),
        2: lambda: joint_error_test(seed=TRAIN_TRAJ_SEED, _engine=engine, arm_name=arm_name),
        3: lambda: dlc_predict_test(config_path=DLC_CONFIG_PATH, test_image_dir=TEST_IMAGE_FILE_DIR),
        4: lambda: init_and_test(engine, arm_name, TRAIN_TRAJ_SEED, 'train'),
        5: lambda: init_and_test(engine, arm_name, TEST_TRAJ_SEED, 'test'),
        6: lambda: print(make_dataset(data_dir=Path(ERROR_DATA_DIR) / 'test_ft')),
        7: lambda: make_features(Path(ERROR_DATA_DIR) / arm_name / 'test', Path(ERROR_DATA_DIR) / arm_name / 'test_ft', DLC_CONFIG_PATH_dict[arm_name]),
        8: lambda: train_mlp(Path(ERROR_DATA_DIR) / arm_name / 'train_ft', Path(ERROR_DATA_DIR) / arm_name / 'test_ft', Path(ERROR_DATA_DIR) / arm_name),
        9: lambda: random_image_predict(Path(ERROR_DATA_DIR) / arm_name / 'train'),
        10: lambda: predict_and_print_error(engine, arm_name, ONLINE_TEST_TRAJ_SEED, True),
        11: lambda: calibrate_and_print_error(engine, arm_name)
    }

    if args.p in operations:
        operations[args.p]()

    if not is_no_engine and engine:
        engine.close()

def init_and_test(engine, arm_name, seed, phase):
    engine.add_clients([arm_name])
    engine.start()
    engine.clients[arm_name].open_jaw()
    engine.clients[arm_name].wait()
    data_dir = Path(ERROR_DATA_DIR) / arm_name / phase
    joint_error_test(seed=seed, _engine=engine, save_data_dir=data_dir, arm_name=arm_name)

def random_image_predict(base_path):
    import random
    image_dir_list = sorted(base_path.glob('**/*.npz'))
    image_dir = random.choice(image_dir_list)
    with image_dir.open('rb') as f:
        data = np.load(f, allow_pickle=True)
    dlc_predict_test(config_path=DLC_CONFIG_PATH, test_image_dir=data['image'])

def predict_and_print_error(engine, arm_name, seed, is_predict):
    load_dict = {
        'dlc_config_path': DLC_CONFIG_PATH,
        'keras_model_path': str(Path(ERROR_DATA_DIR) / arm_name / 'model.hdf5'),
        'scalers_path': str(Path(ERROR_DATA_DIR) / arm_name / 'scalers.pkl')
    }
    error = joint_error_test(seed=seed, _engine=engine, load_dict=load_dict, is_predict=is_predict, arm_name=arm_name)
    print("Predict error deg:", np.rad2deg(error))

def calibrate_and_print_error(engine, arm_name):
    load_dict = {
        'dlc_config_path': DLC_CONFIG_PATH_dict[arm_name],
        'keras_model_path': str(Path(ERROR_DATA_DIR) / arm_name / 'model.hdf5'),
        'scalers_path': str(Path(ERROR_DATA_DIR) / arm_name / 'scalers.pkl')
    }
    error = calibrate_joint_error(_engine=engine, load_dict=load_dict, arm_name=arm_name)
    error_gt = np.deg2rad([1, 2, 0, 0, 0, 0])
    engine.clients[arm_name].reset_pose()
    engine.clients[arm_name].wait()
    print("Predict error (value)  (ground truth error):", error, error - error_gt[:3])
    print("Predict error deg  (value)  (ground truth error):", np.rad2deg(error), np.rad2deg(error - error_gt[:3]))