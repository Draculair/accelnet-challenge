## Download

Open a terminal and run the following commands:

```sh
git clone https://github.com/linhongbin-ws/accel-challenge.git
cd accel-challenge
git clone https://github.com/collaborative-robotics/surgical_robotics_challenge.git
```

Since the trained model is large, please contact the author at draculaaair@gmail.com to obtain the model files and merge them into the `model` folder at `<path/to/accel-challenge>/model/`.

## Install

1. Install [ambf](https://github.com/WPI-AIM/ambf).
2. Create a conda virtual environment with Python 3.7:

```sh
conda create -n accel_challenge python=3.7
conda activate accel_challenge
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip install scikit-image
pip install -r requirements.txt
```

3. Install PyKDL on the virtual environment from source. Before installing PyKDL, make sure to uninstall any existing ros-kdl packages:

```sh
sudo find / -iname PyKDL.so # Print out all paths to PyKDL.so
sudo rm -rf <path/to>/PyKDL.so
```

4. Install surgical_robotics_challenge:

```sh
cd <path/to/surgical_robotics_challenge>/scripts/
pip install -e .
```

5. Install accel-challenge:

```sh
cd <path/to/accel-challenge>
conda install -c conda-forge wxpython # Required by deeplabcut
pip install -r requirements.txt
pip install -e .
```

6. Modify the original surgical_robotics_challenge by editing the file `<path/to/surgical_robotics_challenge>/scripts/surgical_robotics_challenge/launch_crtk_interface.py`. Replace the line:

```py
import psm_arm
import ecm_arm
import scene
```

with:

```py
from surgical_robotics_challenge import psm_arm
from surgical_robotics_challenge import ecm_arm
from surgical_robotics_challenge import scene
```

7. Modify the file `<path/to/accel-challenge>/accel_challenge/bash/user_var.sh` and update the path variables according to your environment:

```sh
AMBF_PATH="/path/to/ambf"
SURGICAL_CHALLENGE_PATH='/path/to/accel-challenge/surgical_robotics_challenge'
ANACONDA_PATH="/path/to/anaconda3"
ENV_NAME="accel_challenge" # conda virtual environment name
```

## How to Run

- To run Challenge #1, please refer to the [README](https://github.com/Draculair/accelnet-challenge/tree/master/accel_challenge/challenge1).
- To run Challenge #2, please refer to the [README](https://github.com/Draculair/accelnet-challenge/tree/master/accel_challenge/challenge2).

## Results
The results are showed in "challenge1.mkv" and "challenge2.mkv".
