# CORE: Canonical Object Representations for Enhanced Manipulation

This repository contains the official implementation of CORE, a method for manipulating articulated objects with generalization across object translations, scaling, and robot kinematics. We propose an object-centric representation that decouples vision-based decision making from robot execution, enabling stable predictions under varying camera views, object placements, and robot configurations.

**Key Features:**
- Object-centric imitation learning for articulated object manipulation
- Generalization across camera perspectives, object translations, and scaling
- Compatible with different robot kinematics
- State-of-the-art performance on manipulation tasks

For more details, see our [thesis](thesis/) or the [paper](#).

---

## Record demos
Set up
```bash
# In terminal 1
# pip install --user virtualenv
# virtualenv CORE
# source ENV/bin/activate
# conda activate env
git clone --recurse-submodules https://github.com/tongmiaoxu/CORE.git

cd CORE
 . venv/bin/activate
pip install --upgrade pip setuptools


export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget --no-check-certificate https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

pip install git+https://github.com/stepjam/RLBench.git

pip install gymnasium 
pip install testresources

pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
# Install the headless version of OpenCV
pip install opencv-python-headless
pip install PyQt5
```
This is how to build task:
You can configure the saved_path and task in the scripts
```bash
cd CORE/
python3 tools/task_builder.py

```
This is how to collect data
```bash
#Open MobaXterm
source ENV/bin/activate
export ..
pip install opencv-python-headless
cd CORE/
bash scripts/generate_dataset_IL.sh
#variations have to be larger than 0

```
To see the collected image in MobaXterm
``` bash
eog /home/tongmiao/CORE/data/open_door/episode_0/front_rgb/0.png 


```
To train:
```bash
git clone https://github.com/RobotIL-rls/RobotIL.git --recursive
git clone https://github.com/RobotIL-rls/robomimic.git
cd RobotIL/
pip install -e .
pip install -e robomimic
pip install opencv-python
bash scripts/train_policy.sh

```
To infer:
```bash
source ENV/bin/activate
export ..
pip install opencv-python-headless
cd CORE/
bash scripts/inference.sh

```
To read npy data:
```bash 
echo "import numpy as np; data = np.load('data/open_door/episode_0/task_data.npy');print(data[0])" > print_first_line.py
python3 print_first_line.py
```
To obtain contact_label and door pose:
```bash
python utils/process_pcd.py
```
To validate traning data in object-centric:
```bash
python utils/validate_training_data_in_obc.py
python utils/validate_training.py
```
