
## 1. Data Preparation
```shell
V2X-Radar
├── data
│   ├── v2x-radar
│   │   ├── v2x-radar-i   # KITTI Format
│   │   │   ├── training
│   │   │   │   ├── velodyne
│   │   │   │   ├── radar
│   │   │   │   ├── calib
│   │   │   │   ├── image_1
│   │   │   │   ├── image_2
│   │   │   │   ├── image_3
│   │   │   │   ├── label_2
│   │   │   ├── ImageSets
│   │   │   │   ├── train.txt
│   │   │   │   ├── trainval.txt
│   │   │   │   ├── val.txt
│   │   │   │   ├── test.txt
│   │   ├── v2x-radar-v   # KITTI Format
│   │   │   ├── training
│   │   │   │   ├── velodyne
│   │   │   │   ├── radar
│   │   │   │   ├── calib
│   │   │   │   ├── image_2
│   │   │   │   ├── label_2
│   │   │   ├── ImageSets
│   │   │   │   ├── train.txt
│   │   │   │   ├── trainval.txt
│   │   │   │   ├── val.txt
│   │   │   │   ├── test.txt
│   │   ├── v2x-radar-c  # OpenV2V Format
│   │   │   ├── train
│   │   │   │   ├── 2024-05-15-16-28-09
│   │   │   │   │   ├── -1  # RoadSide
│   │   │   │   │   │   ├── 00000.pcd - 00250.pcd # LiDAR point clouds from timestamp 0 to 250
│   │   │   │   │   │   ├── 00000_radar.pcd - 00250_radar.pcd # the 4D Radar point clouds from timestamp 0 to 250
│   │   │   │   │   │   ├── 00000.yaml - 00250.yaml # metadata for each timestamp
│   │   │   │   │   │   ├── 00000_camera0.jpg - 00250_camera0.jpg # left camera images
│   │   │   │   │   │   ├── 00000_camera1.jpg - 00250_camera1.jpg # front camera images
│   │   │   │   │   │   ├── 00000_camera2.jpg - 00250_camera2.jpg # right camera images
│   │   │   │   │   ├── 142 # Vehicle Side  
│   │   │   ├── validate
│   │   │   ├── test
│   ├── other datasets 
```

## 2. Installation
### 2.1. Basic Installation
```bash
conda create -n v2x-radar python=3.8
conda activate v2x-radar
pip install torch==1.12.0+cu113 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r docs/requirements.txt
python setup.py develop
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmdet==2.26.0
```

### 2.2. Install Spconv and other dependencies(1.2.1 or 2.x)
```bash
pip install spconv-cu113 # match your cudatoolkit version
python opencood/utils/setup.py build_ext --inplace
python packages/pcdet_utils/setup.py build_ext --inplace
cd packages/pypcd && python setup.py install && cd ../../
python packages/Voxelization/setup.py develop
python packages/Voxelization/setup_v2.py develop
```

### 2.3. Install within CUDA 118 (optional)
```bash
pip install torch==2.0.0 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install spconv-cu118
```
### 2.4. Download Pretrained Model
```bash
Download "https://drive.google.com/file/d/1VjGlWEC5FeYMkb-bPXFNtRqp0POdDnCd/view?usp=drive_link"
mv efficientnet_depth_v2xradar CodeBase/OpenCOOD/opencood/pretrained
```


## 3. Basic Train / Test Command
### 3.1. Train the model
We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `-y` or `hypes_yaml` : the path of the training configuration file, e.g. `opencood/hypes_yaml/lidar_only/collab_lidaronly_lidarpillarnet_coalign.yaml`, meaning you want to train
a FCooper model. 
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,

### 3.2. Train the model in DDP
```python
cd CodeBase/OpenCOOD
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
CUDA_VISIBLE_DEVICES=0,1 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_coalign.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_coalign.out
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_coalign.yaml 4 100 > v2xradar-collab_cameraonly_bevdepth_coalign.out
```
### 3.3. Test the model
```python
python opencood/tools/test.py --hypes_yaml ${CONFIG_FILE} --ckpt_path ${CHECKPOINT_PATH} --fusion_method ${FUSION_METHOD} 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/test_ddp.py --hypes_yaml ${CONFIG_FILE} --ckpt_path ${CHECKPOINT_PATH} --fusion_method ${FUSION_METHOD}
bash dist_test.sh 4 opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign.yaml opencood/work_dirs/v2x_radar_collab_radaronly_radarpillarnet_coalign_2025_12_29_21_30_23/net_epoch24.pth intermediate 100
```
- `test.py / test_ddp.py` has more optional args, you can inspect into this file.
- `[--fusion_method intermediate]` the default fusion method is intermediate fusion. According to your fusion strategy in training, available fusion_method can be:
  - **single**: only ego agent's detection, only ego's gt box. *[only for late fusion dataset]*
  - **no**: only ego agent's detection, all agents' fused gt box.  *[only for late fusion dataset]*
  - **late**: late fusion detection from all agents, all agents' fused gt box.  *[only for late fusion dataset]*
  - **early**: early fusion detection from all agents, all agents' fused gt box. *[only for early fusion dataset]*
  - **intermediate**: intermediate fusion detection from all agents, all agents' fused gt box. *[only for intermediate fusion dataset]*
