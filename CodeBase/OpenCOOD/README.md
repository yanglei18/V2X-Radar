
## 1. Data Preparation

- V2X-Radar: Please refer to [this website](http://openmpd.com/column/V2X-Radar). We release the mini version, full version will be released soon.
- DAIR-V2X-C: Download the data from [this website](https://thudair.baai.ac.cn/index). We use complemented annotation, so please also follow the instruction of [this page](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/). 
- OPV2V: Please refer to [this repo](https://github.com/DerrickXuNu/OpenCOOD). You also need to download `additional-001.zip` which stores data for camera modality.
- V2XSet: Please refer to [this repo](https://github.com/DerrickXuNu/v2x-vit).

```shell
V2X-Radar
├── data
│   ├── v2x-radar 
│   │   ├── v2x-radar-i
│   │   ├── v2x-radar-v
│   │   ├── v2x-radar-c  # OpenV2V Format
│   │   │   ├── train
│   │   │   │   ├── 2024-05-15-16-28-09
│   │   │   │   │   ├── -1  # RoadSide
│   │   │   │   │   │   ├── 00000.pcd - 00250.pcd # the LiDAR point clouds data from timestamp 0 to 250
│   │   │   │   │   │   ├── 00000_radar.pcd - 00250_radar.pcd # the 4D Radar point clouds data transformed on the LiDAR frame from timestamp 0 to 250
│   │   │   │   │   │   ├── 00000.yaml - 00250.yaml # corresponding metadata for each timestamp
│   │   │   │   │   │   ├── 00000_camera0.jpg - 00250_camera0.jpg # left camera images
│   │   │   │   │   │   ├── 00000_camera1.jpg - 00250_camera1.jpg # front camera images
│   │   │   │   │   │   ├── 00000_camera2.jpg - 00250_camera2.jpg # right camera images
│   │   │   │   │   ├── 142 # Vehicle Side
│   │   │   ├── validate
│   │   │   ├── test
│   ├── dair-v2x-c 
│   ├── openv2v
│   │   ├── additional
│   │   ├── test
│   │   ├── train
│   │   └── validate
│   ├── v2x-set
│   │   ├── test
│   │   ├── train
│   │   └── validate
```

## 2. Installation
### 2.1. Basic Installation
```bash
conda create -n v2x-radar python=3.8
conda activate v2x-radar
conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 cudatoolkit=11.3 -c pytorch
cd V2X-Radar/CodeBase/OpenCOOD
pip install -r requirements.txt
python setup.py develop
```

### 2.2. Install Spconv (1.2.1 or 2.x)
We use spconv 1.2.1 or spconv 2.x to generate voxel features. spconv 2.x has much convenient installation.

To install **spconv 2.x**, check the [table](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library) to run the installation command. For example we have cudatoolkit 11.6, then we should run
```bash
pip install spconv-cu113 # match your cudatoolkit version
```

To install **spconv 1.2.1**, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.
You can also get a detailed installation guide in [CoAlign Installation Doc](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie#doxcn5rISC6NcfXIUnWFnXhTEzd).


### 2.3. Bbx IoU cuda version compile
Install bbx nms calculation cuda version
  
```bash
cd V2X-Radar/CodeBase/OpenCOOD
python opencood/utils/setup.py build_ext --inplace
```

### 2.4. Dependencies for FPV-RCNN (optional)
Install the dependencies for fpv-rcnn.
  
```bash
cd V2X-Radar/CodeBase/OpenCOOD
python opencood/pcdet_utils/setup.py build_ext --inplace
```


---
To align with our agent-type assignment in our experiments, please make a copy of the assignment file under the logs folder
```bash
# in HEAL directory
mkdir V2X-Radar/CodeBase/OpenCOOD/opencood/logs
cp -r opencood/modality_assign opencood/logs/heter_modality_assign
```


## 3. Basic Train / Test Command
### 3.1. Train the model
We uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `-y` or `hypes_yaml` : the path of the training configuration file, e.g. `opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml`, meaning you want to train
a FCooper model. 
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,

### 3.2. Train the model in DDP
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
`--nproc_per_node` indicate the GPU number you will use.

### 3.3. Test the model
```python
python opencood/tools/inference_modify.py --model_dir ${CHECKPOINT_FOLDER} [--fusion_method intermediate]
```
- `inference_modify.py` has more optional args, you can inspect into this file.
- `[--fusion_method intermediate]` the default fusion method is intermediate fusion. According to your fusion strategy in training, available fusion_method can be:
  - **single**: only ego agent's detection, only ego's gt box. *[only for late fusion dataset]*
  - **no**: only ego agent's detection, all agents' fused gt box.  *[only for late fusion dataset]*
  - **late**: late fusion detection from all agents, all agents' fused gt box.  *[only for late fusion dataset]*
  - **early**: early fusion detection from all agents, all agents' fused gt box. *[only for early fusion dataset]*
  - **intermediate**: intermediate fusion detection from all agents, all agents' fused gt box. *[only for intermediate fusion dataset]*
