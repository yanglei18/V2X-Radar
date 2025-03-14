## 1. Installation

### 1.1. Docker Installation
```shell
docker pull yanglei2024/op-bevheight:base
cd V2X-Radar/CodeBase/BEVHeight
python setup.py develop
```

### 1.2. Local Installation
**a.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**b.** Install mmcv-full==1.4.0  mmdet==2.19.0 mmdet3d==0.18.1.

**c.** Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

**d.** Install requirements.
```shell
pip install -r requirements.txt
```
**e.** Install V2X-Radar/CodeBase/BEVHeight (gpu required).
```shell
python setup.py develop
```

## 2. Data Preparation
### 2.1. V2X-Radar-I / V2X-Radar-V  Dataset
**a.** Download V2X-Radar-I / V2X-Radar-V dataset from official [website](http://openmpd.com/column/V2X-Radar).

**We provide mini and full versions (release soon)**

```shell
V2X-Radar
├── data
│   ├── v2x-radar
│   │   ├── mini
│   │   │   ├── v2x-radar-i # KITTI Format
│   │   │   │   ├── training
│   │   │   │   │   ├── velodyne
│   │   │   │   │   ├── radar
│   │   │   │   │   ├── calib
│   │   │   │   │   ├── image_1
│   │   │   │   │   ├── image_2
│   │   │   │   │   ├── image_3
│   │   │   │   │   ├── label_2
│   │   │   │   ├── ImageSets
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   │   ├── v2x-radar-v # KITTI Format
│   │   │   │   ├── training
│   │   │   │   │   ├── velodyne
│   │   │   │   │   ├── radar
│   │   │   │   │   ├── calib
│   │   │   │   │   ├── image_2
│   │   │   │   │   ├── label_2
│   │   │   │   ├── ImageSets
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   ├── trainval-full  # release soon
│   │   ├── ...
```
**b.** Prepare infos for V2X-Radar-I / V2X-Radar-V datasets.
```shell
python scripts/gen_info_v2x_radar_i.py
python scripts/gen_info_v2x_radar_v.py
```

### 2.2. DAIR-V2X-I Dataset
**a.** Download DAIR-V2X-I dataset from official [website](http://openmpd.com/column/V2X-Radar).

**b.** Convert the dataset to KITTI format.
```
ln -s [single-infrastructure-side root] ./data/dair-v2x
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
```

The directory will be as follows.
```shell
V2X-Radar
├── data
│   ├── dair-v2x-i
│   │   ├── velodyne
│   │   ├── image
│   │   ├── calib
│   │   ├── label
|   |   └── data_info.json
|   ├── dair-v2x-i-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   |...
|...
```
**c.** Prepare infos for DAIR-V2X-I dataset.
```shell
python scripts/gen_info_dair.py
```


### 2.3. Rope3D Dataset
**a.** Download Rope3D dataset from official [website](https://thudair.baai.ac.cn/index).

**b.** Convert the dataset to KITTI format.
```
ln -s [rope3d root] ./data/rope3d
python scripts/data_converter/rope2kitti.py --source-root data/rope3d --target-root data/rope3d-kitti
```
The directory will be as follows.
```shell
V2X-Radar
├── data
|   ├── rope3d
|   |   ├── training
|   |   ├── validation
|   |   ├── training-image_2a
|   |   ├── training-image_2b
|   |   ├── training-image_2c
|   |   ├── training-image_2d
|   |   └── validation-image_2
|   ├── rope3d-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── map_token2id.json
|   |...  
├── ...
```
**c.** Prepare infos for Rope3D dataset.
```shell
python scripts/gen_info_rope3d.py
```

### 2.4. KITTI Dataset
**a.** Download KITTI dataset from official [website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).


**b.** Prepare infos for KITTI dataset.
```shell
python scripts/gen_info_kitti.py --data_root data/kitti
```


### 2.5. Visualize the dataset in KITTI format
```shell
python scripts/data_converter/visual_tools_kitti.py --data_root ../../data/kitti --demo_dir ./demo
python scripts/data_converter/visual_tools_v2x_radar.py --data_root ../../data/v2x-radar/mini/v2x-radar-i --demo_dir ./demo
```

## 3. Train and Eval
### 3.1. Eval  `BEVDepth / BEVHeight / BEVHeight_Plus` with 8 GPUs

```shell
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
### 3.2. Eval  `BEVDepth / BEVHeight / BEVHeight_Plus` with 8 GPUs
```shell
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```