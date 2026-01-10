<p align="center">
  <h1 align="center">V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception</h1>
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=EUnI2nMAAAAJ&hl=zh-CN&oi=sra"><strong>Lei Yang</strong></a>
    · 
    <a href="https://scholar.google.com.hk/citations?user=0Q7pN4cAAAAJ&hl=zh-CN"><strong>Xinyu Zhang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Jun Li</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Chen Wang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=S3cQz1AAAAAJ&hl=zh-CN&oi=ao"><strong>Jiaqi Ma</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=joReSgYAAAAJ&hl=zh-CN&oi=sra"><strong>Zhiying Song</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=tTnWi_EAAAAJ&hl=zh-CN"><strong>Tong Zhao</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=tIjCAKEAAAAJ&hl=zh-CN"><strong>Ziying Song</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=pmzKjcUAAAAJ&hl=zh-CN"><strong>Li Wang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong> Mo Zhou</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong> Yang Shen</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=ElfT3eoAAAAJ"><strong> Kai Wu</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=UKVs2CEAAAAJ&hl=zh-CN"><strong> Chen Lv</strong></a>
</p>

<div align="center">
  <img src="./assets/teaser-v2.jpg" alt="Logo" width="100%">
</div>

<p align="center">
  <br>
    <a href="https://neurips.cc/virtual/2025/poster/121426"><img alt="website" src="https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome"></a>
    <a href="https://arxiv.org/pdf/2411.10962"><img alt="paper" src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
    <a href="https://huggingface.co/datasets/yanglei18/V2X-Radar">
    <img alt="huggingface" src="https://img.shields.io/badge/Hugging%20Face-Dataset-gray?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=gray&color=0078D7">
  </a>
  <a href="https://pan.baidu.com/s/1Yw7PQKB4TFOL-JOUZ6kfOg ">
    <img alt="baidunetdisk" src="https://img.shields.io/badge/Baidu%20Netdisk-Dataset-white?style=flat&logo=baidu&logoColor=0066FF&labelColor=white&color=0066FF">
  </a>
    <br></br>
    </a>
  </p>
</p>

This is the official implementation of **"V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception"**.

<p align="center">
<img src="imgs/scene1.png" width="600" alt="" class="img-responsive">
</p>

## Overview
- [Codebase Features](#codebase-features)
- [Data Download](#data-download)
- [Changelog](#changelog)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgment](#known-issues)

## CodeBase Features
- Multiple Tasks supported
    - [x] Cooperative 3D Object Detection
    - [x] Single-agent 3D Object Detection
- Support cooperative perception dataset
    - [x] V2X-Radar
- Support multi real-world single-agent dataset
    - [x] V2X-Radar-I
    - [x] V2X-Radar-V
    - [x] DAIR-V2X-I
    - [x] Rope3D
    - [x] KITTI
- SOTA model supported
    - [x] [BEVHeight [CVPR2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVHeight_A_Robust_Framework_for_Vision-Based_Roadside_3D_Object_Detection_CVPR_2023_paper.pdf)
    - [x] [BEVHeight++ [T-PAMI2025]](https://arxiv.org/pdf/2309.16179)
    - [x] [BEVDepth [AAAI2023]](https://arxiv.org/pdf/2206.10092)
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [CoAlign [ICRA 2023]](https://arxiv.org/abs/2211.07214)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
    - [x] [CoBEVT [CoRL2022]](https://arxiv.org/abs/2207.02202)
    - [x] [HEAL [ICLR 2024]](https://arxiv.org/abs/1905.05265)

## Data Download
Please check our [website](https://huggingface.co/datasets/yanglei18/V2X-Radar) to download the data ([OPV2V](https://github.com/DerrickXuNu/OpenCOOD/blob/main/docs/md_files/data_annotation_tutorial.md) / [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) format).

After downloading the data, please put the data in the following structure:
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
## Changelog
- Jan. 16, 2026: Refactor the codebase, more concise and easier to extend. 🧩🧩🧩
- Jan. 16, 2026: Release the pretrained model weights.
- Jan. 13, 2026: Reuploaded a more complete and refined version of the dataset.📦📦📦
- Oct. 23, 2025: The full Dataset data is released: [Hugging Face](https://huggingface.co/datasets/yanglei18/V2X-Radar) | [Baidu Netdisk](https://pan.baidu.com/s/1Yw7PQKB4TFOL-JOUZ6kfOg ) (Code: **cefq**).
- Sep. 19, 2025: Our V2X-Radar paper was accepted as a <span style="color:red">**NeuIPS 2025 Spotlight**</span> (top ≈ 2.8 %)! 🎉🎉🎉
- Mar. 15, 2025: Tha paper and supplementary is released.
- Mar. 14, 2025: The codebase is released.
- Nov. 7, 2024: Tha paper is released.

## Quick Start
### Cooperative Perception
Please refer to [CodeBase/BEVHeight](CodeBase/BEVHeight/README.md).

### Single-agent Perception
Please refer to [CodeBase/OpenCOOD](CodeBase/OpenCOOD/README.md).

## Models Zoo

###  Cooperative 3D Object Detection Benchmarks
Vehicle category includes car, bus, truck | Metrics: AP@IoU = 0.7 / 0.5 (↑)

| Method      | M | Overall              | 0–30 m               | 30–50 m              | 50–100 m             | Config | Model |
|-------------|---|----------------------|----------------------|----------------------|----------------------|--------|-------|
| Late Fusion | C | 13.59 / 32.88        | 16.16 / 40.58        | 13.29 / 30.37        | 11.71 / 20.00        | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_late.yaml)     | [√](https://drive.google.com/file/d/1p_G_nHQE1Ttx93SamjHG7-3eiOgb2Yht/view?usp=drive_link)     |
| F-Cooper    | C | 15.56 / 44.43        | 23.22 / 61.97        | 10.98 / 31.24        | 4.15 / 15.38         | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_fcooper.yaml)      | [√](https://drive.google.com/file/d/1eeZ9R8ZqfwlEuX__eUNnQARtdxquEN0e/view?usp=drive_link)     |
| CoAlign     | C | 24.26 / 46.89        | **36.49 / 63.70**    | 12.75 / 32.77        | 11.36 / 23.17        | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_coalign.yaml)      | [√](https://drive.google.com/file/d/1tA4bbcvrhZUo6JbSdWrf9C2XiMC1jDci/view?usp=drive_link)     |
| HEAL        | C | **25.05 / 46.94**    | 35.18 / 60.40        | **13.48 / 33.63**    | **15.80 / 26.88**    | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_pyramid.yaml)     | [√](https://drive.google.com/file/d/1eDQ0amU7IjSzSFMAhyhZLSDcjRJiNCeX/view?usp=drive_link)     |
| Late Fusion | L | 39.37 / 65.75        | 50.92 / 80.59        | 31.59 / 58.64        | 18.54 / 31.62        | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_late.yaml)     | [√](https://drive.google.com/file/d/1L8gFb8_DAfa41POajwKb2dqvX_xGnIcW/view?usp=drive_link)     |
| F-Cooper    | L | 50.04 / 73.44        | 70.29 / 89.52        | 38.10 / 69.72        | 17.44 / 34.50        | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_fcooper.yaml)      | [√](https://drive.google.com/file/d/1_WkYo43ZdTnjb59_q_5rpxlq3rkQ0YHT/view?usp=drive_link)     |
| CoAlign     | L | 60.18 / 80.42        | 75.42 / 91.08        | 48.10 / 76.48        | 29.62 / 45.51        | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_coalign.yaml)      | [√](https://drive.google.com/file/d/1TQnRBV23z8PX6HWDoe19K-fFcWf8AQhg/view?usp=drive_link)     |
| HEAL        | L | **67.57 / 83.00**    | **82.76 / 92.19**    | **57.70 / 80.51**    | **34.79 / 51.93**    | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_pyramid.yaml)      | [√](https://drive.google.com/file/d/1Tqqx5j61Hcd0nli1CgoUd0Yxq7v_vkw0/view?usp=drive_link)     |
| Late Fusion | R | 3.77 / 17.24         | 6.27 / 25.97         | 1.44 / 11.19         | 0.13 / 0.69          | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_late.yaml)     | [√](https://drive.google.com/file/d/1_U8tDDLwjQJLzqlMI24I0vuvlxfQ2gwg/view?usp=drive_link)     |
| F-Cooper    | R | 6.84 / 23.16         | 11.80 / 34.98        | 2.74 / 16.87         | 0.38 / 2.05          | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_fcooper.yaml)      | [√](https://drive.google.com/file/d/1L82Y5yJmxeDWWIOQZQanwsdGQ1pF3OA_/view?usp=drive_link)     |
| CoAlign     | R | 11.46 / 26.34        | **18.01 / 38.46**    | **5.75 / 16.55**     | 0.28 / 2.83          | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign.yaml)      | [√](https://drive.google.com/file/d/1KWlMzoakT4rno_l5BdTPm3208-G8FGAu/view?usp=drive_link)     |
| HEAL        | R | **12.50 / 29.04**    | **23.02 / 44.83**    | 5.16 / 17.85         | **0.45 / 2.98**      | [√](CodeBase/OpenCOOD/opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_pyramid.yaml)      | [√](https://drive.google.com/file/d/1pBVCiSstwCFR_rv-EZPGLaqcqAS5WQbL/view?usp=drive_link)     |


# Acknowledgment
This project is not possible without the following codebases.
* [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)
* [BEVHeight++](https://github.com/yanglei18/BEVHeight_Plus)
* [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
* [HEAL](https://github.com/yifanlu0227/HEAL)
* [pypcd](https://github.com/dimatura/pypcd)

## Citation
```shell
@article{yang2024v2x,
  title={V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception},
  author={Yang, Lei and Zhang, Xinyu and Li, Jun and Wang, Chen and Ma, Jiaqi and Song, Zhiying and Zhao, Tong and Song, Ziying and Wang, Li and Zhou, Mo and Shen, Yang and Lv, Chen},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
