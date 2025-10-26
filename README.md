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

Supported by the [THU OpenMDP Lab](http://openmpd.com/column/V2X-Radar).

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
- Support both simulation and real-world cooperative perception dataset
    - [x] V2X-Radar
    - [x] DAIR-V2X
    - [x] V2XSet
    - [x] OpenV2V
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
    - [x] [Cooper [ICDCS]](https://arxiv.org/abs/1905.05265)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2VNet [ECCV2020]](https://arxiv.org/abs/2008.07519)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
    - [x] [CoBEVT [CoRL2022]](https://arxiv.org/abs/2207.02202)
    - [x] [HEAL [ICLR 2024]](https://arxiv.org/abs/1905.05265)

## Data Download
Please check our [website](https://huggingface.co/datasets/yanglei18/V2X-Radar) to download the data (OPV2V / KITTI format).

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
- Oct. 23, 2025: The full Dataset data is released: [Hugging Face](https://huggingface.co/datasets/yanglei18/V2X-Radar) | [Baidu Netdisk](https://pan.baidu.com/s/1Yw7PQKB4TFOL-JOUZ6kfOg ) (Code: **cefq**)
- Sep. 19, 2025: Our V2X-Radar paper was accepted as a <span style="color:red">**NeuIPS 2025 Spotlight**</span> (top ≈ 2.8 %)! 🎉🎉🎉
- Mar. 18, 2025: The [mini sample](https://drive.google.com/drive/folders/11zq-v9GBdFv_tnpKd3EkS1mbSNzZf0RT?usp=sharing) data is released.
- Mar. 15, 2025: Tha paper and supplementary is released.
- Mar. 14, 2025: The codebase is released.
- Nov. 7, 2024: Tha paper is released.

## Quick Start
### Cooperative Perception
Please refer to [CodeBase/BEVHeight](CodeBase/BEVHeight/README.md).

### Single-agent Perception
Please refer to [CodeBase/OpenCOOD](CodeBase/OpenCOOD/README.md).


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
