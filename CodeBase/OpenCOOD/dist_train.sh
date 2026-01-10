#!/bin/bash
# tmux new -s train3d
# conda activate RaCP
# TORCH_DISTRIBUTED_DEBUG=DETAIL

CONFIG=$1
GPUS=$2
VISUALIZE=$3
RESUME_DIR=$4

# CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=$GPUS \
    --master_port=$((10000 + RANDOM % 55536)) \
    $(dirname "$0")/opencood/tools/train_ddp.py \
    -y $CONFIG \
    --resume_dir "$RESUME_DIR" \
    --visualize $VISUALIZE

# v2x-radar R or C
# CUDA_VISIBLE_DEVICES=2,3 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_coalign.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_coalign.out
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_attfuse.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_attfuse.out
# CUDA_VISIBLE_DEVICES=8,9 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_cobevt.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_cobevt.out
# CUDA_VISIBLE_DEVICES=6,7 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_fcooper.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_fcooper.out

# CUDA_VISIBLE_DEVICES=0,1 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_pyramid.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_pyramid.out
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_v2xvit.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_v2xvit.out
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/single_cameraonly_bevdepth.yaml 2 100 > v2xradar-single_cameraonly_bevdepth.out

# LIDAR
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/single_lidaronly_lidarpillarnet.yaml 2 100 > v2xradar-single_lidaronly_lidarpillarnet.out
# CUDA_VISIBLE_DEVICES=2,3 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_fcooper.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_fcooper.out
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_coalign.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_coalign.out
# CUDA_VISIBLE_DEVICES=6,7 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_pyramid.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_pyramid.out
# CUDA_VISIBLE_DEVICES=8,9 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/lidar_only/collab_lidaronly_lidarpillarnet_v2xvit.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_v2xvit.out

# Radar
# CUDA_VISIBLE_DEVICES=0,1 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/single_radaronly_radarpillarnet.yaml 2 100 > v2xradar-single_radaronly_radarpillarnet.out
# CUDA_VISIBLE_DEVICES=2,3 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_pyramid.yaml 2 100 > v2xradar-collab_radaronly_radarpillarnet_pyramid.out
# CUDA_VISIBLE_DEVICES=4,5 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign.yaml 2 100 > v2xradar-collab_radaronly_radarpillarnet_coalign.out
# CUDA_VISIBLE_DEVICES=0,1 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_v2xvit.yaml 2 100 > v2xradar-collab_radaronly_radarpillarnet_v2xvit.out
# CUDA_VISIBLE_DEVICES=8,9 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_lidaronly_lidarpillarnet_early.yaml 2 100 > v2xradar-collab_lidaronly_lidarpillarnet_early.out

# CUDA_VISIBLE_DEVICES=6,7,8,9 nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_late.yaml 4 100 > v2xradar-collab_radaronly_radarpillarnet_late.out





# v2x-radar R or C
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_coalign.yaml 2 100 > v2xradar-collab_cameraonly_bevdepth_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign.yaml 4 100 > v2xradar-collab_radaronly_radarpillarnet_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_pyramid.yaml 4 100 > v2xradar-collab_cameraonly_bevdepth_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_pyramid.yaml 4 100 > v2xradar-collab_radaronly_radarpillarnet_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/camera_only/collab_cameraonly_bevdepth_attfuse.yaml 4 100 > v2xradar-collab_cameraonly_bevdepth_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_attfuse.yaml 4 100 > v2xradar-collab_radaronly_radarpillarnet_attfuse.out

# v2x-radar R + C
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_bevfusion_attfuse.yaml 4 100 > v2xradar-collab_bevfusion_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_bevfusion_coalign.yaml 4 100 > v2xradar-collab_bevfusion_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_bevfusion_pyramid.yaml 4 100 > v2xradar-collab_bevfusion_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_lxl_attfuse.yaml 4 100 > v2xradar-collab_lxl_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_lxl_coalign.yaml 4 100 > v2xradar-collab_lxl_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_lxl_pyramid.yaml 4 100 > v2xradar-collab_lxl_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_rcfusion_attfuse.yaml 4 100 > v2xradar-collab_rcfusion_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_rcfusion_coalign.yaml 4 100 > v2xradar-collab_rcfusion_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-radar/rccross_fusion/collab_rcfusion_pyramid.yaml 4 100 > v2xradar-collab_rcfusion_pyramid.out

# v2x-r R or C
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/camera_only/collab_cameraonly_bevdepth_coalign.yaml 4 100 > v2xr-collab_cameraonly_bevdepth_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/radar_only/collab_radaronly_radarpillarnet_coalign.yaml 4 100 > v2xr-collab_radaronly_radarpillarnet_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/camera_only/collab_cameraonly_bevdepth_pyramid.yaml 4 100 > v2xr-collab_cameraonly_bevdepth_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/radar_only/collab_radaronly_radarpillarnet_pyramid.yaml 4 100 > v2xr-collab_radaronly_radarpillarnet_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/camera_only/collab_cameraonly_bevdepth_attfuse.yaml 4 100 > v2xr-collab_cameraonly_bevdepth_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/radar_only/collab_radaronly_radarpillarnet_attfuse.yaml 4 100 > v2xr-collab_radaronly_radarpillarnet_attfuse.out

# v2x-r R + C
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_bevfusion_attfuse.yaml 4 100 > v2xr-collab_bevfusion_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_bevfusion_coalign.yaml 4 100 > v2xr-collab_bevfusion_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_bevfusion_pyramid.yaml 4 100 > v2xr-collab_bevfusion_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_lxl_attfuse.yaml 4 100 > v2xr-collab_lxl_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_lxl_coalign.yaml 4 100 > v2xr-collab_lxl_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_lxl_pyramid.yaml 4 100 > v2xr-collab_lxl_pyramid.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_rcfusion_attfuse.yaml 4 100 > v2xr-collab_rcfusion_attfuse.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_rcfusion_coalign.yaml 4 100 > v2xr-collab_rcfusion_coalign.out
# nohup bash dist_train.sh opencood/hypes_yaml/v2x-r/rccross_fusion/collab_rcfusion_pyramid.yaml 4 100 > v2xr-collab_rcfusion_pyramid.out