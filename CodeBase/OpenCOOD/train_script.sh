# python -m torch.distributed.launch  --master_port 5179 --nproc_per_node=8 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2x-radar/CameraOnly/camera_coalign.yaml

python -m torch.distributed.launch  --master_port 5179 --nproc_per_node=8 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2x-radar/LiDAROnly/lidar_coalign.yaml


python opencood/tools/inference_modify.py --model_dir opencood/logs/v2x_radar_lidar_coalign_2025_03_16_07_44_06
