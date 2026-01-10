GPUS=$1
CONFIG_FILE=$2
CKPT_PATH=$3
FUSION_METHOD=$4
VISUALIZE=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    --use_env \
    $(dirname "$0")/opencood/tools/test_ddp.py \
    --hypes_yaml "$CONFIG_FILE" \
    --ckpt_path "$CKPT_PATH" \
    --fusion_method "$FUSION_METHOD" \
    --save_vis_interval "$VISUALIZE"

# Usage examples:
# bash dist_test.sh 4 opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign.yaml opencood/work_dirs/v2x_radar_collab_radaronly_radarpillarnet_coalign_2025_12_29_21_30_23/net_epoch24.pth intermediate 100
# bash dist_test.sh 4 opencood/hypes_yaml/v2x-radar/LiDAR_only/collab_lidaronly_pointpillar_attfuse.yaml opencood/work_dirs/v2x_radar_single_lidaronly_pointpillar_2025_12_10_19_38_49/net_epoch32.pth no 25
# bash dist_test.sh 4 opencood/hypes_yaml/v2x-radar/radar_only/collab_radaronly_radarpillarnet_coalign_whole.yaml opencood/work_dirs/v2x_radar_collab_radaronly_radarpillarnet_coalign_whole_2026_01_02_09_00_26/net_epoch24.pth intermediate 100