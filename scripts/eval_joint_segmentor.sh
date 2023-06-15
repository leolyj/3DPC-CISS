# Evaluate Joint Training on different datasets of various split
# S3DIS DataSet
GPU_ID=0

DATASET='s3dis'
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

args=(--phase 'jointeval'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"  --tasks "Joint"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

SPLIT=0
MODEL_CHECKPOINT='./log_s3dis/log_joint_s3dis_cv0'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_s3dis/log_joint_s3dis_cv1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

# ScanNet DataSet
GPU_ID=0

DATASET='scannet'
DATA_PATH='./datasets/ScanNet/blocks_bs1_s1/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
#
args=(--phase 'jointeval'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"  --tasks "Joint"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

SPLIT=0
MODEL_CHECKPOINT='./log_scannet/log_joint_scannet_cv0'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_scannet/log_joint_scannet_cv1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"
