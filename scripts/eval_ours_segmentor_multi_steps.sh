# Evaluate our method on different datasets of multi-step increments
# S3DIS DataSet
GPU_ID=0

DATASET='s3dis'
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

args=(--phase 'increeval_multi'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

SPLIT=0
MODEL_CHECKPOINT='./log_s3dis/log_ours_s3dis_cv0_tasks8-1'
TASKS='8-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_s3dis/log_ours_s3dis_cv1_tasks8-1'
TASKS='8-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

# ScanNet DataSet
GPU_ID=0

DATASET='scannet'
DATA_PATH='./datasets/ScanNet/blocks_bs1_s1/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

args=(--phase 'increeval_multi'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

SPLIT=0
MODEL_CHECKPOINT='./log_scannet/log_ours_scannet_cv0_tasks15-1'
TASKS='15-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_scannet/log_ours_scannet_cv1_tasks15-1'
TASKS='15-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"
