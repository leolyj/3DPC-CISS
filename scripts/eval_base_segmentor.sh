# Evaluate the base model on different datasets of various splits and task settings
# S3DIS DataSet
GPU_ID=0

DATASET='s3dis'
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

args=(--phase 'baseeval'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

# ****** Need to modify the xxx to the base model folder, e.g. xxx -> ours, ... ******
SPLIT=0
TASKS='12-1'
MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv0_tasks12-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv0_tasks10-3'
TASKS='10-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv0_tasks8-5'
TASKS='8-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv1_tasks12-1'
TASKS='12-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv1_tasks10-3'
TASKS='10-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_s3dis/log_xxx_s3dis_cv1_tasks8-5'
TASKS='8-5'
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

args=(--phase 'baseeval'  --dataset "${DATASET}"
      --data_path  "$DATA_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K --dgcnn_mlp_widths "$MLP_WIDTHS")

# ****** Need to modify the xxx to the base model folder, e.g. xxx -> ours, ... ******
SPLIT=0
MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv0_tasks19-1'
TASKS='19-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv0_tasks17-3'
TASKS='17-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv0_tasks15-5'
TASKS='15-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

SPLIT=1
MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv1_tasks19-1'
TASKS='19-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv1_tasks17-3'
TASKS='17-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"

MODEL_CHECKPOINT='./log_scannet/log_xxx_scannet_cv1_tasks15-5'
TASKS='15-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --save_path "$MODEL_CHECKPOINT" --model_checkpoint_path "$MODEL_CHECKPOINT"
