# Training our method on different datasets of various splits and task settings
# S3DIS Dataset
GPU_ID=0

DATASET='s3dis'
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1/'
SAVE_PATH='./log_s3dis/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

EVAL_INTERVAL=3
BATCH_SIZE=32
NUM_WORKERS=16
NUM_EPOCHS=100
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5
T=0.0065

args=(--phase 'increOurs' --dataset "${DATASET}"
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --uncertain_t $T
      --n_epochs $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --base_lr $LR --base_weight_decay $WEIGHT_DECAY
      --base_decay_size $DECAY_STEP --base_gamma $DECAY_RATIO)

SPLIT=0
TASKS='12-1'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv0_tasks12-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='10-3'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv0_tasks10-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='8-5'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv0_tasks8-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

# *******************************************************************
# Tasks '8-1' for multi-step increments on S3DIS of Split 0, for 8 base classes, 5 novel classes, increments 5 steps
# If you have a trained base model in above '8-5' setting(SPLIT 0), you only need to put the pretrained base checkpoints in this log folder, and modify
# the 'for step in range(0, STEP):' -> 'for step in range(1, STEP):' in runs/train_ours.py for only incre model training
TASKS='8-1'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv0_tasks8-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

SPLIT=1
TASKS='12-1'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv1_tasks12-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='10-3'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv1_tasks10-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='8-5'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv1_tasks8-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

# *******************************************************************
# Tasks '8-1' for multi-step increments on S3DIS of Split 1, for 8 base classes, 5 novel classes, increments 5 steps
# If you have a trained base model in above '8-5' setting(SPLIT 1), you only need to put the pretrained base checkpoints in this log folder, and modify
# the 'for step in range(0, STEP):' -> 'for step in range(1, STEP):' in runs/train_ours.py for only incre model training
TASKS='8-1'
BASE_MODEL_PATH='./log_s3dis/log_ours_s3dis_cv1_tasks8-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

# ScanNet Dataset
GPU_ID=0

DATASET='scannet'
DATA_PATH='./datasets/ScanNet/blocks_bs1_s1/'
SAVE_PATH='./log_scannet/'

NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20

EVAL_INTERVAL=3
BATCH_SIZE=32
NUM_WORKERS=16
NUM_EPOCHS=100
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5
T=0.0045

args=(--phase 'increOurs' --dataset "${DATASET}"
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --uncertain_t $T
      --n_epochs $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --base_lr $LR --base_weight_decay $WEIGHT_DECAY
      --base_decay_size $DECAY_STEP --base_gamma $DECAY_RATIO)

SPLIT=0
TASKS='19-1'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv0_tasks19-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='17-3'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv0_tasks17-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='15-5'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv0_tasks15-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

# *******************************************************************
# Tasks '15-1' for multi-step increments on ScanNet of Split 0, for 15 base classes, 5 novel classes, increments 5 steps
# If you have a trained base model in above '15-5' setting(SPLIT 0), you only need to put the pretrained base checkpoints in this log folder, and modify
# the 'for step in range(0, STEP):' -> 'for step in range(1, STEP):' in runs/train_ours.py for only incre model training
TASKS='15-1'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv0_tasks15-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

SPLIT=1
TASKS='19-1'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv1_tasks19-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='17-3'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv1_tasks17-3'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

TASKS='15-5'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv1_tasks15-5'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"

# *******************************************************************
# Tasks '15-1' for multi-step increments on ScanNet of Split 1, for 15 base classes, 5 novel classes, increments 5 steps
# If you have a trained base model in above '15-5' setting (SPLIT 1), you only need to put the pretrained base checkpoints in this log folder, and modify
# the 'for step in range(0, STEP):' -> 'for step in range(1, STEP):' in runs/train_ours.py for only incre model training
TASKS='15-1'
BASE_MODEL_PATH='./log_scannet/log_ours_scannet_cv0_tasks15-1'
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT --tasks "${TASKS}" --base_model_checkpoint_path "${BASE_MODEL_PATH}"
