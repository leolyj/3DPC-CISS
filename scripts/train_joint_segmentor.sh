# Joint Training on different datasets of various splits
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

args=(--phase 'jointtrain' --dataset "${DATASET}"
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --tasks "Joint"
      --n_epochs $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --joint_lr $LR --joint_weight_decay $WEIGHT_DECAY
      --joint_decay_size $DECAY_STEP --joint_gamma $DECAY_RATIO)

SPLIT=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT

SPLIT=1
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT

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

args=(--phase 'jointtrain' --dataset "${DATASET}"
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --tasks "Joint"
      --n_epochs $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --joint_lr $LR --joint_weight_decay $WEIGHT_DECAY
      --joint_decay_size $DECAY_STEP --joint_gamma $DECAY_RATIO)

SPLIT=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT

SPLIT=1
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}" --cvfold $SPLIT