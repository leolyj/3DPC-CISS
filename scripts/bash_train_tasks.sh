# Train the designed baselines and our model
# Joint Training: Upper bound
bash ./scripts/train_joint_segmentor.sh

# Setting: '12-1', '10-3', '8-5' for S3DIS dataset, '19-1', '17-3', '15-5' for ScanNet dataset
# See different *.sh files for setting details

# Noticing: Our training code provide a overall training process for all individual model, step 0 for base,
# step 1,...STEP for incre. if you have trained a incremental model in a specific setting of splits, you only
# need to put its base model checkpoints into other models folders (to be trained in the same setting) and
# modify the corresponding training file in 'runs' folder, to make the 'for step in range(0, STEP):' ->
# 'for step in range(1, STEP):', (only train the incre model) in order to avoid the base model
# re-train and for equal comparision.

# *********** Direct Adaptation Methods ***********
# Freeze-and-Add
bash ./scripts/freeze_and_add_incre_segmentor.sh

# Finetuning
bash ./scripts/finetuning_incre_segmentor.sh

# *********** Forgetting-Prevention Methods ***********
# Elastic Weight Consolidation: EWC
bash ./scripts/train_ewc_segmentor.sh

# Learning without Forgetting: LwF
bash ./scripts/train_lwf_segmentor.sh

# 3DPC-CISS: Ours -> Extra multi-step increments '8-1' for S3DIS dataset, '15-1' for ScanNet dataset
bash ./scripts/train_ours_segmentor.sh






