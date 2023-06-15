# Eval the designed baselines and our model: see different *.sh files for setting details
# Joint Training: Upper bound
bash ./scripts/eval_joint_segmentor.sh

# *********** Direct Adaptation Methods ***********
# Freeze-and-Add
bash ./scripts/eval_freeze_and_add_segmentor.sh

# Finetuning
bash ./scripts/eval_finetuning_segmentor.sh

# *********** Forgetting-Prevention Methods ***********
# Elastic Weight Consolidation: EWC
bash ./scripts/eval_ewc_segmentor.sh

# Learning without Forgetting: LwF
bash ./scripts/eval_lwf_segmentor.sh

# 3DPC-CISS: Evaluate our method
bash ./scripts/eval_ours_segmentor.sh

# 3DPC-CISS: Evaluate our method for mutli-steps increments, '8-1' for S3DIS dataset, '15-1' for ScanNet dataset
bash ./scripts/eval_ours_segmentor_multi_steps.sh

# Only Evaluate the base model (not the Increment model) -> see eval_base_segmentor.sh for details, need to modify the folder root
bash ./scripts/eval_base_segmentor.sh




