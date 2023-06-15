"""Main function for 3DPC-CISS"""
import ast
import argparse

# ********************************************************************************************
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--phase', type=str, default='incretrain', choices=['jointtrain', 'freeze_and_add', 'finetuning', 'increEWC', 'increLwF', 'increOurs', 'baseeval', 'jointeval', 'increeval', 'othereval', 'increeval_multi'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold for class order split, Options:{0,1}')
    parser.add_argument('--tasks', type=str, default='12-1', help='Incremental setting for tasks')
    parser.add_argument('--data_path', type=str, default='./datasets/S3DIS/blocks_bs1_s1', help='Directory to the source data')

    parser.add_argument('--base_model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of base model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of test model for resuming')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=3, help='epoch interval to evaluate model')

    #optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')

    parser.add_argument('--base_lr', type=float, default=0.001, help='base train learning rate [default: 0.001]')
    parser.add_argument('--base_weight_decay', type=float, default=0.0001, help='weight decay for regularization')
    parser.add_argument('--base_decay_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--base_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    parser.add_argument('--incre_lr', type=float, default=0.001, help='incre train learning rate [default: 0.001]')
    parser.add_argument('--incre_weight_decay', type=float, default=0.0001, help='weight decay for regularization')
    parser.add_argument('--incre_decay_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--incre_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--uncertain_t', type=float, default=0.0065, help='The uncertain threshold for classes')

    parser.add_argument('--joint_lr', type=float, default=0.001, help='joint train learning rate [default: 0.001]')
    parser.add_argument('--joint_weight_decay', type=float, default=0.0001, help='weight decay for regularization')
    parser.add_argument('--joint_decay_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--joint_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')

    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64,64], [64,64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]', help='DGCNN MLP (following stacked Edgeconv) widths')
    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.pc_in_dim = len(args.pc_attribs)

    # Start trainer
    if args.phase=='jointtrain':
        args.log_dir = args.save_path + 'log_joint_%s_cv%d' % (args.dataset, args.cvfold)
        from runs.joint_train import train
        train(args)
    elif args.phase=='freeze_and_add':
        args.log_dir = args.save_path + 'log_freeze_and_add_%s_cv%d_tasks%s' % (args.dataset, args.cvfold, args.tasks)
        from runs.freeze_and_add import freeze_and_add
        freeze_and_add(args)
    elif args.phase=='finetuning':
        args.log_dir = args.save_path + 'log_finetuning_%s_cv%d_tasks%s' % (args.dataset, args.cvfold, args.tasks)
        from runs.finetuning import finetuning
        finetuning(args)
    elif args.phase=='increEWC':
        args.log_dir = args.save_path + 'log_ewc_%s_cv%d_tasks%s' % (args.dataset, args.cvfold, args.tasks)
        from runs.train_ewc import train_EWC
        train_EWC(args)
    elif args.phase=='increLwF':
        args.log_dir = args.save_path + 'log_lwf_%s_cv%d_tasks%s' % (args.dataset, args.cvfold, args.tasks)
        from runs.train_lwF import train_LwF
        train_LwF(args)
    elif args.phase=='increOurs':
        args.log_dir = args.save_path + 'log_ours_%s_cv%d_tasks%s' % (args.dataset, args.cvfold, args.tasks)
        from runs.train_ours import train
        train(args)
    elif args.phase=='baseeval':
        args.log_dir = args.model_checkpoint_path
        from runs.eval_base import eval_base
        eval_base(args)
    elif args.phase=='jointeval' or args.phase=='increeval' or args.phase=='othereval':
        args.log_dir = args.model_checkpoint_path
        from runs.eval import eval
        eval(args)
    elif args.phase=='increeval_multi':
        args.log_dir = args.model_checkpoint_path
        from runs.eval_multi_steps import eval
        eval(args)
    else:
        raise ValueError('Please set correct phase.')