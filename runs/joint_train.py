""" Joint Training for 3D Point Cloud semantic Segmentation """
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dgcnn_seg_joint import DGCNNSeg
from dataloaders.joint_loader import MyDataset
from utils.logger import init_logger
from utils.checkpoint_util import save_train_checkpoint

def metric_evaluate(predicted_label, gt_label, NUM_CLASS, class_id, logger, dataset):
    '''Caluate the mIoU for Classes'''
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]
    if isinstance(class_id, int):
        class_id = list([class_id])

    for i in range(gt_label.size()[0]):
        pred_pc = predicted_label[i]
        gt_pc = gt_label[i]

        for j in range(gt_pc.shape[0]):
            gt_l = int(gt_pc[j])
            pred_l = int(pred_pc[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    OA = sum(true_positive_classes)/float(sum(positive_classes))

    IoU_list = []
    for i in range(NUM_CLASS):
        iou_class = true_positive_classes[i] / float(
            gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        IoU_list.append(iou_class)
        logger.cprint('Class_%d IoU: %f' % (i, iou_class))

    if dataset == 's3dis':
        mean_IoU = np.sum(IoU_list) / len(class_id)
    else:
        mean_IoU = np.sum(IoU_list[1:]) / len(class_id)

    return OA, mean_IoU, IoU_list

def train(args):
    # Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
        VALID_SET = 'Area_5'

        BASE_CLASSES = DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_ALL_CLASSES = DATASET.classes
        STEP = 2
        CLASSES = BASE_CLASSES + INCRE_CLASSES
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
        VALID_SET = []
        lines = open(os.path.join(os.path.dirname(args.data_path[:-14]), 'scannetv2_val.txt')).readlines()
        for line in lines:
            VALID_SET.append(line.strip('\n'))

        BASE_CLASSES = DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_ALL_CLASSES = DATASET.classes
        STEP = 2
        CLASSES = [0] + BASE_CLASSES + INCRE_CLASSES
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    TRAIN_DATASET = MyDataset(args.data_path, CLASSES, STEP, mode='train', valid_set=VALID_SET,
                                        num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                        pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    VALID_DATASET = MyDataset(args.data_path, CLASSES, STEP, mode='test', valid_set=VALID_SET,
                                        num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                        pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    LOG_DIR = args.log_dir
    logger = init_logger(LOG_DIR, args)
    logger.cprint('=== Train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(
                                                         CLASSES, len(TRAIN_DATASET), len(VALID_DATASET)))

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                                  drop_last=True)
    VALID_LOADER = DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                  drop_last=True)
    WRITER = SummaryWriter(log_dir=LOG_DIR)

    logger.cprint('*******************Training the Joint Model*******************')
    # Initial the joint model and optimizer
    model = DGCNNSeg(args, num_classes=NUM_ALL_CLASSES)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    optimizer_joint = optim.Adam([{'params': model.encoder.parameters(), 'lr': args.base_lr}, \
                                   {'params': model.segmentor.parameters(), 'lr': args.base_lr}], \
                                    weight_decay=args.base_weight_decay)

    # Set learning rate scheduler
    lr_scheduler_joint = optim.lr_scheduler.StepLR(optimizer_joint, step_size=args.base_decay_size, gamma=args.base_gamma)

    # train
    best_iou = 0
    global_iter = 0
    for epoch in range(args.n_epochs):
        for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
            if torch.cuda.is_available():
                ptclouds = ptclouds.cuda()
                labels = labels.cuda()

            logits = model(ptclouds)

            if args.dataset == 'scannet':
                loss = F.cross_entropy(logits, labels, ignore_index=0) # For scannet ignore labels 0 -> 1 : unannotated
            else:
                loss = F.cross_entropy(logits, labels)

            # Loss backwards and optimizer updates
            optimizer_joint.zero_grad()
            loss.backward()
            optimizer_joint.step()

            WRITER.add_scalar('Train/loss', loss, global_iter)
            logger.cprint('=====[Train] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, batch_idx, loss.item()))
            global_iter += 1

        lr_scheduler_joint.step()

        if (epoch+1) % args.eval_interval == 0:
            pred_total = []
            gt_total = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    model.eval()
                    logits = model(ptclouds)
                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(logits, labels, ignore_index=0)  # For scannet ignore labels 0
                    else:
                        loss = F.cross_entropy(logits, labels)

                    # 　Compute predictions
                    _, preds = torch.max(logits.detach(), dim=1, keepdim=False)
                    pred_total.append(preds.cpu().detach())

                    WRITER.add_scalar('Valid/loss', loss, global_iter)
                    logger.cprint('=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

            pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
            accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_ALL_CLASSES, CLASSES, logger, dataset=args.dataset)
            logger.cprint('===== EPOCH [%d]: Accuracy: %f | mIoU: %f =====\n' % (epoch, accuracy, mIoU))
            WRITER.add_scalar('Valid/overall_accuracy', accuracy, global_iter)
            WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

            if mIoU > best_iou:
                best_iou = mIoU
                logger.cprint('*******************Model Saved*******************')
                save_train_checkpoint(model, args.log_dir, 'best_joint_model')

    WRITER.close()
