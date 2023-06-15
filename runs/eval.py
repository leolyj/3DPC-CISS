""" Evaluate the joint model and 2-step incremental model of class-incremental 3D point cloud semantic segmentation """
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.dgcnn_seg import DGCNNSeg, Classifer
from models.dgcnn_seg_joint import DGCNNSeg as DGCNNSeg_Joint
from dataloaders.loader import MyTestDataset
from dataloaders.joint_loader import MyTestDataset as MyTestDataset_Joint
import torch.nn.functional as F
from utils.logger import init_logger
from utils.checkpoint_util import load_trained_checkpoint

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

    if dataset == 'scannet':
        mean_IoU = np.sum(IoU_list[1:]) / (len(class_id) - 1) # Exclude the ignore labels
    else:
        mean_IoU = np.sum(IoU_list) / len(class_id)

    return OA, mean_IoU, IoU_list

def eval(args):
    logger = init_logger(args.log_dir, args)

    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
        TEST_SET = 'Area_5'
        BASE_CLASSES = DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
        print('test classes:', TEST_CLASSES)
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
        TEST_SET = []
        lines = open(os.path.join(os.path.dirname('./datasets/ScanNet/'), 'scannetv2_val.txt')).readlines()
        for line in lines:
            TEST_SET.append(line.strip('\n'))

        BASE_CLASSES = [0] + DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
        print('test classes:', TEST_CLASSES)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    # Init dataset, dataloader
    if args.tasks == "Joint":
        TOTAL_STEP = 2
    else:
        INCRE = int((args.tasks).split('-')[1])
        TOTAL_STEP = int(len(INCRE_CLASSES) / INCRE) + 1

    if args.phase != 'jointeval':
        TEST_DATASET = MyTestDataset(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                     num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                     pc_augm=False, pc_augm_config=None)

        TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                 drop_last=True)
        model = DGCNNSeg(args)
        if TOTAL_STEP == 2:
            classifer_base = Classifer(num_classes = len(BASE_CLASSES) + 1)
            model = load_trained_checkpoint(model, args.model_checkpoint_path, name='end_incre_model_checkpoint.tar')
            if args.phase == 'increeval':
                classifer_incre = Classifer(num_classes=len(BASE_CLASSES) + len(INCRE_CLASSES))
                classifer_incre = load_trained_checkpoint(classifer_incre, args.model_checkpoint_path, name='end_incre_model_classifer_checkpoint.tar')
                classifer_weights = (classifer_incre.classifer_weights).cuda()
            else:
                classifer_incre = Classifer(num_classes=len(INCRE_CLASSES) + 1)
                classifer_base = load_trained_checkpoint(classifer_base, args.model_checkpoint_path, name='end_base_model_classifer_checkpoint.tar')
                classifer_incre = load_trained_checkpoint(classifer_incre, args.model_checkpoint_path, name='end_incre_model_classifer_checkpoint.tar')
                classifer_weights_base = torch.empty_like(classifer_base.classifer_weights).copy_(classifer_base.classifer_weights.detach())
                classifer_weights = classifer_incre.enroll_weights(classifer_weights_base)
        else:
            return NotImplementedError('See eval_multi_steps.py for implementation!')

        model.cuda()
        pred_total = []
        gt_total = []
        with torch.no_grad():
            for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
                labels = labels - int(1)
                gt_total.append(labels.detach())

                if torch.cuda.is_available():
                    ptclouds = ptclouds.cuda()
                    labels = labels.cuda()
                model.eval()
                _, logits = model(ptclouds)
                logits_new = classifer_incre(logits, classifer_weights)
                loss = F.cross_entropy(logits_new, labels)

                # Compute predictions
                _, preds = torch.max(logits_new.detach(), dim=1, keepdim=False)
                pred_total.append(preds.cpu().detach())

                logger.cprint(
                    '=====[Test] Iter: %d | Loss: %.4f =====' % (i, loss.item()))

        pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
        gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)

        accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, len(TEST_CLASSES), TEST_CLASSES, logger, args.dataset)

        logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
    else:
        TEST_DATASET = MyTestDataset_Joint(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                     num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                     pc_augm=False, pc_augm_config=None)

        TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                 drop_last=True)
        model = DGCNNSeg_Joint(args, num_classes =len(TEST_CLASSES))
        model = load_trained_checkpoint(model, args.model_checkpoint_path, name='best_joint_model_checkpoint.tar')

        model.cuda()
        pred_total = []
        gt_total = []
        with torch.no_grad():
            for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
                gt_total.append(labels.detach())

                if torch.cuda.is_available():
                    ptclouds = ptclouds.cuda()
                    labels = labels.cuda()
                model.eval()
                logits = model(ptclouds)
                loss = F.cross_entropy(logits, labels)

                # Compute predictions
                _, preds = torch.max(logits.detach(), dim=1, keepdim=False)
                pred_total.append(preds.cpu().detach())

                logger.cprint(
                    '=====[Test] Iter: %d | Loss: %.4f =====' % (i, loss.item()))

        pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
        gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)

        accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, len(TEST_CLASSES), TEST_CLASSES, logger, args.dataset)

        logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
