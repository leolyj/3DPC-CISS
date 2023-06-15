""" Evaluate the multi-steps incremental model of class-incremental 3D point cloud semantic segmentation """
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.dgcnn_seg import DGCNNSeg, Classifer
from dataloaders.loader import MyTestDataset
import torch.nn.functional as F
from utils.logger import init_logger
from utils.checkpoint_util import load_trained_checkpoint

def metric_evaluate(predicted_label, gt_label, NUM_CLASS, TOTAL_CLASS, eval_mode, class_id, logger, dataset):
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

    if eval_mode == 'eval_Base':
        IoU_list = []
        for i in range(NUM_CLASS):
            iou_class = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
            IoU_list.append(iou_class)
        logger.cprint('background class IoU: %f' % (IoU_list[0]))
        for j in range(TOTAL_CLASS):
            if j not in class_id:
                logger.cprint('Class_%d IoU: X' % j)
            else:
                ind = class_id.index(j)
                logger.cprint('Class_%d IoU: %f' % (class_id[ind], IoU_list[ind+1]))
        if dataset == 'scannet':
            mean_IoU = np.array(IoU_list[2:]).mean()  # Caluate the Mean IoU for classes exclude background and unannotated
        else:
            mean_IoU = np.array(IoU_list[1:]).mean()  # Caluate the Mean IoU for classes exclude background
    elif eval_mode == 'eval_Incre':
        IoU_list = []
        for i in range(NUM_CLASS):
            if i in class_id:
                iou_class = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                IoU_list.append(iou_class)
                logger.cprint('Class_%d IoU: %f' % (i, iou_class))
            else:
                IoU_list.append(0.0)
                logger.cprint('Class_%d IoU: X' % i)
        mean_IoU = np.sum(IoU_list)/len(class_id)
    else:
        return NotImplementedError('Unknown eval mode!')

    return OA, mean_IoU, IoU_list

# -------------------------------------------------------------------------------
def eval(args):
    logger = init_logger(args.log_dir, args)

    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
        TEST_SET = 'Area_5'
        BASE_CLASSES = DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_ALL_CLASSES = DATASET.classes
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
        TEST_SET = []
        lines = open(os.path.join(os.path.dirname('./datasets/ScanNet/'), 'scannetv2_val.txt')).readlines()
        for line in lines:
            TEST_SET.append(line.strip('\n'))

        BASE_CLASSES = [0] + DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_ALL_CLASSES = DATASET.classes
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    # Init dataset, dataloader
    INCRE = int((args.tasks).split('-')[1])
    TOTAL_STEP = int(len(INCRE_CLASSES) / INCRE) + 1

    for step in range(0, TOTAL_STEP - 1):
        if step != TOTAL_STEP - 2:
            TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES[:step+1]
            print('test classes:', TEST_CLASSES)
        else:
            TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
            print('test classes:', TEST_CLASSES)

        TEST_DATASET = MyTestDataset(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                         num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                         pc_augm=False, pc_augm_config=None)
        TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                     drop_last=True)
        model = DGCNNSeg(args)

        # --------------------------------------------------------------------------
        if args.phase == 'increeval_multi':
            if step == TOTAL_STEP - 2:
                classifer_incre = Classifer(num_classes=len(BASE_CLASSES) + len(INCRE_CLASSES))
                classifer_incre = load_trained_checkpoint(classifer_incre, args.model_checkpoint_path, name='end_incre_step_'+str(step+1)+'_model'+'_classifer_checkpoint.tar')
            else:
                classifer_incre = Classifer(num_classes = len(TEST_CLASSES) + 1)
                classifer_incre = load_trained_checkpoint(classifer_incre, args.model_checkpoint_path, name='end_incre_step_' + str(step+1) + '_model' + '_classifer_checkpoint.tar')

            model = load_trained_checkpoint(model, args.model_checkpoint_path, name='end_incre_step_'+str(step+1)+'_model'+'_checkpoint.tar')
            classifer_weights = (classifer_incre.classifer_weights).cuda()

            model.cuda()
            pred_total = []
            gt_total = []
            with torch.no_grad():
                for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
                    if step == TOTAL_STEP - 2:
                        labels = labels - int(1)
                    else:
                        labels = labels
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

            if step != TOTAL_STEP - 2:
                accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, len(TEST_CLASSES) + 1, NUM_ALL_CLASSES,
                                                               'eval_Base', TEST_CLASSES, logger, args.dataset)
            else:
                accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, len(TEST_CLASSES), NUM_ALL_CLASSES,
                                                               'eval_Incre', TEST_CLASSES, logger, args.dataset)

            logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
        else:
            return NotImplementedError('Unknown eval mode!')
