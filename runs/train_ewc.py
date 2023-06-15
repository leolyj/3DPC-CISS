import os
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dgcnn_seg import DGCNNSeg, Classifer
from dataloaders.loader import MyDataset
from utils.logger import init_logger
from utils.checkpoint_util import save_train_checkpoint, load_trained_checkpoint, save_classifer_checkpoint

def get_regularizer(model, model_old, device, old_state):
    resume = False
    name = 'ewc'
    if old_state is not None:
        if name != old_state['name']:
            print(f"Warning: The regularizer you passed {name}"
                  f" is different from the state one {old_state['name']}")
        resume = True

    if name is None:
        return None
    elif name == 'ewc':
        fisher = old_state["fisher"] if resume else None
        return EWC(model, model_old, device, fisher=fisher,
                   alpha=0.9,
                   normalize= not False)
    else:
        raise NotImplementedError

def normalize_fn(mat):
    EPS = 1e-8
    return (mat - mat.min()) / (mat.max() - mat.min() + EPS)

class Regularizer:
    def update(self):
        """ Stub method """
        raise NotImplementedError

    def penalty(self):
        """ Stub method """
        raise NotImplementedError

    def state_dict(self):
        """ Stub method """
        raise NotImplementedError

    def load_state_dict(self, state):
        """ Stub method """
        raise NotImplementedError

class EWC(Regularizer):
    def __init__(self, model, model_old, device='cuda:0', fisher=None, alpha=0.9, normalize=True):

        self.model = model
        self.device = device
        self.alpha = alpha
        self.normalize = normalize

        # store old model for penalty step
        if model_old is not None:
            self.model_old = model_old
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
        else:
            self.penalize = False

        # make the fisher matrix for the estimate of parameter importance
        # store the old fisher matrix (if exist) for penalize step
        if fisher is not None:  # initialize the old Fisher Matrix
            self.fisher_old = fisher
            self.fisher = {}
            for key, par in self.fisher_old.items():
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = normalize_fn(par) if normalize else par
                self.fisher_old[key] = self.fisher_old[key].to(device)
                self.fisher[key] = torch.clone(par).to(device)
        else:  # initialize a new Fisher Matrix and don't penalize
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}

        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

    def update(self):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in self.model.named_parameters():
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n])

    def penalty(self):
        if not self.penalize:
            return 0.
        else:
            loss = 0.
            for n, p in self.model.named_parameters():
                if n in self.model_old_dict and p.requires_grad:
                    loss += (self.fisher_old[n].cuda() * (p.cuda() - self.model_old_dict[n].cuda()) ** 2).sum()
            return loss

    def get(self):
        return self.fisher  # return the new Fisher matrix

    def state_dict(self):
        state = {"name": "ewc", "fisher": self.fisher, "alpha": self.alpha,}
        return state

    def load_state_dict(self, state):
        self.fisher = state["fisher"]
        for k,p in self.fisher.items():
            self.fisher[k] = p.to(self.device)
        self.alpha = state["alpha"]

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

    if eval_mode == 'eval_Base': # Include the background classes for eval_base or eval_incre
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

def train_EWC(args):
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
        NUM_BASE_CLASSES = len(BASE_CLASSES)
        NUM_ALL_CLASSES = DATASET.classes
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
        VALID_SET = []
        lines = open(os.path.join(os.path.dirname(args.data_path[:-14]), 'scannetv2_val.txt')).readlines()
        for line in lines:
            VALID_SET.append(line.strip('\n'))

        BASE_CLASSES = [0] + DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_BASE_CLASSES = len(BASE_CLASSES)
        NUM_ALL_CLASSES = DATASET.classes
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    INCRE = int((args.tasks).split('-')[1]) # Incre x classes for 1 step
    STEP = int(len(INCRE_CLASSES) / INCRE) + 1 # Total steps
    CLASS2SCAN = DATASET.class2scans
    Old_Regularizer = None

    for step in range(0, STEP):
        SAMPLE_CLASS = BASE_CLASSES.copy()  # intital as old class
        if step == 0: # Train the model for base classes
            CLASSES = BASE_CLASSES
            CURRENT_CLASS = BASE_CLASSES
            LOG_DIR = args.log_dir + '/base_model'
        else: # Train the model for incremental classes
            LOG_DIR = args.log_dir + '/incre_model'

            if step==1 and int((args.tasks).split('-')[1]) == (NUM_ALL_CLASSES-NUM_BASE_CLASSES):
                CLASSES = INCRE_CLASSES
                CURRENT_CLASS = INCRE_CLASSES
                SAMPLE_CLASS.extend(CLASSES)
            else: # multi-steps increments
                SAMPLE_CLASS.extend(INCRE_CLASSES[:step*INCRE])
                CLASSES = INCRE_CLASSES[(step - 1) * INCRE:step * INCRE]
                CURRENT_CLASS = INCRE_CLASSES[(step - 1) * INCRE:step * INCRE]

        TRAIN_DATASET = MyDataset(args.data_path, CLASSES, CURRENT_CLASS, CLASS2SCAN, step, mode='train', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        VALID_DATASET = MyDataset(args.data_path, CLASSES, CURRENT_CLASS, CLASS2SCAN, step, mode='test', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)
        logger = init_logger(LOG_DIR, args)
        logger.cprint('=== Train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(
                                                         CLASSES, len(TRAIN_DATASET), len(VALID_DATASET)))

        TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                                  drop_last=True)

        VALID_LOADER = DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                  drop_last=True)

        WRITER = SummaryWriter(log_dir=LOG_DIR)

        if step == 0: # Train the base model on the base classes
            logger.cprint('*******************Training the Model on Base Classes: %d Classes | Step %d'
                          '*******************' % (NUM_BASE_CLASSES,step))
            # Init base model and optimizer
            model = DGCNNSeg(args)
            regularizer = get_regularizer(model, None, 'cuda:0', None)
            classifer = Classifer(NUM_BASE_CLASSES + 1)
            print(model)
            if torch.cuda.is_available():
                model.cuda()
                classifer.cuda()

            optimizer_base = optim.Adam([{'params': model.parameters(), 'lr': args.base_lr}, \
                                   {'params': classifer.parameters(), 'lr': args.base_lr}], \
                                    weight_decay=args.base_weight_decay)

            # Set learning rate scheduler
            lr_scheduler_base = optim.lr_scheduler.StepLR(optimizer_base, step_size=args.base_decay_size, gamma=args.base_gamma)

            # train
            best_iou = 0
            global_iter = 0
            for epoch in range(args.n_epochs):
                for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    _, logits = model(ptclouds)
                    cls_logits = classifer(logits)

                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(cls_logits, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1
                    else:
                        loss = F.cross_entropy(cls_logits, labels)

                    # Loss backwards and optimizer updates
                    optimizer_base.zero_grad()
                    loss.backward()
                    regularizer.update()  # update the regularizer
                    l_reg = 1.0 * regularizer.penalty()
                    if l_reg != 0.0: l_reg.backward()
                    optimizer_base.step()

                    WRITER.add_scalar('Train/loss', loss, global_iter)
                    logger.cprint('=====[Train] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, batch_idx, loss.item()))
                    global_iter += 1

                lr_scheduler_base.step()

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
                            classifer.eval()

                            _, logits = model(ptclouds)
                            logits_cls = classifer(logits)

                            if args.dataset == 'scannet':
                                loss = F.cross_entropy(logits_cls, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1 : unannotated
                            else:
                                loss = F.cross_entropy(logits_cls, labels)

                            l_reg = regularizer.penalty()
                            loss += l_reg

                            # 　Compute predictions
                            _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                            pred_total.append(preds.cpu().detach())

                            WRITER.add_scalar('Valid/loss', loss, global_iter)
                            logger.cprint(
                                '=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

                    pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
                    gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
                    accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES, 'eval_Base', BASE_CLASSES, logger, args.dataset)
                    logger.cprint('===== EPOCH [%d]: Accuracy: %f | mIoU: %f =====\n' % (epoch, accuracy, mIoU))
                    WRITER.add_scalar('Valid/overall_accuracy', accuracy, global_iter)
                    WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

                    if mIoU > best_iou:
                        best_iou = mIoU
                        logger.cprint('*******************Model Saved*******************')
                        save_train_checkpoint(model, args.log_dir, 'best_base_model')
                        save_classifer_checkpoint(classifer, args.log_dir, 'best_base_model')

            logger.cprint('*******************End of Training the Base Model*******************')
            logger.cprint('*******************Eval the Base Model*******************')
            pred_end_total = []
            gt_end_total = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_end_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    model.eval()
                    classifer.eval()
                    _, logits = model(ptclouds)
                    logits_cls = classifer(logits)

                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(logits_cls, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1 : unannotated
                    else:
                        loss = F.cross_entropy(logits_cls, labels)

                    l_reg = 1.0 * regularizer.penalty()
                    loss += l_reg

                    # 　Compute predictions
                    _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                    pred_end_total.append(preds.cpu().detach())

                    WRITER.add_scalar('Valid/loss', loss, global_iter)
                    logger.cprint('=====[Valid] End | Iter: %d | Loss: %.4f =====' % (i, loss.item()))

            pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
            accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES,
                                                           'eval_Base', BASE_CLASSES, logger, args.dataset)
            logger.cprint('===== Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
            logger.cprint('*******************Model Saved*******************')
            save_train_checkpoint(model, args.log_dir, 'end_base_model')
            save_classifer_checkpoint(classifer, args.log_dir, 'end_base_model')
            Old_Regularizer = regularizer

            WRITER.close()
        elif step > 0 and STEP==2:
            logger.cprint('*******************Training the Model on Incremental Classes: %d Classes | Total 2 Tasks | Step %d'
                          '*******************' % (len(INCRE_CLASSES), step))
            # train the model for incremental classes: step > 0 and tasks == 2
            # Init the old model
            model_old = DGCNNSeg(args)
            model_old = load_trained_checkpoint(model_old, args.base_model_checkpoint_path,
                                                'end_base_model_checkpoint.tar')  # Load the last base model
            # Init the new model
            model_new = DGCNNSeg(args)
            classifer_new = Classifer(num_classes=NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1)
            model_new = load_trained_checkpoint(model_new, args.base_model_checkpoint_path,
                                                    'end_base_model_checkpoint.tar')  # Load the last base model
            print('New model:\n', model_new)

            new_regularizer = get_regularizer(model_new, model_old, 'cuda:0', Old_Regularizer.state_dict())

            # Freeze the old model
            for param in model_old.parameters():
                param.requires_grad = False

            if torch.cuda.is_available():
                model_old.cuda()
                model_new.cuda()
                classifer_new.cuda()

            optimizer_incre = optim.Adam([{'params': model_new.parameters(), 'lr': args.incre_lr},
                                    {'params': classifer_new.parameters(), 'lr': args.incre_lr}], \
                                   weight_decay=args.incre_weight_decay)

            # Set learning rate scheduler
            lr_scheduler_incre = optim.lr_scheduler.StepLR(optimizer_incre, step_size=args.incre_decay_size, gamma=args.incre_gamma)

            # train
            best_iou = 0
            global_iter = 0
            for epoch in range(args.n_epochs):
                for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()
                    model_old.eval()

                    _, logits_new = model_new(ptclouds)
                    logits_new_cls = classifer_new(logits_new)

                    loss_incre = F.cross_entropy(logits_new_cls, labels.cuda())
                    del labels, logits_new

                    # Loss backwards and optimizer updates
                    optimizer_incre.zero_grad()
                    loss_incre.backward()

                    # Update the new regularizer
                    new_regularizer.update()
                    l_reg = 1.0 * new_regularizer.penalty()
                    l_reg.backward()
                    optimizer_incre.step()

                    WRITER.add_scalar('Train/loss', loss_incre, global_iter)
                    logger.cprint(
                        '=====[Train] Epoch: %d | Iter: %d | Loss: %.8f =====' % (epoch, batch_idx, loss_incre.item()))
                    global_iter += 1

                lr_scheduler_incre.step()

                if (epoch + 1) % args.eval_interval == 0:
                    pred_total = []
                    gt_total = []
                    with torch.no_grad():
                        for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                            gt_total.append(labels.detach())
                            if torch.cuda.is_available():
                                ptclouds = ptclouds.cuda()
                                labels = labels.cuda()
                            model_new.eval()
                            classifer_new.eval()

                            _, logits_new = model_new(ptclouds)
                            logits_new_cls = classifer_new(logits_new)
                            loss_incre = F.cross_entropy(logits_new_cls, labels)
                            l_reg = 1.0 * new_regularizer.penalty()
                            loss_incre += l_reg

                            # 　Compute predictions
                            _, preds = torch.max(logits_new_cls.detach(), dim=1, keepdim=False)
                            pred_total.append(preds.cpu().detach())

                            WRITER.add_scalar('Valid/loss', loss_incre, global_iter)
                            logger.cprint(
                                '=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss_incre.item()))

                    pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
                    gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
                    _, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1,
                                                            NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1, 'eval_Incre', range(1, len(INCRE_CLASSES)+1), logger, args.dataset)
                    logger.cprint('===== EPOCH [%d]: mIoU: %f =====\n' % (epoch, mIoU))
                    WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

                    if mIoU > best_iou:
                        best_iou = mIoU
                        logger.cprint('*******************Model Saved*******************')
                        save_train_checkpoint(model_new, args.log_dir, 'best_incre_model')
                        save_classifer_checkpoint(classifer_new, args.log_dir, 'best_incre_model')

            logger.cprint('*******************End of Training the Incre Model*******************')
            logger.cprint('*******************Eval the Incre Model*******************')
            pred_end_total = []
            gt_end_total = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_end_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()
                    model_new.eval()
                    classifer_new.eval()

                    _, logits_new = model_new(ptclouds)
                    logits_new_cls = classifer_new(logits_new)
                    loss_incre = F.cross_entropy(logits_new_cls, labels)
                    l_reg = 1.0 * new_regularizer.penalty()
                    loss_incre += l_reg

                    # 　Compute predictions
                    _, preds = torch.max(logits_new_cls.detach(), dim=1, keepdim=False)
                    pred_end_total.append(preds.cpu().detach())

                    WRITER.add_scalar('Valid/loss', loss_incre, global_iter)
                    logger.cprint('=====[Valid] End | Iter: %d | Loss: %.4f =====' % (i, loss_incre.item()))

            pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
            _, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1,
                                                    NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1, 'eval_Incre',
                                                    range(1, len(INCRE_CLASSES)+1), logger, args.dataset)
            logger.cprint('===== End Model | mIoU: %f =====\n' % (mIoU))
            logger.cprint('*******************Model Saved*******************')
            save_train_checkpoint(model_new, args.log_dir, 'end_incre_model')
            save_classifer_checkpoint(classifer_new, args.log_dir, 'end_incre_model')

            WRITER.close()
        else: # For Multi-step Increments -> Similar to train_ours.py
            return NotImplementedError('Refer to train_ours.py for Implementation!')