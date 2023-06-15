"""Data Loader for Joint Training"""

import os
import random
import math
import glob
import numpy as np
import transforms3d

import torch
from torch.utils.data import Dataset

def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      classes, step):
    if isinstance(classes, int):
        classes = list([classes])
    else:
        classes = list(classes)

    data = np.load(os.path.join(data_path, 'data', '%s.npy' %scan_name))
    N = data.shape[0] #number of points in this scan

    # Sampled points index
    sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point))

    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(np.int)

    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)

    if step == 0:
        groundtruth = np.zeros_like(labels)
        for i, label in enumerate(labels):
            if label in classes:
                groundtruth[i] = classes.index(label)+1
    else:
        groundtruth = np.zeros_like(labels)
        for i, label in enumerate(labels):
            if label in classes:
                groundtruth[i] = classes[classes.index(label)]

    return ptcloud, groundtruth

def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P

################################################  Dataset ################################################
class MyDataset(Dataset):
    def __init__(self, data_path, classes, step, mode='train', valid_set='Area_5',
                 num_point=2048, pc_attribs='xyz', pc_augm=False, pc_augm_config=None):
        super(MyDataset).__init__()
        self.data_path = data_path
        self.classes = classes
        self.step = step
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        all_block_names = []
        for file in glob.glob(os.path.join(self.data_path, 'data', '*.npy')):
            all_block_names.append(os.path.basename(file)[:-4])

        test_block_names = []
        for test_file in all_block_names:
            if isinstance(valid_set, list):
                file = str(test_file.split('_')[0] + '_' + test_file.split('_')[1])
                if file in valid_set:
                    test_block_names.append(test_file)
            else:
                if test_file.split('_')[0] + '_' + test_file.split('_')[1] == valid_set:
                    test_block_names.append(test_file)

        if mode == 'train':
            self.block_names = list(set(all_block_names) - set(test_block_names))
        elif mode == 'test':
            self.block_names = list(set(test_block_names))
        else:
            raise NotImplementedError('Mode is unknown!')

        print('[Training Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]

        ptcloud, label = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                           self.pc_augm_config, block_name, self.classes, self.step)

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64))

class MyTestDataset(Dataset):
    def __init__(self, data_path, classes, step, test_set='Area_5', num_point=2048, pc_attribs='xyz', pc_augm=False, pc_augm_config=None):
        super(MyDataset).__init__()
        self.data_path = data_path
        self.classes = classes
        self.step = step
        self.num_point = num_point
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        test_block_names = []
        if isinstance(test_set, list):
            for test in test_set:
                for test_file in glob.glob(os.path.join(self.data_path, 'data', test +'_*.npy')):
                    test_block_names.append(os.path.basename(test_file)[:-4])
        else:
            for test_file in glob.glob(os.path.join(self.data_path, 'data', test_set+'_*.npy')):
                test_block_names.append(os.path.basename(test_file)[:-4])

        self.block_names = list(set(test_block_names))
        print('[Test Dataset] Num_blocks: {0}'.format(len(self.block_names)))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]

        ptcloud, label = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                           self.pc_augm_config, block_name, self.classes, self.step)

        return block_name, torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64))