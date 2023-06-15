"""Class Split (Base and Incre) for ScanNet Dataset"""
import os
import glob
import numpy as np
import pickle

class ScanNetDataset(object):
    def __init__(self, cvfold, tasks,  data_path):
        self.data_path = data_path
        self.classes = 21
        # self.class2type = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk', 6:'bed', 7:'bookshelf',
        #                    8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet', 12:'curtain', 13:'counter', 14:'door',
        #                    15:'window', 16:'shower curtain', 17:'refrigerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
        class_names = open(os.path.join(os.path.dirname(data_path[:-14]), 'meta', 'scannet_classnames.txt')).readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()

        if cvfold == 0:
            if tasks == '19-1':
                self.base = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet']
                self.incre = ['otherfurniture']
            elif tasks == '17-3':
                self.base = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator']
                self.incre = ['picture', 'cabinet', 'otherfurniture']
            elif tasks == '15-5':
                self.base = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window']
                self.incre = ['shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
            elif tasks == '15-1':
                self.base = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window']
                self.incre = ['shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
            elif tasks == "Joint": # For Joint training, combine the base+novel classes, random
                self.base = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
                self.incre = []
            else:
                NotImplementedError('Unknown tasks mode!')
        elif cvfold == 1:
            if tasks == '19-1':
                self.base = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'floor', 'otherfurniture', 'picture', 'refrigerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall']
                self.incre = ['window']
            elif tasks == '17-3':
                self.base = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'floor', 'otherfurniture', 'picture', 'refrigerator', 'shower curtain', 'sink', 'sofa', 'table']
                self.incre = ['toilet', 'wall', 'window']
            elif tasks == '15-5':
                self.base = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'floor', 'otherfurniture', 'picture', 'refrigerator', 'shower curtain', 'sink']
                self.incre = ['sofa', 'table', 'toilet', 'wall', 'window']
            elif tasks == '15-1':
                self.base = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'floor', 'otherfurniture', 'picture', 'refrigerator', 'shower curtain', 'sink']
                self.incre = ['sofa', 'table', 'toilet', 'wall', 'window']
            elif tasks == "Joint": # For Joint training, combine the base+novel classes, random
                self.base = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'floor', 'otherfurniture', 'picture', 'refrigerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall', 'window']
                self.incre = []
            else:
                NotImplementedError('Unknown tasks mode!')
        else:
            raise NotImplementedError('Unknown cvfold (%s). [Options: 0,1]' %cvfold)

        self.base_classes = [self.type2class[i] for i in self.base]
        if len(self.incre) != 0:
            self.incre_classes = [self.type2class[i] for i in self.incre]
            print('base_class:{0}'.format(self.base_classes))
            print('incre_class:{0}'.format(self.incre_classes))
        else:  # For Joint Training
            self.incre_classes = []
            print('all_class:{0}'.format(self.base_classes))

        self.class2scans = self.get_class2scans()

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_path, 'class2scans.pkl')
        if os.path.exists(class2scans_file):
            # load class2scans (dictionary)
            with open(class2scans_file, 'rb') as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = .05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k:[] for k in range(self.classes)}

            for file in glob.glob(os.path.join(self.data_path, 'data', '*.npy')):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:,6].astype(np.int)
                classes = np.unique(labels)
                print('{0} | shape: {1} | classes: {2}'.format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    # if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0]*min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print('==== class to scans mapping is done ====')
            for class_id in range(self.classes):
                print('\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}'.format(
                          class_id,  min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])))

            with open(class2scans_file, 'wb') as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans


if __name__ == '__main__':
    dataset = ScanNetDataset(0, '19-1', '../datasets/ScanNet/blocks_bs1_s1')
    dataset1 = ScanNetDataset(0, '17-3', '../datasets/ScanNet/blocks_bs1_s1')
    dataset2 = ScanNetDataset(0, '15-5', '../datasets/ScanNet/blocks_bs1_s1')
    dataset3 = ScanNetDataset(0, '15-1', '../datasets/ScanNet/blocks_bs1_s1')
    dataset4 = ScanNetDataset(0, 'Joint', '../datasets/ScanNet/blocks_bs1_s1')
    # -----------------------------------------------------------------------
    dataset5 = ScanNetDataset(1, '19-1', '../datasets/ScanNet/blocks_bs1_s1')
    dataset6 = ScanNetDataset(1, '17-3', '../datasets/ScanNet/blocks_bs1_s1')
    dataset7 = ScanNetDataset(1, '15-5', '../datasets/ScanNet/blocks_bs1_s1')
    dataset8 = ScanNetDataset(1, '15-1', '../datasets/ScanNet/blocks_bs1_s1')
    dataset9 = ScanNetDataset(1, 'Joint', '../datasets/ScanNet/blocks_bs1_s1')